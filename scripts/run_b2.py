"""SPRINT Block B2 (14 Nov 2026) — steering along the saved C1-only gov-vs-social direction. New file.
  python run_b2.py MODEL OUTDIR --dirs runs/A/dirs_c1only_<tag>.npz [--layers Lstar,final]
Adds s*alpha*sigma*d_hat to the residual (output of decoder layer l-1 = hidden_states[l]) at the final token; s = +1 if the gov value
is option A else -1 (oracle sign). alpha* on SELECTION (smallest alpha with C2k4 P_low <= upper 95% CI of base C1). TEST at alpha*.
Controls on the first 100 test pairs: 20 random unit directions, the neutral Blue/Green binding direction (fit here), and a positive
control (replace the final-position residual at layer l with the C1 residual of the same item key).
"""
import json, os, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
MODEL, OUT = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
def arg(n, d): return type(d)(args[args.index(n) + 1]) if n in args else d
HERE = os.path.dirname(os.path.abspath(__file__)); tag = MODEL.split("/")[-1]; dev = "cuda"; t0 = time.time(); os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
mm = model.model.language_model if hasattr(model.model, "language_model") else model.model
LAYERS = mm.layers
def lid(L):
    s = set()
    for x in (L, " " + L):
        e = tok.encode(x, add_special_tokens=False)
        if len(e) == 1: s.add(e[0])
    return sorted(s)
LID = {"A": lid("A"), "B": lid("B")}
D = np.load(arg("--dirs", f"runs/A/dirs_c1only_{tag}.npz")); Ls = list(D["layers"])
which = {"Lstar": int(D["Lstar"]), "final": int(D["final"])}
layers = [which[x] for x in arg("--layers", "Lstar,final").split(",")]
STATE = {"vec": None, "sign": None, "replace": None, "layer": None}
def hook(li):
    def h(mod, inp, out):
        if STATE["layer"] != li + 1: return None
        hs = out[0] if isinstance(out, tuple) else out
        if STATE["vec"] is not None:
            hs = hs.clone(); hs[:, -1] += (STATE["sign"][:, None] * STATE["vec"][None]).to(hs.dtype)
        elif STATE["replace"] is not None:
            hs = hs.clone(); hs[:, -1] = STATE["replace"].to(hs.dtype)
        else: return None
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return h
for li, l in enumerate(LAYERS): l.register_forward_hook(hook(li))
def render(r):
    return tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True)
def fwd(R, layer=None, vec=None, replace=None, want_h=None, bs=24):
    """returns p_low list (and hidden_states[want_h][:, -1] if asked). Batches keep input order (needed for sign/replace)."""
    pl, hs_out = [], []
    for s in range(0, len(R), bs):
        B = R[s:s + bs]; enc = tok([render(r) for r in B], return_tensors="pt", padding=True, add_special_tokens=False)
        m = enc["attention_mask"].to(dev); pos = (m.cumsum(-1) - 1).clamp(min=0)
        STATE.update(layer=layer, vec=vec, replace=None if replace is None else replace[s:s + bs].to(dev),
                     sign=torch.tensor([1.0 if r["letter_low"] == "B" else -1.0 for r in B], device=dev))
        with torch.no_grad():
            o = model(input_ids=enc["input_ids"].to(dev), attention_mask=m, position_ids=pos, use_cache=False, logits_to_keep=1,
                      output_hidden_states=want_h is not None)
        lp = torch.log_softmax(o.logits[:, -1].float(), -1); pa = lp[:, LID["A"]].exp().sum(-1); pb = lp[:, LID["B"]].exp().sum(-1)
        for b, r in enumerate(B): pl.append(((pb[b] if r["letter_low"] == "B" else pa[b]) / (pa[b] + pb[b])).item())
        if want_h is not None: hs_out.append(o.hidden_states[want_h][:, -1].float().cpu())
    STATE.update(layer=None, vec=None, replace=None)
    return (pl, torch.cat(hs_out)) if want_h is not None else pl
def pm(R, pl):
    by = {}
    for r, p in zip(R, pl): by.setdefault(r["pair_id"], []).append(p)
    return np.array([np.mean(v) for v in by.values()])
def ci(x):
    rs = np.random.default_rng(0); bs = [x[rs.integers(0, len(x), len(x))].mean() for _ in range(2000)]; return [float(v) for v in np.percentile(bs, [2.5, 97.5])]
rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "prompts_v2.jsonl"))]
sel = lambda f, ck, split, n=None: [r for r in rows if r["fmt"] == f and r["ck"] == ck and r["split"] == split and (n is None or r["pair_id"] in n)]
test_p = sorted({r["pair_id"] for r in rows if r["split"] == "test"}); sub = set(test_p[:100])
NR = [json.loads(l) for l in open(os.path.join(HERE, "data", "prompts_neutral.jsonl"))]
res = {"model": MODEL, "job": os.environ.get("SLURM_JOB_ID"), "layers": {}}
ALPHAS = [0, 1, 2, 4, 8, 16, 32]
for L in layers:
    li = Ls.index(L); w = torch.tensor(D["w"][li], dtype=torch.float32); nw = w.norm()
    dhat = (w / nw).to(dev); sigma = float(D["sd_logit"][li]) / float(nw)
    out = {"layer": L, "sigma": sigma}
    # --- alpha* on SELECTION ---
    S1, S2 = sel("F1", "C1k1", "selection"), sel("F1", "C2k4", "selection")
    c1 = pm(S1, fwd(S1)); hi = ci(c1)[1]; out["sel_C1"] = float(c1.mean()); out["sel_C1_ci_hi"] = hi; curve = {}
    for a in ALPHAS:
        curve[a] = float(pm(S2, fwd(S2, L, dhat * a * sigma)).mean()); print("L", L, "alpha", a, "sel C2k4", round(curve[a], 3), f"{time.time()-t0:.0f}s", flush=True)
    ok = [a for a in ALPHAS if curve[a] <= hi]; astar = ok[0] if ok else min(ALPHAS, key=lambda a: curve[a])
    out.update(sel_curve=curve, alpha_star=astar, alpha_star_reached=bool(ok)); vec = dhat * astar * sigma
    # --- TEST at alpha* (all 400 pairs) ---
    test = {}
    for f, cks in (("F1", ["C1k1", "C2k4", "C0k1", "C3k4", "C7k4", "C2k2", "C2k8", "C7k2"]), ("F3", ["C1k1", "C2k4", "C7k4"])):
        for ck in cks:
            T = sel(f, ck, "test"); b = pm(T, fwd(T)); s = pm(T, fwd(T, L, vec))
            test[f"{f}_{ck}"] = {"base": float(b.mean()), "base_ci": ci(b), "steer": float(s.mean()), "steer_ci": ci(s)}
            print("L", L, f, ck, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in test[f'{f}_{ck}'].items()}, f"{time.time()-t0:.0f}s", flush=True)
    out["test"] = test
    t = test["F1_C2k4"]; c1t = test["F1_C1k1"]["base_ci"]
    out["crit_i"] = bool(c1t[0] <= t["steer"] <= c1t[1]); out["crit_i_overshoot"] = bool(t["steer"] < c1t[0])
    # --- controls on the first 100 test pairs (C2k4) ---
    T = sel("F1", "C2k4", "test", sub); b100 = pm(T, fwd(T)).mean(); d100 = pm(T, fwd(T, L, vec)).mean() - b100
    rng = torch.Generator().manual_seed(0); rnd = []
    for i in range(20):
        u = torch.randn(w.shape[0], generator=rng); u = (u / u.norm()).to(dev)
        rnd.append(float(pm(T, fwd(T, L, u * astar * sigma)).mean() - b100))
    # neutral binding direction at the same layer (fit on neutral-selection)
    NS = [r for r in NR if r["split"] == "neutral_sel"]; NT = [r for r in NR if r["split"] == "neutral_test"]
    _, Hs = fwd(NS, want_h=L); _, Ht = fwd(NT, want_h=L)
    ys = torch.tensor([r["blue_is_A"] for r in NS]); mu, sd = Hs.mean(0), Hs.std(0) + 1e-4
    wn = torch.zeros(Hs.shape[1], requires_grad=True); bn = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([wn, bn], max_iter=200, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(((Hs - mu) / sd) @ wn + bn, ys) + 1e-3 * (wn * wn).sum(); loss.backward(); return loss
    opt.step(cl)
    wraw = (wn.detach() / sd); out["neutral_auc_test"] = float(roc_auc_score([r["blue_is_A"] for r in NT], (((Ht - mu) / sd) @ wn.detach()).numpy()))
    un = (wraw / wraw.norm()).to(dev); dneu = float(pm(T, fwd(T, L, un * astar * sigma)).mean() - b100)
    # positive control: C1 residual of the same key at layer L, final position
    key = lambda r: (r["pair_id"], r["lowidx"], r["order"], r["ans"])
    C1k = {key(r): r for r in sel("F1", "C1k1", "test", sub)}; Tm = [r for r in T if key(r) in C1k]
    _, Hc = fwd([C1k[key(r)] for r in Tm], want_h=L)
    dpos = float(pm(Tm, fwd(Tm, L, replace=Hc)).mean() - pm(Tm, fwd(Tm)).mean())
    out["controls_100pairs"] = {"base_C2k4": float(b100), "delta_direction": float(d100), "delta_random": rnd,
                                "mean_abs_delta_random": float(np.mean(np.abs(rnd))), "delta_neutral_dir": dneu, "delta_positive_control": dpos,
                                "direction_fraction_of_positive": float(d100 / dpos) if dpos else None}
    out["crit_ii"] = bool(np.mean(np.abs(rnd)) < abs(d100) / 3)
    out["B2D_met"] = bool(out["crit_i"] and out["crit_ii"])
    print("L", L, "alpha*", astar, "crit_i", out["crit_i"], "crit_ii", out["crit_ii"], out["controls_100pairs"], flush=True)
    res["layers"][L] = out
    json.dump(res, open(f"{OUT}/b2_steer_{tag}.json", "w"), indent=1)
print("DONE b2", time.time() - t0)
