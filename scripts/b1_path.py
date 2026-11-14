"""SPRINT Block B1 (14 Nov 2026) — upstream path patching into the decision (receiver) heads. New file.
  python b1_path.py MODEL OUTDIR --fmt F1 --set F1 --corrupt C2k4 [--extra_data f --extra_beh f] [--nrank 64 --nverify 160]
Receivers = top-10 of the set's Block A selection ranking. Sender pool = top-50 heads (excluding receivers) + top-10 MLPs by Block A
selection AtP, in layers below the deepest receiver. Ranking: single-sender path patching on SELECTION items; top-10 senders tested
jointly on TEST items against 20 random 10-sender sets from the full eligible head+MLP set. Shared-suffix positions only (FM F12).
"""
import json, os, sys, time, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
MODEL, OUT = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
def arg(n, d): return type(d)(args[args.index(n) + 1]) if n in args else d
HERE = os.path.dirname(os.path.abspath(__file__)); tag = MODEL.split("/")[-1]; dev = "cuda"; t0 = time.time(); os.makedirs(OUT, exist_ok=True)
FMT, SET, CORR = arg("--fmt", "F1"), arg("--set", "F1"), arg("--corrupt", "C2k4"); NR, NV = arg("--nrank", 64), arg("--nverify", 160)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
mm = model.model.language_model if hasattr(model.model, "language_model") else model.model
LAYERS = mm.layers; cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; NL = len(LAYERS)
def lid(L):
    s = set()
    for x in (L, " " + L):
        e = tok.encode(x, add_special_tokens=False)
        if len(e) == 1: s.add(e[0])
    return sorted(s)
LID = {"A": lid("A")[0], "B": lid("B")[0]}
ST = {"S": 0, "setz": {}, "setm": {}, "cachez": set(), "cachem": False}; CZ = {}; CM = {}
def zhook(li):
    def h(mod, inp):
        x = inp[0]; S = ST["S"]; changed = False
        if li in ST["setz"]:
            x = x.clone(); changed = True
            for hh, val in ST["setz"][li]: x[0, -S:, hh * HD:(hh + 1) * HD] = val
        if ST["cachez"] == "all" or li in ST["cachez"]: CZ[li] = x[0, -S:].clone()
        return (x,) if changed else None
    return h
def mhook(li):
    def h(mod, inp, out):
        S = ST["S"]
        if ST["cachem"]: CM[li] = out[0, -S:].clone()
        if li in ST["setm"]:
            out = out.clone(); out[0, -S:] = ST["setm"][li]; return out
    return h
for li, l in enumerate(LAYERS):
    l.self_attn.o_proj.register_forward_pre_hook(zhook(li)); l.mlp.register_forward_hook(mhook(li))
def enc(r):
    t = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True)
    return tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
def LD(ids, r, setz=None, setm=None, cachez=(), cachem=False):
    ST.update(setz=setz or {}, setm=setm or {}, cachez=cachez, cachem=cachem)
    lg = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    ST.update(setz={}, setm={}, cachez=set(), cachem=False)
    lo = LID[r["letter_low"]]; hi = LID["A" if r["letter_low"] == "B" else "B"]
    return (lg[lo] - lg[hi]).item()
# ---------------- data / items (same eligibility as Block A patching) ----------------
rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "prompts_v2.jsonl"))]
for xf in [x for x in arg("--extra_data", "").split(",") if x]: rows += [json.loads(l) for l in open(xf)]
rows = [r for r in rows if r["fmt"] == FMT]
beh = {}
for bf in [f"runs/A/behaviour_{tag}.jsonl"] + [x for x in arg("--extra_beh", "").split(",") if x]:
    for l in open(bf): x = json.loads(l); beh[x["id"]] = x
key = lambda r: (r["pair_id"], r["lowidx"], r["order"], r["ans"])
c1 = {key(r): r for r in rows if r["ck"] == "C1k1"}; c2 = {key(r): r for r in rows if r["ck"] == CORR}
def eligible(split):
    ks = [k for k in sorted(c1) if c1[k]["split"] == split and k in c2 and c1[k]["id"] in beh and c2[k]["id"] in beh and beh[c2[k]["id"]]["ld"] - beh[c1[k]["id"]]["ld"] > 1.0]
    random.Random(0).shuffle(ks); return ks
sel_keys, test_keys = eligible("selection")[:NR], eligible("test")[:NV]
R = json.load(open(f"runs/A/patch_{SET}_{tag}.json")); order = [tuple(h) for h in R["order_selection"]]
recv = order[:10]; Lmax = max(l for l, _ in recv)
mlp_attr = np.mean([p["mlp_attr"] for p in R["atp_selection"]], 0)
pool = [("h", l, h) for l, h in order if (l, h) not in recv and l < Lmax][:50] + [("m", int(l), None) for l in np.argsort(mlp_attr) if l < Lmax][:10]
full = [("h", l, h) for l in range(Lmax) for h in range(NH) if (l, h) not in recv] + [("m", l, None) for l in range(Lmax)]
print(f"receivers {recv}; pool {len(pool)}; full eligible {len(full)}; items sel {len(sel_keys)} test {len(test_keys)}", flush=True)
rz = {}
for l, h in recv: rz.setdefault(l, []).append(h)
def item_state(k):
    """clean caches, corrupt LD, receivers-clean LD"""
    rc, rk = c1[k], c2[k]; ic, ik = enc(rc), enc(rk)
    a, b = ic[0].tolist(), ik[0].tolist(); S = 0
    while S < min(len(a), len(b)) and a[-1 - S] == b[-1 - S]: S += 1
    ST["S"] = S; CZ.clear(); CM.clear()
    LD(ic, rc, cachez="all", cachem=True); cz, cm = dict(CZ), dict(CM)
    ldk = LD(ik, rk)
    ldr = LD(ik, rk, setz={l: [(h, cz[l][:, h * HD:(h + 1) * HD]) for h in hs] for l, hs in rz.items()})
    return {"ik": ik, "rk": rk, "S": S, "cz": cz, "cm": cm, "ldk": ldk, "ldr": ldr}
def path_ld(st, senders):
    ST["S"] = st["S"]; setz, setm = {}, {}
    for kind, l, h in senders:
        if kind == "h": setz.setdefault(l, []).append((h, st["cz"][l][:, h * HD:(h + 1) * HD]))
        else: setm[l] = st["cm"][l]
    CZ.clear(); LD(st["ik"], st["rk"], setz=setz, setm=setm, cachez=set(rz)); rec = {l: CZ[l] for l in rz}
    return LD(st["ik"], st["rk"], setz={l: [(h, rec[l][:, h * HD:(h + 1) * HD]) for h in hs] for l, hs in rz.items()})
# ---------------- rank pool on SELECTION ----------------
med = np.zeros(len(pool)); dk = dr = 0.0
for n, k in enumerate(sel_keys):
    st = item_state(k); dk += st["ldk"]; dr += st["ldr"]
    for j, s in enumerate(pool): med[j] += st["ldk"] - path_ld(st, [s])
    if n % 8 == 0: print(f"rank {n}/{len(sel_keys)} {time.time()-t0:.0f}s", flush=True)
med /= len(sel_keys); top = [pool[j] for j in np.argsort(-med)[:10]]
print("top senders", top, "sel total effect", (dk - dr) / len(sel_keys), flush=True)
# ---------------- verify on TEST ----------------
rs = random.Random(1); rand_sets = [rs.sample(full, 10) for _ in range(20)]
ver = []
for n, k in enumerate(test_keys):
    st = item_state(k); v = {"pair_id": k[0], "S": st["S"], "ldk": st["ldk"], "ldr": st["ldr"], "top10": path_ld(st, top)}
    for i, s in enumerate(rand_sets): v[f"rand_{i:02d}"] = path_ld(st, s)
    ver.append(v)
    if n % 20 == 0: print(f"verify {n}/{len(test_keys)} {time.time()-t0:.0f}s", flush=True)
def frac(name, V=ver): return (np.mean([v["ldk"] for v in V]) - np.mean([v[name] for v in V])) / (np.mean([v["ldk"] for v in V]) - np.mean([v["ldr"] for v in V]))
by = {}
for i, v in enumerate(ver): by.setdefault(v["pair_id"], []).append(i)
P = list(by); bs = []
for _ in range(2000):
    idx = [i for p in np.random.default_rng(len(bs)).integers(0, len(P), len(P)) for i in by[P[p]]]; bs.append(frac("top10", [ver[i] for i in idx]))
rnd = [frac(f"rand_{i:02d}") for i in range(20)]
res = {"model": MODEL, "fmt": FMT, "set": SET, "corrupt": CORR, "receivers": recv, "pool": pool, "pool_mediated_sel": med.tolist(), "top_senders": top,
       "mediated_top10": frac("top10"), "mediated_top10_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "random_mediated": rnd,
       "random_p95": float(np.percentile(rnd, 95)), "n_test": len(ver), "mean_S": float(np.mean([v["S"] for v in ver])),
       "receivers_effect_test": float(np.mean([v["ldk"] - v["ldr"] for v in ver])), "verify": ver, "seconds": time.time() - t0, "job": os.environ.get("SLURM_JOB_ID")}
res["B1C_met"] = bool(res["mediated_top10"] >= .40 and res["mediated_top10"] > res["random_p95"])
json.dump(res, open(f"{OUT}/b1_path_{FMT}_{CORR}_{tag}.json", "w"))
print("B1", FMT, CORR, "mediated top10", round(res["mediated_top10"], 3), res["mediated_top10_ci"], "random p95", round(res["random_p95"], 3), "met", res["B1C_met"], flush=True)
print("DONE b1", time.time() - t0)
