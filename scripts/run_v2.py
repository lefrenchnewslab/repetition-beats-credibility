"""SPRINT Block A (13 Nov 2026) — GPU script. PILOT-DERIVED from pilot_illusory/scripts/run.py (11 Nov), modified in sprint.
  python run_v2.py behave MODEL OUTDIR [--fmts F1,F2,F3] [--splits selection,test] [--cks C1k1,...] [--limit N] [--maxtok T] [--nohidden]
  python run_v2.py patch  MODEL OUTDIR --beh BEHAVIOUR_JSONL [--fmt F1] [--n 96] [--nverify 160] [--nrand 20]
Changes vs pilot (see README change log): data v2 path + format/split/condition filters; hidden states stored in one
preallocated bf16 tensor, every 2nd layer + last (FM E15); behaviour written before probes; OOM back-off (E16);
probes fit on the SELECTION split and evaluated on TEST (layer + lambda chosen inside selection, E4), probe directions
saved; patching ranks heads on SELECTION, verifies on TEST, mean-ablation reference from SELECTION (E3), >=20 random
draws (E13), per-item AtP stored for rank-stability (E12), attention breakdown on TEST items.
"""
import json, os, sys, time, random, hashlib
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
mode, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
args = sys.argv[4:]
def arg(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default
DATA = arg("--data", os.path.join(HERE, "data", "prompts_v2.jsonl"))
os.makedirs(OUT, exist_ok=True)
tag = MODEL.split("/")[-1]
dev = "cuda"
t_start = time.time()

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
attn_impl = "eager" if mode == "patch" else "sdpa"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation=attn_impl).to(dev).eval()
for p_ in model.parameters(): p_.requires_grad_(False)

def letter_ids(L):
    ids = set()
    for s in (L, " " + L):
        e = tok.encode(s, add_special_tokens=False)
        if len(e) == 1: ids.add(e[0])
    return sorted(ids)
LID = {"A": letter_ids("A"), "B": letter_ids("B")}
print("letter ids", LID, flush=True)

def render(r):
    msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text, text.index(r["user"])

fmts = arg("--fmts", "F1,F2,F3").split(",")
splits = arg("--splits", "selection,test").split(",")
cks = arg("--cks", "")
cks = set(cks.split(",")) if cks else None
rows = []
for l in open(DATA):
    r = json.loads(l)
    if r["fmt"] in fmts and r["split"] in splits and (cks is None or r["ck"] in cks): rows.append(r)
for xf in [x for x in arg("--extra_data", "").split(",") if x]:  # amendment 5: extra prompt files (e.g. F3 selection C7k4); patch mode only
    for l in open(xf):
        r = json.loads(l)
        if r["fmt"] in fmts and r["split"] in splits and (cks is None or r["ck"] in cks): rows.append(r)
print(f"{len(rows)} prompts after filters fmts={fmts} splits={splits} cks={cks}", flush=True)

def get_layers():
    m = model.model
    if hasattr(m, "language_model"): m = m.language_model
    return m.layers
LAYERS = get_layers()
cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
NH = cfg.num_attention_heads
HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
NL = len(LAYERS)
DM = cfg.hidden_size
print(f"{tag}: layers={NL} heads={NH} head_dim={HD} d={DM}", flush=True)

def lettermass(logits_last):
    lp = torch.log_softmax(logits_last.float(), -1)
    return lp[:, LID["A"]].exp().sum(-1), lp[:, LID["B"]].exp().sum(-1)

def batch_encode(texts):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    m = enc["attention_mask"]; pos = (m.cumsum(-1) - 1).clamp(min=0)
    return enc["input_ids"].to(dev), m.to(dev), pos.to(dev)

# probe conditions whose final-token states are stored (per format)
PROBE_CK = {"F1": {"C0k1", "C0rk4", "C1k1", "C2k2", "C2k4", "C2k8", "C3k2", "C3k4", "C7k2", "C7k4"},
            "F2": {"C0k1", "C1k1", "C2k4", "C7k4"}, "F3": {"C0k1", "C1k1", "C2k4", "C7k4"}}
STORE_L = sorted(set(range(0, NL + 1, 2)) | {NL})  # hidden_states index (0 = embeddings)

# =====================================================================================
if mode == "behave":
    parts = arg("--from_parts", "")  # dev-QoS split (13 Nov): merge per-part outputs, then fit probes
    part = arg("--part", "")
    if not parts:
        limit = arg("--limit", 0)
        if limit:
            keep = {}
            for r in rows:
                keep.setdefault(r["split"], set())
                if len(keep[r["split"]]) < limit: keep[r["split"]].add(r["pair_id"])
            rows = [r for r in rows if r["pair_id"] in keep[r["split"]]]
        nohidden = "--nohidden" in args
        texts = [render(r)[0] for r in rows]
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        order = np.argsort(lens)[::-1]
        ck_order = arg("--ck_order", "")  # amendment 2 (Qwen-32B): process conditions in priority order, longest-first inside each
        if ck_order:
            order = np.array([j for c in ck_order.split(",") for j in order if rows[j]["ck"] == c])
            print("condition priority order:", ck_order, len(order), "prompts", flush=True)
        maxtok = arg("--maxtok", 24000)
        want_h = [(not nohidden) and (r["ck"] in PROBE_CK[r["fmt"]] or "--hidden_all" in args) for r in rows]  # --hidden_all: amendment 6 (C6)
        hidx = {}; n_h = 0
        for j, w in enumerate(want_h):
            if w: hidx[rows[j]["id"]] = n_h; n_h += 1
        H = torch.zeros((n_h, len(STORE_L), DM), dtype=torch.bfloat16) if n_h else None
        print(f"storing hidden states for {n_h} prompts x {len(STORE_L)} layers ({n_h*len(STORE_L)*DM*2/1e9:.1f} GB)", flush=True)
        res = [None] * len(rows)
        i = 0; nb = 0; scale = 1.0
        fout = open(f"{OUT}/behaviour_{tag}{'.' + part if part else ''}.jsonl", "w"); written = 0  # streamed: a timeout keeps every finished prompt
        while i < len(order):
            L = lens[order[i]]; bs = max(1, int(maxtok * scale) // L)
            idx = order[i:i + bs]
            need = any(want_h[j] for j in idx)
            try:
                ids, m, pos = batch_encode([texts[j] for j in idx])
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=m, position_ids=pos, output_hidden_states=need, use_cache=False, logits_to_keep=1)
            except torch.cuda.OutOfMemoryError:
                out = ids = m = pos = None; torch.cuda.empty_cache(); scale /= 2
                print(f"OOM at bs={bs} L={L}; scale -> {scale}", flush=True)
                if bs == 1: raise
                continue
            i += len(idx)
            pa, pb = lettermass(out.logits[:, -1]); la = out.logits[:, -1].float()
            for b, j in enumerate(idx):
                r = rows[j]
                p_low_raw = (pb[b] if r["letter_low"] == "B" else pa[b]).item(); mass = (pa[b] + pb[b]).item()
                ldl = la[b, LID[r["letter_low"]][0]].item(); ldh = la[b, LID["A" if r["letter_low"] == "B" else "B"][0]].item()
                res[j] = {"id": r["id"], "fmt": r["fmt"], "pair_id": r["pair_id"], "split": r["split"], "ck": r["ck"], "lowidx": r["lowidx"],
                          "order": r["order"], "ans": r["ans"], "letter_low": r["letter_low"], "ntok": lens[j],
                          "p_low": p_low_raw / mass, "mass": mass, "ld": ldl - ldh}
                if need and want_h[j]:
                    H[hidx[r["id"]]] = torch.stack([out.hidden_states[l][b, -1] for l in STORE_L]).to(torch.bfloat16).cpu()
            del out
            while written < i: fout.write(json.dumps(res[order[written]]) + "\n"); written += 1
            fout.flush()
            nb += 1
            if nb % 25 == 0: print(f"batch {nb} done {i}/{len(order)} L={L} bs={bs} {time.time()-t_start:.0f}s", flush=True)
        fout.close()  # E15: behaviour on disk before probes
        res = [x for x in res if x is not None]
        meta = {"model": MODEL, "n_prompts": len(rows), "seconds_behaviour": time.time() - t_start, "gpu": torch.cuda.get_device_name(),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "letter_ids": LID, "layers": NL, "heads": NH,
                "stored_layers": STORE_L, "job": os.environ.get("SLURM_JOB_ID"), "data_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest(),
                "filters": {"fmts": fmts, "splits": splits, "cks": sorted(cks) if cks else None, "limit": limit}}
        json.dump(meta, open(f"{OUT}/meta_behave_{tag}{'.' + part if part else ''}.json", "w"), indent=1)
        print("behaviour written", time.time() - t_start, flush=True)
    else:
        res, Hs, hid_list = [], [], []
        for p_ in parts.split(","):
            res += [json.loads(l) for l in open(f"{OUT}/behaviour_{tag}.{p_}.jsonl")]
            d_ = torch.load(f"{OUT}/hidden_{tag}.{p_}.pt"); Hs.append(d_["H"]); hid_list += d_["ids"]
            assert d_["stored_layers"] == STORE_L
        H = torch.cat(Hs) if Hs else None; hidx = {i_: k for k, i_ in enumerate(hid_list)}; n_h = len(hid_list)
        ids_set = {x["id"] for x in res}; rows = [r for r in rows if r["id"] in ids_set]
        assert len(rows) == len(res) == len(ids_set), (len(rows), len(res))
        with open(f"{OUT}/behaviour_{tag}.jsonl", "w") as f_:
            for x in res: f_.write(json.dumps(x) + "\n")
        meta = {"model": MODEL, "n_prompts": len(rows), "parts": parts.split(","), "gpu": torch.cuda.get_device_name(), "layers": NL, "heads": NH,
                "stored_layers": STORE_L, "letter_ids": LID, "job": os.environ.get("SLURM_JOB_ID"),
                "data_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest()}
        print("merged parts", parts, len(res), "prompts,", n_h, "hidden states", flush=True)
    if part:  # save hidden states of this part and stop (probes run in the merge step)
        torch.save({"H": H, "ids": list(hidx), "stored_layers": STORE_L}, f"{OUT}/hidden_{tag}.{part}.pt")
        json.dump(meta, open(f"{OUT}/meta_behave_{tag}.{part}.json", "w"), indent=1)
        print("DONE behave part", part, time.time() - t_start, flush=True); sys.exit(0)
    if n_h == 0:
        print("DONE behave (no probes)", flush=True); sys.exit(0)

    # ---------------- probes: fit on SELECTION, evaluate on TEST ----------------
    from sklearn.metrics import roc_auc_score
    byid = {r["id"]: r for r in rows}; resid = {x["id"]: x for x in res}
    hid_ids = list(hidx)
    def sel(fmt, ckset, split, filt=None, pairs=None):
        return [i_ for i_ in hid_ids if byid[i_]["fmt"] == fmt and byid[i_]["ck"] in ckset and byid[i_]["split"] == split
                and (filt is None or filt(resid[i_])) and (pairs is None or byid[i_]["pair_id"] in pairs)]
    def X_at(ids, li):
        return torch.nan_to_num(H[[hidx[i_] for i_ in ids], li].float())
    def Y(ids, fn): return torch.tensor([fn(byid[i_]) for i_ in ids], dtype=torch.float32)
    def fit(X, y, lam):
        mu, sd = X.mean(0), X.std(0) + 1e-4
        Xs = ((X - mu) / sd).to(dev); ys = y.to(dev)
        w = torch.zeros(X.shape[1], device=dev, requires_grad=True); b0 = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([w, b0], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b0, ys) + lam * (w * w).sum()
            loss.backward(); return loss
        with torch.enable_grad(): opt.step(closure)
        wr = (w.detach().cpu() / sd); br = (b0.detach().cpu() - (mu * wr).sum())  # raw-space direction
        return wr, br
    def score(wb, X): return (X @ wb[0] + wb[1]).numpy()
    def auc(wb, X, y):
        if X is None or len(y) == 0 or len(set(y.tolist())) < 2: return None
        return float(roc_auc_score(y.numpy(), score(wb, X)))
    gov_is_A = lambda r: float(r["letter_low"] == "B")
    flipped = lambda x: x["p_low"] > 0.5
    notflipped = lambda x: x["p_low"] <= 0.5
    sel_pids = sorted({byid[i_]["pair_id"] for i_ in hid_ids if byid[i_]["split"] == "selection"})
    val_p = set(sel_pids[int(.75 * len(sel_pids)):]); trn_p = set(sel_pids[:int(.75 * len(sel_pids))])
    LAB_CKS = ["C1k1", "C2k2", "C2k4", "C2k8", "C3k2", "C3k4", "C7k2", "C7k4"]
    specs = {
        # (a) credibility, trained on all labelled F1 conditions (prompt Step 4a)
        "cred_all": dict(train=LAB_CKS, label=gov_is_A, choose="C2k8",
                         tests=[("F1", [c], None, c) for c in LAB_CKS] + [("F1", [c], flipped, c + "_flipped") for c in ["C2k4", "C2k8", "C7k4", "C3k4"]]
                         + [("F1", [c], notflipped, c + "_notflipped") for c in ["C2k4", "C2k8"]]
                         + [("F1", ["C0k1"], None, "C0k1_control"), ("F1", ["C0rk4"], None, "C0rk4_control")]
                         + [(f, [c], None, f"{f}_{c}") for f in ("F2", "F3") for c in ("C1k1", "C2k4", "C7k4", "C0k1")]
                         + [(f, ["C2k4"], flipped, f"{f}_C2k4_flipped") for f in ("F2", "F3")]),
        # pilot-style length-shift probe: train on k<=2, test at k=8
        "cred_lowk": dict(train=["C1k1", "C2k2"], label=gov_is_A, choose="C1k1",
                          tests=[("F1", ["C2k8"], None, "C2k8"), ("F1", ["C2k8"], flipped, "C2k8_flipped"), ("F1", ["C1k1", "C2k2"], None, "indist")]),
        # (c) control: same label, no source labels
        "cred_control_C0": dict(train=["C0k1"], label=gov_is_A, choose="C0k1",
                                tests=[("F1", ["C0k1"], None, "C0k1"), ("F1", ["C0rk4"], None, "C0rk4")]),
        # (b) copies vs independent vs paraphrased, pairwise; train k=2, test k=2 and k=4
        "C2_vs_C3": dict(train=["C2k2", "C3k2"], label=lambda r: float(r["cond"] == "C3"), choose=None,
                         tests=[("F1", ["C2k2", "C3k2"], None, "k2"), ("F1", ["C2k4", "C3k4"], None, "k4")]),
        "C2_vs_C7": dict(train=["C2k2", "C7k2"], label=lambda r: float(r["cond"] == "C7"), choose=None,
                         tests=[("F1", ["C2k2", "C7k2"], None, "k2"), ("F1", ["C2k4", "C7k4"], None, "k4")]),
        "C3_vs_C7": dict(train=["C3k2", "C7k2"], label=lambda r: float(r["cond"] == "C7"), choose=None,
                         tests=[("F1", ["C3k2", "C7k2"], None, "k2"), ("F1", ["C3k4", "C7k4"], None, "k4")]),
    }
    probes = {}; dirs = {}
    for name, sp in specs.items():
        tr_ids = sel("F1", set(sp["train"]), "selection")
        if not tr_ids: continue
        ytr = Y(tr_ids, sp["label"])
        tr_mask = torch.tensor([byid[i_]["pair_id"] in trn_p for i_ in tr_ids])
        ch_ids = sel("F1", {sp["choose"]}, "selection", pairs=val_p) if sp["choose"] else None
        probes[name] = {}; W = []
        for li, layer in enumerate(STORE_L):
            Xl = X_at(tr_ids, li)
            best = None
            for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
                wb = fit(Xl[tr_mask], ytr[tr_mask], lam)
                a = auc(wb, Xl[~tr_mask], ytr[~tr_mask])
                if a is not None and (best is None or a > best[0]): best = (a, lam, wb)
            d = {"lam": best[1], "val_auc_selection": best[0]}
            if ch_ids:  # E4: layer-choice statistic, computed inside selection (val fold) with the train-fold probe
                Xc, yc = X_at(ch_ids, li), Y(ch_ids, sp["label"])
                d["choose_auc_selection"] = auc(best[2], Xc, yc)
                d["choose_logloss_selection"] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.from_numpy(score(best[2], Xc)).float().clamp(-30, 30), yc))  # amendment 1: continuous, no ties
            wb = fit(Xl, ytr, best[1])
            W.append(torch.cat([wb[0], wb[1].view(1)]))
            for fmt, cl, filt, tn in sp["tests"]:
                te = sel(fmt, set(cl), "test", filt)
                d[tn] = auc(wb, X_at(te, li), Y(te, sp["label"])) if te else None
                d[tn + "_n"] = len(te)
            probes[name][layer] = d
        if sp["choose"]:
            probes[name]["chosen_layer"] = min(STORE_L, key=lambda l: probes[name][l].get("choose_logloss_selection", 1e9))
            probes[name]["chosen_layer_by_auc_first_tie"] = max(STORE_L, key=lambda l: probes[name][l].get("choose_auc_selection") or 0)
        dirs[name] = torch.stack(W).numpy()
        print("probe", name, "done", time.time() - t_start, flush=True)
    json.dump(probes, open(f"{OUT}/probes_{tag}.json", "w"), indent=1)
    np.savez(f"{OUT}/probe_dirs_{tag}.npz", layers=np.array(STORE_L), **dirs)  # rows: [w_raw (d), b_raw]
    meta["seconds_total"] = time.time() - t_start
    json.dump(meta, open(f"{OUT}/meta_behave_{tag}.json", "w"), indent=1)
    print("DONE behave", meta, flush=True)

# =====================================================================================
if mode == "patch":
    FMT = arg("--fmt", "F1"); N = arg("--n", 96); NV = arg("--nverify", 160); NRAND = arg("--nrand", 20)
    CORR = arg("--corrupt", "C2k4")  # amendment 5: corrupt condition (C2k4 exact copies, or C7k4 paraphrased copies)
    beh = {}
    for bf in [arg("--beh", "")] + [x for x in arg("--extra_beh", "").split(",") if x]:
        for l in open(bf): x = json.loads(l); beh[x["id"]] = x
    key = lambda r: (r["pair_id"], r["lowidx"], r["order"], r["ans"])
    rows = [r for r in rows if r["fmt"] == FMT]
    c1 = {key(r): r for r in rows if r["ck"] == "C1k1"}
    c2 = {key(r): r for r in rows if r["ck"] == CORR}
    def eligible(split):  # FM B3: items with a real repetition effect in the behaviour run
        ks = [k_ for k_ in sorted(c1) if c1[k_]["split"] == split and k_ in c2 and c1[k_]["id"] in beh and c2[k_]["id"] in beh
              and beh[c2[k_]["id"]]["ld"] - beh[c1[k_]["id"]]["ld"] > 1.0]
        random.Random(0).shuffle(ks); return ks
    sel_keys = eligible("selection"); test_keys = eligible("test")
    print(f"eligible selection {len(sel_keys)} test {len(test_keys)}", flush=True)
    atp_sel = sel_keys[:N]; atp_test = test_keys[:N]; vkeys = test_keys[:NV]; mkeys = sel_keys[N:N + 24] or sel_keys[:24]

    o_projs = [l.self_attn.o_proj for l in LAYERS]
    mlps = [l.mlp for l in LAYERS]
    state = {"mode": None}
    cache = {}
    def enc1(r):
        t, off = render(r)
        e = tok(t, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"].to(dev), e["offset_mapping"][0], off
    def suffix_len(a, b):
        a, b = a[0].tolist(), b[0].tolist(); n = 0
        while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]: n += 1
        return n
    def pre_hook(li):
        def h(mod, inp):
            x = inp[0]; S = state["S"]
            if state["mode"] == "cache":
                cache[("z", li)] = x[0, -S:].detach().clone()
            elif state["mode"] == "grad":
                x.retain_grad(); cache[("zc", li)] = x
            elif state["mode"] == "patch" and li in state["heads"]:
                x = x.clone(); src = state["src"][li]
                for hh in state["heads"][li]:
                    x[0, -S:, hh * HD:(hh + 1) * HD] = src[-S:, hh * HD:(hh + 1) * HD]
                return (x,)
        return h
    def mlp_hook(li):
        def h(mod, inp, out):
            S = state["S"]
            if state["mode"] == "cache": cache[("m", li)] = out[0, -S:].detach().clone()
            elif state["mode"] == "grad": out.retain_grad(); cache[("mc", li)] = out
        return h
    hs = [o.register_forward_pre_hook(pre_hook(i)) for i, o in enumerate(o_projs)]
    hs += [m.register_forward_hook(mlp_hook(i)) for i, m in enumerate(mlps)]
    emb = model.get_input_embeddings()
    hs.append(emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True) if state["mode"] == "grad" else None))
    def LD(logits, r):
        lo = LID[r["letter_low"]][0]; hi = LID["A" if r["letter_low"] == "B" else "B"][0]
        return logits[0, -1, lo] - logits[0, -1, hi]

    def atp(keys):
        per = []
        for n_, k_ in enumerate(keys):
            rc, rk = c1[k_], c2[k_]
            ic, _, _ = enc1(rc); ik, _, _ = enc1(rk)
            S = suffix_len(ic, ik); state["S"] = S
            state["mode"] = "cache"
            with torch.no_grad(): lc = LD(model(input_ids=ic, use_cache=False).logits, rc).item()
            clean = dict(cache)
            state["mode"] = "grad"
            with torch.enable_grad():
                out = model(input_ids=ik, use_cache=False); ld = LD(out.logits, rk); ld.backward()
            lk = ld.item()
            ha = torch.zeros(NL, NH); ma = torch.zeros(NL)
            for li in range(NL):
                z = cache[("zc", li)]; g = z.grad[0, -S:].float(); zc = z[0, -S:].detach().float()
                ha[li] = ((clean[("z", li)].float() - zc) * g).view(S, NH, HD).sum((0, 2)).cpu()
                mo = cache[("mc", li)]; mg = mo.grad[0, -S:].float()
                ma[li] = ((clean[("m", li)].float() - mo[0, -S:].detach().float()) * mg).sum().cpu()
            cache.clear(); model.zero_grad(set_to_none=True); del out, ld
            per.append({"key": k_, "S": S, "ld_clean": lc, "ld_corrupt": lk, "ans": rk["ans"], "order": rk["order"],
                        "head_attr": ha.numpy().round(5).tolist(), "mlp_attr": ma.numpy().round(5).tolist()})
            if n_ % 16 == 0: print(f"atp {n_}/{len(keys)} S={S} ldc={lc:.2f} {time.time()-t_start:.0f}s", flush=True)
        return per
    def rank(per):
        ha = np.mean([p["head_attr"] for p in per], 0)
        flat = sorted((ha[l, h], l, h) for l in range(NL) for h in range(NH))  # most negative first
        return [(l, h) for _, l, h in flat], ha
    per_sel = atp(atp_sel)
    order_sel, ha_sel = rank(per_sel)
    top = order_sel[:20]
    print("top AtP heads (selection)", top[:10], flush=True)
    per_test = atp(atp_test)  # for rank stability only (E12); never used for selection

    # mean-ablation reference from SELECTION items (E3)
    lens_m = []
    for k_ in mkeys:
        for r in (c1[k_], c2[k_]): lens_m.append(enc1(r)[0].shape[1])
    SM = min(min(lens_m), 256)
    acc = None; cnt = 0
    for k_ in mkeys:
        for r in (c1[k_], c2[k_]):
            ic, _, _ = enc1(r); state["S"] = SM; state["mode"] = "cache"
            with torch.no_grad(): model(input_ids=ic, use_cache=False)
            cur = [cache[("z", li)].float() for li in range(NL)]; cache.clear()
            acc = cur if acc is None else [a + c for a, c in zip(acc, cur)]; cnt += 1
    state["mode"] = None
    MEAN = {li: (acc[li] / cnt).to(torch.bfloat16) for li in range(NL)}

    rng = random.Random(1)
    allheads = [(l, h) for l in range(NL) for h in range(NH)]
    hsets = {"top1": top[:1], "top5": top[:5], "top10": top[:10], "top20": top[:20],
             "mean_top1": top[:1], "mean_top5": top[:5], "mean_top10": top[:10], "mean_top20": top[:20]}
    for i_ in range(NRAND): hsets[f"rand10_{i_:02d}"] = rng.sample(allheads, 10)
    for i_ in range(5): hsets[f"mean_rand10_{i_:02d}"] = rng.sample(allheads, 10)
    for hh in top[:10]: hsets[f"single_{hh[0]}_{hh[1]}"] = [hh]
    SUN = {"Llama-3.1-8B-Instruct": (17, 24), "gemma-3-12b-it": (29, 11)}  # Sun et al. 2605.09314
    SUN_RANK = None
    if tag in SUN and SUN[tag][0] < NL and SUN[tag][1] < NH:
        hsets["sun_decision_head"] = [SUN[tag]]; SUN_RANK = order_sel.index(SUN[tag])
        print("Sun head AtP rank (selection)", SUN_RANK, flush=True)

    def run_patch(r_c, r_k):
        ic, _, _ = enc1(r_c); ik, _, _ = enc1(r_k); S = suffix_len(ic, ik)
        if S > SM: return None
        state["S"] = S; state["mode"] = "cache"
        with torch.no_grad(): lc = LD(model(input_ids=ic, use_cache=False).logits, r_c).item()
        src = {li: cache[("z", li)] for li in range(NL)}; cache.clear()
        state["mode"] = None
        with torch.no_grad(): lk = LD(model(input_ids=ik, use_cache=False).logits, r_k).item()
        outs = {}
        for name, hd in hsets.items():
            hmap = {}
            for (l, h) in hd: hmap.setdefault(l, []).append(h)
            state["mode"] = "patch"; state["heads"] = hmap
            state["src"] = MEAN if name.startswith("mean_") else src
            with torch.no_grad(): outs[name] = LD(model(input_ids=ik, use_cache=False).logits, r_k).item()
            state["mode"] = None
        return S, lc, lk, outs
    ver = []; skipped = 0
    for n_, k_ in enumerate(vkeys):
        o = run_patch(c1[k_], c2[k_])
        if o is None: skipped += 1; continue
        S, lc, lk, outs = o
        ver.append({"key": k_, "pair_id": k_[0], "S": S, "ld_clean": lc, "ld_corrupt": lk, **outs, "ans": c2[k_]["ans"], "order": c2[k_]["order"]})
        if n_ % 20 == 0: print(f"verify {n_}/{len(vkeys)} {time.time()-t_start:.0f}s", flush=True)
    print("verify done", len(ver), "skipped (S > SM)", skipped, time.time() - t_start, flush=True)

    # attention of top-10 (+Sun) heads from the final token, on TEST corrupt prompts (C2k4)
    for hk in hs: hk.remove()
    att_heads = top[:10] + ([SUN[tag]] if "sun_decision_head" in hsets and SUN[tag] not in top[:10] else [])
    att = []
    for k_ in vkeys[:24]:
        r = c2[k_]
        ids, offs, off0 = enc1(r)
        with torch.no_grad(): out = model(input_ids=ids, use_cache=False, output_attentions=True)
        def tokmask(st, en):
            a, b = off0 + st, off0 + en
            return ((offs[:, 0] < b) & (offs[:, 1] > a)).to(dev)
        groups = {}
        for sp in r["spans"]:
            groups.setdefault(f'{sp["kind"]}_{sp["role"]}', []).append(tokmask(sp["start"], sp["end"]))
        u = r["user"]; q0 = u.rindex("What is"); ia = u.rindex("\n(A) "); ib = u.rindex("\n(B) ")
        groups["question"] = [tokmask(q0, ia)]
        optA, optB = tokmask(ia + 1, ib), tokmask(ib + 1, len(u))
        groups["opt_low"] = [optB if r["letter_low"] == "B" else optA]
        groups["opt_high"] = [optA if r["letter_low"] == "B" else optB]
        groups["chat_tail"] = [(offs[:, 0] >= off0 + len(u)).to(dev)]
        groups["bos_system"] = [(offs[:, 1] <= off0).to(dev)]
        masks = {g: torch.stack(v).any(0) for g, v in groups.items()}
        rec = {"key": k_, "order": r["order"], "ntok": int(ids.shape[1])}
        for (l, h) in att_heads:
            a = out.attentions[l][0, h, -1].float()
            rec[f"L{l}H{h}"] = {g: (a[m].sum().item(), int(m.sum().item())) for g, m in masks.items()}
        att.append(rec)
        del out; torch.cuda.empty_cache()

    res = {"model": MODEL, "fmt": FMT, "corrupt": CORR, "n_atp_selection": len(per_sel), "n_atp_test": len(per_test), "eligible": [len(sel_keys), len(test_keys)],
           "atp_selection": per_sel, "atp_test": per_test, "head_attr_selection": ha_sel.tolist(),
           "top20": top, "order_selection": order_sel[:200], "sun_rank": SUN_RANK, "SM": SM, "verify": ver, "verify_skipped": skipped,
           "hsets": hsets, "attention": att, "layers": NL, "heads": NH,
           "seconds": time.time() - t_start, "gpu": torch.cuda.get_device_name(), "job": os.environ.get("SLURM_JOB_ID"),
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    json.dump(res, open(f"{OUT}/patch_{FMT}{'' if CORR == 'C2k4' else '_' + CORR}_{tag}.json", "w"))
    print("DONE patch", FMT, CORR, time.time() - t_start, flush=True)

# =====================================================================================
# Amendment 2 (Amr, 13 Nov): probe controls. Uses saved probe directions; no refit of the main probe.
#  python run_v2.py extras MODEL OUTDIR --beh BEHAVIOUR_JSONL --dirs PROBE_DIRS_NPZ --probes PROBES_JSON
if mode == "extras":
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    beh = {}
    for l in open(arg("--beh", "")): x = json.loads(l); beh[x["id"]] = x
    D = np.load(arg("--dirs", "")); assert list(D["layers"]) == STORE_L, (list(D["layers"]), STORE_L)
    W = torch.tensor(D["cred_all"], dtype=torch.float32)  # [n_layers, d+1]
    chosen = json.load(open(arg("--probes", "")))["cred_all"]["chosen_layer"]
    res = {"model": MODEL, "chosen_layer": chosen, "stored_layers": STORE_L}
    # ---- (b) graded credibility: project C1-shaped prompts with source X in {gov, newspaper, person, social} ----
    G = [json.loads(l) for l in open(arg("--graded", os.path.join(HERE, "data", "prompts_graded.jsonl")))]
    texts = [render(r)[0] for r in G]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = list(np.argsort(lens)[::-1]); maxtok = arg("--maxtok", 16000)
    grow = [None] * len(G); i = 0
    while i < len(order):
        bs = max(1, maxtok // lens[order[i]]); idx = order[i:i + bs]; i += len(idx)
        ids, m, pos = batch_encode([texts[j] for j in idx])
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=m, position_ids=pos, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        pa, pb = lettermass(out.logits[:, -1])
        hsl = torch.stack([out.hidden_states[l][:, -1].float() for l in STORE_L], 1).cpu()  # [b, nL, d]
        sc = (hsl * W[None, :, :-1]).sum(-1) + W[None, :, -1]  # probe logit for "option A is gov-sourced"
        for b, j in enumerate(idx):
            r = G[j]; xA = r["letter_low"] == "B"  # X holds the non-social value; X is option A iff the social letter is B
            sgn = 1.0 if xA else -1.0
            pX = ((pa[b] if xA else pb[b]) / (pa[b] + pb[b])).item()
            grow[j] = {"pair_id": r["pair_id"], "level": r["level"], "rank": r["level_rank"], "order": r["order"], "ans": r["ans"],
                       "p_X": pX, "mass": (pa[b] + pb[b]).item(), "score": (sgn * sc[b]).numpy().round(4).tolist()}
        del out
    res["graded_rows"] = grow
    per_layer = {}
    named = [g for g in grow if g["rank"] is not None]  # H3 uses the four named levels only (amendment 4)
    rk = np.array([g["rank"] for g in named])
    for li, layer in enumerate(STORE_L):
        s = np.array([g["score"][li] for g in named])
        means = {lev: float(np.mean([g["score"][li] for g in grow if g["level"] == lev]))
                 for lev in ("gov", "newspaper", "person", "social", "neutral") if any(g["level"] == lev for g in grow)}
        pos = (means["neutral"] - means["social"]) / (means["gov"] - means["social"]) if "neutral" in means and means["gov"] != means["social"] else None
        per_layer[layer] = {"spearman": float(spearmanr(rk, s).correlation), "means": means, "neutral_position": pos,
                            "ordered": means["gov"] > means["newspaper"] > means["person"] > means["social"]}
    res["graded"] = per_layer
    res["graded_behaviour_pX"] = {lev: float(np.mean([g["p_X"] for g in grow if g["level"] == lev])) for lev in ("gov", "newspaper", "person", "social", "neutral")
                                  if any(g["level"] == lev for g in grow)}
    print("graded done", {k: per_layer[chosen][k] for k in ("spearman", "ordered")}, time.time() - t_start, flush=True)
    # ---- (a) lexical baseline: mean-pooled INPUT embeddings over the whole prompt, same labels/splits as cred_all ----
    LAB_CKS = ["C1k1", "C2k2", "C2k4", "C2k8", "C3k2", "C3k4", "C7k2", "C7k4"]
    R_ = [r for r in rows if r["fmt"] == "F1" and r["ck"] in LAB_CKS + ["C0k1"]]
    emb = model.get_input_embeddings()
    E = torch.zeros(len(R_), DM)
    texts = [render(r)[0] for r in R_]
    for s0 in range(0, len(R_), 64):
        ids, m, _ = batch_encode(texts[s0:s0 + 64])
        with torch.no_grad():
            e = emb(ids).float(); E[s0:s0 + 64] = ((e * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu()
    gov_is_A = np.array([float(r["letter_low"] == "B") for r in R_])
    is_sel = np.array([r["split"] == "selection" for r in R_]); is_lab = np.array([r["ck"] in LAB_CKS for r in R_])
    sel_pids = sorted({r["pair_id"] for r in R_ if r["split"] == "selection"}); val_p = set(sel_pids[int(.75 * len(sel_pids)):])
    is_val = np.array([r["pair_id"] in val_p for r in R_])
    def fitp(X, y, lam):
        X = torch.tensor(X); y = torch.tensor(y, dtype=torch.float32)
        mu, sd = X.mean(0), X.std(0) + 1e-4; Xs = ((X - mu) / sd).to(dev); ys = y.to(dev)
        w = torch.zeros(X.shape[1], device=dev, requires_grad=True); b0 = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([w, b0], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b0, ys) + lam * (w * w).sum(); loss.backward(); return loss
        with torch.enable_grad(): opt.step(closure)
        wr = w.detach().cpu() / sd; br = b0.detach().cpu() - (mu * wr).sum()
        return lambda Z: (torch.tensor(Z) @ wr + br).numpy()
    En = E.numpy()
    tr = is_sel & is_lab & ~is_val; va = is_sel & is_lab & is_val
    best = max([1e-4, 1e-3, 1e-2, 1e-1], key=lambda lam: roc_auc_score(gov_is_A[va], fitp(En[tr], gov_is_A[tr], lam)(En[va])))
    f_ = fitp(En[is_sel & is_lab], gov_is_A[is_sel & is_lab], best)
    lex = {"lam": best}
    flip = np.array([beh[r["id"]]["p_low"] > .5 if r["id"] in beh else False for r in R_])
    for ck in ["C1k1", "C2k4", "C2k8", "C0k1"]:
        mk = ~is_sel & np.array([r["ck"] == ck for r in R_])
        lex[ck] = float(roc_auc_score(gov_is_A[mk], f_(En[mk]))); lex[ck + "_n"] = int(mk.sum())
        if ck in ("C2k4", "C2k8"):
            mf = mk & flip
            lex[ck + "_flipped"] = float(roc_auc_score(gov_is_A[mf], f_(En[mf]))) if len(set(gov_is_A[mf])) == 2 else None
            lex[ck + "_flipped_n"] = int(mf.sum())
    res["lexical_meanpool_embedding"] = lex
    print("lexical baseline", lex, flush=True)
    # ---- amendment 3: neutral-label (Blue/Green) binding control + projection onto the gov-vs-social direction ----
    NR = [json.loads(l) for l in open(arg("--neutral", os.path.join(HERE, "data", "prompts_neutral.jsonl")))]
    texts = [render(r)[0] for r in NR]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = list(np.argsort(lens)[::-1]); HN = torch.zeros(len(NR), len(STORE_L), DM); i = 0
    while i < len(order):
        bs = max(1, maxtok // lens[order[i]]); idx = order[i:i + bs]; i += len(idx)
        ids, m, pos = batch_encode([texts[j] for j in idx])
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=m, position_ids=pos, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hsl = torch.stack([out.hidden_states[l][:, -1].float() for l in STORE_L], 1).cpu()
        for b, j in enumerate(idx): HN[j] = hsl[b]
        del out
    yb = np.array([r["blue_is_A"] for r in NR]); nsel = np.array([r["split"] == "neutral_sel" for r in NR])
    npid = sorted({r["pair_id"] for r in NR if r["split"] == "neutral_sel"}); nval = set(npid[int(.75 * len(npid)):])
    nv = np.array([r["pair_id"] in nval for r in NR])
    gov_ref = {li: float(np.mean([g["score"][li] for g in grow if g["level"] == "gov"])) for li in range(len(STORE_L))}
    neutral = {}
    for li, layer in enumerate(STORE_L):
        Xl = HN[:, li].numpy()
        tr, va = nsel & ~nv, nsel & nv
        best = max([1e-4, 1e-3, 1e-2, 1e-1], key=lambda lam: roc_auc_score(yb[va], fitp(Xl[tr], yb[tr], lam)(Xl[va])))
        f_ = fitp(Xl[nsel], yb[nsel], best)
        d = {"lam": best, "binding_auc_test": float(roc_auc_score(yb[~nsel], f_(Xl[~nsel]))), "n_test": int((~nsel).sum())}
        raw = (HN[~nsel, li] @ W[li, :-1] + W[li, -1]).numpy()  # gov-vs-social logit, "option A is gov"
        ori = np.where(yb[~nsel] == 1, raw, -raw)                 # oriented toward the Blue option
        d.update({"proj_mean_toward_blue": float(ori.mean()), "proj_mean_abs": float(np.abs(raw).mean()),
                  "gov_vs_social_ref": gov_ref[li], "abs_ratio_to_gov_ref": float(np.abs(raw).mean() / abs(gov_ref[li])) if gov_ref[li] else None,
                  "proj_mean_sigmoid_A": float((1 / (1 + np.exp(-raw))).mean())})
        neutral[layer] = d
    res["neutral"] = neutral
    print("neutral done", neutral[chosen], time.time() - t_start, flush=True)
    res["seconds"] = time.time() - t_start; res["job"] = os.environ.get("SLURM_JOB_ID")
    json.dump(res, open(f"{OUT}/extras_{tag}.json", "w"))
    print("DONE extras", time.time() - t_start, flush=True)
