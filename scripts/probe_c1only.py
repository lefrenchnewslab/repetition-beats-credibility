"""SPRINT Block A, amendment 6 (13 Nov) — C1-only credibility probe on saved final-token states (CPU).
Train: F1 selection C1k1 (1 vs 1, no count asymmetry). Same recipe as cred_all: standardise, L2 logistic via LBFGS,
lambda from a 75/25 entity fold inside selection. Test (held-out entities): C1k1, C2k4, C2k8 (all / flipped / not flipped),
C0rk4, C0k1, and C6k4/C6k8 if hidden_<tag>.c6.pt exists.
  python probe_c1only.py RUNDIR MODEL_TAG  -> RUNDIR/probe_c1only_<tag>.json
"""
import json, sys, os, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score
RD, tag = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
meta = {}; meta_letter = {}
for f in [os.path.join(HERE, "data", "prompts_v2.jsonl"), os.path.join(HERE, "data", "prompts_c6.jsonl")]:
    for l in open(f):
        r = json.loads(l)
        if r["fmt"] == "F1": meta[r["id"]] = (r["ck"], r["split"], r["pair_id"], float(r["letter_low"] == "B")); meta_letter[r["id"]] = r["letter_low"]
beh = {}
for f in [f"{RD}/behaviour_{tag}.jsonl", f"{RD}/behaviour_{tag}.f1sel.jsonl", f"{RD}/behaviour_{tag}.f1test.jsonl", f"{RD}/behaviour_{tag}.c6.jsonl"]:  # merged or per-part (same rows)
    if os.path.exists(f):
        for l in open(f): x = json.loads(l); beh[x["id"]] = x["p_low"]
parts = [p for p in ("f1sel", "f1test", "c6") if os.path.exists(f"{RD}/hidden_{tag}.{p}.pt")]
Hs, ids = [], []
for p in parts:
    d = torch.load(f"{RD}/hidden_{tag}.{p}.pt", map_location="cpu"); Hs.append(d["H"]); ids += d["ids"]; L = d["stored_layers"]
H = torch.cat(Hs); idx = {i: k for k, i in enumerate(ids)}
print("loaded", parts, H.shape, f"{time.time()-t0:.0f}s", flush=True)
def pick(ck, split, filt=None):
    return [i for i in ids if i in meta and meta[i][0] == ck and meta[i][1] == split and (filt is None or (i in beh and filt(beh[i])))]
tr = pick("C1k1", "selection")
BAL = "--balanced" in sys.argv  # amendment 10: decision-balanced C1 training set
bal_info = None
if BAL:
    cells = {}
    for i in tr:
        if i not in beh: continue
        cells.setdefault((meta[i][3], beh[i] > .5), []).append(i)
    n_dis = sum(len(v) for (yy, dd), v in cells.items() if dd)
    bal_info = {"n_disagree_total": n_dis, "cells": {f"y{int(yy)}_{'dis' if dd else 'agr'}": len(v) for (yy, dd), v in cells.items()}}
    if n_dis < 200 or len(cells) < 4:
        json.dump({"tag": tag, "balanced": bal_info, "stopped": f"only {n_dis} disagreeing C1 selection items (< 200) or a missing cell"},
                  open(f"{RD}/probe_c1bal_{tag}.json", "w"), indent=1)
        print("STOP: too few disagreeing items", bal_info); sys.exit(0)
    n_cell = min(len(v) for v in cells.values()); rs = np.random.default_rng(0)
    tr = sorted(i for v in cells.values() for i in rs.choice(v, n_cell, replace=False).tolist())
    bal_info["n_cell"] = int(n_cell); bal_info["n_train"] = len(tr)
    print("balanced", bal_info, flush=True)
pids = sorted({meta[i][2] for i in tr}); val = set(pids[int(.75 * len(pids)):])
trm = torch.tensor([meta[i][2] not in val for i in tr])
y = torch.tensor([meta[i][3] for i in tr])
flip = lambda p: p > .5; notflip = lambda p: p <= .5
tests = {"C1k1": pick("C1k1", "test"), "C2k4": pick("C2k4", "test"), "C2k8": pick("C2k8", "test"),
         "C2k8_flipped": pick("C2k8", "test", flip), "C2k8_notflipped": pick("C2k8", "test", notflip),
         "C0rk4": pick("C0rk4", "test"), "C0k1": pick("C0k1", "test"), "C6k4": pick("C6k4", "test"), "C6k8": pick("C6k8", "test")}
def fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-4; Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + lam * (w * w).sum(); loss.backward(); return loss
    opt.step(cl)
    wr = w.detach() / sd; br = b.detach() - (mu * wr).sum()
    f = lambda Z: (Z @ wr + br).numpy(); f.w = wr; f.b = br   # Block B2: keep raw-space direction
    return f
# amendment 7: decision-disagreeing C1 items in the held-out selection fold (model picks the social value: P_low > .5)
dis = torch.tensor([(meta[i][2] in val) and (beh.get(i, 0) > .5) for i in tr])
res = {"tag": tag, "parts": parts, "n_train": len(tr), "n_disagree_val": int(dis.sum()), "balanced": bal_info, "layers": {}}
rng = np.random.default_rng(0)
def auc_ci(f, te, B=1000):  # cluster bootstrap over pairs
    yt = np.array([meta[i][3] for i in te]); s = f(H[[idx[i] for i in te], li].float())
    by = {}
    for k, i in enumerate(te): by.setdefault(meta[i][2], []).append(k)
    P = list(by); bs = []
    for _ in range(B):
        ix = np.concatenate([by[P[j]] for j in rng.integers(0, len(P), len(P))])
        if len(set(yt[ix])) == 2: bs.append(roc_auc_score(yt[ix], s[ix]))
    return [float(x) for x in np.percentile(bs, [2.5, 97.5])]
fits = {}
for li, layer in enumerate(L):
    X = H[[idx[i] for i in tr], li].float()
    best = None
    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        f = fit(X[trm], y[trm], lam); a = roc_auc_score(y[~trm].numpy(), f(X[~trm]))
        if best is None or a > best[0]: best = (a, lam, f)
    fdis = best[2](X[dis]); ydis = y[dis].numpy()
    d = {"lam": best[1], "val_auc_selection": best[0],
         "val_auc_disagree": float(roc_auc_score(ydis, fdis)) if len(set(ydis)) == 2 else None}
    f = fit(X, y, best[1]); fits[layer] = (li, f)
    for name, te in tests.items():
        yt = np.array([meta[i][3] for i in te])
        d[name] = float(roc_auc_score(yt, f(H[[idx[i] for i in te], li].float()))) if len(set(yt)) == 2 else None
        d[name + "_n"] = len(te)
    res["layers"][layer] = d
    print(layer, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if not k.endswith("_n")}, flush=True)
Lstar = max(L, key=lambda l: res["layers"][l]["val_auc_disagree"] or 0); Lfin = L[-1]
te = tests["C2k8_flipped"]
li, f = fits[Lstar]; ci_a = auc_ci(f, te)
li, f = fits[Lfin]; ci_c = auc_ci(f, te)
a_ = res["layers"][Lstar]["C2k8_flipped"]; b_ = res["layers"][Lstar]["C0rk4"]; c_ = res["layers"][Lfin]["C2k8_flipped"]
res["H4"] = {"Lstar": Lstar, "final_layer": Lfin,
             "a_flipped_auc_Lstar": a_, "a_ci": ci_a, "a_met": bool(a_ >= .75 and ci_a[0] > .5),
             "b_C0r_auc_Lstar": b_, "b_met": bool(.40 <= b_ <= .60),
             "c_flipped_auc_final": c_, "c_ci": ci_c, "c_met": bool(c_ < .5 and ci_c[1] < .5)}
res["H4"]["met"] = res["H4"]["a_met"] and res["H4"]["b_met"] and res["H4"]["c_met"]
res["H4"]["fallback_mid_stack_only"] = res["H4"]["a_met"] and res["H4"]["b_met"] and not res["H4"]["c_met"]
# amendment 9 (H5): flipped-item AUC + CI at the last 3 stored layers, and the model's choice rate at C2k8
last3 = {}
for layer in L[-3:]:
    li, f = fits[layer]; last3[layer] = {"auc": res["layers"][layer]["C2k8_flipped"], "ci": auc_ci(f, te)}
c2 = tests["C2k8"]; choice = float(np.mean([beh[i] > .5 for i in c2 if i in beh]))
fin = last3[Lfin]
res["H5"] = {"final_layer": Lfin, "auc": fin["auc"], "ci": fin["ci"], "choice_rate_social_C2k8": choice, "n_C2k8": len(c2),
             "n_flipped": len(te), "last3": last3,
             "met": bool(fin["auc"] >= .70 and fin["ci"][0] > .5 and choice >= .5)}
print("H5", res["H5"], flush=True)
print("H4", res["H4"], flush=True)
json.dump(res, open(f"{RD}/probe_{'c1bal' if BAL else 'c1only'}_{tag}.json", "w"), indent=1)
# Block B2: save every layer's final-fit direction (raw space) + the C1 selection projection SD, for steering
W_ = np.stack([fits[l][1].w.numpy() for l in L]); B_ = np.array([float(fits[l][1].b) for l in L])
SD_ = np.array([float(np.std(fits[l][1](H[[idx[i] for i in tr], fits[l][0]].float()))) for l in L])
np.savez(f"{RD}/dirs_{'c1bal' if BAL else 'c1only'}_{tag}.npz", layers=np.array(L), w=W_, b=B_, sd_logit=SD_, Lstar=res["H4"]["Lstar"], final=res["H4"]["final_layer"])
print("DONE", f"{time.time()-t0:.0f}s")

# Block F3 (--chosen): same fitted directions; AUC against the model's CHOSEN option (preferred letter is A) instead of the gov label
if "--chosen" in sys.argv:
    chosenA = lambda i: float((beh[i] > .5) == (meta_letter[i] == "A"))
    outc = {"tag": tag, "Lstar": Lstar, "final": Lfin, "layers": {}}
    for layer in (Lstar, Lfin):
        li_, f_ = fits[layer]; d = {}
        for name in ("C2k8", "C2k8_flipped", "C2k8_notflipped", "C1k1"):
            te_ = [i for i in tests[name] if i in beh]
            sc = f_(H[[idx[i] for i in te_], li_].float())
            yg = np.array([meta[i][3] for i in te_]); yc = np.array([chosenA(i) for i in te_])
            d[name] = {"n": len(te_), "auc_gov_label": float(roc_auc_score(yg, sc)) if len(set(yg)) == 2 else None,
                       "auc_chosen_answer": float(roc_auc_score(yc, sc)) if len(set(yc)) == 2 else None,
                       "share_chosen_equals_gov": float(np.mean(yg == yc))}
        outc["layers"][layer] = d
    json.dump(outc, open(f"{RD}/probe_c1chosen_{tag}.json", "w"), indent=1); print("chosen-answer AUCs written", flush=True)
