"""SPRINT Block F - POST HOC (requested by Amr after the Qwen-32B F1 data had been seen; not pre-registered, no verdict change).
Per-template C7 k=1 P_low (t0 = C1 by construction), mean over templates 1-3, and D = R_p - R_e recomputed against
(a) that mean and (b) the single strongest template (highest C7k1 P_low). Cluster bootstrap over pairs (2000, seed 0).
  python3 scripts/posthoc_f1_templates.py  -> prints a markdown block; writes numbers_F_posthoc.json
"""
import json, os, collections
import numpy as np
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RF, RA = f"{S}/results/F", f"{S}/results/A"
SH = {"Qwen2.5-32B-Instruct": "Qwen-32B", "Qwen2.5-7B-Instruct": "Qwen-7B", "Llama-3.1-8B-Instruct": "Llama-8B", "gemma-3-12b-it": "Gemma-12B"}
NUM = {}; out = []
for fmt, sub in (("F3", ""), ("F1", "/f1")):
    for tag, short in SH.items():
        f = f"{RF}{sub}/behaviour_{tag}.jsonl"
        if not os.path.exists(f): continue
        D = collections.defaultdict(lambda: collections.defaultdict(list))
        for path in (f"{RA}/behaviour_{tag}.jsonl", f):
            for l in open(path):
                x = json.loads(l)
                if x.get("split", "test") == "test" and x["fmt"] == fmt: D[x["ck"]][x["pair_id"]].append(x["p_low"])
        cks = ["C1k1", "C2k4", "C7k4", "C7k1t1", "C7k1t2", "C7k1t3"]
        if any(c not in D for c in cks): continue
        P = np.array(sorted(set.intersection(*[set(D[c]) for c in cks]))); M = {c: np.array([np.mean(D[c][p]) for p in P]) for c in cks}
        M["C7k1t0"] = M["C1k1"]                                    # template 0 is byte-identical to C1
        tm = {t: M[f"C7k1t{t}"].mean() for t in range(4)}; tstar = max((1, 2, 3), key=lambda t: tm[t])
        mean123 = (M["C7k1t1"] + M["C7k1t2"] + M["C7k1t3"]) / 3
        Re = M["C2k4"] - M["C1k1"]
        Da = (M["C7k4"] - mean123) - Re; Db = (M["C7k4"] - M[f"C7k1t{tstar}"]) - Re
        rng = np.random.default_rng(0); idx = rng.integers(0, len(P), (2000, len(P)))
        ci = lambda v: np.percentile(v[idx].mean(1), [2.5, 97.5]) * 100
        out.append(f"| {short} | {fmt} | " + " | ".join(f"{tm[t]:.3f}" for t in range(4)) + f" | {mean123.mean():.3f} | t{tstar} | "
                   f"{Da.mean()*100:+.1f} [{ci(Da)[0]:+.1f}, {ci(Da)[1]:+.1f}] | {Db.mean()*100:+.1f} [{ci(Db)[0]:+.1f}, {ci(Db)[1]:+.1f}] |")
        b = f"F1posthoc.{short}.{fmt}"
        for t in range(4): NUM[f"{b}.C7k1_t{t}.p_low"] = {"value": float(tm[t]), "n": int(len(P))}
        NUM[f"{b}.C7k1_mean_t123.p_low"] = {"value": float(mean123.mean()), "n": int(len(P))}; NUM[f"{b}.strongest_template"] = {"value": tstar}
        NUM[f"{b}.D_vs_mean_t123_pts"] = {"value": float(Da.mean() * 100), "ci": [float(x) for x in ci(Da)], "n": int(len(P))}
        NUM[f"{b}.D_vs_strongest_pts"] = {"value": float(Db.mean() * 100), "ci": [float(x) for x in ci(Db)], "n": int(len(P))}
hdr = ("| model | format | C7k1 t0 (=C1) | t1 | t2 | t3 | mean t1–3 | strongest | D vs mean t1–3 [CI] | D vs strongest template [CI] |\n"
       "|---|---|---|---|---|---|---|---|---|---|")
print(hdr + "\n" + "\n".join(out)); json.dump(NUM, open(f"{S}/numbers_F_posthoc.json", "w"), indent=1)
