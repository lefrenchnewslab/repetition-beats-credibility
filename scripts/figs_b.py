"""SPRINT Block B figures + numbers for B0/B1/B2 (14 Nov 2026). New file. Run after analyze_b.py (it extends numbers_B.json).
  python3 figs_b.py results/B
"""
import json, os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RB = os.path.abspath(sys.argv[1]); SPRINT = os.path.abspath(os.path.join(RB, "..", "..")); FIG = os.path.join(SPRINT, "figs")
NP = os.path.join(SPRINT, "numbers_B.json"); NUM = json.load(open(NP)) if os.path.exists(NP) else {}
def put(k, v, ci=None, n=None, src=None): NUM[k] = {"value": float(v), "ci": ci, "n": n, "source": src}
def save(fig, name): fig.tight_layout(); fig.savefig(f"{FIG}/{name}.png", dpi=180); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig)
SH = {"Llama-3.1-8B-Instruct": "Llama-8B", "Qwen2.5-7B-Instruct": "Qwen-7B", "gemma-3-12b-it": "Gemma-12B"}
# --- B0: prefix-matching vs copying, all heads, top-10 highlighted ---
fs = sorted(glob.glob(f"{RB}/b0_induction_*.json")); fig, axs = plt.subplots(1, len(fs), figsize=(4.2 * len(fs), 3.6))
for ax, f in zip(np.atleast_1d(axs), fs):
    R = json.load(open(f)); tag = R["model"].split("/")[-1]; P = np.array(R["all"]["prefix"]).ravel(); C = np.array(R["all"]["copy"]).ravel()
    cp = np.array([(C < c).mean() for c in C])
    ax.scatter(cp, P, s=4, color="#bbbbbb", label="all heads")
    top = R["sets"]["F1"]; ax.scatter([x["copy_pct"] for x in top], [x["prefix"] for x in top], s=40, color="#d1495b", label="top-10 repetition heads (F1)")
    if "F3_C7k4" in R["sets"]:
        t7 = R["sets"]["F3_C7k4"]; ax.scatter([x["copy_pct"] for x in t7], [x["prefix"] for x in t7], s=40, marker="^", facecolors="none", edgecolors="#3b6fb6", label="top-10 paraphrase heads (F3/C7)")
    ax.axhline(.3, color="k", ls="--", lw=.7); ax.axvline(.9, color="k", ls="--", lw=.7)
    ax.set_title(f"{SH[tag]}: induction region = top right", fontsize=8); ax.set_xlabel("copying score (percentile among heads)"); ax.set_ylabel("prefix-matching score")
    put(f"B0.{SH[tag]}.F1.n_induction", sum(x["induction"] for x in top), n=10, src=os.path.relpath(f, SPRINT))
    put(f"B0.{SH[tag]}.F1.mean_prefix", np.mean([x["prefix"] for x in top]), src=os.path.relpath(f, SPRINT))
    put(f"B0.{SH[tag]}.null_p95_mean_prefix", R["null_mean_prefix_10"]["p95"], src=os.path.relpath(f, SPRINT))
np.atleast_1d(axs)[0].legend(fontsize=6); save(fig, "figB0_induction")
# --- B1: mediated fraction vs random ---
fs = sorted(glob.glob(f"{RB}/b1_path_*.json"))
if fs:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, f in enumerate(fs):
        R = json.load(open(f)); lab = f"{R['fmt']}/{R['corrupt']}"; m = R["mediated_top10"]; ci = R["mediated_top10_ci"]
        ax.bar(i, m * 100, yerr=[[m * 100 - ci[0] * 100], [ci[1] * 100 - m * 100]], color="#3b6fb6", capsize=4)
        ax.scatter(i + .25 + np.random.default_rng(0).uniform(-.05, .05, 20), np.array(R["random_mediated"]) * 100, s=8, color="grey")
        put(f"B1.{lab}.mediated_top10_pct", m * 100, [c * 100 for c in ci], R["n_test"], os.path.relpath(f, SPRINT))
        put(f"B1.{lab}.random_p95_pct", R["random_p95"] * 100, n=20, src=os.path.relpath(f, SPRINT)); put(f"B1.{lab}.met", float(R["B1C_met"]))
    ax.axhline(40, color="k", ls="--", lw=.7); ax.set_xticks(range(len(fs)))
    ax.set_xticklabels([f"{json.load(open(f))['fmt']}/{json.load(open(f))['corrupt']}" for f in fs]); ax.set_ylabel("% of decision-head effect mediated (test)")
    ax.set_title("Qwen-7B: top-10 upstream senders (bars) vs 20 random sets (dots)", fontsize=8); save(fig, "figB1_path")
# --- B2: steering curves + positive control ---
f = f"{RB}/b2_steer_Qwen2.5-7B-Instruct.json"
if os.path.exists(f):
    R = json.load(open(f)); fig, ax = plt.subplots(figsize=(5, 3.4))
    for L, o in R["layers"].items():
        a = sorted(o["sel_curve"], key=float); ax.plot([float(x) for x in a], [o["sel_curve"][x] for x in a], "-o", ms=3, label=f"L{L} credibility direction (selection C2k4)")
        c = o["controls_100pairs"]; ax.scatter([34], [c["base_C2k4"] + c["delta_positive_control"]], marker="*", s=90, label=f"L{L} positive control (C1 residual swap)")
        put(f"B2.L{L}.delta_direction", c["delta_direction"], n=800, src=os.path.relpath(f, SPRINT)); put(f"B2.L{L}.delta_positive_control", c["delta_positive_control"], n=800, src=os.path.relpath(f, SPRINT))
        put(f"B2.L{L}.mean_abs_delta_random", c["mean_abs_delta_random"], n=20, src=os.path.relpath(f, SPRINT)); put(f"B2.L{L}.met", float(o["B2D_met"]))
    ax.axhspan(.228, .254, color="#2e9e6a", alpha=.2, label="C1 test 95% CI (target)"); ax.set_xscale("symlog")
    ax.set_xlabel("α (units of probe-projection SD)"); ax.set_ylabel("P_low, C2k4"); ax.legend(fontsize=6); ax.set_title("Qwen-7B steering (oracle sign): no effect; residual swap restores at L28", fontsize=8)
    save(fig, "figB2_steering")
json.dump(NUM, open(NP, "w"), indent=1); print("figs + numbers written")
