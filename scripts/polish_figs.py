r"""SPRINT Block C (14 Nov 2026) — print-ready versions of fig1, fig2, fig5, figB3 (same data, same filenames, PDF+PNG). New file.
Sizes are set to the width each figure is printed at (\linewidth = 6.5 in; figB3 at 0.8\linewidth), so 9-pt fonts stay 9 pt.
  python3 scripts/polish_figs.py
"""
import json, os, glob, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FIG = f"{S}/figs"; RA = f"{S}/results/A"; RB = f"{S}/results/B"
A = json.load(open(f"{S}/numbers_A.json")); B = json.load(open(f"{S}/numbers_B.json"))
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
M = [("Llama-8B", "Llama-3.1-8B", "#3b6fb6"), ("Qwen-7B", "Qwen2.5-7B", "#d1495b"), ("Gemma-12B", "Gemma-3-12B", "#2e9e6a"), ("Qwen-32B", "Qwen2.5-32B", "#8a5cc2")]
def val(k):
    x = A.get(k); return None if x is None else x["value"]
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- fig1: dose-response ----------
STY = {"C2": ("-", "o", "exact copies (C2)"), "C7": (":", "^", "paraphrased copies (C7)"), "C3": ("--", "s", "independent users (C3)")}
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.55), sharey=True)
for ax, (fmt, title) in zip(axs, [("F1", "Tables (F1)"), ("F2", "News snippets (F2)"), ("F3", "Retrieved documents (F3)")]):
    for short, _, col in M:
        c1 = val(f"{short}.{fmt}.test.C1k1.p_low")
        if c1 is None: continue
        for fam, (ls, mk, _) in STY.items():
            ks = [k for k in (2, 4, 8) if val(f"{short}.{fmt}.test.{fam}k{k}.p_low") is not None]
            if not ks: continue
            ax.plot([1] + ks, [c1] + [val(f"{short}.{fmt}.test.{fam}k{k}.p_low") for k in ks], ls=ls, marker=mk, ms=3.2, lw=1.3, color=col)
        c5 = val(f"{short}.{fmt}.test.C5k4.p_low")
        if c5 is not None: ax.scatter([4], [c5], marker="x", s=28, color=col, lw=1.4, zorder=5)
    ax.axhline(.5, color="grey", lw=.7, ls="-", alpha=.6)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_xlabel("copies / users k")
    ax.set_title(title); ax.set_ylim(-.02, 1.02)
axs[0].set_ylabel(r"$P_{\mathrm{low}}$ (low-credibility value)")
h1 = [Line2D([], [], color=c, lw=2, label=l) for _, l, c in M]
h2 = [Line2D([], [], color="k", ls=ls, marker=mk, ms=3.5, label=lab) for ls, mk, lab in STY.values()] + [Line2D([], [], color="k", marker="x", ls="none", label="length control (C5, k=4)")]
fig.legend(handles=h1 + h2, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.2), columnspacing=1.2, handlelength=2.2)
fig.tight_layout(); save(fig, "fig1_dose_response")

# ---------- fig2: head recovery ----------
def recovery(V, name):
    lc = np.mean([v["ld_clean"] for v in V]); lk = np.mean([v["ld_corrupt"] for v in V]); return (lk - np.mean([v[name] for v in V])) / (lk - lc) * 100
panels = [("Llama-3.1-8B-Instruct", "F1", "Llama-8B", "Llama-8B, tables", "#3b6fb6"), ("Qwen2.5-7B-Instruct", "F1", "Qwen-7B", "Qwen-7B, tables", "#d1495b"),
          ("gemma-3-12b-it", "F1", "Gemma-12B", "Gemma-12B, tables", "#2e9e6a"), ("Qwen2.5-7B-Instruct", "F3_C7k4", "Qwen-7B", "Qwen-7B, documents,\nparaphrased copies", "#d1495b")]
fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.5), sharey=True)
labels = ["top-1", "top-5", "top-10", "top-20", "mean-ablate", "random"]
for ax, (tag, fset, short, title, col) in zip(axs, panels):
    R = json.load(open(f"{RA}/patch_{fset}_{tag}.json")); V = R["verify"]
    fk = fset.replace("_", "-"); vals = []; errs = []
    for n in ("top1", "top5", "top10", "top20", "mean_top10"):
        x = A[f"{short}.{fk}.patch.{n}.recovery_pct"]; vals.append(x["value"]); errs.append([x["value"] - x["ci"][0], x["ci"][1] - x["value"]])
    ax.bar(range(5), vals, yerr=np.array(errs).T, color=[col] * 4 + ["#aaaaaa"], capsize=2, width=.7, error_kw={"lw": .8})
    rnd = [recovery(V, n) for n in R["hsets"] if n.startswith("rand10_")]
    ax.scatter(5 + np.random.default_rng(0).uniform(-.18, .18, len(rnd)), rnd, s=7, color="k", alpha=.6)
    ax.axhline(40, color="k", ls="--", lw=.7); ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8.5)
    ax.set_title(title, fontsize=8.5); ax.set_ylim(-15, 108)
axs[0].set_ylabel("% of repetition effect\nrecovered (held-out)")
fig.text(.5, -.04, "grey bar: mean-ablation of the top-10 heads;  dots: 20 random sets of 10 heads;  dashed line: 40% criterion", ha="center", fontsize=8)
fig.tight_layout(); save(fig, "fig2_head_recovery")

# ---------- fig5: C1-only probe by layer ----------
fig, ax = plt.subplots(figsize=(6.5, 2.7))
for tag, lab, col in (("Llama-3.1-8B-Instruct", "Llama-8B (exploratory)", "#3b6fb6"), ("Qwen2.5-7B-Instruct", "Qwen-7B", "#d1495b"), ("gemma-3-12b-it", "Gemma-12B", "#2e9e6a")):
    Q = json.load(open(f"{RA}/probe_c1only_{tag}.json")); Ls = sorted(Q["layers"], key=int); x = [int(l) / int(Ls[-1]) for l in Ls]
    ax.plot(x, [Q["layers"][l]["C2k8_flipped"] for l in Ls], "-o", ms=3, lw=1.4, color=col, label=f"{lab}: flipped items")
    ax.plot(x, [Q["layers"][l]["C0rk4"] for l in Ls], ":", lw=1.2, color=col, label=f"{lab}: no labels (C0r)")
    ax.axvline(Q["H4"]["Lstar"] / int(Ls[-1]), color=col, lw=.9, ls="--", alpha=.8)
ax.axhline(.5, color="grey", lw=.8); ax.axhline(.75, color="k", lw=.6, ls="--", alpha=.6)
ax.set_xlabel("relative depth (layer / last layer); dashed verticals = pre-registered L*"); ax.set_ylabel("probe AUC (held-out)")
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.25))
fig.tight_layout(); save(fig, "fig5_c1only_flipped_by_layer")

# ---------- figB3: fix trade-off ----------
CV = json.load(open(f"{RB}/b3_curve_Qwen2.5-7B-Instruct.json"))
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for name, lab, col in (("U", "union of both head sets", "#8a5cc2"), ("F3_C7k4", "paraphrase head set", "#d1495b"), ("Feeders", "paraphrase-only (feeder) heads", "#2e9e6a")):
    pts = sorted((float(g), p["BR_C7k4"], p["SP_F3"]) for g, p in CV["curves"][name]["points"].items())
    ax.plot([p[1] * 100 for p in pts], [p[2] * 100 for p in pts], "-o", ms=4, lw=1.4, color=col, label=lab)
    for g, xx, yy in pts:
        if g <= 0.25: ax.annotate(f"γ={g:g}", (xx * 100, yy * 100), fontsize=7.5, xytext=(4, 3), textcoords="offset points", color=col)
    ax.scatter([B[f"B3.{name}.F3.BR_C7k4"]["value"] * 100], [B[f"B3.{name}.F3.pref_kept"]["value"] * 100], s=140, facecolors="none", edgecolors=col, lw=1.8)
ax.axvline(50, color="k", ls="--", lw=.7); ax.axhline(70, color="k", ls="--", lw=.7)
ax.fill_between([50, 100], 70, 102, color="#2e9e6a", alpha=.07)
ax.scatter([79.2], [72.5], marker="*", s=110, color="k", label="trained adapter, Schuster et al. (other model)")
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.set_xlabel("paraphrase bias removed (%), documents, test"); ax.set_ylabel("source preference kept (%)")
ax.legend(fontsize=7.5, frameon=False, loc="lower left", title="damping γ = 0, 0.25, 0.5, 0.75 (right to left);\ncircles = pre-registered chosen points", title_fontsize=7.5)
fig.tight_layout(); save(fig, "figB3_tradeoff_F3")
print("polished: fig1, fig2, fig5, figB3")
