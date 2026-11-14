"""Figure 5: paraphrase bias removed vs source preference kept, Qwen2.5-7B, retrieved documents, test (head damping curves + same-model LoRA)."""
import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = sys.argv[1] if len(sys.argv) > 1 else "."
CV = json.load(open(f"{S}/results/B/b3_curve_Qwen2.5-7B-Instruct.json")); D = json.load(open(f"{S}/numbers_D.json")); B = json.load(open(f"{S}/numbers_B.json"))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for name, lab, col in (("U", "head damping: union", "#8a5cc2"), ("F3_C7k4", "head damping: paraphrase heads", "#d1495b"), ("Feeders", "head damping: feeder heads", "#2e9e6a")):
    pts = sorted((float(g), p["BR_C7k4"], p["SP_F3"]) for g, p in CV["curves"][name]["points"].items())
    ax.plot([p[1] * 100 for p in pts], [p[2] * 100 for p in pts], "-o", ms=3.5, lw=1.3, color=col, label=lab)
br, sp, c = D["D.F3.BR_C7k4"], D["D.F3.pref_kept"], D["D.F3.corrob_kept"]["value"]
ax.errorbar([br["value"] * 100], [sp["value"] * 100], xerr=[[(br["value"] - br["ci"][0]) * 100], [(br["ci"][1] - br["value"]) * 100]],
            yerr=[[(sp["value"] - sp["ci"][0]) * 100], [(sp["ci"][1] - sp["value"]) * 100]], fmt="D", ms=7, color="#e8a33d", ecolor="#e8a33d", capsize=3, label="trained LoRA adapter (same model)")
ax.axvline(50, color="k", ls="--", lw=.7); ax.axhline(70, color="k", ls="--", lw=.7); ax.fill_between([50, 100], 70, 115, color="#2e9e6a", alpha=.07)
ax.annotate(f"distinct users: {c*100:.0f}% kept", (br["value"] * 100, sp["value"] * 100), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=7.5, color="#b07415")
x, y, cc = B["B3.F3_C7k4.F3.BR_C7k4"]["value"], B["B3.F3_C7k4.F3.pref_kept"]["value"], B["B3.F3_C7k4.F3.corrob_kept"]["value"]
ax.annotate(f"distinct users: {cc*100:.0f}% kept", (x * 100, y * 100), xytext=(8, 4), textcoords="offset points", fontsize=7.5, color="#d1495b")
ax.set_xlim(0, 100); ax.set_ylim(0, 115); ax.set_xlabel("paraphrase effect removed (%), documents, test"); ax.set_ylabel("source preference kept (%)")
ax.legend(fontsize=7.5, frameon=False, loc="lower left"); fig.tight_layout()
fig.savefig(f"{S}/figs/figD_tradeoff_F3.pdf", bbox_inches="tight"); fig.savefig(f"{S}/figs/figD_tradeoff_F3.png", dpi=200, bbox_inches="tight"); print("ok")
