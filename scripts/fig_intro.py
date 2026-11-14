"""Figure 1 (intro): Qwen2.5-32B, retrieved documents, test entities: share of items answered with the social-media value."""
import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = sys.argv[1] if len(sys.argv) > 1 else "."
A = json.load(open(f"{S}/numbers_A.json"))
k = lambda c: A[f"Qwen-32B.F3.test.{c}.flip_pct"]["value"]
rows = [("one post", k("C1k1")), ("4 exact copies\nof one post", k("C2k4")), ("4 distinct users,\nsame text", k("C3k4")), ("4 reworded copies\nof one post", k("C7k4"))]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.2, 2.1))
cols = ["#9aa5b1", "#9aa5b1", "#5b8db8", "#d1495b"]
y = range(len(rows))[::-1]
ax.barh(list(y), [v for _, v in rows], color=cols, height=0.62)
for yy, (_, v) in zip(y, rows): ax.text(v + 1.5, yy, f"{v:.0f}%", va="center", fontsize=9)
ax.set_yticks(list(y)); ax.set_yticklabels([r for r, _ in rows]); ax.set_xlim(0, 100)
ax.set_xlabel("items answered with the social-media value (%), against one government source")
fig.tight_layout(); fig.savefig(f"{S}/figs/fig0_headline.pdf", bbox_inches="tight"); fig.savefig(f"{S}/figs/fig0_headline.png", dpi=200, bbox_inches="tight")
print([(r.replace(chr(10), ' '), round(v, 2)) for r, v in rows])
