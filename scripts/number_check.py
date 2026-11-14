"""Check every hand-typed number in the paper's main text (paper/main.tex, abstract to Section 6) against the
numbers files and results. Sprint Block C (14 Nov 2026), updated after Block D to the final paper text.
  python3 scripts/number_check.py [--out FILE]   -> paper/NUMBER_CHECK.md (or FILE); exit code 1 if any check fails

Each claim is (where, quote, printed, source, value). A claim PASSES only if
  (1) `quote` occurs verbatim in paper/main.tex (whitespace-normalised), so a claim about stale text cannot pass, and
  (2) the source value, rounded to the printed precision, equals the printed value.
Paths are relative to the repository root (the parent of scripts/), so reproduce.py can run it on regenerated numbers.
"""
import json, os, re, sys
import numpy as np

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{S}/paper/NUMBER_CHECK.md"
J = lambda p: json.load(open(f"{S}/{p}"))
A, B, D = J("numbers_A.json"), J("numbers_B.json"), J("numbers_D.json")
BG = J("results/B/gemma/numbers_B_gemma-3-12b-it.json")
M3 = ("Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-3-12b-it")
SH = ("Llama-8B", "Qwen-7B", "Gemma-12B")
B0 = {m: J(f"results/B/b0_induction_{m}.json") for m in M3}
B2 = J("results/B/b2_steer_Qwen2.5-7B-Instruct.json")
MD = J("data/manifest_d.json")
TEX = re.sub(r"\s+", " ", open(f"{S}/paper/main.tex").read())
TEX = TEX[TEX.index(r"\begin{abstract}"):TEX.index(r"\section*{Code and Data}")]   # main text only


def g(Dct, k, i=None):
    x = Dct[k]
    return x["value"] if i is None else x["ci"][i]


a = lambda k, i=None: g(A, k, i)
b = lambda k, i=None: g(B, k, i)
d = lambda k, i=None: g(D, k, i)
C = []


def add(where, quote, printed, src, val, scale=1.0):
    C.append((where, quote, str(printed), src, val * scale))


# ---------------- Abstract ----------------
top10 = [a(f"{m}.F1.patch.top10.recovery_pct") for m in SH]
p95 = [a(f"{m}.F1.patch.rand10.recovery_p95") for m in SH]
add("Abstract", r"four paraphrased copies of one post flip 86\% of items", 86, "Qwen-32B.F3.test.C7k4.flip_pct", a("Qwen-32B.F3.test.C7k4.flip_pct"))
add("Abstract", r"against 3\% for four exact copies", 3, "Qwen-32B.F3.test.C2k4.flip_pct", a("Qwen-32B.F3.test.C2k4.flip_pct"))
add("Abstract", r"recover 92--95\% of the repetition effect", 92, "*.F1.patch.top10.recovery_pct (min of 3 models)", min(top10))
add("Abstract", r"recover 92--95\% of the repetition effect", 95, "*.F1.patch.top10.recovery_pct (max of 3 models)", max(top10))
add("Abstract", r"95th percentile of 14--31\% for random heads", 14, "*.F1.patch.rand10.recovery_p95 (min)", min(p95))
add("Abstract", r"95th percentile of 14--31\% for random heads", 31, "*.F1.patch.rand10.recovery_p95 (max)", max(p95))
add("Abstract", r"removes 95\% of the paraphrase effect", 95, "D.F3.BR_C7k4", d("D.F3.BR_C7k4"), 100)

# ---------------- Section 3 (design constants) ----------------
add("§3.5", "on 1,500 items about new fictional entities", "1500", "data/manifest_d.json items.train_d", MD["items"]["train_d"])

# ---------------- Section 4.1 + Table 2 ----------------
add("§4.1", r"reproduces at most 40\% of the shift in Qwen-7B", 40, "Qwen-7B.F1.test.C5_pct_of_C2k4_shift", a("Qwen-7B.F1.test.C5_pct_of_C2k4_shift"))
c4 = [a(f"{m}.F1.test.C4k4.shift_vs_C1_pts") for m in SH]
add("§4.1", "(+3 to +18 points)", 3, "*.F1.test.C4k4.shift_vs_C1_pts (min)", min(c4))
add("§4.1", "(+3 to +18 points)", 18, "*.F1.test.C4k4.shift_vs_C1_pts (max)", max(c4))
ROWS = {"Llama-8B": "Llama-3.1-8B", "Qwen-7B": "Qwen2.5-7B", "Gemma-12B": "Gemma-3-12B", "Qwen-32B": "Qwen2.5-32B"}
cols = [("F1", "C1k1"), ("F1", "C2k4"), ("F1", "C3k4"), ("F1", "C7k4"), ("F3", "C1k1"), ("F3", "C2k4"), ("F3", "C7k4")]
for m, name in ROWS.items():
    vals = [a(f"{m}.{f}.test.{ck}.p_low") for f, ck in cols]
    row = name + " & " + " & ".join(f"{v:.2f}".lstrip("0") for v in vals) + r" \\"   # the table row rebuilt from the data
    for (f, ck), v in zip(cols, vals):
        add("Table 2", row, f"{v:.2f}", f"{m}.{f}.test.{ck}.p_low", v)

# ---------------- Section 4.2 ----------------
q = "Gemma: 0.73 exact vs 0.83 paraphrased and 0.83 independent"
add("§4.2", q, "0.73", "Gemma-12B.F1.test.C2k4.p_low", a("Gemma-12B.F1.test.C2k4.p_low"))
add("§4.2", q, "0.83", "Gemma-12B.F1.test.C7k4.p_low", a("Gemma-12B.F1.test.C7k4.p_low"))
add("§4.2", q, "0.83", "Gemma-12B.F1.test.C3k4.p_low", a("Gemma-12B.F1.test.C3k4.p_low"))
add("§4.2", "near 0.5 (0.49--0.59)", "0.49", "Llama-8B.F3.test.C2k4.p_low", a("Llama-8B.F3.test.C2k4.p_low"))
add("§4.2", "near 0.5 (0.49--0.59)", "0.59", "Qwen-7B.F3.test.C2k4.p_low", a("Qwen-7B.F3.test.C2k4.p_low"))
q = "is 0.11 with four exact copies and 0.71 with four paraphrased copies"
add("§4.2", q, "0.11", "Qwen-32B.F3.test.C2k4.p_low", a("Qwen-32B.F3.test.C2k4.p_low"))
add("§4.2", q, "0.71", "Qwen-32B.F3.test.C7k4.p_low", a("Qwen-32B.F3.test.C7k4.p_low"))
for p, i in (("60.3", None), ("58.3", 0), ("62.2", 1)):
    add("§4.2", "(+60.3 points [58.3, 62.2])", p, "Qwen-32B.F3.test.C7k4_minus_C2k4_pts", a("Qwen-32B.F3.test.C7k4_minus_C2k4_pts", i))
q = r"Paraphrased copies flip 86\% of items, against 3\% for exact copies and 52\% for four independent users"
add("§4.2", q, 86, "Qwen-32B.F3.test.C7k4.flip_pct", a("Qwen-32B.F3.test.C7k4.flip_pct"))
add("§4.2", q, 3, "Qwen-32B.F3.test.C2k4.flip_pct", a("Qwen-32B.F3.test.C2k4.flip_pct"))
add("§4.2", q, 52, "Qwen-32B.F3.test.C3k4.flip_pct", a("Qwen-32B.F3.test.C3k4.flip_pct"))

# ---------------- Section 4.3 ----------------
for m, (v, lo, hi, pp) in {"Llama-8B": ("91.9", "90.4", "93.4", "30.9"), "Qwen-7B": ("95.4", "92.3", "99.0", "19.4"),
                           "Gemma-12B": ("93.1", "91.2", "95.2", "13.9")}.items():
    k = f"{m}.F1.patch.top10.recovery_pct"; quote = rf"{v}\% [{lo}, {hi}]"
    add("§4.3", quote, v, k, a(k)); add("§4.3", quote, lo, k + " ci lo", a(k, 0)); add("§4.3", quote, hi, k + " ci hi", a(k, 1))
    add("§4.3", r"the 95th percentile is 30.9\%, 19.4\% and 13.9\%", pp, f"{m}.F1.patch.rand10.recovery_p95", a(f"{m}.F1.patch.rand10.recovery_p95"))
ma = [a(f"{m}.F1.patch.mean_top10.recovery_pct") for m in SH]
add("§4.3", r"by mean ablation recovers less (57--79\%)", 57, "*.F1.patch.mean_top10.recovery_pct (min)", min(ma))
add("§4.3", r"by mean ablation recovers less (57--79\%)", 79, "*.F1.patch.mean_top10.recovery_pct (max)", max(ma))
q = r"the top ten recover 77\% (exact) and 78\% (paraphrased)"
add("§4.3", q, 77, "Qwen-7B.F3.patch.top10.recovery_pct", a("Qwen-7B.F3.patch.top10.recovery_pct"))
add("§4.3", q, 78, "Qwen-7B.F3-C7k4.patch.top10.recovery_pct", a("Qwen-7B.F3-C7k4.patch.top10.recovery_pct"))
add("§4.3", r"recovers 56\% alone", 56, "Gemma-12B.F1.patch.top1.recovery_pct", a("Gemma-12B.F1.patch.top1.recovery_pct"))
add("§4.3", "the top head, L29H11, is the persuasion decision head", 0, "Gemma-12B.F1.patch.sun_head_atp_rank (0 = top)", a("Gemma-12B.F1.patch.sun_head_atp_rank"))
add("§4.3", "their head L17H24 ranks second", 1, "Llama-8B.F1.patch.sun_head_atp_rank (1 = second)", a("Llama-8B.F1.patch.sun_head_atp_rank"))
add("§4.3", "is an induction head (0/10)", 0, "B0.*.F1.n_induction (max over models)", max(b(f"B0.{m}.F1.n_induction") for m in SH))
strong = lambda R: int(((np.array(R["all"]["prefix"]) >= .3) & (np.array(R["all"]["copy"]) > R["copy_p90_all"])).sum())
n_ind = [strong(R) for R in B0.values()]
add("§4.3", "each model contains 7--10 strong induction heads", 7, "b0_induction_*.json, all heads (min over models)", min(n_ind))
add("§4.3", "each model contains 7--10 strong induction heads", 10, "b0_induction_*.json, all heads (max over models)", max(n_ind))
for p, i in (("40.6", None), ("36.5", 0), ("44.7", 1)):
    add("§4.3", r"mediate 40.6\% [36.5, 44.7]", p, "B1.F1/C2k4.mediated_top10_pct", b("B1.F1/C2k4.mediated_top10_pct", i))
add("§4.3", r"random sets of ten mediate 7.0\%", "7.0", "B1.F1/C2k4.random_p95_pct", b("B1.F1/C2k4.random_p95_pct"))
add("§4.3", r"For paraphrases the share is 46.9\%", "46.9", "B1.F3/C7k4.mediated_top10_pct", b("B1.F3/C7k4.mediated_top10_pct"))

# ---------------- Section 4.4 ----------------
for lab, key, (q7, g12) in (("government exceeds newspaper", "gov_minus_newspaper", ("2.15", "1.98")),
                            ("newspaper exceeds person", "newspaper_minus_person", ("0.64", "1.19")),
                            ("person exceeds social media", "person_minus_social", ("3.39", "2.97"))):
    quote = f"{lab} (+{q7} and +{g12})"
    add("§4.4", quote, q7, f"Qwen-7B.controls.H3.{key}", a(f"Qwen-7B.controls.H3.{key}"))
    add("§4.4", quote, g12, f"Gemma-12B.controls.H3.{key}", a(f"Gemma-12B.controls.H3.{key}"))
for lev, p in zip(("gov", "newspaper", "person", "social"), ("0.57", "0.55", "0.48", "0.50")):
    add("§4.4", "is 0.57, 0.55, 0.48 and 0.50", p, f"Gemma-12B.controls.graded.behaviour_pX.{lev}", a(f"Gemma-12B.controls.graded.behaviour_pX.{lev}"))
q = "the probe scores 0.23 (Qwen-7B) and 0.13 (Gemma)"
add("§4.4", q, "0.23", "Qwen-7B.H4.a_flipped_auc_Lstar", a("Qwen-7B.H4.a_flipped_auc_Lstar"))
add("§4.4", q, "0.13", "Gemma-12B.H4.a_flipped_auc_Lstar", a("Gemma-12B.H4.a_flipped_auc_Lstar"))
add("§4.4", "(0.32 and 0.00)", "0.32", "Qwen-7B.H4.b_C0r_auc_Lstar", a("Qwen-7B.H4.b_C0r_auc_Lstar"))
add("§4.4", "(0.32 and 0.00)", "0.00", "Gemma-12B.H4.b_C0r_auc_Lstar", a("Gemma-12B.H4.b_C0r_auc_Lstar"))
q = r"at its last layer (0.79) while 90\% of answers flip"
add("§4.4", q, "0.79", "Llama-8B.H5.final_flipped_auc", a("Llama-8B.H5.final_flipped_auc"))
add("§4.4", q, 90, "Llama-8B.H5.choice_rate", a("Llama-8B.H5.choice_rate"), 100)
add("§4.4, Lim. (v)", "up to 32 standard deviations", 32, "b2_steer L28 alpha grid (max)", max(float(x) for x in B2["layers"]["28"]["sel_curve"]))
add("§4.4", "moves it by $-0.44$", "-0.44", "B2.L28.delta_positive_control", b("B2.L28.delta_positive_control"))
t = B2["layers"]["28"]["test"]
C.append(("§4.4", "by less than 0.001", "<0.001", "b2_steer L28 test: max |steer - base|", max(abs(x["steer"] - x["base"]) for x in t.values())))

# ---------------- Section 4.5 ----------------
for p, i in (("54", None), ("52", 0), ("57", 1)):
    add("§4.5", r"removes 54\% [52, 57] of the exact-copy effect", p, "B3.F1.F1.BR_C2k4", b("B3.F1.F1.BR_C2k4", i), 100)
for p, i in (("80", None), ("77", 0), ("82", 1)):
    add("§4.5", r"keeps 80\% [77, 82] of the source preference", p, "B3.F1.F1.pref_kept", b("B3.F1.F1.pref_kept", i), 100)
add("§4.5", r"60\% of the effect of independent users", 60, "B3.F1.F1.corrob_kept", b("B3.F1.F1.corrob_kept"), 100)
for p, i in (("-1.8", None), ("-3.6", 0), ("0.0", 1)):
    add("§4.5", r"costs 1.8 MMLU percentage points [$-3.6$, 0.0]", p, "B3.F1.mmlu_delta_pts (printed as a cost)", b("B3.F1.mmlu_delta_pts", i))
add("§4.5", r"raises perplexity by 0.3\%", "0.3", "B3.F1.ppl_increase_pct", b("B3.F1.ppl_increase_pct"))
add("§4.5", r"It removes 11\% of the paraphrase effect", 11, "B3.F1.F1.BR_C7k4", b("B3.F1.F1.BR_C7k4"), 100)
add("§4.5", r"or in Gemma (32\% removed)", 32, "results/B/gemma numbers: B3.F1.F1.BR_C2k4", g(BG, "B3.F1.F1.BR_C2k4"), 100)
for p, i in (("95", None), ("94", 0), ("96", 1)):
    add("§4.5", r"removes 95\% [94, 96] of the paraphrase effect", p, "D.F3.BR_C7k4", d("D.F3.BR_C7k4", i), 100)
C.append(("§4.5", "while keeping all of the source preference", ">=100", "D.F3.pref_kept (x100)", d("D.F3.pref_kept") * 100))
add("§4.5", "It costs 0.6 MMLU percentage points", "-0.6", "D.mmlu_delta_pts (printed as a cost)", d("D.mmlu_delta_pts"))
add("§4.5", r"a format it never saw (81\%)", 81, "D.F2.BR_C7k4", d("D.F2.BR_C7k4"), 100)
q = r"four independent users keep 2\% of their effect in documents and 11\% in tables"
add("§4.5", q, 2, "D.F3.corrob_kept", d("D.F3.corrob_kept"), 100)
add("§4.5", q, 11, "D.F1.corrob_kept", d("D.F1.corrob_kept"), 100)
damp = [v["value"] for k, v in B.items() if k.startswith("B3.") and k.endswith("corrob_kept")]
C.append(("§4.5", r"against 55\% or more under damping", ">=55", f"min over {len(damp)} Qwen-7B B3.*.corrob_kept (x100)", min(damp) * 100))
add("Lim. (vi)", r"an MMLU interval reaching $-3.6$ points", "-3.6", "B3.F1.mmlu_delta_pts ci lo", b("B3.F1.mmlu_delta_pts", 0))


# ---------------- evaluate ----------------
def check(printed, val):
    if printed.startswith("<"):
        return val < float(printed[1:]), f"{val:.4f}"
    if printed.startswith(">="):
        return round(val) >= float(printed[2:]), f"{val:.2f}"
    dec = len(printed.split(".")[1]) if "." in printed else 0
    return abs(val - float(printed)) <= 0.5 * 10 ** (-dec) + 1e-9, f"{val:.{dec + 2}f}"


rows, nbad, nquote = [], 0, 0
for where, quote, printed, src, val in C:
    in_text = quote in TEX
    ok_val, shown = check(printed, val)
    status = "PASS" if (in_text and ok_val) else ("**NOT IN TEXT**" if not in_text else "**MISMATCH**")
    nbad += status != "PASS"; nquote += not in_text
    rows.append((where, quote, printed, src, shown, status))
out = ["# NUMBER_CHECK: paper main text vs numbers files",
       "Generated by `scripts/number_check.py`. PASS = the quoted text occurs in `paper/main.tex` and the source value, "
       "rounded to the printed precision, equals the printed value.\n",
       f"**{len(C)} numeric claims checked: {len(C) - nbad} pass, {nbad} fail ({nquote} not found in the text).**\n",
       "| where | quoted text | printed | source | source value | check |", "|---|---|---|---|---|---|"]
out += [f"| {w} | {q.replace('|', '/')} | {p} | `{s}` | {v} | {st} |" for w, q, p, s, v, st in rows]
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
open(OUT, "w").write("\n".join(out) + "\n")
print(f"number_check: {len(C)} claims; {len(C) - nbad} pass; {nbad} fail")
for w, q, p, s, v, st in rows:
    if st != "PASS":
        print(" ", st.strip("*"), "|", w, "|", q, "| printed", p, "| source", v)
sys.exit(1 if nbad else 0)
