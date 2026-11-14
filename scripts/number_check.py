"""Check every hand-typed number in the paper's main text (paper/main.tex, abstract to Section 6) against the
numbers files and results. Sprint Block C (14 Nov 2026), updated after Block D and again after Block F (reviewer
controls) to the revised paper text.
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
A, B, D, F = J("numbers_A.json"), J("numbers_B.json"), J("numbers_D.json"), J("numbers_F.json")
BG = J("results/B/gemma/numbers_B_gemma-3-12b-it.json")
M3 = ("Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-3-12b-it")
SH = ("Llama-8B", "Qwen-7B", "Gemma-12B")
B0 = {m: J(f"results/B/b0_induction_{m}.json") for m in M3}
B2 = J("results/B/b2_steer_Qwen2.5-7B-Instruct.json")
MD = J("data/manifest_d.json")
TEX = re.sub(r"\s+", " ", open(f"{S}/paper/main.tex").read())
TEX = TEX[TEX.index(r"\begin{abstract}"):TEX.index(r"\input{app_prereg}")]   # main text + the verdicts table (Appendix A, typed by hand)


def g(Dct, k, i=None):
    x = Dct[k]
    return x["value"] if i is None else x["ci"][i]


a = lambda k, i=None: g(A, k, i)
b = lambda k, i=None: g(B, k, i)
d = lambda k, i=None: g(D, k, i)
fv = lambda k, i=None: g(F, k, i)
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
add("Abstract", r"(random sets: 95th percentile $\le$31\%)", 31, "*.F1.patch.rand10.recovery_p95 (max)", max(p95))
add("Abstract", r"removes 95\% of the paraphrase effect", 95, "D.F3.BR_C7k4", d("D.F3.BR_C7k4"), 100)
add("Abstract", r"four distinct users then keep only 2--11\% of their effect", 2, "D.F3.corrob_kept", d("D.F3.corrob_kept"), 100)
add("Abstract", r"four distinct users then keep only 2--11\% of their effect", 11, "D.F1.corrob_kept", d("D.F1.corrob_kept"), 100)

# ---------------- Table 2 (pre-registered criteria, Appendix A), Block F rows ----------------
for m in ("Qwen-32B", "Qwen-7B"):   # P1 (F1-main): D's CI lower bound > 0 and R_p > L_p (Qwen-7B is the table's column)
    met = float(fv(f"F1.{m}.F3.D_pts", 0) > 0 and fv(f"F1.{m}.F3.R_p_pts") > fv(f"F1.{m}.F3.L_p_pts"))
    add("Table 2", r"P1 paraphrase gain exceeds exact-copy gain & -- & \textbf{met} & --", 1, f"F1.{m}.F3 D ci lo > 0 and R_p > L_p (1 = met)", met)
for m in SH:
    add("Table 2", "P2 top heads specific to the conflict & not met & not met & not met", 0, f"F2.{m}.specific (0 = not met)", fv(f"F2.{m}.specific"))

# ---------------- Section 3 (design constants) ----------------
add("§3.5", "on 1,500 items about new fictional entities", "1500", "data/manifest_d.json items.train_d", MD["items"]["train_d"])

# ---------------- Section 4.1 + Table 1 ----------------
add("§4.1", r"reproduces at most 40\% of the shift in Qwen-7B", 40, "Qwen-7B.F1.test.C5_pct_of_C2k4_shift", a("Qwen-7B.F1.test.C5_pct_of_C2k4_shift"))
c4 = [a(f"{m}.F1.test.C4k4.shift_vs_C1_pts") for m in SH]
add("§4.1", "moves it only 3--18 points", 3, "*.F1.test.C4k4.shift_vs_C1_pts (min)", min(c4))
add("§4.1", "moves it only 3--18 points", 18, "*.F1.test.C4k4.shift_vs_C1_pts (max)", max(c4))
ROWS = {"Llama-8B": "Llama-3.1-8B", "Qwen-7B": "Qwen2.5-7B", "Gemma-12B": "Gemma-3-12B", "Qwen-32B": "Qwen2.5-32B"}
cols = [("F1", "C1k1"), ("F1", "C2k4"), ("F1", "C3k4"), ("F1", "C7k4"), ("F3", "C1k1"), ("F3", "C2k4"), ("F3", "C7k4")]
for m, name in ROWS.items():
    vals = [a(f"{m}.{f}.test.{ck}.p_low") for f, ck in cols]
    row = name + " & " + " & ".join(f"{v:.2f}".lstrip("0") for v in vals) + r" \\"   # the table row rebuilt from the data
    for (f, ck), v in zip(cols, vals):
        add("Table 1", row, f"{v:.2f}", f"{m}.{f}.test.{ck}.p_low", v)

# ---------------- Section 4.2 ----------------
q = r"four paraphrased copies of one post flip 86\% of Qwen-32B's items, against 3\% for four exact copies and 52\% for four users posting identical text"
add("§4.2", q, 86, "Qwen-32B.F3.test.C7k4.flip_pct", a("Qwen-32B.F3.test.C7k4.flip_pct"))
add("§4.2", q, 3, "Qwen-32B.F3.test.C2k4.flip_pct", a("Qwen-32B.F3.test.C2k4.flip_pct"))
add("§4.2", q, 52, "Qwen-32B.F3.test.C3k4.flip_pct", a("Qwen-32B.F3.test.C3k4.flip_pct"))
q = "In Qwen-32B one paraphrased copy moves the answer by 1 point and filler as long as four paraphrases by 0.3"
add("§4.2 (Block F)", q, 1, "F1.Qwen-32B.F3.W_pts", fv("F1.Qwen-32B.F3.W_pts"))
add("§4.2 (Block F)", q, "0.3", "F1.Qwen-32B.F3.L_p_pts", fv("F1.Qwen-32B.F3.L_p_pts"))

# ---------------- Section 4.3 ----------------
q = r"recover 92--95\% of the repetition effect on held-out entities, against at most 31\% for random sets of ten"
add("§4.3", q, 92, "*.F1.patch.top10.recovery_pct (min)", min(top10))
add("§4.3", q, 95, "*.F1.patch.top10.recovery_pct (max)", max(top10))
add("§4.3", q, 31, "*.F1.patch.rand10.recovery_p95 (max)", max(p95))
for lab, cn, lo, hi in (("credibility alone", "C0k1->C1k1", 72, 93), ("unlabelled repetition", "C0k1->C0rk4", 90, 91),
                        ("a repeated government source", "C1k1->C6k4", 93, 97)):
    r = [fv(f"F2.{m}.{cn}.recovery_top10_pct") for m in SH]; q = rf"{lab} ({lo}--{hi}\%)"
    add("§4.3 (Block F)", q, lo, f"F2.*.{cn}.recovery_top10_pct (min of 3 models)", min(r))
    add("§4.3 (Block F)", q, hi, f"F2.*.{cn}.recovery_top10_pct (max of 3 models)", max(r))
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
add("§4.3", r"the random p95 is 7.0\%", "7.0", "B1.F1/C2k4.random_p95_pct", b("B1.F1/C2k4.random_p95_pct"))
add("§4.3", r"For paraphrases the share is 46.9\%", "46.9", "B1.F3/C7k4.mediated_top10_pct", b("B1.F3/C7k4.mediated_top10_pct"))

# ---------------- Section 4.4 ----------------
h3lo = [a(f"{m}.controls.H3.{k}", 0) for m in ("Qwen-7B", "Gemma-12B") for k in ("gov_minus_newspaper", "newspaper_minus_person", "person_minus_social")]
C.append(("§4.4", r"each step with a 95\% CI above zero", ">0", "min CI lower bound over the 6 H3 contrasts (Qwen-7B, Gemma)", min(h3lo)))
q = "probe AUC is 0.23 (Qwen-7B) and 0.13 (Gemma)"
add("§4.4", q, "0.23", "Qwen-7B.H4.a_flipped_auc_Lstar", a("Qwen-7B.H4.a_flipped_auc_Lstar"))
add("§4.4", q, "0.13", "Gemma-12B.H4.a_flipped_auc_Lstar", a("Gemma-12B.H4.a_flipped_auc_Lstar"))
q = "A decision-balanced probe reverses too in Gemma (confirmatory) and at Qwen-7B's last layer"
C.append(("§4.4", q, "<0.5", "Gemma-12B.c1bal.H4a (flipped-item AUC at L*)", a("Gemma-12B.c1bal.H4a")))
C.append(("§4.4", q, "<0.5", "Qwen-7B.c1bal.H5 (flipped-item AUC, last layer)", a("Qwen-7B.c1bal.H5")))
add("§4.4", "but not at its selected layer (0.58, exploratory", "0.58", "Qwen-7B.c1bal.H4a", a("Qwen-7B.c1bal.H4a"))
q = r"(0.80 and 0.79) while 90\% of answers flip"
add("§4.4", q, "0.80", "Llama-8B.H4.a_flipped_auc_Lstar", a("Llama-8B.H4.a_flipped_auc_Lstar"))
add("§4.4", q, "0.79", "Llama-8B.H5.final_flipped_auc", a("Llama-8B.H5.final_flipped_auc"))
add("§4.4", q, 90, "Llama-8B.H5.choice_rate", a("Llama-8B.H5.choice_rate"), 100)
add("§4.4, Lim. (v)", "up to 32 standard deviations", 32, "b2_steer L28 alpha grid (max)", max(float(x) for x in B2["layers"]["28"]["sel_curve"]))
add("§4.4", "moves it by $-0.44$", "-0.44", "B2.L28.delta_positive_control", b("B2.L28.delta_positive_control"))
t = B2["layers"]["28"]["test"]
C.append(("§4.4", "by less than 0.001", "<0.001", "b2_steer L28 test: max |steer - base|", max(abs(x["steer"] - x["base"]) for x in t.values())))

# ---------------- Section 4.5 ----------------
add("§4.5", r"removes 54\% of the exact-copy effect", 54, "B3.F1.F1.BR_C2k4", b("B3.F1.F1.BR_C2k4"), 100)
q = r"It keeps 80\% of the source preference and 60\% of the effect of distinct users"
add("§4.5", q, 80, "B3.F1.F1.pref_kept", b("B3.F1.F1.pref_kept"), 100)
add("§4.5", q, 60, "B3.F1.F1.corrob_kept", b("B3.F1.F1.corrob_kept"), 100)
for p, i in (("-1.8", None), ("-3.6", 0), ("0.0", 1)):
    add("§4.5", r"costs 1.8 MMLU percentage points [$-3.6$, 0.0]", p, "B3.F1.mmlu_delta_pts (printed as a cost)", b("B3.F1.mmlu_delta_pts", i))
add("§4.5", r"raises perplexity by 0.3\%", "0.3", "B3.F1.ppl_increase_pct", b("B3.F1.ppl_increase_pct"))
add("§4.5", r"It removes 11\% of the paraphrase effect", 11, "B3.F1.F1.BR_C7k4", b("B3.F1.F1.BR_C7k4"), 100)
add("§4.5", r"or in Gemma (32\% removed)", 32, "results/B/gemma numbers: B3.F1.F1.BR_C2k4", g(BG, "B3.F1.F1.BR_C2k4"), 100)
add("§4.5", r"removes 95\% of the paraphrase effect in documents", 95, "D.F3.BR_C7k4", d("D.F3.BR_C7k4"), 100)
C.append(("§4.5", "while keeping all of the source preference", ">=100", "D.F3.pref_kept (x100)", d("D.F3.pref_kept") * 100))
add("§4.5", "It costs 0.6 MMLU percentage points", "-0.6", "D.mmlu_delta_pts (printed as a cost)", d("D.mmlu_delta_pts"))
add("§4.5", r"a format it never saw (81\%)", 81, "D.F2.BR_C7k4", d("D.F2.BR_C7k4"), 100)
q = r"four distinct users keep 2\% of their effect in documents and 11\% in tables"
add("§4.5", q, 2, "D.F3.corrob_kept", d("D.F3.corrob_kept"), 100)
add("§4.5", q, 11, "D.F1.corrob_kept", d("D.F1.corrob_kept"), 100)
damp = [v["value"] for k, v in B.items() if k.startswith("B3.") and k.endswith("corrob_kept")]
C.append(("§4.5", r"against 55\% or more under damping", ">=55", f"min over {len(damp)} Qwen-7B B3.*.corrob_kept (x100)", min(damp) * 100))
add("Lim. (vi)", r"an MMLU interval reaching $-3.6$ points", "-3.6", "B3.F1.mmlu_delta_pts ci lo", b("B3.F1.mmlu_delta_pts", 0))


# ---------------- evaluate ----------------
def check(printed, val):
    if printed.startswith(">") and not printed.startswith(">="):
        return val > float(printed[1:]), f"{val:.4f}"
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
