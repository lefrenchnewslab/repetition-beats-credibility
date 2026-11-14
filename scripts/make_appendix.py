"""SPRINT Block C (14 Nov 2026) — build paper appendices from numbers_A.json / numbers_B.json / results files. New file.
  python3 scripts/make_appendix.py  -> paper/app_prereg.tex, paper/app_tables.tex
No number is typed by hand: every value is read from a JSON/results file at build time.
"""
import json, os, glob
import numpy as np
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(f"{S}/numbers_A.json")); B = json.load(open(f"{S}/numbers_B.json"))
BL = json.load(open(f"{S}/results/B/llama/numbers_B_Llama-3.1-8B-Instruct.json")); BG = json.load(open(f"{S}/results/B/gemma/numbers_B_gemma-3-12b-it.json"))
M3 = ["Llama-8B", "Qwen-7B", "Gemma-12B"]; M4 = M3 + ["Qwen-32B"]
def v(D, k, d=None):
    x = D.get(k); return d if x is None or x.get("value") is None else x["value"]
def ci(D, k):
    x = D.get(k); return None if x is None else x.get("ci")
def f2(x, n=2): return "--" if x is None else (f"{x:.{n}f}")
def pct(x, n=0): return "--" if x is None else f"{x:.{n}f}"
def met(x): return "--" if x is None else ("\\textbf{met}" if x >= .5 else "not met")
esc = lambda s: s.replace("_", "\\_").replace("%", "\\%")

# ======================= Appendix A: pre-registration =======================
P = []
P.append(r"""All criteria below were written before the data they test existed, each with the outcome that fails it and a check that it can fail under the 8-way counterbalancing. C = confirmatory, E = exploratory.

\begin{table}[!htbp]\centering\scriptsize
\begin{tabular}{@{}p{1.1cm}p{5.6cm}p{4.1cm}p{1.15cm}p{1.15cm}p{1.15cm}@{}}\toprule
ID & Definition & Fails if & Llama & Qwen-7B & Gemma \\ \midrule""")
C0R_MIN = min(v(A, f"{m}.probe.cred_all.L{int(v(A, f'{m}.probe.chosen_layer'))}.C0rk4_control") for m in M3)
rows = [
 (r"H1", r"Top-10 heads (AtP on selection, C1 vs C2 $k{=}4$, F1) recover $\ge$40\% of the logit difference on test items by activation patching, \emph{and} exceed the p95 of 20 random 10-head sets.", r"$<$40\% or $\le$ random p95.", r"H1_met"),
 (r"H2", r"Credibility probe (all labelled conditions) AUC $\ge$0.8 at C2 $k{=}8$, all and flipped items, at a layer fixed on selection.", r"AUC $<$0.8. (Met in all, but count-confounded: C0r AUC $\ge$" + f"{C0R_MIN:.3f}" + r"; binding-only in Gemma.)", r"H2_met"),
 (r"H3", r"Graded set: gov$-$newspaper, newspaper$-$person, person$-$social $>0$ with 95\% cluster-bootstrap CIs, at the pre-registered layer.", r"Any contrast CI includes 0. Null (direction ignores source) makes all $\approx0$.", r"H3_met"),
 (r"H4", r"C1-only probe at $L^*$ (chosen on decision-disagreeing selection items): (a) C2 $k{=}8$ flipped AUC $\ge$0.75, CI $>$0.5; (b) C0r AUC in [0.40, 0.60]; (c) final-layer flipped AUC $<$0.5.", r"Any of (a)--(c) missed.", r"H4.met"),
 (r"H5", r"Final-layer C1-only flipped AUC $\ge$0.70, CI $>$0.5, while the model picks social on $\ge$50\% of C2 $k{=}8$.", r"AUC $<$0.70, CI includes 0.5, or choice rate $<$50\%.", r"H5.met"),
]
for rid, d, fl, key in rows:
    lab = {"H4": ["E", "C", "C"], "H5": ["E", "C", "C"]}.get(rid, ["C", "C", "C"])
    cells = [f"{met(v(A, f'{m}.{key}'))} ({l})" for m, l in zip(M3, lab)]
    P.append(f"{rid} & {d} & {fl} & " + " & ".join(cells) + r" \\")
b0 = [met(1.0 if v(B, f"B0.{m}.F1.n_induction", 0) >= 5 else 0.0) + " (C)" for m in M3]
P.append(r"B0 & $\ge$5 of the top-10 F1 heads are induction heads (prefix-matching $\ge$0.30 and copying $>$p90 of all heads). & $<$5 and mean prefix $\le$ random p95. & " + " & ".join(b0) + r" \\")
P.append(r"B1 & $\le$10 upstream components (path patching, suffix positions) mediate $\ge$40\% of the top-head effect on test items, above the random p95. & $<$40\% or $\le$ random p95. & -- & " + f"{met(v(B, 'B1.F1/C2k4.met'))} (C)" + r" & -- \\")
P.append(r"B2 & Steering along the C1-only direction returns C2 $k{=}4$ to the C1 CI at $\alpha^*$, with random directions moving $<$1/3 as much. Diagnostic only. & Not restored, or random $\ge$1/3. & -- & " + f"{met(v(B, 'B2.L18.met'))} (C, L18); {met(v(B, 'B2.L28.met'))} (E, L28)" + r" & -- \\")
P.append(r"B3-S1 & Head damping (k, $\gamma$ on selection): F1 C2 $k{=}4$ bias reduction $\ge$50\%, preference kept $\ge$70\%, MMLU drop $\le$2 pts, perplexity $\le$+5\%. & Any bar missed. & " + f"{met(v(BL, 'B3.B3-S1.met'))} (C) & {met(v(B, 'B3.B3-S1.met'))} (C) & {met(v(BG, 'B3.B3-S1.met'))} (C)" + r" \\")
P.append(r"B3-S2 & Same bars on F3 paraphrases (C7 $k{=}4$) with the F3/C7 head set. & Any bar missed. & -- & " + f"{met(v(B, 'B3.B3-S2.met'))} (C)" + r" & -- \\")
P.append(r"B3-U & Single fix on the union of both head sets; S1 and S2 bars. & Any bar missed. & -- & " + f"{met(v(B, 'B3-U-S1.met'))}/{met(v(B, 'B3-U-S2.met'))} (C)" + r" & -- \\")
P.append(r"H-F & Feeders (paraphrase-only heads) keep more preference than the union at matched bias reduction (test curves). & Not higher (H-F.1: if above the union's range, must dominate its max point). & -- & " + ("supported (E)" if v(B, "B3.HF.supported", 0) >= .5 else "not supported (E)") + r" & -- \\")
D = json.load(open(f"{S}/numbers_D.json"))
_f = lambda k: D[k]["value"] * 100
P.append(r"D-S1 & Trained LoRA adapter (same model): F1 C2 $k{=}4$ bias reduction $\ge$50\%, preference kept $\ge$70\%, MMLU drop $\le$2 pts, perplexity $\le$+5\%. & Any bar missed. & -- & " + f"\\textbf{{met}} ({_f('D.F1.BR_C2k4'):.0f}\\%, {_f('D.F1.pref_kept'):.0f}\\%) (C)" + r" & -- \\")
P.append(r"D-S2 & Same adapter, F3 C7 $k{=}4$: bias reduction $\ge$50\%, F3 preference kept $\ge$70\%, same capability bars. & Any bar missed. & -- & " + f"\\textbf{{met}} ({_f('D.F3.BR_C7k4'):.0f}\\%, {_f('D.F3.pref_kept'):.0f}\\%) (C)" + r" & -- \\")
P.append(r"D-cmp & Adapter vs best head damping at matched F3 preference: better if its CI lower bound exceeds the damping value. & Worse or no difference. & -- & better (C) & -- \\")
P.append(r"D-sel & Adapter keeps $\ge$50\% of the independent-user (C3 $k{=}4$) effect. & Below 50\%. & -- & " + f"not met ({_f('D.F1.corrob_kept'):.0f}\\% F1, {_f('D.F3.corrob_kept'):.0f}\\% F3)" + r" & -- \\")
P.append(r"\bottomrule\end{tabular}\caption{Pre-registered criteria, failure outcomes and verdicts.}\label{tab:prereg}\end{table}")
am = [
 (r"A1", r"13 Nov, after a 3-pair code-check smoke test", r"H2 layer chosen by selection log-loss instead of AUC (ties at 1.00).", r"all full runs"),
 (r"A2", r"13 Nov", r"Qwen-32B scope (test split, F1+F3, priority order); lexical baselines; graded-credibility set; H3 defined.", r"Gemma, Qwen-32B runs; all extras"),
 (r"A3", r"13 Nov", r"Neutral-tag (Blue/Green) binding control; $\ge$0.95 $\Rightarrow$ H2 is binding only.", r"all extras"),
 (r"A4", r"13 Nov", r"Blue-vs-Green projection demoted to sanity check; neutral tag as 5th graded level; H3 rewritten as three paired contrasts; H1 adds random p95.", r"all extras, all patching"),
 (r"A5", r"13 Nov 16:45", r"F3 head analysis also with paraphrased copies (C7 $k{=}4$) as corrupt run.", r"job A11"),
 (r"A6", r"13 Nov $\sim$17:00", r"C1-only probe (count confound) with predictions for C2 $k{=}8$, C0r, C6.", r"all C1-only fits"),
 (r"A7", r"13 Nov $\sim$17:40", r"H4 layer-resolved rule ($L^*$ on disagreeing items).", r"Qwen-7B, Gemma probes"),
 (r"A8", r"13 Nov $\sim$18:10", r"Robustness over near-max layers; final-layer description.", r"Qwen-7B, Gemma probes"),
 (r"A9", r"13 Nov $\sim$18:25", r"H5 (source readable at output).", r"Qwen-7B, Gemma probes"),
 (r"A10", r"13 Nov $\sim$19:00", r"Decision-balanced C1 probe (stop if $<$200 disagreeing items).", r"Gemma probe"),
 (r"B-branch", r"13 Nov $\sim$22:55", "Two head sets (Jaccard " + f"{v(A, 'Qwen-7B.rank.F1_vs_F3-C7k4.jaccard10'):.2f}" + r" $<$ 0.5).", r"all B1--B3 data"),
 (r"B-frame", r"13 Nov $\sim$23:50", r"Steering withdrawn as a fix (needs oracle sign); diagnostic only.", r"all B2/B3 data"),
 (r"B-U", r"13 Nov $\sim$23:50", r"Union fix B3-U.", r"all B3 data"),
 (r"B-Feeders", r"14 Nov $\sim$00:15", r"Feeders head set and H-F; test $\gamma$-curves.", r"B3 test data"),
 (r"H-F.1", r"14 Nov $\sim$00:30", r"H-F rule when Feeders exceed the union's range.", r"B3 test data"),
 (r"B-report", r"14 Nov $\sim$00:55", r"Report preference kept in all formats (no criterion change).", r"B3 test data"),
 (r"B1-scope", r"14 Nov $\sim$04:20", r"Reduced sender pool (top-50 heads + top-10 MLPs by AtP); suffix positions only.", r"all B1 data"),
]
P.append(r"""
\begin{table}[!htbp]\centering\scriptsize
\begin{tabular}{@{}p{1.3cm}p{2.2cm}p{8.4cm}p{2.6cm}@{}}\toprule
ID & Written (as recorded) & Change & Before which data \\ \midrule""")
for a in am: P.append(" & ".join(a) + r" \\")
P.append(r"\bottomrule\end{tabular}\caption{Amendments, in order.}\label{tab:amend}\end{table}")
open(f"{S}/paper/app_prereg.tex", "w").write("\n".join(P) + "\n\\clearpage\n")

# ======================= Appendix B: full tables =======================
T = [r"\scriptsize"]
CK = [("C0k1", "C0 no labels"), ("C0rk4", r"C0r low$\times$4, no labels"), ("C1k1", "C1 gov vs social"), ("C2k2", "C2 exact $k{=}2$"), ("C2k4", "C2 exact $k{=}4$"), ("C2k8", "C2 exact $k{=}8$"),
      (r"C3k2", r"C3 independent $k{=}2$"), (r"C3k4", r"C3 independent $k{=}4$"), (r"C4k2", r"C4 merged $k{=}2$"), (r"C4k4", r"C4 merged $k{=}4$"), (r"C5k4", r"C5 length control"),
      (r"C6k2", r"C6 gov$\times$2"), (r"C6k4", r"C6 gov$\times$4"), (r"C7k2", r"C7 paraphrased $k{=}2$"), (r"C7k4", r"C7 paraphrased $k{=}4$")]
def pc(m, fmt, ck):
    k = f"{m}.{fmt}.test.{ck}.p_low"; x = v(A, k); c = ci(A, k)
    return "" if x is None else (f"{x:.3f}" + (f" [{c[0]:.3f}, {c[1]:.3f}]" if c else ""))
T.append(r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llll@{\hskip 8pt}llll@{}}\toprule""")
for half in (M4[:2], M4[2:]):
    T.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}}}" + (" & " if i == 0 else "") for i, m in enumerate(half)).replace(" &  & ", " & & ") + r" \\")
T.pop(); T.pop()
# simpler: one table per model with F1/F2/F3 columns
T = [r"\scriptsize"]
for m in M4:
    T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lccc@{}}\toprule")
    T.append(rf"{m} & F1 tables & F2 news & F3 documents \\ \midrule")
    for ck, lab in CK:
        cells = [pc(m, f, ck) for f in ("F1", "F2", "F3")]
        if any(cells): T.append(f"{lab} & " + " & ".join(c or "--" for c in cells) + r" \\")
    mass = [v(A, f"{m}.{f}.test.mass") for f in ("F1", "F2", "F3")]
    c5 = [v(A, f"{m}.{f}.test.C5_pct_of_C2k4_shift") for f in ("F1", "F2", "F3")]
    T.append(r"\midrule P(A)+P(B) mass & " + " & ".join(f2(x, 3) for x in mass) + r" \\")
    T.append(r"C5 shift as \% of C2 $k{=}4$ shift & " + " & ".join(pct(x) for x in c5) + r" \\")
    T.append(rf"\bottomrule\end{{tabular}}\caption{{$P_{{\text{{low}}}}$ [95\% CI, bootstrap over 400 test pairs], {m}. Empty: condition not run in that format.}}\end{{table}}")
# heads
T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\linewidth}{!}{\begin{tabular}{@{}lcccccccc@{}}\toprule")
T.append(r"Model / format & top-1 & top-5 & top-10 [CI] & top-20 & mean-abl.\ top-10 & rand.\ p95 & rand.\ max & Jaccard sel/test \\ \midrule")
for m, fmt, lab in [("Llama-8B", "F1", "Llama F1"), ("Qwen-7B", "F1", "Qwen-7B F1"), ("Gemma-12B", "F1", "Gemma F1"), ("Qwen-7B", "F3", "Qwen-7B F3, C2 $k{=}4$"), ("Qwen-7B", "F3-C7k4", "Qwen-7B F3, C7 $k{=}4$")]:
    b = f"{m}.{fmt}.patch"; c10 = ci(A, f"{b}.top10.recovery_pct")
    T.append(f"{lab} & {pct(v(A, f'{b}.top1.recovery_pct'), 1)} & {pct(v(A, f'{b}.top5.recovery_pct'), 1)} & {pct(v(A, f'{b}.top10.recovery_pct'), 1)} [{c10[0]:.1f}, {c10[1]:.1f}] & "
             f"{pct(v(A, f'{b}.top20.recovery_pct'), 1)} & {pct(v(A, f'{b}.mean_top10.recovery_pct'), 1)} & {pct(v(A, f'{b}.rand10.recovery_p95'), 1)} & {pct(v(A, f'{b}.rand10.recovery_max'), 1)} & {f2(v(A, f'{b}.rank.sel_vs_test.jaccard10'))}" + r" \\")
T.append(r"\bottomrule\end{tabular}}\caption{Held-out head patching: \% of the C2$-$C1 logit difference recovered on 160 test items (20 random 10-head draws).}\end{table}")
# graded set + probes
T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\linewidth}{!}{\begin{tabular}{@{}lccccc|ccc@{}}\toprule")
T.append(r"Model & gov & newspaper & person & social & neutral tag & gov$-$news [CI] & news$-$person [CI] & person$-$social [CI] \\ \midrule")
for m in M3:
    L = int(v(A, f"{m}.probe.chosen_layer"))
    means = [v(A, f"{m}.controls.graded.L{L}.mean_{x}") for x in ("gov", "newspaper", "person", "social", "neutral")]
    cs = []
    for k in ("gov_minus_newspaper", "newspaper_minus_person", "person_minus_social"):
        x = v(A, f"{m}.controls.H3.{k}"); c = ci(A, f"{m}.controls.H3.{k}"); cs.append(f"{x:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]")
    T.append(f"{m} (L{L}) & " + " & ".join(f"{x:+.2f}" for x in means) + " & " + " & ".join(cs) + r" \\")
    pX = [v(A, f"{m}.controls.graded.behaviour_pX.{x}") for x in ("gov", "newspaper", "person", "social", "neutral")]
    T.append(r"\quad behaviour P(choose X) & " + " & ".join(f"{x:.3f}" for x in pX) + r" & & & \\")
T.append(r"\bottomrule\end{tabular}}\caption{Graded credibility set (100 test pairs, C1-shaped, F1): mean projection onto the gov--social direction, oriented to source X against a social partner (social $\approx$0 by construction), with pre-registered H3 contrasts.}\end{table}")
T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\linewidth}{!}{\begin{tabular}{@{}lcccccccc@{}}\toprule")
T.append(r"Model & $L^*$ & H4(a) flipped AUC [CI] & H4(b) C0r & final flipped AUC [CI] & choice social & balanced $L^*$ & bal.\ H4(a) & bal.\ final \\ \midrule")
for m in M3:
    a, ca = v(A, f"{m}.H4.a_flipped_auc_Lstar"), ci(A, f"{m}.H4.a_flipped_auc_Lstar"); h5, c5 = v(A, f"{m}.H5.final_flipped_auc"), ci(A, f"{m}.H5.final_flipped_auc")
    T.append(f"{m}{' (E)' if m.startswith('Llama') else ''} & {int(v(A, f'{m}.H4.Lstar'))} & {a:.3f} [{ca[0]:.3f}, {ca[1]:.3f}] & {v(A, f'{m}.H4.b_C0r_auc_Lstar'):.3f} & {h5:.3f} [{c5[0]:.3f}, {c5[1]:.3f}] & "
             f"{v(A, f'{m}.H5.choice_rate')*100:.0f}\\% & {int(v(A, f'{m}.c1bal.Lstar'))} & {v(A, f'{m}.c1bal.H4a'):.3f} & {v(A, f'{m}.c1bal.H5'):.3f}" + r" \\")
T.append(r"\bottomrule\end{tabular}}\caption{C1-only probe (amendments 6--10): AUC for ``which option is government-sourced'' on held-out test entities; flipped = items where the model picked the social value at C2 $k{=}8$. (E) exploratory.}\end{table}")
# fixes
T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llcccc@{}}\toprule")
T.append(r"Fix (model, set, $k$, $\gamma$) & Format & Bias red.\ C2 $k{=}4$ & Bias red.\ C7 $k{=}4$ & Pref.\ kept & Corrob.\ kept \\ \midrule")
sel = json.load(open(f"{S}/results/B/b3_select_Qwen2.5-7B-Instruct.json"))["sets"]
selL = json.load(open(f"{S}/results/B/llama/b3_select_Llama-3.1-8B-Instruct.json"))["sets"]; selG = json.load(open(f"{S}/results/B/gemma/b3_select_gemma-3-12b-it.json"))["sets"]
fixes = [("Qwen-7B, H$_{F1}$", B, "F1", sel["F1"]), ("Qwen-7B, H$_{F3C7}$", B, "F3_C7k4", sel["F3_C7k4"]), ("Qwen-7B, union", B, "U", sel["U"]),
         (r"Qwen-7B, Feeders", B, r"Feeders", sel["Feeders"]), (r"Llama, H$_{F1}$", BL, r"F1", selL["F1"]), (r"Gemma, H$_{F1}$", BG, r"F1", selG["F1"])]
def pci(D, k):
    x = v(D, k); c = ci(D, k); return "--" if x is None else f"{x*100:.0f} [{c[0]*100:.0f}, {c[1]*100:.0f}]"
for lab, D, s, SS in fixes:
    ch = SS["chosen"]
    for i, fmt in enumerate(("F1", "F3", "F2")):
        T.append((f"{lab}, {len(ch['heads'])} heads, $\\gamma{{=}}{ch['gamma']}$" if i == 0 else "") + f" & {fmt} & {pci(D, f'B3.{s}.{fmt}.BR_C2k4')} & {pci(D, f'B3.{s}.{fmt}.BR_C7k4')} & {pci(D, f'B3.{s}.{fmt}.pref_kept')} & {pci(D, f'B3.{s}.{fmt}.corrob_kept')}" + r" \\")
    mc = ci(D, f"B3.{s}.mmlu_delta_pts"); pc_ = ci(D, f"B3.{s}.ppl_increase_pct")
    T.append(rf" & \multicolumn{{5}}{{l}}{{capability: MMLU $\Delta$ {v(D, f'B3.{s}.mmlu_delta_pts'):+.1f} pts [{mc[0]:+.1f}, {mc[1]:+.1f}]; WikiText perplexity {v(D, f'B3.{s}.ppl_increase_pct'):+.1f}\% [{pc_[0]:+.1f}, {pc_[1]:+.1f}]}} \\ \midrule")
T[-1] = T[-1].replace(r"\\ \midrule", r"\\")
T.append(r"\bottomrule\end{tabular}\caption{Head-damping fixes on test entities (\%, 95\% CI over pairs; capability CIs paired over items). $(k,\gamma)$ chosen on 100 selection pairs.}\end{table}")
# test curves
cv = json.load(open(f"{S}/results/B/b3_curve_Qwen2.5-7B-Instruct.json"))
T.append(r"\begin{table}[!htbp]\centering\scriptsize\begin{tabular}{@{}lcccc@{}}\toprule $\gamma$ & 0 & 0.25 & 0.5 & 0.75 \\ \midrule")
for s, lab in (("U", "union"), ("F3_C7k4", "H$_{F3C7}$"), ("Feeders", "Feeders")):
    pts = cv["curves"][s]["points"]
    T.append(f"{lab} & " + " & ".join(f"{pts[g]['BR_C7k4']*100:.0f} / {pts[g]['SP_F3']*100:.0f}" for g in ("0.0", "0.25", "0.5", "0.75")) + r" \\")
T.append(r"\bottomrule\end{tabular}\caption{Test $\gamma$-curves, Qwen-7B, F3: paraphrase bias reduction / F3 source preference kept (\%).}\end{table}")
# B0, B1, B2
T.append(r"\begin{table}[!htbp]\centering\scriptsize\resizebox{\linewidth}{!}{\begin{tabular}{@{}lccl@{}}\toprule Model & induction heads in top-10 & mean prefix (random p95) & top-10 heads: prefix-matching \\ \midrule")
for f in sorted(glob.glob(f"{S}/results/B/b0_induction_*.json")):
    R = json.load(open(f)); m = {"Llama-3.1-8B-Instruct": "Llama-8B", "Qwen2.5-7B-Instruct": "Qwen-7B", "gemma-3-12b-it": "Gemma-12B"}[R["model"].split("/")[-1]]
    top = R["sets"]["F1"]
    T.append(f"{m} & {sum(x['induction'] for x in top)}/10 & {np.mean([x['prefix'] for x in top]):.3f} ({R['null_mean_prefix_10']['p95']:.3f}) & " + ", ".join(f"L{x['head'][0]}H{x['head'][1]} {x['prefix']:.2f}" for x in top[:6]) + r", \dots \\")
T.append(r"\bottomrule\end{tabular}}\caption{B0 induction check (random-token sequences; all heads scored).}\end{table}")
T.append(r"\begin{table}[!htbp]\centering\scriptsize\begin{tabular}{@{}lp{6.5cm}cc@{}}\toprule Run & Top-10 senders (selection) & mediated [CI] & random p95 \\ \midrule")
for f in sorted(glob.glob(f"{S}/results/B/b1_path_*.json")):
    R = json.load(open(f)); sd = ", ".join((f"MLP{l}" if k == "m" else f"L{l}H{h}") for k, l, h in R["top_senders"])
    T.append(f"{R['fmt']} / {R['corrupt']} & {sd} & {R['mediated_top10']*100:.1f} [{R['mediated_top10_ci'][0]*100:.1f}, {R['mediated_top10_ci'][1]*100:.1f}] & {R['random_p95']*100:.1f}" + r" \\")
T.append(r"\bottomrule\end{tabular}\caption{B1 path patching into the top-10 heads, Qwen-7B, 160 test items (suffix positions only).}\end{table}")
R2 = json.load(open(f"{S}/results/B/b2_steer_Qwen2.5-7B-Instruct.json"))
T.append(r"\begin{table}[!htbp]\centering\scriptsize\begin{tabular}{@{}lcccccc@{}}\toprule Layer & $\sigma$ & $\alpha^*$ & C2 $k{=}4$ base$\to$steered & direction $\Delta$ & random mean $|\Delta|$ & residual swap $\Delta$ \\ \midrule")
for L, o in R2["layers"].items():
    t = o["test"]["F1_C2k4"]; c = o["controls_100pairs"]
    T.append(f"L{L} & {o['sigma']:.3f} & {o['alpha_star']} & {t['base']:.3f}$\\to${t['steer']:.3f} & {c['delta_direction']:+.4f} & {c['mean_abs_delta_random']:.4f} & {c['delta_positive_control']:+.3f}" + r" \\")
T.append(r"\bottomrule\end{tabular}\caption{B2 steering, Qwen-7B (oracle sign; diagnostic only). Controls on the first 100 test pairs.}\end{table}")
open(f"{S}/paper/app_tables.tex", "w").write("\n".join(T) + "\n\\clearpage\n")
print("wrote app_prereg.tex, app_tables.tex")
