"""SPRINT Block C (14 Nov 2026) — build paper appendices from numbers_A.json / numbers_B.json / results files. New file.
Block F (14 Nov, evening): pre-registration rows P1/P2, amendment rows, and the Block F tables from numbers_F.json.
  python3 scripts/make_appendix.py  -> paper/app_prereg.tex, paper/app_tables.tex, paper/app_extra.tex
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
P.append(r"B1 & $\le$10 upstream components (path patching, suffix positions) mediate $\ge$40\% of the top-head effect on test items, above the random p95. & $<$40\% or $\le$ random p95. & -- & " + f"{met(v(B, 'B1.F1/C2k4.met'))} (C, marginal)" + r" & -- \\")
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
F = json.load(open(f"{S}/numbers_F.json"))
fv = lambda k: F[k]["value"]
p1 = lambda m: met(float(F[f"F1.{m}.F3.D_pts"]["ci"][0] > 0 and fv(f"F1.{m}.F3.R_p_pts") > fv(f"F1.{m}.F3.L_p_pts")))
PF = []
PF.append(r"P1 & Block F, documents (F3): the paraphrase gain beyond one paraphrased copy exceeds the exact-copy gain, $D = R_p - R_e > 0$ (CI lower bound $>0$), with $R_p = P_{\text{low}}(\text{C7 } k{=}4) - P_{\text{low}}(\text{C7 } k{=}1)$, $R_e = P_{\text{low}}(\text{C2 } k{=}4) - P_{\text{low}}(\text{C1})$, \emph{and} $R_p >$ the length-matched filler effect $L_p$. Confirmatory in Qwen-32B and Qwen-7B. & $D$'s CI includes 0 or $D<0$ (a one-shot wording effect), or $R_p \le L_p$ (a length effect). & "
         + f"{p1('Llama-8B')} (E) & {p1('Qwen-7B')} (C) & {p1('Gemma-12B')} (E) & {p1('Qwen-32B')} (C)" + r" \\")
PF.append(r"P2 & Block F, tables (F1): the fixed top-10 heads are specific to the credibility--repetition conflict: recovery on C1$\to$C2 $k{=}4$ exceeds the other three contrasts (C0$\to$C1, C0$\to$C0r $k{=}4$, C1$\to$C6 $k{=}4$) by $>$20 points. & Any other contrast within 20 points. & "
         + " & ".join(f"{met(fv(f'F2.{m}.specific'))} (C)" for m in M3) + r" & -- \\")
P.append(r"\bottomrule\end{tabular}\caption{Pre-registered criteria, failure outcomes and verdicts. Unless a CI is named, bars apply to point estimates.}\label{tab:prereg}\end{table}")
P.append(r"""
\begin{table}[!htbp]\centering\scriptsize
\begin{tabular}{@{}p{0.6cm}p{5.4cm}p{3.6cm}p{1.15cm}p{1.15cm}p{1.15cm}p{1.15cm}@{}}\toprule
ID & Definition & Fails if & Llama & Qwen-7B & Gemma & Qwen-32B \\ \midrule""")
P += PF
P.append(r"\bottomrule\end{tabular}\caption{Block F (reviewer controls): pre-registered criteria, failure outcomes and verdicts.}\label{tab:prereg_f}\end{table}")
am = [
 (r"A1", r"13 Nov, after a 3-pair code-check smoke test", r"H2 layer chosen by selection log-loss instead of AUC (ties at 1.00).", r"all full runs"),
 (r"A2", r"13 Nov", r"Qwen-32B scope (test split, F1+F3, priority order); lexical baselines; graded-credibility set; H3 defined.", r"Gemma, Qwen-32B runs; all extras"),
 (r"A3", r"13 Nov", r"Neutral-tag (Blue/Green) binding control; $\ge$0.95 $\Rightarrow$ H2 is binding only.", r"all extras"),
 (r"A4", r"13 Nov", r"Blue-vs-Green projection demoted to sanity check; neutral tag as 5th graded level; H3 rewritten as three paired contrasts; H1 adds random p95.", r"all extras, all patching"),
 (r"A5", r"13 Nov", r"F3 head analysis also with paraphrased copies (C7 $k{=}4$) as corrupt run.", r"F3 head patching"),
 (r"A6", r"13 Nov", r"C1-only probe (count confound) with predictions for C2 $k{=}8$, C0r, C6.", r"all C1-only fits"),
 (r"A7", r"13 Nov", r"H4 layer-resolved rule ($L^*$ on disagreeing items).", r"Qwen-7B, Gemma probes"),
 (r"A8", r"13 Nov", r"Robustness over near-max layers; final-layer description.", r"Qwen-7B, Gemma probes"),
 (r"A9", r"13 Nov", r"H5 (source readable at output).", r"Qwen-7B, Gemma probes"),
 (r"A10", r"13 Nov", r"Decision-balanced C1 probe (stop if $<$200 disagreeing items).", r"Gemma probe"),
 (r"B-branch", r"13 Nov", "Two head sets (Jaccard " + f"{v(A, 'Qwen-7B.rank.F1_vs_F3-C7k4.jaccard10'):.2f}" + r" $<$ 0.5).", r"all B1--B3 data"),
 (r"B-frame", r"13 Nov", r"Steering withdrawn as a fix (needs oracle sign); diagnostic only.", r"all B2/B3 data"),
 (r"B-U", r"13 Nov", r"Union fix B3-U.", r"all B3 data"),
 (r"B-Feeders", r"14 Nov", r"Feeders head set and H-F; test $\gamma$-curves.", r"B3 test data"),
 (r"H-F.1", r"14 Nov", r"H-F rule when Feeders exceed the union's range.", r"B3 test data"),
 (r"B-report", r"14 Nov", r"Report preference kept in all formats (no criterion change).", r"B3 test data"),
 (r"B1-scope", r"14 Nov", r"Reduced sender pool (top-50 heads + top-10 MLPs by AtP); suffix positions only.", r"all B1 data"),
 (r"F", r"14 Nov", r"Block F (reviewer controls): C7 $k{=}1$ per paraphrase template, C8 $k{=}4$, C5p; P1, P2 and their failing outcomes.", r"all Block F data"),
 (r"F-C5p", r"14 Nov", r"F3 filler blocks lengthened to 4 sentences so that C5p matches C7 $k{=}4$ in length.", r"all Block F data"),
 (r"F-posthoc", r"14 Nov, after Qwen-32B data", r"Per-template C7 $k{=}1$ values and $D$ against templates 1--3 or the strongest template; labelled post hoc, verdicts unchanged.", r"none (post hoc)"),
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
T.append(r"\bottomrule\end{tabular}}\caption{Held-out head patching: \% of the C2$-$C1 logit difference (C7$-$C1 in the last row) recovered on 160 test items (20 random 10-head draws).}\end{table}")
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
T.append(r"\bottomrule\end{tabular}}\caption{Graded credibility set (100 test pairs, C1-shaped, F1): mean projection onto the gov--social direction, oriented to source X against a social partner (social $\approx$0 by construction), with pre-registered H3 contrasts. The direction is the all-conditions probe of H2, not the C1-only probe of H4, so this ordering does not validate a pure C1-only credibility direction; in Gemma the neutral tag projects above the person level (binding caveat, amendment A3).}\end{table}")
TG = {"Llama-8B": "Llama-3.1-8B-Instruct", "Qwen-7B": "Qwen2.5-7B-Instruct", "Gemma-12B": "gemma-3-12b-it"}
PO = {m: json.load(open(f"{S}/results/A/probe_c1only_{t}.json")) for m, t in TG.items()}; PB = {m: json.load(open(f"{S}/results/A/probe_c1bal_{t}.json")) for m, t in TG.items()}
LL = PO["Llama-8B"]["layers"]; bst = max(o["val_auc_disagree"] for o in LL.values()); near = [l for l, o in LL.items() if o["val_auc_disagree"] >= bst - 0.005]
PRB_TXT = (", ".join(f"{PO[m]['n_disagree_val']} ({m.replace('-8B', '').replace('-12B', '')})" for m in M3) + r"; the balanced probe's stopping rule ($<$200) applies to its training pool of "
           + ", ".join(str(PB[m]["balanced"]["n_disagree_total"]) for m in M3[:2]) + f" and {PB[M3[2]]['balanced']['n_disagree_total']}. "
           + r"C0r keeps the orientation of the singleton side after the labels are removed, so an AUC below 0.40 means the probe marks the repeated value as government-held. "
           + f"Llama is exploratory because its per-layer results were seen before the rule; its layers within 0.005 AUC of the best (L{', '.join(near)}) average "
           + f"{np.mean([LL[l]['C2k8_flipped'] for l in near]):.3f} flipped AUC and {np.mean([LL[l]['C0rk4'] for l in near]):.3f} C0r AUC.")
T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\linewidth}{!}{\begin{tabular}{@{}lcccccccc@{}}\toprule")
T.append(r"Model & $L^*$ & H4(a) flipped AUC [CI] & H4(b) C0r & final flipped AUC [CI] & choice social & balanced $L^*$ & bal.\ H4(a) & bal.\ final \\ \midrule")
for m in M3:
    a, ca = v(A, f"{m}.H4.a_flipped_auc_Lstar"), ci(A, f"{m}.H4.a_flipped_auc_Lstar"); h5, c5 = v(A, f"{m}.H5.final_flipped_auc"), ci(A, f"{m}.H5.final_flipped_auc")
    T.append(f"{m}{' (E)' if m.startswith('Llama') else ''} & {int(v(A, f'{m}.H4.Lstar'))} & {a:.3f} [{ca[0]:.3f}, {ca[1]:.3f}] & {v(A, f'{m}.H4.b_C0r_auc_Lstar'):.3f} & {h5:.3f} [{c5[0]:.3f}, {c5[1]:.3f}] & "
             f"{v(A, f'{m}.H5.choice_rate')*100:.0f}\\% & {int(v(A, f'{m}.c1bal.Lstar'))} & {v(A, f'{m}.c1bal.H4a'):.3f} & {v(A, f'{m}.c1bal.H5'):.3f}" + r" \\")
T.append(r"\bottomrule\end{tabular}}\caption{C1-only probe (amendments 6--10): AUC for ``which option is government-sourced'' on held-out test entities; flipped = variant-level prompt runs where the model picked the social value at C2 $k{=}8$ (not entity-level choices as in Figure 1). Decision-disagreeing selection runs used to choose $L^*$: " + PRB_TXT + r" (E) exploratory.}\end{table}")
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
# Block D: trained LoRA adapter on the same model, same columns (numbers_D.json); training settings read from train_lora.py and the logs
import re
ND = json.load(open(f"{S}/numbers_D.json")); TL = open(f"{S}/scripts/train_lora.py").read()
lora_r = re.search(r"LoraConfig\(r=(\d+)", TL).group(1); lora_mods = re.search(r"target_modules=\[([^\]]+)\]", TL).group(1).replace('"', "").replace("_proj", "").replace(", ", "/")
ck = open(f"{S}/results/D/chosen_adapter.txt").read().strip().split("/")[-2:]; LG = {d.split("/")[-2]: json.load(open(d)) for d in sorted(glob.glob(f"{S}/results/D/lr*/train_log.json"))}
LC = LG[ck[0]]; ep = ck[1].replace("ckpt_", ""); lams = (LC["lambda1"], 1 - LC["lambda1"])
for i, fmt in enumerate(("F1", "F3", "F2")):
    T.append((r"Qwen-7B, LoRA adapter" if i == 0 else "") + f" & {fmt} & {pci(ND, f'D.{fmt}.BR_C2k4')} & {pci(ND, f'D.{fmt}.BR_C7k4')} & {pci(ND, f'D.{fmt}.pref_kept')} & {pci(ND, f'D.{fmt}.corrob_kept')}" + r" \\")
mc = ci(ND, "D.mmlu_delta_pts"); pc_ = ci(ND, "D.ppl_increase_pct")
T.append(rf" & \multicolumn{{5}}{{l}}{{capability: MMLU $\Delta$ {v(ND, 'D.mmlu_delta_pts'):+.1f} pts [{mc[0]:+.1f}, {mc[1]:+.1f}]; WikiText perplexity {v(ND, 'D.ppl_increase_pct'):+.1f}\% [{pc_[0]:+.1f}, {pc_[1]:+.1f}]}} \\")
lr_s = lambda x: f"{x:.0e}".replace("e-0", r"\times10^{-") + "}"
T.append(r"\bottomrule\end{tabular}\caption{Head-damping fixes on test entities (\%, 95\% CI over pairs; capability CIs paired over items). $(k,\gamma)$ chosen on 100 selection pairs from $k \in \{5,10,20\} \times \gamma$: the largest bias reduction among settings keeping $\ge$80\% of source preference; if none does (Llama), the setting keeping the most. "
         + f"Last block: a rank-{lora_r} LoRA adapter on the {lora_mods} projections of every layer, trained ({LC['n_train']:,} items, new fictional entities) with a symmetric KL loss that keeps the A/B distribution on the repeated prompt and on the single-claim C1 prompt equal to the base model's on C1 (weights {lams[1]:.2f} and {lams[0]:.2f}), "
         + f"learning rate ${lr_s(LC['lr'])}$, checkpoint (two learning rates $\\times$ epochs 1, 2 and a refinement pass) chosen by the lowest loss on {LC['n_val']} validation items from held-out training entities: epoch {ep[1:]}, loss {LC['checkpoints'][ep]['loss']:.3f}." + r"}\end{table}")
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
T.append(r"\bottomrule\end{tabular}\caption{B2 steering, Qwen-7B (oracle sign; diagnostic only). Controls on the first 100 test pairs. The $\alpha$ grid went up to " + str(max(int(a) for o in R2["layers"].values() for a in o["sel_curve"])) + r" at both layers; $\alpha^*$ is the value chosen on selection (" + ", ".join(f"{o['alpha_star']} at L{L}" for L, o in R2["layers"].items()) + r") and applied to test pairs. Each layer uses its own probe vector, with the sign set by the known government answer; only the final token is edited. The residual swap restores C1 at L28 but moves the answer the wrong way at L18.}\end{table}")
# Block F: F1-main quantities and F2 recovery
FM = [("Qwen-32B", "F3", "C"), ("Qwen-7B", "F3", "C"), ("Llama-8B", "F3", "E"), ("Gemma-12B", "F3", "E"), ("Llama-8B", "F1", "E"), ("Qwen-7B", "F1", "E"), ("Gemma-12B", "F1", "E")]
T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\linewidth}{!}{\begin{tabular}{@{}llccccccc|cccc@{}}\toprule")
T.append(r"Model & Format & $W$ & $R_p$ & $R_e$ & $D = R_p - R_e$ [CI] & C7$-$C2 & $L_p$ & interaction [CI] & C2 $k{=}4$ & C3 $k{=}4$ & C7 $k{=}4$ & C8 $k{=}4$ \\")
T.append(r" & & & & & & & & & same, exact & different, exact & same, paraphr. & different, paraphr. \\ \midrule")
for m, fmt, lab in FM:
    b = f"F1.{m}.{fmt}"; dc, ic = F[f"{b}.D_pts"]["ci"], F[f"{b}.I_2x2_pts"]["ci"]
    T.append(f"{m} ({lab}) & {'documents' if fmt == 'F3' else 'tables'} & {fv(b + '.W_pts'):+.1f} & {fv(b + '.R_p_pts'):+.1f} & {fv(b + '.R_e_pts'):+.1f} & {fv(b + '.D_pts'):+.1f} [{dc[0]:+.1f}, {dc[1]:+.1f}] & {(fv(b + '.C7k4.p_low') - fv(b + '.C2k4.p_low')) * 100:+.1f} & {fv(b + '.L_p_pts'):+.1f} & {fv(b + '.I_2x2_pts'):+.1f} [{ic[0]:+.1f}, {ic[1]:+.1f}] & "
             + " & ".join(f"{fv(f'{b}.{ck}.p_low'):.3f}" for ck in ("C2k4", "C3k4", "C7k4", "C8k4")) + r" \\")
T.append(r"\bottomrule\end{tabular}}\caption{Block F: is it repetition or wording? Effects in percentage points of $P_{\text{low}}$ on 400 test pairs (95\% CI, cluster bootstrap over pairs). $W$: one paraphrased copy (mean over the four templates) minus C1; $R_p$: four paraphrased copies minus one; $R_e$: four exact copies minus C1; C7$-$C2: four paraphrased minus four exact copies of one post. $W$ is negative in tables, where a prose paraphrase inside a table cell lowers $P_{\text{low}}$, so there $D$ exceeds C7$-$C2. $L_p$: one social block plus filler matched in length to C7 $k{=}4$, minus C1. Right: $P_{\text{low}}$ in the source identity $\times$ wording $2{\times}2$ at $k{=}4$; interaction = (C8$-$C7)$-$(C3$-$C2). (C) confirmatory, (E) exploratory.}\end{table}")
CON = [("C1k1->C2k4", r"C1$\to$C2 $k{=}4$ (reference)"), ("C0k1->C1k1", r"C0$\to$C1 (credibility)"), ("C0k1->C0rk4", r"C0$\to$C0r $k{=}4$ (unlabelled repetition)"), ("C1k1->C6k4", r"C1$\to$C6 $k{=}4$ (government repeated)")]
T.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llccccc@{}}\toprule")
T.append(r"Model & Contrast (clean$\to$corrupt) & eligible & mean gap & flip rate & top-10 recovery [CI] & random p95 \\ \midrule")
for m in M3:
    for i, (c, lab) in enumerate(CON):
        b = f"F2.{m}.{c}"; rc = F[f"{b}.recovery_top10_pct"]["ci"]
        T.append(f"{m if i == 0 else ''} & {lab} & {int(fv(b + '.n_eligible'))} & {abs(fv(b + '.mean_gap')):.2f} & {fv(b + '.flip_rate') * 100:.0f}\\% & {fv(b + '.recovery_top10_pct'):.1f} [{rc[0]:.1f}, {rc[1]:.1f}] & {fv(b + '.random_p95_pct'):.1f}" + r" \\")
    if m != M3[-1]: T[-1] += r" \midrule"
T.append(r"\bottomrule\end{tabular}\caption{Block F: the fixed top-10 heads of criterion H1 (tables, F1), patched for four contrasts. Eligible: variant runs (of 400 test pairs $\times$ 8 variants) with a gap $>$1 logit in the expected direction; 160 of them, drawn at random, were patched. Recovery = share of the mean logit-difference gap restored (95\% CI over pairs); random p95 over the same 20 random 10-head sets.}\end{table}")
# Example item (verbatim from the released prompt files; label pools and template lists read from build_data_v2.py)
import ast, gzip
BD = {t.targets[0].id: t.value for t in ast.parse(open(f"{S}/scripts/build_data_v2.py").read()).body if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)}
cst = lambda n: ast.literal_eval(BD[n])
def rows_of(f, **kw):
    fp = f"{S}/data/{f}"; fh = gzip.open(fp + ".gz", "rt") if os.path.exists(fp + ".gz") else open(fp)
    for l in fh:
        r = json.loads(l)
        if all(r.get(k) == x for k, x in kw.items()): fh.close(); return r
PID = min(json.loads(l)["pair_id"] for l in open(f"{S}/data/pairs_v2.jsonl") if json.loads(l)["split"] == "test")
one = lambda f, fmt, ck: rows_of(f, pair_id=PID, fmt=fmt, ck=ck, lowidx=0, order="HF", ans="AH")
def tt(txt):
    m = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}", "|": r"\textbar{}", "-": "-{}"}
    return r"\texttt{" + r"\newline ".join("".join(m.get(c, c) for c in L) or r"\mbox{}" for L in txt.split("\n")) + "}"
blocks = lambda u: u.split("\n\n")
EX = {fmt: one("prompts_v2.jsonl", fmt, "C1k1") for fmt in ("F1", "F2", "F3")}
c7 = blocks(one("prompts_v2.jsonl", "F3", "C7k4")["user"])[2:6]; c5 = blocks(one("prompts_v2.jsonl", "F3", "C5k4")["user"])[2]; c5p = blocks(one("prompts_f.jsonl", "F3", "C5pk4")["user"])[2]
GR = {lev: rows_of("prompts_graded.jsonl", pair_id=PID, level=lev)["source_X"] for lev in ("gov", "newspaper", "person", "social")}
NB = rows_of("prompts_neutral.jsonl", split="neutral_test", tagH="Blue")["user"]
gov_t = sorted({x.replace("{P}", "<Place>") for v_ in cst("GOV").values() for x in v_}); news_t = [x.replace("{P}", "<Place>") for x in cst("NEWS_T")]
HEADS, NAMEL, FR2 = cst("HEAD"), cst("NAMELAB"), cst("FRAME2")
T.append(r"\begin{table}[p]\centering\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}p{0.30\linewidth}p{0.33\linewidth}p{0.33\linewidth}@{}}\toprule")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textbf{(a) One complete item, condition C1, test pair " + str(PID) + r" (government block first, government value as option A)}} \\ \midrule")
T.append(r"F1 table & F2 news snippet & F3 retrieved documents \\")
T.append(" & ".join(tt("System: " + EX[f]["system"]) for f in ("F1", "F2", "F3")) + r" \\[2pt]")
T.append(" & ".join(tt(EX[f]["user"].rstrip()) for f in ("F1", "F2", "F3")) + r" \\")
T.append(r"\bottomrule\end{tabular}\caption{Example item. Conditions used in Block F: \textbf{C7 $k{=}1$} = the government block plus one paraphrased social block (one item per template; template 0 is identical to C1); "
         r"\textbf{C8 $k{=}4$} = the government block plus four social blocks from four different users, user $i$ written with template $i$ (same length, attribution and placement as C7 $k{=}4$, which repeats one user); "
         r"\textbf{C5p} = one social block plus three neutral filler blocks about other invented entities, written with templates 1--3 and four sentences each in F3, so that the total length matches C7 $k{=}4$ (C5 matches C2 $k{=}4$ with three-sentence fillers). "
         r"Every item comes in 8 variants: which value is the social one $\times$ block order $\times$ answer-letter order.}\label{tab:example}\end{table}")
T.append(r"\begin{table}[p]\centering\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}p{0.30\linewidth}p{0.33\linewidth}p{0.33\linewidth}@{}}\toprule")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textbf{(b) Source label strings (\texttt{<Place>}: an invented place name)}} \\ \midrule")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{Government: " + "; ".join(r"\texttt{" + x.replace("<", r"\textless{}").replace(">", r"\textgreater{}") + "}" for x in gov_t)
         + r" (title-cased). Newspaper: " + "; ".join(r"\texttt{" + x.replace("<", r"\textless{}").replace(">", r"\textgreater{}") + "}" for x in news_t)
         + r". Person: an invented full name. Social: \texttt{User @<Adjective><Noun><100--9999>}, written \texttt{(Source: User @\dots)} in F1, \texttt{social media user @\dots} in F2 and \texttt{Social media post by user @\dots} in F3. "
         + r"Graded set, same pair: " + "; ".join(rf"{k} \texttt{{{x}}}" for k, x in GR.items()) + r". Neutral control: \texttt{(Source: Blue)} vs \texttt{(Source: Green)} in place of the two sources, e.g.\ \texttt{"
         + [L for L in NB.split("\n") if "(Source: Blue)" in L][0].replace("_", r"\_") + r"}.} \\ \midrule")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textbf{(c) The four paraphrase templates: the social block of the same F3 item under templates 0--3 (C7 $k{=}4$ uses all four)}} \\ \midrule")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{" + r"\newline ".join(rf"{t}: " + tt(b) for t, b in enumerate(c7)) + r"} \\[2pt]")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{Template $t$ rotates the sentence order by $t$ and uses wording $t$ for every sentence. In F1 it uses the header \texttt{" + " / ".join(f"{a}|{b}" for a, b in HEADS).replace("|", r"\textbar{}")
         + r"}, the name label \texttt{" + " / ".join(NAMEL) + r"} and the field label $t$ (e.g.\ \texttt{" + " / ".join(cst("LAB")[(EX["F1"]["cls"], json.loads(open(f"{S}/data/pairs_v2.jsonl").readlines()[PID])["cfield"])]) + r"}), with the rows rotated by $t$; in F2 the attribution frame \texttt{"
         + " / ".join(x.replace("{s}", "<source>") for x in FR2).replace("<", r"\textless{}").replace(">", r"\textgreater{}") + r"}.} \\ \midrule")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textbf{(d) Length fillers (no source line), same F3 item: first filler block}} \\ \midrule")
T.append(r"\multicolumn{3}{@{}p{\linewidth}@{}}{C5: " + tt(c5) + r"\newline C5p: " + tt(c5p) + r"} \\")
T.append(r"\bottomrule\end{tabular}\caption{Example item (continued): source label strings, the four paraphrase templates and the length fillers, for the same test pair as Table~\ref{tab:example}.}\label{tab:example2}\end{table}")
# ---- app_extra.tex: headline metrics, template sensitivity, block order, LoRA training, conditions and prompts. Every number from a results/numbers file. ----
import collections, textwrap
def beh(tag):
    fp = f"{S}/results/A/behaviour_{tag}.jsonl"; fh = gzip.open(fp + ".gz", "rt") if os.path.exists(fp + ".gz") else open(fp)
    out = [json.loads(l) for l in fh]; fh.close(); return [x for x in out if x.get("split", "test") == "test"]
X = []
B32 = beh("Qwen2.5-32B-Instruct")
X.append(r"\begin{table}[!htbp]\centering\scriptsize\begin{tabular}{@{}lcc@{}}\toprule")
X.append(r"Condition & Entity choices for the low source & Mean $P_{\text{low}}$ \\ \midrule")
for ck, lab in (("C1k1", "C1"), ("C2k4", "C2 $k{=}4$"), ("C3k4", "C3 $k{=}4$"), ("C7k4", "C7 $k{=}4$")):
    per = collections.defaultdict(list)
    for x in B32:
        if x["fmt"] == "F3" and x["ck"] == ck: per[x["pair_id"]].append(x["p_low"])
    m = np.array([np.mean(a) for a in per.values()]); n = int((m > .5).sum())
    X.append(f"{lab} & {n}/{len(m)} ({n / len(m) * 100:.3g}\\%) & {v(A, f'Qwen-32B.F3.test.{ck}.p_low'):.3f}" + r" \\")
X.append(r"\bottomrule\end{tabular}\caption{Two metrics for the headline result, Qwen-32B, retrieved documents (F3), 400 test entities. Entity choice: the entity counts when its $P_{\text{low}}$, averaged over the 8 variants, exceeds 0.5 (Figure 1). Mean $P_{\text{low}}$: average over entities (Tables 9 and 18).}\label{tab:twometrics}\end{table}")
X.append("")
PH = json.load(open(f"{S}/numbers_F_posthoc.json")); ph = lambda k: PH[k]["value"]
sg = lambda x: f"{x:+.1f}".replace("-", "$-$")
X.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\linewidth}{!}{\begin{tabular}{@{}llcccccccc@{}}\toprule")
X.append(r"Model & Format & t0 & t1 & t2 & t3 & $D$ (pre-specified) & $D$ vs mean t1--3 [CI] & $D$ vs strongest [CI] & strongest \\ \midrule")
for m, fmt, lab in FM:
    b = f"F1posthoc.{m}.{fmt}"; c1, c2 = PH[f"{b}.D_vs_mean_t123_pts"]["ci"], PH[f"{b}.D_vs_strongest_pts"]["ci"]
    X.append(f"{m} ({lab}) & {'documents' if fmt == 'F3' else 'tables'} & " + " & ".join(f"{ph(f'{b}.C7k1_t{t}.p_low'):.3f}" for t in range(4))
             + f" & {sg(fv(f'F1.{m}.{fmt}.D_pts'))} & {sg(ph(b + '.D_vs_mean_t123_pts'))} [{sg(c1[0])}, {sg(c1[1])}] & {sg(ph(b + '.D_vs_strongest_pts'))} [{sg(c2[0])}, {sg(c2[1])}] & t{int(ph(b + '.strongest_template'))}" + r" \\")
X.append(r"\bottomrule\end{tabular}}\caption{Post-hoc template sensitivity of $D$ (amendment F-posthoc). Left: $P_{\text{low}}$ for one paraphrased copy in each template (template 0 is the C1 wording). Right, in percentage points: the pre-specified $D$ (C7 $k{=}1$ baseline averaged over templates 0--3), $D$ against the mean of templates 1--3, and $D$ against the strongest of templates 1--3 (number in brackets). 95\% CIs, cluster bootstrap over 400 test pairs. The pre-specified verdicts stand, but not every sensitivity contrast has the same sign: Gemma's document advantage reverses against its strongest template.}\label{tab:tplsens}\end{table}")
X.append("")
B7 = [x for x in beh("Qwen2.5-7B-Instruct") if x["fmt"] == "F1"]
X.append(r"\begin{table}[!htbp]\centering\scriptsize\begin{tabular}{@{}lccc@{}}\toprule")
X.append(r"Condition & Government first: mean $P_{\text{low}}$ & Social first: mean $P_{\text{low}}$ & Mean A+B mass \\ \midrule")
nord = set()
for ck, lab in (("C1k1", "C1"), ("C2k2", "C2 $k{=}2$"), ("C2k4", "C2 $k{=}4$"), ("C2k8", "C2 $k{=}8$")):
    rr = [x for x in B7 if x["ck"] == ck]; o = {q: [x["p_low"] for x in rr if x["order"] == q] for q in ("HF", "LF")}; nord |= {len(o["HF"]), len(o["LF"])}
    X.append(f"{lab} & {np.mean(o['HF']):.3f} & {np.mean(o['LF']):.3f} & {np.mean([x['mass'] for x in rr]):.3f}" + r" \\")
assert len(nord) == 1
X.append(r"\bottomrule\end{tabular}\caption{Block order in Qwen-7B tables (F1), test split, " + f"{nord.pop():,}" + r" variant runs per order and condition. At $k{=}8$ the answer depends strongly on which block comes first, which lowers the average in Table 7. A+B mass (probability on the two answer letters, both orders) is not lowest at $k{=}8$.}\label{tab:order}\end{table}")
X.append("")
MD = json.load(open(f"{S}/data/manifest_d.json")); rx = lambda pat: re.search(pat, TL).group(1)
lr_t = lambda x: f"{x:.0e}".replace("e-0", "e-")
X.append(r"\subsection*{LoRA training}")
X.append(f"The adapter (rank {lora_r}, $\\alpha={rx(r'lora_alpha=(\d+)')}$, dropout {rx(r'lora_dropout=([0-9.]+)')}, on the q, k, v, o, gate, up and down projections) follows the objective of Schuster et al. \\citep{{schuster2026whose}}. "
         r"Each training item pairs a repeated prompt with its C1 prompt. With $S(p,q)=\tfrac12[\mathrm{KL}(p\Vert q)+\mathrm{KL}(q\Vert p)]$ over the two answer letters and $p_0$ the frozen base model on the C1 prompt, the loss is $\lambda S(p_0, p_\theta(\text{C1})) + (1-\lambda) S(p_0, p_\theta(\text{repeated}))$ with "
         + f"$\\lambda = {LC['lambda1']:.3f}$. Training data: {MD['items']['train_d']:,} items about {MD['entities']['train_d']} new fictional entities (validation: {MD['items']['val_d']} items, {MD['entities']['val_d']} entities), "
         + f"{MD['items']['train_d'] // MD['entities']['train_d']} per entity drawn from C2 and C7 at $k \\in \\{{2,4\\}}$ in tables and documents; no distinct users. AdamW (weight decay {rx(r'weight_decay=([0-9.]+)')}), gradient accumulation {rx(r'GA = (\d+)')}, {float(rx(r'WARM = ([0-9.]+)')) * 100:.0f}\\% warm-up, "
         + f"two epochs, then one refinement pass on the repetition term only; one seed. Learning rate in \\{{{', '.join(lr_t(LG[d]['lr']) for d in LG)}\\}} and checkpoint (epoch 1, epoch 2, refinement) were chosen by the lowest validation loss on held-out training entities: {lr_t(LC['lr'])} after epoch {ep[1:]}. "
         + r"The adapter is merged into the base weights for evaluation. On 500 MMLU items its accuracy moves from " + f"{np.mean(json.load(open(f'{S}/results/D/b3_cap_LORA_Qwen2.5-7B-Instruct.json'))['base']['mmlu']) * 100:.1f}\\% to {np.mean(json.load(open(f'{S}/results/D/b3_cap_LORA_Qwen2.5-7B-Instruct.json'))['fixed']['mmlu']) * 100:.1f}\\%; it never saw news snippets (F2) or distinct users (C3) in training (Table 13).")
X.append("")
X.append(r"\subsection*{Conditions and prompts}")
X.append(r"C0: no source labels. C0r: the low value repeated four times, no labels. C1: one government and one social block. C2: $k$ exact copies of the social block. C3: $k$ different users, identical text. C4: one block listing $k$ users. C5: one social block plus filler blocks about other entities, as long as C2 $k{=}4$. C6: the government block repeated. C7: $k$ paraphrased copies from one user, copy $i$ in template $i$ (template 0 is the C1 wording). C7 $k{=}1$: one paraphrased copy, averaged over templates 0--3. C8: $k$ different users, each in a different template. C5p: one social block plus three filler blocks in templates 1--3, as long as C7 $k{=}4$ (in documents, four attribute sentences with no source line). Government sources are fictional ministries or statistics offices (``Ministry Of Interior Of Fikylteix''); social sources are usernames (``User @BrightMeadow7155''). "
         + f"The graded set adds a newspaper (``{GR['newspaper']}''), a person (``{GR['person']}'') and neutral tags (``Blue'', ``Green''). The system prompt reads: ``{EX['F3']['system']}'' (F1 and F2 say ``tabular data'' and ``news excerpts''). "
         + r"Tables~\ref{tab:example} and~\ref{tab:example2} give the full C1 item in all three formats, the label strings, the four templates and the fillers. Below, the same test entity in tables with four different users and in news snippets with four paraphrased copies, government block first (long lines wrapped).")
assert "Ministry Of Interior Of Fikylteix" in open(f"{S}/data/pairs_v2.jsonl").read() and "BrightMeadow7155" in open(f"{S}/data/pairs_v2.jsonl").read()
wrap = lambda u: "\n".join(("\n".join(textwrap.wrap(L, 96, subsequent_indent="    ")) if L else "") for L in u.split("\n"))
for fn, fmt, ck, head in (("prompts_f.jsonl", "F1", "C8k4", r"Tables (F1), C8 $k{=}4$: four different users, templates 0--3"), ("prompts_v2.jsonl", "F2", "C7k4", r"News snippets (F2), C7 $k{=}4$: one user, templates 0--3")):
    X.append("\n\\paragraph*{" + head + "}\n{\\scriptsize\n\\begin{verbatim}\n" + wrap(one(fn, fmt, ck)["user"]) + "\n\\end{verbatim}\n}")
open(f"{S}/paper/app_extra.tex", "w").write("\n".join(X) + "\n")
open(f"{S}/paper/app_tables.tex", "w").write("\n".join(T) + "\n\\clearpage\n")
print("wrote app_prereg.tex, app_tables.tex, app_extra.tex")
