# RESULTS_B — Block B: mechanism + training-free fix
**SPRINT WORK (Apart AI x Epistemics, 13–15 Nov 2026).** Primary model: Qwen2.5-7B-Instruct (chosen by Block A's pre-registered best-F1-recovery rule). Replications: Llama-3.1-8B-Instruct, Gemma-3-12B-it. Budget ≤ 15 A100-h. Failure modes: `FAILURE_MODES.md` section F.
Every criterion below is written before its data exists. For each one, the outcome that fails it is stated, along with whether it can fail under the 8-way counterbalancing.

## Pre-registration

### B0 — induction-head check (written 2026-11-13 ~22:55 CET; job 165344 was submitted at ~22:50 and **none of its output has been read**)
- *Recipe:* BOS + 64 random tokens (ids 1,000–30,000) repeated twice; 48 sequences; every head scored. **Prefix-matching** = mean attention from second-copy tokens to the token after their previous occurrence. **Copying** = direct logit attribution of the head output to the correct next token minus its vocabulary mean, through the final norm's per-position scale. **Previous-token** score reported as a control.
- *Induction head (per head):* prefix-matching ≥ 0.30 **and** copying score above the 90th percentile of all heads in that model.
- **B0-a "repetition heads are induction heads"** (confirmatory, per model; head set = Block A F1/C2 top-10): met if **≥ 5 of the top-10** are induction heads.
- **B0-b "enriched"**: the top-10 mean prefix-matching score exceeds the 95th percentile of 1,000 random 10-head sets from the same model.
- Reading: B0-a met → "induction heads". B0-a not met but B0-b met → "enriched for induction-like attention, not induction heads". Both not met → "not induction heads". The 20 random heads are listed individually as asked.
- *C2 vs C7 (exploratory, Qwen-7B):* compare the induction scores of the F3/C2-only, F3/C7-only and shared top-10 heads (from A11). Expectation to test: if induction-style copying drives exact-copy repetition but not paraphrase, the C2-only heads should score higher on prefix-matching than the C7-only heads. Only 5 heads per group, so this is descriptive (no test).
- *Fails if* < 5 induction heads in the top-10 (B0-a), or the top-10 mean prefix score is ≤ the random p95 (B0-b). Both are attainable: Block A's top heads attend mainly to answer options and the chat tail. *Counterbalancing:* not applicable; the scores come from synthetic sequences, not from the prompts.

### Branch decision (written 2026-11-13 ~22:55 CET, from A11's head-overlap report, before any B1–B3 data exists)
Qwen-7B top-10 Jaccard(F1/C2 selection ranking, F3/C7 selection ranking) = **0.33** (shared: L17H26, L18H4, L18H27, L19H23, L26H26; F1-only: L10H24, L17H21, L17H22, L18H18, L19H18; F3/C7-only: L12H16, L14H12, L16H0, L16H1, L20H27). All-head AtP Spearman 0.43. **0.33 < 0.5 → two head sets.** **H_F1** = the F1/C2 selection ranking, used for table/exact-copy evaluation. **H_F3C7** = the F3/C7 selection ranking, used for document/paraphrase evaluation. Both are reported. (For the record, F3/C7's own held-out top-10 recovery is 78.4% [76.3, 80.2] vs random p95 17.7%. F3/C2 vs F3/C7 Jaccard is also 0.33.)

### Framing rule (Amr, 2026-11-13 ~23:50 CET; written before any B2/B3 data exists)
**B3 head scaling is the deployable fix.** It needs no knowledge of which source is official. **B2 steering** needs that knowledge (the oracle sign: add the direction when gov is option A, subtract it when gov is B), and in deployment that is exactly what the model is supposed to work out. So B2 is reported **only as a causal diagnostic / oracle upper bound, never as a defence or fix**. The B3(b) "steering as a fix" variant below is withdrawn as a fix. Its B2 numbers appear only in the B2 section.

### B3-U — single deployable fix: scale the union H_F1 ∪ H_F3C7 (written ~23:50 CET; B3a has not started, so no B3 data exists)
- *Head set:* for k ∈ {5, 10, 20}, U_k = top-k(H_F1) ∪ top-k(H_F3C7), between k and 2k heads (Qwen-7B: the 5 shared heads count once). γ ∈ {0, 0.25, 0.5, 0.75}, at every position, as in fix (a).
- *Choosing (k, γ) on SELECTION* (the same 100 pairs): maximise **min(bias reduction F1 C2k4, bias reduction F3 C7k4)**, subject to source preference retained ≥ 0.80 on F1 selection. Ties go to the larger γ, then the smaller k.
- *Evaluated on TEST in all formats (F1, F3, F2) and conditions* (C0, C1, C2k2/k4/k8 [F1 only for k8], C3k4, C5, C7k4), with the same capability checks (MMLU 500, WikiText 200).
- **B3-U-S1 (confirmatory):** the S1 bars on F1 (C2k4 bias reduction ≥ 50%, preference ≥ 70%). **B3-U-S2 (confirmatory):** the S2 bars on F3 (C7k4 bias reduction ≥ 50%, F3 preference ≥ 70%). Both need MMLU drop ≤ 2 points and perplexity increase ≤ 5%. The selectivity bar (C3k4 corroboration ≥ 50%) is reported for each format.
- **If B3-U meets both S1 and S2, it is the paper's headline fix.** The per-set fixes (B3-S1 with H_F1, B3-S2 with H_F3C7) stay as reported comparisons.
- *Fails if* any bar is missed in either format. The counterbalancing argument is as for B3-S1/S2: a noise-like fix fails the preference bar.

### B3-Feeders — exploratory (Amr, written 2026-11-14 ~00:15 CET; B3a is running its selection grid; **no B3 test data exists**)
- *Head set:* **Feeders** = top-10(H_F3C7) minus the heads shared with top-10(H_F1). For Qwen-7B that is {L12H16, L14H12, L16H0, L16H1, L20H27}, which excludes the hub heads (L18H4, L17H26, L18H27, L19H23, L26H26). Fixed set; γ ∈ {0, 0.25, 0.5, 0.75}.
- *γ on SELECTION* (the same 100 pairs, same rule as H_F3C7): maximise F3 C7k4 bias reduction subject to F1 source preference ≥ 0.80; ties go to the larger γ. Run as a short extra job (B3a2) after B3a.
- *Test:* the chosen Feeders point in B3d alongside U (all formats + capability). **Test γ-curves** (job B3e): U at its chosen k, H_F3C7 at its chosen k, and Feeders, each at γ ∈ {0, 0.25, 0.5, 0.75}, plus base, on F3 TEST conditions C0, C1, C7k4 (all 400 pairs).
- **Hypothesis H-F (exploratory, written now):** on F3 TEST, Feeders reduce paraphrase repetition bias with **less loss of source preference** than U or H_F3C7. *Matched comparison:* let BR\* = Feeders' C7k4 bias reduction at its chosen γ. Interpolate U's test curve linearly (preference vs bias reduction) at BR\*, and the same for H_F3C7. **H-F holds if Feeders' F3 preference retained is higher than U's interpolated value at BR\*** (and the H_F3C7 comparison is reported the same way). **Fails if** it is not higher than U's. If BR\* lies outside U's test-curve range (no matching possible), that is reported as "not matchable" and H-F counts as not supported.
  - **Amendment H-F.1 (Amr, 2026-11-14 ~00:30 CET; before any B3 test data):** if BR\* **exceeds U's maximum** test bias reduction (over U's test-curve points γ ∈ {0, 0.25, 0.5, 0.75} at its chosen k), H-F is **supported** if Feeders' F3 preference retained ≥ U's preference retained **at U's maximum-reduction point** (Feeders dominates), and **not supported** otherwise. Unchanged otherwise. In particular, BR\* **below** U's minimum test reduction is still "not matchable" → not supported. *Can it fail?* Yes: Feeders can exceed U's reduction while keeping less preference than U's strongest point.
- *Descriptive (report):* the full γ trade-off curve for every set, on selection (all k × γ) and on test (the curve job), with the pre-registered chosen points marked.

### Reporting addition — which format "source preference kept" uses (Amr, 2026-11-14 ~00:55 CET; before any B3 test data; **not a criterion change**)
- *Selection (unchanged):* every set (H_F1, H_F3C7, U, Feeders) is constrained on **F1** preference (F1 selection C0/C1), because the F3 selection split has no C0 prompts.
- *Test criteria (unchanged):* S1 and U-S1 use **F1** preference (with F1 C2k4 bias reduction). S2, U-S2 and H-F use **F3** preference (with F3 C7k4 bias reduction).
- *Added to every test report:* source preference kept in **both F1 and F3** (and F2) for every set, next to the bias reduction in each format, so bias and preference can be compared within the same format. These extra numbers are descriptive and do not change any verdict.

### B3 — training-free fix (written 2026-11-13 ~23:05 CET; no B3 data exists). **Main deliverable.**
- *Fix (a), source-agnostic:* multiply the outputs (the `o_proj` input slices) of the top-k heads of a head set by γ **at every position**, with k ∈ {5, 10, 20} and γ ∈ {0, 0.25, 0.5, 0.75}. ~~*Fix (b):* B2 steering at α\*~~ — **withdrawn as a fix** (framing rule above); B2 is a diagnostic only.
- *Head sets (branch):* **H_F1** is evaluated on F1 (tables; exact copies C2 and also C7). **H_F3C7** is evaluated on F3 (documents; paraphrases C7 and also C2). Both are also run on the other format and reported as cross-format transfer (exploratory).
- *Selection size (stated before running):* the **first 100 selection pairs** (800 prompts per condition), so the 13-setting × 2-set grid fits one 2-h dev job.
- *Choosing (k, γ) on SELECTION only:* maximise repetition-bias reduction on selection (H_F1: F1 selection C2k4; H_F3C7: F3 selection C7k4), subject to source preference retained ≥ **0.80** on F1 selection (stricter than the 0.70 test bar, for headroom). Ties go to the larger γ (milder fix), then the smaller k. If no setting meets the constraint, the setting with the highest retained preference is reported and the criterion is expected to fail.
- *Metrics on TEST* (pair means over 8 variants; 95% CI by bootstrap over pairs): bias reduction = 1 − [P_low(C2k4) − P_low(C1)]_fixed / [same]_base (also with C7k4 and C2k2/k8); source preference retained = [P_low(C0) − P_low(C1)]_fixed / [same]_base; corroboration retained = [P_low(C3k4) − P_low(C1)]_fixed / [same]_base; confidence = mean max-option probability and entropy (A/B renormalised) per condition; A+B mass; P_low(C1) base vs fixed (FM F11). *Capability:* accuracy on 500 fixed MMLU test items (0-shot, A–D letter logits; seed 0) and perplexity on 200 WikiText-103 test paragraphs (≥ 50 tokens, truncated to 256), base vs fixed, with paired bootstrap CIs.
- **B3-S1 (confirmatory, Qwen-7B, H_F1 on F1 TEST):** met if bias reduction (C2k4) **≥ 50%**, source preference retained **≥ 70%**, MMLU drop **≤ 2 points** and perplexity increase **≤ 5%** (point estimates; CIs reported).
- **B3-S2 (confirmatory, Qwen-7B, H_F3C7 on F3 TEST, paraphrases):** met if bias reduction (C7k4) **≥ 50%**, source preference retained on F3 **≥ 70%**, and the same capability bars.
- **Corroboration bar:** a fix is called **selective** only if it also keeps **≥ 50%** of the independent-corroboration effect (C3k4). *Why 50%:* several independent sources *should* raise belief, and Schuster et al. treat that as the desired behaviour. A fix that removes most of it has turned "discount copies" into "ignore majorities". 50% is the point where more than half of the legitimate signal survives. Failing it does not fail S1/S2, but the paper must then call the fix non-selective.
- **B3-R (confirmatory replications, run after B2):** S1 with the same rule for Llama-8B and Gemma-12B (their Block A F1 top-20 rankings).
- *Fails if* any bar is missed. *Can fail under counterbalancing?* Yes. A fix that just adds noise pushes every P_low toward 0.5. That would give a large "bias reduction" (C2k4 and C1 both near 0.5), but source preference retained ≈ 0, so it fails the 70% bar. A letter or position artefact cancels in the pair means; HF/LF and AH/AL are reported (FM F9). Nothing in the design forces a pass.
- Reference only (different model, trained fix): Schuster et al. LoRA on Gemma-3-4B, 79.2% bias reduction, ≥ 72.5% preference retained.

### B2 — steering along the credibility direction (written ~23:05 CET; no B2 data exists)
- *Direction d:* the saved C1-only gov-vs-social probe direction (amendment 6 recipe, unbalanced; refit deterministically and saved) at **L\*** (Qwen-7B: L18) — **confirmatory**; and at the final stored layer (L28) — exploratory. Unit-normalised.
- *Intervention:* add s·α·σ·d̂ to the residual stream at the **final token** at that layer. σ = SD of the C1 selection projections on d̂; s = +1 if the gov value is option A, else −1 (oracle sign, FM F4). α ∈ {0, 1, 2, 4, 8, 16, 32}.
- *α\* (SELECTION only):* the smallest α at which F1 selection C2k4 P_low ≤ the upper bound of the 95% CI of base F1 selection C1 P_low. If none, α\* = the α that minimises it.
- *Controls:* 20 random unit directions at the same norm with the same sign rule; the neutral Blue/Green binding direction (refit inside the job on neutral-selection prompts at the same layer); and a **positive control**, replacing the full final-position residual at that layer with the C1 residual of the same item key. The direction's effect is reported as a fraction of the positive control's.
- **B2-D "the credibility direction drives the decision" (confirmatory, Qwen-7B, F1 TEST, L\*):** met if at α\* (i) TEST C2k4 P_low lies **within the 95% CI of base TEST C1 P_low**, and (ii) the 20 random directions move C2k4 P_low by **< 1/3** of the direction's change (mean |ΔP_low|).
- *Subset for the controls (stated before running):* the 20 random directions, the neutral direction and the positive control are evaluated on the **first 100 TEST pairs** (800 C2k4 prompts). Criterion (ii) compares their mean |ΔP_low| with the credibility direction's |ΔP_low| **on the same 100 pairs**. Criterion (i) uses all 400 test pairs. α\* is chosen on all 200 selection pairs.
- Also reported: C1, C2k2/k8, C7k2/k4, C3k4, C0 (side effects) and F3 transfer (exploratory).
- *Interpretation (fixed now):* met → the credibility representation is a decision variable that repetition overrides. Not met → readable but not used at that layer. In either case B2 is a **causal diagnostic / oracle upper bound and is never presented as a defence** (framing rule).
- *Fails if* (i) C2k4 stays above the C1 CI at α\*, or (ii) random directions move P_low by ≥ 1/3 as much. *Counterbalancing:* the oracle sign is balanced across letters. A direction-agnostic perturbation can only push P_low toward 0.5. From Qwen-7B's C2k4 (0.635) that cannot reach C1's CI (about 0.23–0.26), so noise cannot fake (i), and (ii) separately bounds non-specific effects.

### B1 — upstream tracing (written ~23:05 CET; runs last; **skipped if B0–B3 are not done by Sat 14 Nov 18:00 Cairo**)
- *Path patching on TEST:* F1, then F3. Clean = C1; corrupt = C2k4 and C7k4. Receivers = the top-10 heads of the branch head set (H_F1 for F1, H_F3C7 for F3); their q/k/v inputs on the shared suffix receive the patched residual, and everything else stays at the corrupt run (FM F12). Senders = every head and MLP in layers below the deepest receiver.
- *Ranking on SELECTION:* single-sender path patching on 64 selection items; take the top-10 senders.
- **B1-C "counting circuit" (confirmatory, Qwen-7B):** met if those ≤ 10 senders jointly mediate **≥ 40%** of the top-head effect (the recovery from patching the receivers' outputs directly) on held-out TEST items, **and** exceed the 95th percentile of **20 random 10-sender sets**. Otherwise reported as diffuse.
- *Positions (exploratory):* the mediated effect split by sender position group (copy blocks / source-label tokens / options / chat tail).
- *Fails if* < 40% or ≤ the random p95. *Counterbalancing:* the effect is the counterbalanced LD gap (letters swapped), so a letter-bias pathway cancels and cannot produce a pass.

### B1 scope amendment (written 2026-11-14 ~04:20 CET; **no B1 code has run, no B1 data exists**)
- *Sender pool (reduced for cost):* single-sender path patching on all ~750 heads and MLPs below the deepest receiver would take ~48k paired forwards. Instead the pool is **the top-50 heads and top-10 MLPs by Block A's selection-split AtP** (total effect, already computed on selection items, with receivers excluded and only layers below the deepest receiver kept). Each pool member is then ranked by single-sender path patching on **64 selection items**, and the top-10 are taken. Random 10-sender sets (20 draws) are drawn from the **full** layer-eligible head+MLP set, not the pool, so the random baseline is not inflated by pre-selection.
- *Path-patching variant:* step 1 replaces the sender outputs with the clean (C1) values at the shared-suffix positions in a corrupt run, and caches the receivers' outputs. Step 2 runs the corrupt prompt with only the receivers' outputs replaced by those cached values. Paths sender → (intermediate components) → receiver are therefore included (no freezing of intermediates). *Mediated fraction* = (LD_corrupt − LD_step2) / (LD_corrupt − LD_receivers-clean), where the denominator is the Block-A-style top-10 receiver patch on the same items.
- *Position scope (unchanged from F12, made explicit):* clean and corrupt prompts differ in length, so only the shared suffix (≈ the last block, the question, the options and the chat tail) can be patched. **Senders acting at the copy-block positions themselves are invisible to this design.** A null result therefore means "not via suffix positions", not "diffuse".
- *Order:* F1 / C2k4 with H_F1 receivers (confirmatory B1-C); F3 / C7k4 with H_F3C7 receivers (exploratory) if time allows. The criterion is unchanged: ≥ 40% mediated on 160 TEST items by ≤ 10 senders, and above the p95 of the 20 random sets.

## Priority (Amr)
B0 → B3 (Qwen) → B2 → B3 replication (Llama, Gemma) → B1 (skipped if B0–B3 are not done by Sat 18:00 Cairo). Dev-QoS Slurm dependency chain, one GPU job at a time; CPU work in parallel.

---
## Results — summary (job ids, all numbers: `JOBS.md`, `numbers_B.json`; figures `figs/figB0–B3*`)
| criterion | label | result |
|---|---|---|
| B0-a/b repetition heads are induction heads | confirmatory, 3 models | **NOT MET** in all 3 (0/10 induction heads; mean prefix-matching below the random p95) |
| B3-S1 head-scaling fix, exact copies in tables | confirmatory, Qwen-7B | **MET**: 54% bias reduction, 80% preference kept, 60% corroboration kept (selective), MMLU −1.8 (CI to −3.6), ppl +0.3% |
| B3-S1 replication | confirmatory, Llama / Gemma | **NOT MET** / **NOT MET** (Llama: no setting keeps preference, as its baseline credibility gap is ~0.05; Gemma: 32%) |
| B3-S2 paraphrases in documents (H_F3C7) | confirmatory, Qwen-7B | **NOT MET**: 26% bias reduction (85% preference kept) |
| B3-U-S1 / U-S2 single deployable fix (union) | confirmatory, Qwen-7B | **NOT MET** / **NOT MET** (44% / 8%) |
| H-F Feeders keep more preference at matched reduction | exploratory | **NOT SUPPORTED** (89% vs U's 97% at 9% reduction) |
| B2-D credibility direction drives the decision | confirmatory at L18; exploratory at L28 | **NOT MET** at both. At L28 the residual swap restores the answer (Δ −0.44) but the direction does nothing (Δ ≈ 0) → a readout, not a lever (diagnostic only) |
| B1-C ≤ 10 upstream components mediate ≥ 40% | confirmatory, Qwen-7B F1 | **MET (marginal)**: 40.6% [36.5, 44.7] vs random p95 7.0%. F3/C7 (exploratory): 46.9% vs 8.8% |

**What Block B says.** (1) Repetition is not carried by induction heads, and the hubs that read the answer options are fed mainly by **mid-layer MLPs**. Exact copies additionally use one induction-like head (L14H6), which paraphrases do not (exploratory). (2) A training-free fix exists **only for the narrow case** it was tuned on (exact copies, tables, Qwen-7B), and it does not replicate in Llama or Gemma. (3) For **paraphrased repetition in RAG documents**, the realistic threat, no head-scaling setting tested reaches ≥ 50% reduction with ≥ 70% source preference kept: the same heads carry source credibility and repetition, so damping them trades one for the other almost one-for-one (figB3). (4) The C1-trained gov-vs-social direction reads the outcome but is not a causal lever at the tested magnitudes. Steering is reported only as a diagnostic.

**Deviations from the Block B prompt (all pre-registered before the affected data):** two head sets, by the branch rule. B3(b) steering-as-fix was withdrawn (oracle). The union fix B3-U and the Feeders/H-F test were added (Amr). Selection used 100 of the 200 selection pairs (for time). The B2 controls used the first 100 test pairs. B1 used a reduced sender pool (top-50 heads + top-10 MLPs by Block A AtP), with intermediate paths allowed and suffix positions only. F3 C2k8 was not generated, so F3 C2 is k = 2/4 only. B0's first run OOM'd (bug fixed, rerun). Budget: 8.56 / 15 A100-h.

## Results — detail (chronological)

### B3 — H_F1 fix on TEST (job 165398; Qwen-7B; scale 10 heads by γ = 0.5, chosen on selection; `results/B/summary_B.md`)
| format | bias reduction C2k4 | bias reduction C7k4 | source preference kept | corroboration kept (C3k4) |
|---|---|---|---|---|
| **F1 tables** | **54% [52, 57]** | 11% [10, 13] | **80% [77, 82]** | 60% [57, 62] |
| F3 documents | 22% [20, 24] | 7% [5, 8] | 96% [94, 97] | 79% [77, 80] |
| F2 news | 19% [17, 21] | 5% [4, 6] | 93% [91, 95] | 70% [68, 73] |
*Capability:* MMLU 70.8 → 69.0 (**−1.8 pts** [−3.6, +0.0], n = 500); WikiText perplexity 12.79 → 12.83 (**+0.3%** [+0.1, +0.5], n = 200).
**B3-S1: MET (confirmatory).** The fix is also **selective** (corroboration kept 60% ≥ 50%). Caveats: the MMLU point estimate passes the ≤ 2-point bar, but its CI reaches −3.6. The fix is specific to the condition it was chosen on: it removes only 11% of the *paraphrase* bias in the same format and 19–22% of exact-copy bias in news/documents. C1 P_low rises 0.243 → 0.295 (FM F11), and C0 stays at 0.500.

### B3-R — replication on Llama-3.1-8B (job 170008; `results/B/llama/`)
- *Selection:* no setting meets the ≥ 80% preference constraint. Llama's selection baseline preference gap is tiny (C1 0.452 vs C0 0.500), so "preference kept" divides by ≈ 0.05 and every damping that removes bias drives it to ≈ 0 or below (e.g. k10 γ0: 86% bias removed, −4% preference kept). Per the pre-registered fallback, the setting with the highest preference kept was used: **k = 5, γ = 0.75**.
- *TEST:* F1 bias reduction **−2% [−3, −1]**, preference kept 73%, corroboration kept 96%; MMLU +0.4 pts, perplexity −0.1%.
- **B3-S1 (Llama): NOT MET (confirmatory).** The fallback setting is essentially a no-op. *Context:* Llama's baseline credibility preference in tables is weak (Block A: C1 0.434 vs C0 0.500), so there is little source preference for a head-scaling fix to keep while it removes repetition bias. The trade-off seen in Qwen is sharper here.

### B3-R — replication on Gemma-3-12B (jobs 170009 select, 170010 test; `results/B/gemma/`)
- *Selection:* chosen **k = 5, γ = 0.5** (constraint met: selection preference kept 87%, bias reduction 39%). The grid is non-monotone (e.g. k10 γ0: 18%, k10 γ0.5: 62% but preference 64%).
- *TEST:* F1 bias reduction **32% [28, 36]**, preference kept 92% [83, 102], corroboration kept 91%; MMLU −1.2 pts [−3.2, +1.0]; perplexity −2.2%. In news/documents the fix *raises* exact-copy bias (F2 −11%, F3 3%).
- **B3-S1 (Gemma): NOT MET (confirmatory)**, 32% < 50%.
- **Replication summary: B3-S1 is met in Qwen-7B only (1 of 3 models).**

### B1 — upstream path patching, Qwen-7B (job 174087; `results/B/b1_path_*.json`; reduced sender pool + suffix-only scope, see the scope amendment)
| run | receivers | top-10 senders (ranked on 64 selection items) | mediated on 160 TEST items [95% CI] | random 10-sender p95 (20 draws) | verdict |
|---|---|---|---|---|---|
| **F1 / C2k4** (confirmatory) | H_F1 top-10 | MLPs L12, L13, L14, L16, L18; heads **L14H6**, L13H13, L14H18, L4H8, L16H0 | **40.6% [36.5, 44.7]** | 7.0% | **B1-C MET** |
| F3 / C7k4 (exploratory) | H_F3C7 top-10 | MLPs L14, L16, L19, L20; heads L17H24, L11H8, L11H0, L13H1, L17H22, L15H4 | **46.9% [42.1, 51.4]** | 8.8% | met (exploratory) |
**Reading.** Ten upstream components, mostly **mid-layer MLPs (L12–L20)** plus a few heads, carry ~40–47% of the decision heads' repetition effect through the suffix positions, 5–6× the random baseline. The confirmatory pass is marginal: the point estimate is just above 40% and the CI's lower end (36.5%) is below it. *Exploratory link to B0:* **L14H6**, the one strongly induction-like head (prefix-matching 0.95), is a top sender for **exact copies** (F1/C2) but not for **paraphrases** (F3/C7), where the senders are different MLPs and heads. That fits the B0 expectation (induction-style copying feeds exact repetition, not paraphrase), but it is a single head. *Scope:* only suffix positions were patched, so components acting inside the copy blocks are not measured.

### B2 — steering, Qwen-7B, L\* = L18 (confirmatory; job 165440; `results/B/b2_steer_Qwen2.5-7B-Instruct.json`)
- σ (SD of C1 selection projections on d̂) = 0.143. Selection C2k4 P_low vs α = 0, 1, 2, 4, 8, 16, 32: 0.618 → 0.612. The C1 CI upper bound (~0.25) is never reached, so α\* = 32 (minimiser).
- TEST at α\*: C2k4 0.634 → 0.626 (C1 CI [0.228, 0.254]) → **(i) not met**. Other conditions move ≤ 0.01 (C1 0.242 → 0.239, C0 0.500 → 0.491, C7k4 0.915 → 0.914).
- Controls (first 100 test pairs, C2k4): direction Δ = −0.008; 20 random directions mean |Δ| = 0.009 → **(ii) not met**; neutral-binding direction Δ = −0.001 (its probe AUC at L18 = 0.71); **positive control** (C1 final-position residual swapped in) Δ = **+0.076**, the wrong sign.
- **B2-D at L\*: NOT MET.** Per the pre-registered reading, "readable but not used *at this site*". The positive control shows that **the final-position residual at L18 does not carry the repetition decision**: replacing it entirely leaves P_low near the corrupt value. The decision heads at L17–19 read the option tokens *after* this point. So final-token steering at L18 cannot test whether credibility drives the decision; the null result is about the site, not the representation. The final-layer test (exploratory) follows.

### B2 — steering at the final layer L28 (exploratory; same job 165440)
- Selection C2k4 stays at **0.618 at every α up to 32σ** (α\* = 8, a tie). TEST: every condition is within ±0.001 of base in F1 and F3 (e.g. C2k4 0.634 → 0.634; C7k4 F3 0.955 → 0.954) → (i) not met.
- Controls (first 100 test pairs): direction Δ ≈ 0.0001; random mean |Δ| = 0.002 → (ii) not met; neutral direction Δ = +0.003. **Positive control: Δ = −0.438.** Swapping in the C1 final-position residual at L28 moves C2k4 P_low from about 0.65 to about 0.21, essentially all the way to C1.
- **Reading:** at L28 the final-position residual *does* carry the repetition decision (positive control ≈ full restoration), yet the C1-trained gov-vs-social direction has **no causal effect** at magnitudes up to 32 SDs of its own projection. Under repetition this direction is a **readout that correlates with the answer** (Block A: flipped-item AUC 0.17 at L28), **not a lever** at the tested norms. *Limitation:* the α grid was pre-registered in units of the probe-projection SD, which is small relative to the residual norm. A much larger push was not tested.
**B2 overall: B2-D NOT MET at L\* (confirmatory) or at the final layer (exploratory).** Per the fixed interpretation: "readable but not used". B2 is a causal diagnostic only and is never presented as a defence.

### B3-Feeders on TEST (job 165952; 5 F3/C7-only heads, γ = 0.5, chosen on selection; exploratory)
| format | bias reduction C2k4 | bias reduction C7k4 | source preference kept | corroboration kept |
|---|---|---|---|---|
| F1 | 12% [9, 16] | 9% [7, 10] | 85% [82, 88] | 89% [87, 92] |
| **F3** | 3% [1, 6] | **9% [8, 10]** | **89% [86, 90]** | 90% [88, 92] |
| F2 | −11% [−13, −8] | 4% [3, 5] | 92% [90, 94] | 100% [98, 103] |
*Capability:* MMLU −0.4 pts [−1.6, +0.6]; perplexity −0.1%. At the chosen point, Feeders barely move paraphrase bias (9%) and cost ~11% of source preference. In F2 they slightly *increase* exact-copy bias. 

### B3e — test γ-curves on F3 (job 165953; `results/B/b3_curve_*.json`; figure `figs/figB3_tradeoff_F3.png`)
| γ | U: paraphrase BR / F3 pref kept | H_F3C7: BR / pref | Feeders: BR / pref |
|---|---|---|---|
| 0 | 87% / 11% | 74% / 32% | 26% / 61% |
| 0.25 | 56% / 49% | 49% / 64% | 15% / 78% |
| 0.5 | 26% / 81% | 26% / 85% | 9% / 89% |
| 0.75 | 8% / 98% | 10% / 96% | 5% / 95% |
**H-F (exploratory): NOT SUPPORTED.** At Feeders' chosen point (BR\* = 9%, F3 preference kept 89%), U's test curve interpolated at the same reduction keeps **97%**. The Feeders lose *more* source preference per unit of bias removed, not less. (vs H_F3C7: BR\* is below its curve's range → not matchable.)
**Descriptive, not a criterion:** no tested (set, γ) point on these curves reaches the S2 corner (≥ 50% paraphrase reduction with ≥ 70% F3 preference). The closest are H_F3C7 γ = 0.25 (49% / 64%) and U γ = 0.25 (56% / 49%). So S2's failure is not an artefact of the selection rule. Head scaling on these heads trades source preference against paraphrase bias almost one-for-one, well short of Schuster et al.'s trained LoRA (79% / 72.5%, different model).

### B3-U — union fix on TEST (job 165540; 28 heads = top-20 H_F1 ∪ top-20 H_F3C7, γ = 0.75, chosen on selection)
| format | bias reduction C2k4 | bias reduction C7k4 | source preference kept | corroboration kept |
|---|---|---|---|---|
| F1 | 44% [42, 47] | 9% [8, 11] | 88% [86, 91] | 63% [60, 65] |
| F3 | 15% [13, 16] | 8% [7, 9] | 98% [96, 100] | 81% [79, 83] |
| F2 | 11% [9, 12] | 4% [3, 5] | 96% [95, 98] | 80% [77, 82] |
*Capability:* MMLU −0.6 pts [−2.0, +0.8]; perplexity +0.2%.
**B3-U-S1: NOT MET** (44% < 50%). **B3-U-S2: NOT MET** (8% < 50%). Confirmatory. The selection rule (maximise the *worse* of the two reductions under ≥ 80% preference) picks a mild γ = 0.75 on 28 heads, so the union is safe but weak. **There is no single deployable head-scaling fix for Qwen-7B that meets both bars.**

### B3 — H_F3C7 fix on TEST (job 165399; scale 20 heads by γ = 0.5, chosen on selection)
| format | bias reduction C2k4 | bias reduction C7k4 | source preference kept | corroboration kept |
|---|---|---|---|---|
| F1 tables | 49% [46, 52] | 30% [28, 32] | 88% [85, 92] | 55% [53, 58] |
| **F3 documents** | 23% [21, 26] | **26% [24, 28]** | **85% [83, 87]** | 72% [70, 74] |
| F2 news | 14% [12, 16] | 15% [14, 17] | 89% [87, 91] | 77% [74, 80] |
*Capability:* MMLU 70.8 → 69.6 (−1.2 pts [−3.0, +0.4]); perplexity −0.5%.
**B3-S2: NOT MET (confirmatory):** paraphrase bias reduction in documents is 26%, below the 50% bar (preference and capability pass). Paraphrased repetition in the RAG format is not fixed by scaling the heads that carry it, at a setting that keeps ≥ 80% of source preference on selection.

### B0 — induction-head check (job 165448; `results/B/b0_induction_*.json`; the first attempt 165344 OOM'd and produced nothing)
*Sanity:* the method finds induction heads where they exist. Each model has 7–10 heads meeting the induction definition (prefix ≥ 0.30 and copying > p90), and the strongest have prefix-matching 0.96–0.98 (Llama L15H30, Qwen L22H3, Gemma L17H15).
| model (head set) | induction heads in top-10 (B0-a: ≥ 5) | mean prefix-matching, top-10 | random 10-head p95 (B0-b) | verdict |
|---|---|---|---|---|
| Llama F1 | **0/10** | 0.027 | 0.084 | **not induction heads** |
| Qwen-7B F1 | **0/10** | 0.042 | 0.120 | **not induction heads** |
| Gemma F1 | **0/10** | 0.098 | 0.114 | **not induction heads** |
| Qwen-7B F3/C2 (A11) | 0/10 | 0.128 | 0.120 | enriched (B0-b), driven by one head (L14H6, prefix 0.95, copying p74) |
| Qwen-7B F3/C7 (A11) | 0/10 | 0.062 | 0.120 | not induction heads |
| 20 random heads (each model) | 0/20 | 0.017–0.032 | — | — |
**B0-a and B0-b are not met in any model's primary (F1) head set (confirmatory): the repetition heads are not induction heads.** The decision heads (Llama L17H24 prefix 0.13; Qwen L18H4 0.00; Gemma L29H11 0.21) do not do prefix-matching on repeated random tokens. Their copying scores are mixed (Llama L17H24 at p91; Qwen L18H4 p53; Gemma L29H11 p20).
*C2 vs C7 (exploratory, Qwen-7B, 5 heads per group):* mean prefix-matching of the F3/C2-only heads = 0.24 (L14H6 0.95, L19H18 0.20, L16H6 0.06, L14H18 0.01, L17H22 0.00); F3/C7-only heads = 0.11 (L16H0 0.23, L16H1 0.18, L19H23 0.13, L12H16 0.01, L18H27 0.00); shared heads = 0.03. This is in the direction of the expectation (exact-copy-only heads more induction-like), but it rests on a single induction-like head (L14H6) and n = 5, so it is not evidence. **Reading:** repetition is not carried by classical induction/copying circuitry. The hub heads read the answer options (Block A attention), and the count of supporting blocks reaches them through other heads, in line with paraphrases (no token-level repeats) using largely the same hub.
