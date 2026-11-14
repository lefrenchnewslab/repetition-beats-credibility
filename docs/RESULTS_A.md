# RESULTS_A — Block A: scale-up, held-out validation, realistic formats
**SPRINT WORK (Apart AI x Epistemics, 13–15 Nov 2026).** All numbers below come from fresh v2 data (seed 20261113). No pilot numbers are reported as results.

## Pre-registration (written 2026-11-13, before any v2 data was generated or any Block A job ran)
Definitions follow the pilot. P_low = A/B-renormalised probability of the letter carrying the low-credibility (social) value, averaged over the 8 counterbalanced variants of a pair. LD = logit(low letter) − logit(high letter).

**H1 — held-out head verification.** Heads are ranked by attribution patching (AtP) on the **selection** split (200 entities), with clean = C1 and corrupt = C2 k=4, in format F1. On the **test** split (400 entities, disjoint), real activation patching of the clean outputs of the **top-10** heads into the corrupt run must recover **≥ 40%** of the C2(k=4) − C1 logit difference. Recovery = (mean LD_corrupt − mean LD_patched) / (mean LD_corrupt − mean LD_clean), over test items with LD_corrupt − LD_clean > 1 (FM B3). 95% CI by bootstrap over items.

**H2 — held-out credibility probe.** A linear probe for "which answer option is government-sourced" is fit on final-token residuals of the selection split. On held-out test entities at C2 k=8, its AUC must be **≥ 0.8 on all items and ≥ 0.8 on the flipped items** (variant-level P_low > 0.5). The layer is chosen beforehand, as the layer with the highest C2 k=8 AUC on a held-out-entity fold inside the selection split (FM E4).

**How the verdict is scored.** Each criterion is evaluated per model, in F1, for Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and Gemma-3-12B-it. The verdict reports in how many models it is met. Per PLAN.md, the circuit framing stays primary if H1 holds in ≥ 1 model. If it fails in all three, the paper leads with behaviour + probe + steering. Qwen2.5-32B-Instruct is behaviour only and does not count toward H1/H2.

**Named secondary contrasts (reported regardless, with CIs):** C3 k=4 vs C2 k=4 (independent vs copies); C7 k=4 vs C2 k=4 (paraphrased vs exact copies); C5 vs C2 k=4 (length control, in % of the C2 k=4 shift); and the dose-response C1 → C2 k=2/4/8 per model × format.

_No result has been seen at the time of writing._

**Amendment 1 (2026-11-13, after the code-check smoke test, before any full run).** The H2 layer is chosen by the lowest **log-loss** on the selection-internal fold at C2 k=8, not the highest AUC. In smoke job 140845 the selection-fold AUC saturated at 1.00 from layer 6 onward, so "highest AUC" degenerates to "first tied layer". The 0.8 threshold and everything else are unchanged. _Disclosure:_ the smoke outputs (Llama, 3 selection + 3 test pairs, a code check) were read before this amendment. They are not results and are not reported.

**Amendment 2 (Amr, 2026-11-13, before the Gemma and Qwen-32B runs; job 142090, Llama + Qwen-7B behaviour, was running and none of its output had been read).**
- *Qwen2.5-32B scope:* test split only, formats F1 + F3. Conditions in priority order C0, C1, C2 k=2, C2 k=4, C3 k=4, C5, C7 k=4 (both formats per condition). Results are streamed to disk, so a timeout at the 2 A100-h cap drops only the lowest-priority conditions.
- *Probe controls (all three 7–12B models):* (a) **lexical baselines** on the same label and splits as the credibility probe: layer-0 final-token AUC (from the saved probes), mean-pooled input embeddings over the whole prompt, and TF-IDF word 1–3-grams (model-free). (b) **Graded credibility:** 100 test pairs, C1-shaped, F1. Source X ∈ {government, newspaper ("The {Place} Gazette/Tribune/…"), individual person (full name), social-media user} holds one value, and a social-media user holds the other; 8 counterbalanced variants (3,200 prompts). Final-token states are projected onto the saved gov-vs-social direction (`cred_all`, fit on selection), oriented toward X's option. The per-layer Spearman between source rank (gov 3 > newspaper 2 > person 1 > social 0) and projection is reported, along with the per-level mean projections and, for comparison, the behavioural P(X's value).
- **H3 — "credibility is represented" (pre-registered here).** Met for a model if, at its pre-registered layer: (i) the mean projections are strictly ordered gov > newspaper > person > social; (ii) the pooled Spearman ρ > 0, with a 95% CI (cluster bootstrap over pairs) excluding 0; and (iii) every lexical baseline is clearly lower, defined as test AUC at C2 k=8 ≤ 0.65 **and** ≥ 0.20 below the probe's AUC at the chosen layer. The layer-0 final-token AUC is 0.5 by construction (the final token is the same chat-template token in every prompt). It is reported as asked, but it is not informative; the mean-pooled and n-gram baselines carry the comparison. _Note added 2026-11-13, still before any full-run output was read:_ running the n-gram baseline on the data alone showed AUC = 0.500 with zero decision-function variance. This holds by construction. Under the 8-way counterbalancing, every prompt has a mirror with the opposite label and the same token multiset (e.g. lowidx 0/AH vs lowidx 1/AL), so no order-free lexical feature set can carry the label. The same applies to mean-pooled embeddings and the layer-0 final token. Condition (iii) is therefore met by design. It shows the label requires contextual binding, but it is not an empirical hurdle. As a non-trivial shallow reference, the probe AUC at the earliest stored layers (2 and 4) is reported next to it. The criterion is unchanged. **Amendment 3 (Amr, 2026-11-13; no full-run output exists — job 142090 is still queued).** The lexical baseline stays reported with the note above, but it is replaced as a *control* by a **neutral-label binding control**.
- *Data:* C1-shaped prompts in F1. Source labels are replaced by neutral tags "Source: Blue" / "Source: Green". The tag→block assignment is swapped within each pair (16 variants = tag swap × value swap × table order × answer order). **Neutral-selection:** 100 selection-split pairs. **Neutral-test:** 100 test-split pairs. 1,600 prompts each.
- *Binding probe:* the same recipe as `cred_all` (final-token residual, every stored layer, L2 logistic regression, lambda chosen on a 75/25 entity fold inside neutral-selection). Label: "option A is the Blue-tagged value". Fit on neutral-selection, test AUC on neutral-test, reported per layer.
- *Projection:* neutral-test prompts are projected onto the saved gov-vs-social direction (`cred_all`). Expected near the midpoint, defined as: mean probe logit oriented toward the Blue option ≈ 0, and mean |logit| < 25% of the mean oriented logit of graded-set government-vs-social prompts (both at the chosen layer). Reported: mean, |mean|, and that ratio.
- **Interpretation, fixed now:** if the neutral binding AUC reaches **≥ 0.95** at any layer, then a high gov-vs-social AUC (H2) shows only that the model **binds source tags to values**. The claim "credibility is represented" then rests **solely on H3** (graded ordering). If the neutral binding AUC stays < 0.95 at every layer while H2 holds, the gap is reported as evidence that gov-vs-social carries more than tag binding. H3 is still required for the credibility claim. If neutral prompts do **not** sit near the midpoint, the gov-vs-social direction also responds to arbitrary tags, and that is reported as a limitation of the direction.

**Amendment 4 (Amr, 2026-11-13; no output read — job 142090 still queued). Supersedes the conflicting parts of amendments 2–3.**
1. **Neutral midpoint.** The Blue-vs-Green projection is ≈ 0 by symmetry, whether neutral tags sit at the midpoint or at the social end (social vs social is ≈ 0 too). It is **kept as a sanity check only, not a criterion**. Added instead: a **5th graded level, X = "Source: Blue" vs a social user** (same 100 test pairs, C1, F1, 8 variants; +800 prompts per model). Reported: its mean projection relative to the four named levels, its normalised position (neutral − social)/(gov − social), and behavioural P(choose X). **Exploratory.** The neutral Blue/Green **binding AUC** rule from amendment 3 is unchanged.
2. **H3 (rewritten; replaces amendment 2's (i) and (ii)).** The social level (X = a second social user vs the social partner) is ≈ 0 by construction: the two sides are exchangeable, and the score is oriented toward X under counterbalancing. It serves **only as an anchor**. H3 is met for a model only if, at its pre-registered layer, all three hold, each tested on pair-level means (8 variants averaged per pair) with a **95% cluster bootstrap over pairs** (2,000 resamples):
   (a) gov − newspaper > 0, with CI excluding 0;
   (b) newspaper − person > 0, with CI excluding 0;
   (c) person > 0, i.e. person − social anchor > 0, with CI excluding 0.
   The Spearman over the three non-social levels (gov 2, newspaper 1, person 0) is **descriptive only**. Amendment 2's condition (iii) (lexical baselines) is kept as written; it is met by construction (see the note above).
3. **H1 (addition).** Besides top-10 recovery ≥ 40%, the top-10 recovery must **exceed the 95th percentile of the random-10-head recoveries** (20 draws, the same held-out test items). **Both are required.**

**Can each criterion fail under the counterbalancing?** (checked before deployment)
| criterion | outcome that fails it | can counterbalancing force a pass? |
|---|---|---|
| H1 (≥ 40%) | Patching the 10 selection-ranked heads recovers < 40% of the C2k4−C1 logit difference on test items. Happens if the repetition effect is diffuse, or if the selection ranking does not transfer to new entities. | No. Recovery is measured on test items with a real effect (LD gap > 1). The counterbalancing only balances letters and positions. |
| H1 (> p95 random) | Some random 10-head sets recover as much as the top-10 (e.g. patching almost any suffix head moves the logit, or the top-10 recovery is small). | No. The random draws use the same items and the same patching. |
| H2 (AUC ≥ 0.8, C2k8 all + flipped) | The final-token probe gives < 0.8 on held-out C2k8, or on the flipped items (e.g. the representation follows the output letter). | No. The label is balanced (chance = 0.5), and no order-free lexical feature carries it (shown: n-grams exactly 0.5), so it cannot pass on surface tokens. |
| H3(a) gov > newspaper | The newspaper projects as high as the government, e.g. the direction encodes "institutional vs individual" rather than graded credibility. | No. Both levels face the same social partner in the same 8 variants; only X's label differs. |
| H3(b) newspaper > person | Person ≥ newspaper, e.g. the direction tracks label length or name-likeness. | No, for the same reason. |
| H3(c) person > 0 | The person projects at or below the social anchor, e.g. the direction is binary "government vs everything else". **Under the null that the direction ignores source identity, every level's mean is ≈ 0 by the same symmetry as social vs social, so (a)–(c) all fail.** | No. Symmetry makes the null *fail* H3, not pass it. |
| H3 (iii) lexical baselines | **Cannot fail**: every order-free baseline is exactly 0.5 under the mirror-variant design. Kept as a recorded fact, not as evidence. | Yes, by construction (disclosed). |
| Neutral binding AUC ≥ 0.95 (amendment 3) | AUC < 0.95 at every layer (the model does not bind arbitrary tags as cleanly as gov/social labels). This is an interpretive switch, not pass/fail. | No. The label is balanced, with the same mirror symmetry. |

**Amendment 5 (Amr, 2026-11-13, 16:45; before job A11 runs).** For the F3 head analysis of the best model (chosen by the F1 held-out top-10 recovery, FM E19), patching is run twice. Corrupt = C2 k=4 (exact copies, as before), and **corrupt = C7 k=4 (paraphrased copies)**, both against clean = C1. For C7, heads are ranked by AtP on F3 *selection* C7 k=4 prompts. These are generated deterministically in a new file (`prompts_f3selC7.jsonl`, 1,600 prompts = 200 selection pairs × 8 variants; all other files unchanged), and the model's behaviour on them is run first, for the LD-gap > 1 item filter. Verification is on F3 test items. Reported: held-out recovery for each, and **head overlap** (top-10 Jaccard, all-head AtP Spearman, the shared and unique heads) between F1/C2k4, F3/C2k4 and F3/C7k4. This is **exploratory and decides the Block B target**: if F3/C7k4 shares most of its top heads with F1/C2k4, Block B targets one repetition circuit; if not, the paraphrase route is a separate target. No threshold is set, because nothing is accepted or rejected on it. _Scope note:_ in C7 the copy nearest the question is reworded, so the clean/corrupt shared suffix is shorter when the social block comes last (only question + options + chat tail). Mean suffix length is reported per run.

**Amendment 6 (Amr, 2026-11-13, ~17:00; Amr called it "amendment 5". Written before the probe below is fit for any model).** *Why:* the Llama `cred_all` probe scores AUC 0.997 on C0r (no labels; low value ×4). In every training condition with k ≥ 2, the gov value is also the singleton value, so H2's k=8 AUC cannot separate "source identity" from "which option is the minority value". *Test (exploratory):* the same recipe (final-token residual, every stored layer, L2 logistic, lambda chosen on a 75/25 entity fold inside selection), but **trained on C1 only** (selection split, 1 vs 1, so there is no count asymmetry). Evaluated on held-out test entities: C2 k=8 (all, and flipped items), C0r k=4, **C6 k=4 and C6 k=8** (gov repeated k× vs one social block; C6 k=8 is generated new, test split only), plus C1 (in-distribution) and C0 (sanity, which should be ≈ 0.5). Reported at the model's pre-registered H2 layer and per layer. C6 states need a short GPU forward pass, added to job A10. The CPU probe runs in parallel with the GPU chain (Amr's instruction).
*Predictions, stated now:*
| test set | if the probe tracks **source identity** | if it tracks **minority/count** |
|---|---|---|
| C2 k=8 all / flipped | ≥ 0.9 / ≥ 0.9 | ≥ 0.9 (gov is the minority) |
| C0r k=4 (no labels) | 0.4–0.6 (no source to read) | ≥ 0.9 (the singleton side is labelled "gov") |
| C6 k=4 / k=8 (gov is the **majority**) | ≥ 0.9 | ≤ 0.1 (inverted) |
| C1 (in-distribution) | ≥ 0.9 | ~0.5 (no count difference) |
If C0r and C6 both match the source column, H2's AUC is read as source identity. If they match the count column, H2's AUC is reported as confounded with count, and the C1-only numbers replace it in the paper. Anything in between is reported as mixed, with both columns.

**Amendment 7 (Amr, 2026-11-13, ~17:40; Amr called it "amendment 6"). Layer-resolved source claim. Confirmatory for Qwen-7B and Gemma, whose C1-only probe has not been fit (their states have not landed). Exploratory for Llama, because its per-layer table, including L14, was seen first.**
- *Layer choice (selection split only):* fit the C1-only probe per layer on the 75% train fold of selection entities. **L\*** = the layer with the highest AUC on **decision-disagreeing C1 items** (the model's chosen option ≠ the gov option, i.e. variant-level P_low > 0.5) in the 25% held-out-entity fold. The probe is then refit on all selection C1 items with that layer's lambda.
- **H4 "source computed but overridden"** holds for a model if, on the TEST split, all three hold:
  (a) at L\*, AUC on C2 k=8 **flipped** items ≥ 0.75, with a 95% CI (cluster bootstrap over pairs, 1,000 resamples) excluding 0.5;
  (b) at L\*, C0r k=4 AUC within [0.40, 0.60];
  (c) at the **final stored layer**, flipped-item AUC < 0.5, with a CI excluding 0.5 (the override is late).
- *Fails if:* (a) < 0.75 or its CI includes 0.5 → no source signal survives repetition. (b) outside the band → L\* still tracks count or the decision. (c) not met → no late override, and "computed but overridden" is replaced by "source identity decodable mid-stack" only (when (a) and (b) hold).
- *Can each part fail under the counterbalancing?* (a) Yes: the flipped-item labels are balanced across letters, and a decision-tracking direction gives < 0.5 there (Llama L26: 0.29). (b) Yes: both inversion (Llama L26: 0.07) and count-tracking (≥ 0.9) fall outside the band. (c) Yes: a direction that stays source-aligned to the last layer gives ≥ 0.5. Nothing in the design forces a pass, and L\* is chosen without any C2/C0r/test data.
- C6 k=4/8 are reported but are **not a discriminator**: source and decision both point to gov there.
- Figure: flipped-item C2k8 AUC by layer for each model, with the 0.5 line and L\* marked.

**Amendment 8 (Amr, 2026-11-13, ~18:10; Amr called it "amendment 7"). Written before any Qwen-7B or Gemma C1-only probe has run.** The primary H4 test stays at L\*. Added, pre-specified:
- *Robustness report (not a criterion):* the near-max layer set S = every layer whose selection-split decision-disagreeing val AUC is within **0.005** of the maximum. Report (a) the mean C2k8 flipped-item AUC over S, and (b) the mean C0r AUC over S, against the same thresholds (≥ 0.75; 0.40–0.60), together with |S| and the layers in S. Also the full per-layer curves for C2k8-flipped and C0r (table + fig5).
- *Description only:* the C2k8 flipped-item AUC at the **final layer**, next to the model's own choice rate at C2k8 (share of test items where the model picks the social value, P_low > 0.5). This documents whether source information persists at the output while the answer goes the other way.
- Applies to Qwen-7B and Gemma (H4 confirmatory; robustness pre-specified) and to Llama (exploratory throughout).
- Computed in `analyze_v2.py` from the per-layer AUCs the probe already saves; there are no new fits.

**Amendment 9 (Amr, 2026-11-13, ~18:25; written before any Qwen-7B or Gemma C1-only probe has run). Confirmatory for Qwen-7B and Gemma, exploratory for Llama (its value was seen in amendment 8).**
- **H5 "source identity persists at the output"** is met for a model if, at the **final stored layer**, the C1-only probe's AUC on C2 k=8 flipped test items is **≥ 0.70 with a 95% CI excluding 0.5** (cluster bootstrap over pairs, 1,000 resamples), **and** the model picks the social value on **≥ 50%** of C2 k=8 test items (variant-level P_low > 0.5).
- *Fails if* the final-layer flipped AUC is < 0.70 or its CI includes 0.5 (e.g. it tracks the decision and goes < 0.5, as Llama does at L26), or if the model picks the social value on < 50% of C2k8 test items (then there is no "answer goes the other way" to document). Both parts can fail: the first as Llama's L22–L28 show, the second as Qwen-7B's pilot C2k8 rate (~0.52 mean P_low) shows is possible.
- *Robustness (report only):* the same AUC and CI at the **last 3 stored layers**. Only every 2nd layer is stored, so for Llama these are L28/L30/L32, for Qwen-7B L24/L26/L28, and for Gemma L44/L46/L48.
- H4 and amendment 8 are unchanged.

**Amendment 10 (Amr, 2026-11-13, ~19:00; written before any Gemma C1-only output exists). Decision-balanced C1 probe. Confirmatory for Gemma, exploratory for Llama and Qwen-7B (their unbalanced C1-only results were seen).**
- *Training set:* F1 selection C1 items. Each is labelled y = "option A is gov-sourced" and d = agree (the model picks the gov value, P_low ≤ 0.5) or disagree (P_low > 0.5). Subsample (seed 0) so that the four cells y × d have **equal counts** (n_cell = the smallest of the four), which makes the model's choice independent of the label in the training set.
- *Stop rule:* if the C1 selection split has **< 200 decision-disagreeing items** in total, report that and fit nothing for that model.
- *Recipe:* as in H4. A 75/25 entity fold inside the balanced set; lambda by val AUC; **L\*** = the layer with the highest val-fold AUC on decision-disagreeing items; refit on the whole balanced set.
- *Tests and thresholds, unchanged:* H4(a) C2k8 flipped AUC at L\* ≥ 0.75 with CI excluding 0.5; H4(b) C0r at L\* within 0.40–0.60; H5 final-layer flipped AUC ≥ 0.70 with CI excluding 0.5, and the model picks social on ≥ 50% of C2k8. Reported per model alongside the unbalanced results, with n_cell and the last-3-layer report.
- *Can it fail?* Yes, in the same ways as H4/H5. Balancing removes the shortcut "label = predicted answer" from training, so a pass cannot come from decision tracking learned on C1.

If H2 is met but H3 is not, the paper says "the gov-vs-social *source identity* is decodable", not "credibility is represented".

---
## Results (summary; full tables in `results/A/summary_A.md`, every number in `numbers_A.json`)
_All numbers: fresh v2 data, held-out **test** split (400 entities, 8 counterbalanced variants each) unless stated. 95% CIs are bootstrap over pairs. Interim entries (chronological, with job ids) follow below the summary._

### Pre-registered verdicts
| criterion | Llama-3.1-8B | Qwen2.5-7B | Gemma-3-12B |
|---|---|---|---|
| **H1** top-10 heads recover ≥ 40% on held-out items **and** > 95th pct of 20 random 10-head draws | **MET** 91.9% [90.4, 93.4]; random p95 30.9% | **MET** 95.4% [92.3, 99.0]; random p95 19.4% | **MET** 93.1% [91.2, 95.2]; random p95 13.9% |
| **H2** cred_all AUC ≥ 0.8 at C2k8, all + flipped | MET (1.000 / 1.000), **count-confounded** (C0r 0.997) | MET (1.000 / 1.000), count-confounded (C0r 1.000) | MET (1.000 / 1.000); **binding only** (neutral binding 0.972 ≥ 0.95) |
| **H3** graded: gov > newspaper > person > 0 (pair-level CIs) | **NOT MET** (newspaper − person +0.04 [−0.09, 0.17]) | **MET** (+2.15 / +0.64 / +3.45, all CIs > 0) | **MET** (+1.98 / +1.19 / +2.97) |
| **H4** source survives repetition at L\* (C1-only probe) | not met (fallback: mid-stack only); *exploratory* | **NOT MET** (flipped 0.23) | **NOT MET** (flipped 0.13) |
| **H5** source readable at final layer while the answer flips | met (0.79; 90% social); *exploratory* | **NOT MET** (0.17) | **NOT MET** (0.40) |
| amendment 10 (decision-balanced C1 probe): H4(a,b) / H5 | met / met; *exploratory* | not / not; *exploratory* | **not / not** (confirmatory) |

### Behaviour (P_low = probability of the low-credibility value; shift vs C1 in points)
| model | F1 C1 → C2 k2/k4/k8 | F1 C3k4 / C7k4 / C5 | F2 C1 / C2k4 / C7k4 | F3 C1 / C2k4 / C7k4 |
|---|---|---|---|---|
| Llama-8B | .434 → .643/.781/.890 | .905 / .866 / .340 (−27% of the C2k4 shift) | .278 / .496 / **.906** | .273 / .494 / **.870** |
| Qwen-7B | .243 → .555/.635/.518† | .761 / .916 / .399 (40%) | .142 / .568 / **.985** | .124 / .591 / **.955** |
| Gemma-12B | .418 → .629/.727/.766 | .832 / .830 / .286 (−43%) | .051 / .806 / .878 | .222 / .849 / .892 |
| Qwen-32B | .094 → .612/.717/— | .963 / .877 / .076 (−3%) | — | **.001 / .110 / .713** |
† Qwen-7B is strongly order-dependent at k=8 (copies after gov: .946; before: .091), as in the pilot.

**Main behavioural findings.** (1) Repetition flips every model in the table format, and the length control (C5) does not reproduce it (negative or ≤ 40% of the shift). (2) **Paraphrased copies of one social post (C7) are the strongest attack in realistic formats.** In news/RAG, exact copies stall near .5 for Llama and Qwen-7B while paraphrases reach .87–.99. At 32B in RAG, exact copies barely work (.11), but paraphrases flip 86% of pairs (.71; C7 − C2 = +60.3 [58.3, 62.2]), more than four independent users (.52). (3) Gemma is the exception: exact copies already work in news/RAG (.81/.85), with a clean length control (3–5%). (4) Merging all sources into one block (C4) has a small effect (Gemma +3, Llama +18 at k=4), as in Schuster et al.

### Representation (one paragraph)
Every model linearly encodes **which option is government-sourced** when there is no repetition (C1-only probe AUC ≥ 0.95 from mid-depth on). The pre-registered H2 probe (AUC 1.00) is **not** evidence of source identity under repetition: it is count-confounded (C0r ≥ 0.997), and for Gemma tag binding alone reaches 0.97. Trained on C1 only, the direction **follows the model's decision** under 8 copies in both confirmatory models (Qwen-7B, Gemma: flipped-item AUC 0.13–0.23 at L\*; C0r inverted to 0.00–0.32). This holds with decision-balanced training. Llama (exploratory) is the exception: source survives mid-stack (L14) and re-emerges at the output (L32, 0.79 while 90% of answers flip). This rests on a 0.002 selection margin, and the near-max-layer average fails. **Graded credibility is represented in Qwen-7B and Gemma (H3)** but not in Llama, and in Gemma it is much sharper than its behaviour (P(choose X) .57/.55/.48/.50). **Claim supported:** models represent graded source credibility, and under repetition that representation is overwritten by the decision. **Not supported:** "computed but overridden" in the sense of a preserved source signal.

### Heads (attribution patching ranked on SELECTION, verified by activation patching on 160 TEST items; clean = C1, corrupt = C2k4 unless stated)
| model / format | top-1 head (alone) | top-5 / top-10 / top-20 recovery | mean-ablate top-10 | random-10 p95 / max (20 draws) | sel↔test top-10 Jaccard |
|---|---|---|---|---|---|
| Llama F1 | L14H5 (35%); **L17H24** = Sun et al. head, rank 2, 44% | 84.5 / **91.9** / 96.0 | 78.5 | 30.9 / 42.4 | 0.82 |
| Qwen-7B F1 | **L18H4** (41%) | 82.7 / **95.4** / 96.7 | 56.8 | 19.4 / 20.3 | 0.67 |
| Gemma F1 | **L29H11** = Sun et al. head, rank 1 (56%) | 84.4 / **93.1** / 96.1 | 59.4 | 13.9 / 23.9 | 0.82 |
| Qwen-7B F3 (C2k4) | L18H4 (36%) | 61.2 / **76.6** / 90.0 | 32.4 | 14.2 / 15.8 | 0.33 |
| Qwen-7B F3 (C7k4, paraphrases) | L18H4 (19%) | 52.4 / **78.4** / 88.1 | 49.8 | 17.7 / 19.1 | 0.67 |
The repetition effect is concentrated in ~10 of 768–1,024 heads in every model, and the ranking transfers to new entities. Two of three models' top heads are the persuasion "decision heads" Sun et al. found independently. Attention: Llama L17H24 reads the answer-option tokens (0.26 on the low option vs 0.11 on the high one) and the chat tail, not the copies; mid-layer heads (e.g. L13H21) read the low-credibility blocks. **Amendment 5 (exact vs paraphrased copies, Qwen-7B):** top-10 Jaccard(F1/C2, F3/C7) = **0.33**. L18H4, L17H26, L18H27, L19H23 and L26H26 are shared; the paraphrase route adds earlier heads (L12H16, L14H12, L16H0/H1, L20H27). Same hub, partly different feeders. → Block B uses two head sets (RESULTS_B branch decision).

---
## Interim log (chronological; numbers as recorded when each job landed)

### Interim — amendment 6, C1-only probe, Llama-3.1-8B (CPU job 149443; `results/A/probe_c1only_Llama-3.1-8B-Instruct.json`)
At the **pre-registered layer L26** (test AUC; n = 3,200 per condition): C1 0.993 · C2k8 0.365 (flipped 0.292, n = 2,877; not flipped 0.781) · C0r 0.065 · C0 0.500 · C6 pending (A05).
**Against the stated predictions: neither column.** C1 ≥ 0.9 matches *source*, but C2k8 (0.37) fails both columns. C0r is **inverted** (0.07): the C1-trained direction scores the *repeated* value as "gov-like", which neither column predicted (count predicted ≥ 0.9). Per the rule written beforehand, this is reported as **mixed**.
A reading that fits every L26 cell, **not pre-registered**: the C1-trained direction at L26 behaves like the model's **decision**, not like source identity. Its AUC follows how often the model picks the singleton/gov option: C1 0.57, C2k8 0.11, C0r 0.08 (C6k4 0.92, so C6 should come out high). Flipped C2k8 items score 0.29 and non-flipped ones 0.78, which is FM C1 ("the probe tracks the predicted answer").
**Post hoc, by layer (L14 picked after seeing the table; do not cite as confirmatory):** at L14, C2k8 is 0.79 (flipped 0.80) and C0r is 0.52, close to the *source* column. From L18 on, the C2k8-flipped AUC falls (0.62 → 0.29 at L26) and C0r inverts (0.17 → 0.07). Candidate picture: source identity is linearly present mid-stack, and late layers carry the decision, after repetition has overridden it. This is the same "computed but overridden" story as the pilot, but **located by layer**, not shown by one ceiling AUC.
**Consequence for H2 (Llama):** the pre-registered `cred_all` AUC of 1.000 at L26 cannot be read as source identity. The C1-trained direction at the same layer is 0.29 on flipped items, so `cred_all` most likely separates the options by count (gov = singleton in every k ≥ 2 training condition). Per amendment 6, the paper reports H2 as **confounded with count** at L26.

### Interim — amendment 7 / H4, Llama-3.1-8B (EXPLORATORY: the per-layer table was seen before the rule; CPU job 150067; `results/A/probe_c1only_Llama-3.1-8B-Instruct.json`, the previous fit is kept as `…amend6.json`)
- The rule picks **L\* = L14** (val AUC on 173 decision-disagreeing selection C1 items: 0.998). The margin is tiny: L16–L32 all score 0.986–0.996, so L\* is fragile under this criterion.
- (a) C2k8 flipped AUC at L14 = **0.799 [0.788, 0.810]** → met. (b) C0r at L14 = **0.519** → met. (c) C2k8 flipped at the final layer L32 = **0.790 [0.766, 0.815]** → **not met**.
- **H4 not met; fallback per the rule: "source identity decodable mid-stack" only** (exploratory for Llama).
- Descriptive, not a criterion: the flipped-item AUC falls *below* 0.5 at L22–L28 (minimum 0.29 at L26) and recovers to 0.63–0.79 at L30–L32. C0r stays inverted from L18 to L32 (0.07–0.30). So the override shows up in the late-middle layers, not at the last one. The pre-registered (c) tests only the final layer, and redefining it now would be post hoc.
- **Amendment 8 robustness (Llama, exploratory):** near-max set S = {L14, L16, L22, L24, L26, L28} (within 0.005 of 0.998). Mean C2k8-flipped AUC over S = **0.517** (< 0.75); mean C0r over S = **0.248** (outside 0.40–0.60). **Neither (a) nor (b) survives averaging.** The selection criterion cannot tell L14 apart from L22–L28, where the direction tracks the decision, so the L14 pass depends on a 0.002-AUC margin.
- **Amendment 8 description:** at the final layer L32, the flipped-item AUC is 0.790 [0.766, 0.815], while Llama picks the social value on **90%** of C2k8 test items. A linear readout of source identity is present at the output even though the answer goes the other way.

### Interim — amendment 9 / H5, Llama-3.1-8B (EXPLORATORY; CPU job 150687)
- Final layer L32: flipped C2k8 AUC **0.790 [0.766, 0.815]** (n = 2,877); Llama picks the social value on **90%** of C2k8 test items (n = 3,200). **H5 met (exploratory).**
- Last 3 stored layers (robustness): L28 **0.490 [0.454, 0.528]** (CI includes 0.5), L30 **0.626 [0.593, 0.657]**, L32 **0.790 [0.766, 0.815]**. **Only L32 passes the 0.70 bar.** The source readout re-emerges in the last two stored layers, after the decision-tracking dip (L22–L28). This is not a plateau carried through from mid-stack.

### Interim — H4 + H5, Qwen2.5-7B (CONFIRMATORY; CPU job 151289; `results/A/probe_c1only_Qwen2.5-7B-Instruct.json`)
- L\* = **L18** (val AUC 0.996 on 87 decision-disagreeing selection C1 items; S = {L18} only, so L\* is unambiguous).
- **H4 NOT MET.** (a) C2k8 flipped AUC at L18 = **0.227 [0.203, 0.253]**, below chance: at the layer that best separates source from decision on C1, the direction already follows the decision under repetition. (b) C0r at L18 = **0.323**, outside the band. (c) Final layer L28 flipped = **0.167 [0.145, 0.193]** (met, but moot).
- **H5 NOT MET.** Final-layer flipped AUC is 0.167, so the readout follows the decision, while Qwen picks the social value on 52% of C2k8 items. Last 3 stored layers: L24 0.440 [0.410, 0.471], L26 0.216 [0.190, 0.243], L28 0.167 [0.143, 0.191].
- Descriptive: no layer reaches 0.75 on flipped items (max 0.67 at L22). C1 AUC is ≥ 0.95 from L16 on, so source identity is linearly present when there is no repetition; with 8 copies the same direction tracks the decision at every layer from L18. In Qwen-7B there is **no evidence that source identity survives repetition** in this probe.

### Interim — Qwen2.5-32B-Instruct, behaviour only (job 144915, 1 h 22 min of the 2 h cap; all 7 priority conditions × F1+F3 completed; test split, 400 pairs; A+B mass 1.000)
| cond | F1 table P_low (shift vs C1) | F3 retrieved docs P_low (shift vs C1) |
|---|---|---|
| C1 | 0.094 | **0.001** |
| C2k2 exact copies | 0.612 (+51.8) | 0.088 (+8.7) |
| C2k4 exact copies | 0.717 (+62.3) | **0.110 (+11.0)**, 3% of pairs flip |
| C3k4 independent | 0.963 (+86.9) | 0.519 (+51.8) |
| C7k4 paraphrased copies | 0.877 (+78.3) | **0.713 (+71.3), 86% of pairs flip** |
| C5 length control | 0.076 (−1.8; −3% of the C2k4 shift) | 0.002 (+0.2; 2%) |
**Reading:** at 32B in the RAG format, **exact copies barely move the model** (C2k4 0.11), but **reworded copies of the same single social post flip 86% of pairs** (C7k4 − C2k4 = **+60.3 [58.3, 62.2]**). Paraphrased copies even beat four genuinely independent users (+19.4 [17.3, 21.6]). The length control is clean in both formats. Scale buys robustness to verbatim repetition, not to reworded repetition.

### Interim — Gemma-3-12B main results (jobs 144911 + 144912 + 144913; `results/A/summary_A.md`)
- **Behaviour, F1 test:** C1 0.418 · C2k2/k4/k8 0.629 / 0.727 / 0.766 (+21.0 / +30.9 / +34.8) · C3k4 0.832 · C7k4 0.830 · C5 0.286 (−13.2, the opposite direction) · C4k4 0.449 (+3.1) · A+B mass 0.999.
- **F2/F3 (news / documents):** unlike Llama and Qwen-7B, **exact copies work strongly** in Gemma: C2k4 **0.806 / 0.849** (+75.5 / +62.7 vs C1 0.051 / 0.222). Paraphrased copies add only +7.2 / +4.3 (C7k4 0.878 / 0.892). The length control is clean: C5 = **3% / 5%** of the C2k4 shift.
- **H2 met** (cred_all at L26: C2k8 1.000, flipped 1.000, n = 2,450), with the same count confound (C0r 0.998).
- **H3 MET** (confirmatory): gov − newspaper **+1.98 [1.82, 2.16]**, newspaper − person **+1.19 [1.07, 1.30]**, person − social **+2.97 [2.83, 3.12]**; lexical baselines 0.5 by construction (shallow L2 0.54, L4 0.84). Descriptive Spearman over the three non-social levels: 0.87 [0.84, 0.89]. Neutral tag: 0.53 of the social→gov span.
- **Behaviour is only weakly graded:** P(choose X) gov 0.571, newspaper 0.553, person 0.483, social 0.499, neutral 0.477. The representation grades sources much more sharply than the choices do.
- **Neutral Blue/Green binding: max 0.972 at L24 (≥ 0.95).** Per amendment 3, Gemma's gov-vs-social AUC (H2) is **evidence of binding only**, and its credibility claim rests **solely on H3**, which is met.

### Interim — H4, H5, amendment 10 for Gemma-3-12B (CONFIRMATORY; CPU jobs 160278 unbalanced, 160279 balanced; `results/A/probe_c1*_gemma-3-12b-it.json`)
- **C1-only probe:** L\* = **L30** (162 disagreeing val items; the near-max set S spans L22–L42, 11 layers). **H4 NOT MET:** (a) C2k8 flipped at L30 = **0.133 [0.118, 0.151]**; (b) C0r = **0.002**; (c) final L48 = 0.403 [0.382, 0.426] (met, moot). Robustness over S: flipped 0.352, C0r 0.063. **H5 NOT MET:** final-layer flipped AUC **0.403 [0.382, 0.424]**, while Gemma picks social on 77% of C2k8 items; last 3 stored layers 0.315 / 0.383 / 0.403.
- **Balanced probe** (n_cell = 200, 800 training items; 659 disagreeing items, so the stop rule did not trigger): L\* = **L36**. H4(a) **0.220 [0.203, 0.236]**, H4(b) **0.011**; H5 final L48 **0.357 [0.334, 0.381]** → **all not met**.
- Descriptive: C1 AUC is ≥ 0.96 from L20 on, so source identity is linearly present without repetition. Under 8 copies, the same direction scores below 0.5 on flipped items at almost every layer from L20 (max 0.66 at L16, before C1 is decodable), and C0r collapses to ≈ 0 from L28: the direction scores the repeated value as "gov-like" almost perfectly. Gemma's C1-trained direction behaves as a **decision/preference direction under repetition**, more extremely than Qwen-7B.
- **Across the confirmatory models (Qwen-7B, Gemma): H4 and H5 fail under both probes.** The only support for "source computed but overridden" or "persists at the output" is Llama, which is exploratory. Per the fallback rules, the paper cannot claim that source identity survives repetition. What it can claim: source identity is linearly decodable when there is no repetition (all three models, C1 AUC ≥ 0.95 mid-to-late), and under repetition this direction tracks the model's decision.

### Interim — C6 (gov repeated) on the C1-only probes, Llama + Qwen-7B (CPU jobs 153318 unbalanced, 153319 balanced; H4/H5 numbers reproduced exactly)
| probe | model | C6k4 / C6k8 at L\* | C6k4 / C6k8 at final layer |
|---|---|---|---|
| C1-only | Llama (L14 / L32) | 0.801 / 0.731 | 0.834 / 0.777 |
| C1-only | Qwen-7B (L18 / L28) | 0.946 / 0.973 | 0.964 / 0.906 |
| balanced | Llama (L14 / L32) | 0.694 / 0.628 | 0.757 / 0.706 |
| balanced | Qwen-7B (L22 / L28) | 0.871 / 0.782 | 0.975 / 0.908 |
| C1-only | Gemma (L30 / L48) | 1.000 / 1.000 | 0.988 / 0.947 |
| balanced | Gemma (L36 / L48) | 0.988 / 0.983 | 0.981 / 0.948 |

(Gemma rows: CPU jobs 161239/161240; H4/H5 numbers reproduced exactly.) Against amendment 6's table: the **count column (≤ 0.1, inverted) is rejected for all three models**; no C6 cell is below 0.5. Qwen matches the source/decision column (≥ 0.9) at most cells; Llama sits at 0.63–0.83, below the 0.9 the source column predicts. As pre-stated, C6 is not a discriminator between source and decision (both point to gov; the models pick gov on 92% (Llama) of C6k4 items).

### Interim — Qwen2.5-7B main results (jobs 144909 + 144910; `results/A/summary_A.md`)
- **Behaviour, F1 test** (P_low; shift vs C1): C1 0.243 · C2k2 0.555 (+31.2) · C2k4 0.635 (+39.2) · C2k8 0.518 (+27.6; **order-split HF 0.946 / LF 0.091**, the same position dependence as in the pilot) · C3k4 0.761 · C7k4 **0.916** · C5 0.399 (C5 = **40% [36, 44]** of the C2k4 shift, under the 50% WEAK-GO cap from the pilot) · A+B mass 0.81.
- **F2/F3:** paraphrased copies dominate again: C7k4 **0.985 (F2) / 0.955 (F3)**, vs exact copies C2k4 0.568 / 0.591. C5 explains 40% / 46% of the C2k4 shift in F2 / F3.
- **H2 met** (cred_all at L24: C2k8 1.000, flipped 1.000, n = 1,648), with the **same count confound as Llama**: C0r control = **1.000**.
- **H3 MET** (Qwen-7B, confirmatory): gov − newspaper **+2.15 [1.98, 2.34]**, newspaper − person **+0.64 [0.46, 0.83]**, person − social **+3.39 [3.22, 3.55]**; lexical baselines 0.5 by construction (shallow reference L2 0.78, L4 0.90; probe at L24 1.00). Behaviour follows the same order: P(choose X) gov 0.78 > newspaper 0.69 > person 0.56 > social 0.51. The exploratory neutral tag sits at **0.51** of the social→gov span (P(choose) 0.61).
- Neutral Blue/Green binding: max **0.903** (< 0.95), so the gov-vs-social AUC is not reducible to tag binding under amendment 3.
- **Reading:** in Qwen-7B the gov-vs-social direction *grades* sources (H3), but the C1-only probe shows no source signal surviving 8 copies (H4/H5 fail). Graded credibility is represented; under repetition the late layers carry the decision.

### Interim — amendment 10, decision-balanced C1 probe, Llama + Qwen-7B (EXPLORATORY; CPU job 151725; `results/A/probe_c1bal_*.json`)
- *Cell imbalance before balancing* (C1 selection, label y = gov is option A × model agrees/disagrees): Llama y1-disagree 541 / y0-disagree 168 / y0-agree 632 / y1-agree 259; Qwen 259 / 98 / 702 / 541. Disagreement is concentrated in one label, so an unbalanced C1 probe can partly learn "label = answer letter". The stop rule did not trigger (709 and 357 disagreeing items ≥ 200). n_cell = 168 (Llama, 672 training items) and 98 (Qwen, 392). L\* is chosen on only 87 / 47 disagreeing val items, so it is noisier than in H4.
- **Llama:** L\* = L14. H4(a) **0.804 [0.794, 0.815]** met; H4(b) **0.584** met. H5: final L32 **0.783 [0.760, 0.810]**, social chosen on 90% → **met**. Last 3: L28 0.636, L30 0.637, L32 0.783. Same picture as the unbalanced probe; L28 rises from 0.49 to 0.64, but only L32 clears 0.70. The dip below 0.5 remains at L22–L26 (0.36 at L26).
- **Qwen-7B:** L\* = L22. H4(a) **0.579 [0.541, 0.615]** not met; H4(b) **0.291** not met. H5: final L28 **0.171 [0.143, 0.201]** → **not met**. At L22 the flipped AUC is 0.58 (unbalanced probe: 0.67), and no layer clears 0.75; L18–L20 and L24–L28 stay well below 0.5. **Qwen's failure is not an artefact of decision-correlated training labels.**
