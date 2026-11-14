# RESULTS_F — Block F: controls requested by an external reviewer
**SPRINT WORK (14–15 Nov 2026).** Budget ≤ 3 A100-h, A100 only. Failure modes: `FAILURE_MODES.md` section H. Test split only for reported numbers. Every criterion below was written before its data existed.

## Pre-registration (written 2026-11-14 ~15:30 Cairo; no Block F data, code or job exists)

### F1 — repetition or wording? (behaviour only)
**New conditions** (same templates, users, block positions and 8-way counterbalancing as C7 k=4; built with v2's renderers; generator asserts byte-identity of rebuilt C1/C2k4/C3k4/C7k4 with `prompts_v2.jsonl`):
- **C7 k=1, template t ∈ {0,1,2,3}:** gov block + one social block from users[0] rendered with paraphrase template t. Template 0 is the canonical block, so C7k1_t0 is byte-identical to C1; its values are C1's (FM H1).
- **C8 k=4:** four *different* users (users[0..3]), block i rendered with template i, the same as C7 k=4 except for the user names.
- **C5p:** one social block (template 0) + three neutral filler blocks about unrelated fictional entities, rendered with templates 1–3 (to match C7 k=4's length), placed like C5. Mean tokens C5p vs C7k4 are reported (flagged if C5p is > 10% shorter). *Design fix before any data (~15:45 Cairo):* with 3-sentence fillers, F3 C5p was 13% shorter than C7k4 (filler blocks have no source line), so F3 filler blocks carry 4 attribute sentences. Character lengths are now C5p 1,083 vs C7k4 1,098 (F3) and 1,216 vs 1,313 (F1, 7%).

Formats/models: **F3 (retrieved documents), test split, all 4 models** (Qwen-32B: F3 only, as its scope in Block A). F1 tables for Llama-8B, Qwen-7B and Gemma-12B if the budget allows (lowest priority; run last).

**Quantities** (per pair = mean over its 8 variants; 95% CI = cluster bootstrap over pairs, 2,000 resamples, seed 0):
- W = P_low(C7k1, mean over the 4 templates) − P_low(C1). Sensitivity: the mean over templates 1–3.
- R_p = P_low(C7k4) − P_low(C7k1 mean); R_e = P_low(C2k4) − P_low(C1); D = R_p − R_e.
- L_p = P_low(C5p) − P_low(C1) (the length effect matched to C7 k=4).

**F1-main (confirmatory; evaluated separately for Qwen-32B F3 and Qwen-7B F3):** "paraphrasing defeats the duplicate discount" is **supported** if (a) D > 0 with the 95% CI lower bound > 0, **and** (b) R_p > L_p (point estimates; the CI of R_p − L_p is also reported). It **fails** if D's CI includes 0 or D < 0, which would mean the C7 jump is a one-shot wording effect, or if (b) fails, which would mean it's a length effect. *Can it fail under counterbalancing?* Yes: all quantities are paired differences of counterbalanced pair means; letter/position effects cancel, and nothing forces D > 0 (e.g. if C7k1 already carries the jump, R_p ≈ 0 < R_e).
**Exploratory:** the same quantities for Llama-8B and Gemma-12B (F3) and for the three models in F1; sensitivity with templates 1–3.

**F1-2×2 (exploratory):** at k = 4, source identity (same user / different users) × wording (exact / paraphrased): C2 same/exact, C3 different/exact, C7 same/paraphrased, C8 different/paraphrased. Report all four cells, both main effects and the interaction I = (C8 − C7) − (C3 − C2), with CIs.

### F2 — specific heads or generic answer bottleneck? (F1 tables, test split; Llama-8B, Qwen-7B, Gemma-12B)
Fixed head sets from Block A (`results/A/patch_F1_<model>.json`): the top-10 by selection AtP, and the same 20 random 10-head sets (`hsets.rand10_00..19`). No re-ranking (FM H7). Activation patching of the clean head outputs into the corrupt run at the shared question suffix (Block A method), for four clean → corrupt contrasts:
| contrast | expected sign of LD_corrupt − LD_clean |
|---|---|
| C1 → C2k4 (reference) | + |
| C0 → C1 (credibility, no repetition) | − |
| C0 → C0r k=4 (repetition, no labels) | + |
| C1 → C6 k=4 (gov repeated) | − |
Items: test items whose behaviour-run gap (LD_corrupt − LD_clean) is > 1 logit in the expected direction; up to 160 per contrast (shuffled, seed 0, first 160). Recovery = (mean LD_corrupt − mean LD_patched) / (mean LD_corrupt − mean LD_clean) × 100, with a 95% cluster-bootstrap CI over pairs; random p95 over the 20 sets. Also reported per contrast: n eligible items, mean gap, flip rate (share of eligible items where the preferred letter changes between clean and corrupt), and mean suffix length.

**Interpretations (confirmatory, per model; each evaluated independently, so more than one may hold):**
- **"specific to the credibility–repetition conflict":** recovery(C1→C2k4) − max(recovery of the other three) > 20 points.
- **"shared credibility+repetition pathway":** recovery(C0→C1) ≥ 40% **and** > its random p95, **and** recovery(C0→C0r) ≥ 40% **and** > its random p95.
- **"generic answer bottleneck":** all four contrasts (including C1→C6) recover ≥ 40%, and max − min ≤ 20 points.
*Can each fail?* Yes. "Specific" fails if any other contrast comes within 20 points. "Shared" fails if either contrast is < 40% or ≤ random. "Generic" fails if any contrast is < 40% or the spread is > 20 points. The counterbalanced letter assignment means a letter-bias head cannot produce consistent recovery across contrasts with opposite signs.

### F3 — one readout, one layer (CPU, saved states only)
- **Graded set on the C1-only direction:** the graded-set hidden states were **not saved** (Block A projected them on the fly), so this is **not run**. The GPU cost estimate is reported (FM H8).
- **C1-only probe at L\* and at the final layer, C2 k=8 test items:** AUC for "gov is option A" on all, **non-flipped** and flipped items (already in `probe_c1only_*.json`), plus a **chosen-answer AUC**: the same probe score against the label "the model's preferred option is A" (from behaviour). Exploratory. *Reading:* source-tracking = high gov-label AUC on both flipped and non-flipped items; answer-tracking = high chosen-answer AUC and gov-label AUC < 0.5 on flipped items.
- **Positive controls (from B2 outputs):** at which layer the residual patch restores the answer (L28 −0.44; L18 +0.076).

---
## Results
Run 2026-11-14, jobs 188697 (FJ1), 188701 (FJ2), 188704 (FJ3); all on A100 dev; the t3 copies 188699/188702/188705 were cancelled while pending. Full tables: `results/F/summary_F.md`; F2 CIs below are from the final `analyze_f.py` run (all Block F data present; bootstrap draws shift by ≤ 0.4 points from the interim run); numbers: `numbers_F.json` (273 entries) and `numbers_F_posthoc.json`. 95% CIs: cluster bootstrap over pairs (2,000, seed 0). Test split, 400 pairs per model and format.

### F1-main (confirmatory) — **SUPPORTED in both models**
| model (F3) | W | R_p | R_e | **D = R_p − R_e** | L_p | verdict |
|---|---|---|---|---|---|---|
| Qwen-32B | +1.2 [+0.9, +1.5] | +70.0 [+68.1, +72.0] | +11.0 [+9.5, +12.3] | **+59.1 [+57.1, +61.1]** | +0.3 [+0.1, +0.5] | supported |
| Qwen-7B | +6.2 [+5.3, +7.2] | +76.9 [+75.2, +78.5] | +46.7 [+44.5, +49.0] | **+30.2 [+28.3, +32.2]** | +19.3 [+17.6, +20.9] | supported |

C5p/C7k4 token ratio 0.984 in both models. A single paraphrase barely moves P_low (W ≤ 6 points), so the C7 k=4 jump is not a one-shot wording effect, and length-matched filler does not produce it either (L_p ≪ R_p).

### F1 exploratory (other models and formats)
| model | format | W | R_p | R_e | D [CI] | L_p | C5p/C7k4 tokens | I_2x2 [CI] |
|---|---|---|---|---|---|---|---|---|
| Llama-8B | F3 | −0.9 [−2.0, +0.2] | +60.6 | +22.1 | +38.5 [+36.9, +40.2] | +14.5 | 0.991 | −10.5 [−11.6, −9.4] |
| Gemma-12B | F3 | +1.6 [+0.6, +2.7] | +65.3 | +62.7 | **+2.7 [+0.9, +4.4]** | +8.7 | 0.986 | −3.0 [−3.8, −2.1] |
| Qwen-7B | F1 | −12.8 [−13.6, −11.8] | +80.1 | +39.2 | +40.9 [+38.8, +43.0] | +7.6 | 0.920 | −11.9 [−13.4, −10.4] |
| Llama-8B | F1 | −21.9 [−22.9, −20.9] | +65.1 | +34.7 | +30.4 [+29.1, +31.7] | −4.2 | 0.930 | −9.3 [−10.1, −8.5] |
| Gemma-12B | F1 | −26.0 [−26.9, −25.2] | +67.2 | +30.9 | +36.3 [+34.8, +37.9] | +12.1 | 0.929 | −4.2 [−5.2, −3.2] |
| Qwen-32B | F3 (confirmatory, above) | | | | | | | −23.5 [−25.9, −21.1] |

- **Gemma F3 is the weak case.** Exact repetition alone already moves Gemma by 63 points, so there is little duplicate discount for paraphrasing to defeat (D +2.7; −4.4 against its strongest template, post hoc).
- **F1 tables:** one paraphrase *lowers* P_low (W −13 to −26), which inflates R_p. Baseline-free check (descriptive, point estimates): P_low(C7k4) − P_low(C1) vs P_low(C2k4) − P_low(C1) = Llama +43.2 vs +34.7, Qwen-7B +67.3 vs +39.2, Gemma +41.2 vs +30.9. Paraphrased copies still beat exact copies, by less than D suggests. F1 C5p is 7–8% shorter than C7k4 (under the 10% flag).
- **2×2 (source identity × wording at k=4):** the interaction is negative in every model and format (−3 to −24): paraphrasing and multiple users are partly redundant, so each matters less when the other is present. Wording main effect vs source main effect: wording dominates in Qwen-32B F3 (+48.6 vs +29.1), Qwen-7B F3 (+33.3 vs +3.1), Llama F3 (+32.4 vs +14.0) and Qwen-7B F1 (+22.2 vs +6.7); source is larger in Llama F1 (+3.8 vs +7.7) and Gemma F3 (+2.8 vs +4.2); they are equal in Gemma F1 (+8.2 vs +8.4).

### F2 — pre-registered interpretation: **not specific in any model**
Fixed top-10 heads from Block A, recovery as the ratio of means, 160 items per contrast (gap > 1).

| model | C1→C2k4 | C0→C1 | C0→C0rk4 | C1→C6k4 | specific | shared | generic |
|---|---|---|---|---|---|---|---|
| Llama-8B | 92.1 [90.5, 93.7] | 92.6 [91.3, 93.9] | 90.7 [89.4, 91.9] | 92.8 [91.5, 94.0] | ✗ | ✓ | ✓ |
| Qwen-7B | 95.4 [92.5, 98.6] | 72.3 [69.4, 75.2] | 90.2 [86.5, 94.3] | 96.6 [92.3, 101.5] | ✗ | ✓ | ✗ (spread 24.3 > 20) |
| Gemma-12B | 93.2 [91.4, 95.3] | 92.2 [89.3, 95.1] | 90.7 [87.8, 93.9] | 93.7 [91.7, 95.7] | ✗ | ✓ | ✓ |

Random 10-head p95 ≤ 34% for every contrast. Mean gaps (|logit|): Llama 3.2–6.3, Qwen-7B 8.0–15.1, Gemma 3.4–6.6; flip rates 23–59% (per contrast in `summary_F.md`). **Reading:** the top-10 heads carry the final answer whatever moved it, including government repetition (C1→C6). Llama and Gemma meet "generic answer bottleneck"; Qwen-7B meets only "shared" (C0→C1 lower at 72%). Calling them "repetition heads" is not supported; "answer-carrying heads through which the repetition flip passes" is.

### F3 — one readout, one layer (exploratory)
- **Graded set on the C1-only direction: not run.** The graded-set hidden states were never saved (projected on the fly in Block A). Estimated cost to recompute: ~0.2 A100-h (behaviour pass with hidden states on the graded set for Qwen-7B and Gemma).
- **C1-only probe at L\*, C2 k=8 test items (n = 3,200 per model):**

| model, layer | gov-label AUC: all / flipped / non-flipped | chosen-answer AUC: all / flipped | reading |
|---|---|---|---|
| Llama-8B L14 | 0.79 / 0.80 / 0.76 | 0.25 / 0.20 | **source-tracking** (still encodes the gov source after the flip) |
| Qwen-7B L18 | 0.61 / 0.23 / 0.97 | 0.87 / 0.77 | **answer-tracking** |
| Gemma-12B L30 | 0.27 / 0.13 / 0.59 | 0.84 / 0.87 | **answer-tracking** |

Final layers: Llama L32 0.81 / 0.28, Qwen L28 0.62 / 0.92, Gemma L48 0.49 / 0.61 (gov-label / chosen, all items). On non-flipped items the two labels coincide by construction.
- **Positive controls (B2, Qwen-7B, C1 residual swap at the final position):** works at **L28** (ΔP_low −0.438); does **not** work at **L18** (+0.076). **Steering at L18 has no working positive control.**

### Budget and deviations
- **A100-h:** FJ1 0:35:53 + FJ2 1:15:34 + FJ3 0:30:00 = **2.36 A100-h** (cap 3). CPU: 188716, ~2 core-h. No H100.
- **Deviations:** (1) F3 C5p filler blocks lengthened to 4 sentences, before any data. (2) C7 k=1 template 0 is byte-identical to C1, so the 4-template mean includes C1. (3) The F3 graded set was not run (states not saved); cost estimated instead. (4) Per-template post-hoc section added at Amr's request after the Qwen-32B data. (5) Another session's `rlhead` jobs ran on the same allocation while FJ2 ran, so there was more than one GPU job at a time.


### F1 per-template detail — **POST HOC** (requested by Amr on 2026-11-14 after the Qwen-32B F1 data had been seen; not pre-registered; **the pre-registered F1-main verdicts are unchanged**)
C7 k=1 P_low per paraphrase template (t0 is byte-identical to C1), the mean over templates 1–3 only, and D = R_p − R_e recomputed with R_p measured against (a) that mean and (b) the single strongest template (highest C7 k=1 P_low among t1–t3). 95% CIs: cluster bootstrap over pairs (2,000, seed 0). Script: `scripts/posthoc_f1_templates.py`; numbers: `numbers_F_posthoc.json`. Rows are added as each model's data lands.

| model | format | C7k1 t0 (=C1) | t1 | t2 | t3 | mean t1–3 | strongest | D vs mean t1–3 [CI] | D vs strongest template [CI] |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-32B | F3 | 0.001 | 0.006 | 0.033 | 0.011 | 0.017 | t2 | +58.7 [+56.7, +60.7] | +57.0 [+54.8, +59.2] |
| Qwen-7B | F3 | 0.124 | 0.153 | 0.279 | 0.189 | 0.207 | t2 | +28.1 [+25.9, +30.3] | +20.9 [+18.4, +23.6] |
| Llama-8B | F3 | 0.273 | 0.251 | 0.255 | 0.278 | 0.261 | t3 | +38.8 [+37.0, +40.7] | +37.2 [+35.0, +39.4] |
| Gemma-12B | F3 | 0.222 | 0.308 | 0.207 | 0.214 | 0.243 | t1 | +2.1 [+0.1, +4.1] | -4.4 [-6.6, -2.2] |
| Qwen-7B | F1 | 0.243 | 0.112 | 0.051 | 0.054 | 0.072 | t1 | +45.1 [+42.7, +47.4] | +41.2 [+38.7, +43.6] |
| Llama-8B | F1 | 0.434 | 0.191 | 0.061 | 0.174 | 0.142 | t1 | +37.7 [+36.1, +39.3] | +32.8 [+31.0, +34.6] |
| Gemma-12B | F1 | 0.418 | 0.111 | 0.029 | 0.074 | 0.071 | t1 | +45.0 [+43.3, +46.7] | +41.0 [+38.9, +43.0] |

*Post-hoc reading:* in F3 the D values against the mean of t1–3 match the pre-registered D, except that Gemma F3 against its strongest template is negative (−4.4). In F1 tables a single paraphrase **lowers** P_low below C1 for every template (W = −13 to −26), so R_p is measured from a low baseline. A baseline-free comparison is in the Results section.
