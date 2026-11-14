# RESULTS_D — Block D: trained LoRA baseline on Qwen2.5-7B-Instruct
**SPRINT WORK (14 Nov 2026).** Budget ≤ 4 A100-h. Failure modes: `FAILURE_MODES.md` section G. Everything in this pre-registration was written before any Block D training data, training run or test evaluation existed.

## Pre-registration (written 2026-11-14 ~11:35 Cairo)

### Recipe (following Schuster et al.'s `finetuning/train.py`, read from their public repo)
- **LoRA:** r = 16, **α = 16** (their code; the prompt's α = 32 applied only "unless their code says otherwise"), dropout 0.05, targets q/k/v/o/gate/up/down projections, bf16, gradient checkpointing.
- **Objective (theirs):** L = λ1·symKL(f(C1), f′(C1)) + λ2·symKL(f(C1), f′(repeated)). f = frozen base (adapter disabled), f′ = adapted model; distributions = 2-way log-softmax over the "A"/"B" answer tokens at the final position. The repeated prompt is the C2/C7 version of the *same* entity and variant as the C1 prompt. λ1 = 0.6208, λ2 = 1 − λ1. After the main training, 300 optimizer steps of a refinement phase with the repetition term only (λ1 = 0, λ2 = 1).
- **Optimisation (theirs):** AdamW, weight decay 0.01, batch 1, grad accumulation 8, linear schedule with 10% warmup, max grad norm 1.0, 2 epochs.
- **Tuning (prompt rule, on training entities only):** lr ∈ {1e-4, 2e-4}. Checkpoints after epoch 1, after epoch 2, and after refinement. The selected (lr, checkpoint) is the one with the lowest validation loss (λ1·t1 + λ2·t2, their eval loss) on **held-out training entities**. Selection and test are never used for tuning.
- **Training data (prompt spec):** new fictional entities, new seed, place pools disjoint from v2, every v2 and pilot word excluded, with zero name/user/place overlap asserted. **250 training entities + 50 validation entities**; 6 items per entity sampled uniformly from {C2 k=2, C2 k=4, C7 k=2, C7 k=4} × {F1, F3} × 8 variants. That gives **1,500 training / 300 validation items**, each paired with the matched C1 prompt (same entity, format and variant). Not trained on: C0, C0r, C3, C4, C5, C6, F2.

### Evaluation (TEST split, the same scoring and metrics as B3)
F1, F2, F3 × {C0, C1, C2 k=2/4 (k=8 in F1), C3 k=4, C5, C7 k=4}, plus held-out side-effect conditions **C6 k=2/4 and C0r** (F1). Metrics as B3: bias reduction (C2k4, C7k4), source preference kept (C0−C1), corroboration kept (C3k4), all per format with 95% CIs over pairs; mean max-option probability and binary entropy per condition; A+B mass. Capability: the same 500 MMLU items and 200 WikiText paragraphs. The base values come from B3's cap file (identical items, identical base model).

### Criteria
- **D-S1 (confirmatory):** F1 C2k4 bias reduction **≥ 50%**, F1 preference kept **≥ 70%**, MMLU drop **≤ 2 pts**, perplexity increase **≤ 5%**.
- **D-S2 (confirmatory):** F3 C7k4 bias reduction **≥ 50%**, F3 preference kept **≥ 70%**, same capability bars.
- **D-sel (reported per format):** corroboration kept (C3k4) **≥ 50%** → "selective".
- **D-cmp (confirmatory comparison, F3/C7):** let SP_L = LoRA's F3 preference kept. For each B3 test curve (union, H_F3C7, Feeders; γ ∈ {0, 0.25, 0.5, 0.75} plus the base point γ = 1 → (0, 1)), interpolate the F3 C7k4 bias reduction linearly at SP_L. If SP_L is below a curve's minimum preference, take that curve's γ = 0 point. BR_B3\* = the maximum over the three curves. **LoRA is better** if the lower bound of its F3 C7k4 bias-reduction 95% CI > BR_B3\*; **worse** if the upper bound < BR_B3\*; otherwise **no difference**.
- **D-shortcut (exploratory, FM G3):** flag "minority shortcut" if C6 k=4 (F1) P_low under the LoRA is ≥ 0.20 above base.
- *Can each fail under the counterbalancing?* Yes. A flattening LoRA (every P_low → 0.5) gets a large bias reduction but preference kept ≈ 0, so it fails S1/S2. A letter or position artefact cancels in the pair means. D-cmp can come out better, worse or equal. Nothing forces a pass.

### Deviations from Schuster et al. (stated before running)
In-distribution training data (gov vs social, k = 2/4, exact and paraphrased copies, F1 + F3) instead of their other source pairs with one repeat. One seed instead of seven. Qwen-7B instead of Gemma-3-4B. A small lr grid with checkpoint selection on held-out training entities instead of their fixed best hyper-parameters (theirs are used as the defaults). Refinement implemented as one pass over min(300×8, N) items with a fresh linear decay (their code's step counter appears to end refinement after one update; we implement the documented intent). Evaluation uses the adapter merged into the base weights (`merge_and_unload`).

---
## Results

### Training (jobs 182217 lr 2e-4, 182219 lr 1e-4; `results/D/lr*/train_log.json`)
Validation loss on held-out training entities (base 2.334): lr 2e-4: e1 0.482, e2 0.342, refine 0.453. lr 1e-4: e1 0.290, **e2 0.228**, refine 0.296. **Chosen by the pre-registered rule: lr 1e-4, end of epoch 2.** Refinement raised the validation loss at both lrs (it trades C1 preservation for the repetition term).

### Test (job 182221; `results/D/summary_D.md`, `numbers_D.json`, `figs/figD_tradeoff_F3.(pdf|png)`)
| format | bias reduction C2k4 | bias reduction C7k4 | source preference kept | corroboration kept (C3k4) |
|---|---|---|---|---|
| F1 tables | 94% [93, 96] | 99% [98, 100] | 115% [112, 119] | 11% [10, 13] |
| F3 documents | 103% [102, 105] | 95% [94, 96] | 102% [100, 104] | 2% [1, 4] |
| F2 news (never trained on) | 103% [100, 105] | 81% [79, 82] | 97% [95, 99] | -10% [-13, -7] |
*Capability:* MMLU -0.6 pts [-1.6, +0.2]; WikiText perplexity -0.5%.
*Side effects (F1, P_low base -> LoRA):* C3k4 0.761 -> 0.262; C6k4 (gov repeated) 0.030 -> 0.156; C0r (no labels, repeated) 0.974 -> 0.715; C0 stays 0.500. A+B answer mass rises (F1 C1 0.749 -> 0.879).

### Verdicts
- **D-S1: MET** (confirmatory). **D-S2: MET** (confirmatory): the paraphrase bias in documents drops by 95% with source preference fully kept, and it transfers to the untrained F2 format (81%).
- **D-sel: NOT selective** in any format (corroboration kept 11% / 2% / -10%). Four *independent* users now move the model almost no more than one: the LoRA learned "discount every extra low-credibility block", not "discount copies". The head-damping fix for exact copies kept 60% (F1). Training only on copies and paraphrases (as specified) never shows the model a case where extra blocks *should* count.
- **D-cmp: LoRA better** (confirmatory). At its F3 preference (102%, which is at or above every B3 curve point, so the rule compares against the base point, 0% reduction) LoRA's 95% [94, 96] exceeds the best head-damping value (0%).
- **D-shortcut (exploratory): no flag** (C6k4 +0.126 < 0.20). C6 moves *toward* C1-like behaviour, consistent with "ignore repetition on either side", not "minority wins".
- **Reading:** a same-model trained adapter does what head damping could not on paraphrases, at no capability cost. It does so by ignoring corroboration altogether. Neither the training-free nor the trained fix in this sprint separates copies from independent sources.
