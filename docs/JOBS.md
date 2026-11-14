# JOBS — sprint Block A (budget ≤ 25 GPU-h; Qwen2.5-32B ≤ 2 A100-h)
| date | job id | what | GPU | QoS | elapsed | GPUs | GPU-h | state |
|---|---|---|---|---|---|---|---|---|
| 2026-11-13 | 140845 | smoke 1: Llama behave+probes, 3+3 pairs (the command loop stopped after step 1: srun ate stdin; fixed) | A100 | a100-dev | 00:04:03 | 1 | 0.068 | COMPLETED |
| 2026-11-13 | 141014 | smoke 2: Llama patch F1, Qwen-7B + Gemma behave, Gemma patch F3, Qwen-32B fit check (peak 67 GB) | A100 | a100-dev | 00:14:11 | 1 | 0.236 | COMPLETED |
| 2026-11-13 | 142090 | behaviour + probes, Llama + Qwen-7B, t3 QoS | A100 | a100-t3 | 00:00:00 | 1 | 0 | CANCELLED (never started; sat in Priority >1 h; replaced by the dev chain) |
| 2026-11-13 | 143742 | smoke 3: Llama parts → merge → probes → extras (graded, lexical, neutral), 3 pairs | A100 | a100-dev | 00:12:02 | 1 | 0.201 | COMPLETED |
| 2026-11-13 | 144328 | A01 Llama behave F1 test part (48,000 prompts, 32k hidden states; ~32 prompts/s) | A100 | a100-dev | 00:26:16 | 1 | 0.438 | COMPLETED |
| 2026-11-13 | 144906 | A02 Llama behave F2+F3 part (48,000 prompts) | A100 | a100-dev | 00:21:34 | 1 | 0.359 | COMPLETED |
| 2026-11-13 | 144907 | A03 Llama behave F1 sel part + merge (120,000 prompts, 76,800 hidden states) + probes + extras | A100 | a100-dev | 00:25:32 | 1 | 0.426 | COMPLETED |
| 2026-11-13 | 149443 | amendment 6: C1-only probe, Llama (CPU, parallel to the GPU chain per Amr) | CPU (20 cores) | qos_cpu-dev | 00:01:30 | — | 0.5 core-h | COMPLETED |
| 2026-11-13 | 150067 | amendment 7: C1-only probe with the L\* rule + bootstrap CIs, Llama (CPU) | CPU (20 cores) | qos_cpu-dev | ~00:03 | — | ~1 core-h | COMPLETED |
| 2026-11-13 | 150687 | amendment 9: C1-only probe + H5 CIs (last 3 stored layers), Llama (CPU) | CPU (20 cores) | qos_cpu-dev | ~00:03 | — | ~1 core-h | COMPLETED |
| 2026-11-13 | 159248 | A11 Gemma F1 patching + best-model pick (Qwen-7B) + Qwen F3/C2k4 patching + F3 sel C7k4 behaviour + Qwen F3/C7k4 patching | A100 | a100-dev | 00:40:18 | 1 | 0.672 | COMPLETED |
| 2026-11-13 | 159247 | A10 Qwen-7B F1 patching (AtP 96 sel + 96 test; verify 160 test items) | A100 | a100-dev | 00:11:57 | 1 | 0.199 | COMPLETED |
| 2026-11-13 | 144915 | A09 Qwen2.5-32B behaviour only (F1+F3 test, 7 conditions in priority order, 44,800 prompts; bf16, 1× A100 80 GB, peak ~67 GB) | A100 | a100-dev | 01:21:43 | 1 | 1.362 | COMPLETED (within the 2 A100-h cap) |
| 2026-11-13 | 144913 | A08 Gemma F1 sel part + merge + probes + extras + C6 states (+ queued A10/A11) | A100 | a100-dev | 00:59:55 | 1 | 0.999 | COMPLETED |
| 2026-11-13 | 161239, 161240 | C1-only + balanced probe rerun with C6, Gemma (CPU) | CPU (20 cores) | qos_cpu-dev | ~00:06 each | — | ~4 core-h | COMPLETED |
| 2026-11-13 | 160278, 160279 | C1-only + balanced probe, Gemma (confirmatory; CPU; C6 to be added after A08) | CPU (20 cores) | qos_cpu-dev | ~00:06 each | — | ~4 core-h | COMPLETED |
| 2026-11-13 | 144912 | A07 Gemma behave F2+F3 part (48,000 prompts) | A100 | a100-dev | 00:35:47 | 1 | 0.596 | COMPLETED |
| 2026-11-13 | 144911 | A06 Gemma behave F1 test part (48,000 prompts) | A100 | a100-dev | 00:46:15 | 1 | 0.771 | COMPLETED |
| 2026-11-13 | 144910 | A05 Llama C6 states (6,400) + Qwen-7B F2+F3 part + merge + probes + extras + Qwen C6 states | A100 | a100-dev | 00:41:07 | 1 | 0.685 | COMPLETED |
| 2026-11-13 | 153318, 153319 | C1-only probe (unbalanced, balanced) rerun with C6 states, Llama + Qwen-7B (CPU) | CPU (20 cores) | qos_cpu-dev | ~00:05 each | — | ~3 core-h | COMPLETED |
| 2026-11-13 | 151725 | amendment 10: decision-balanced C1 probe, Llama + Qwen-7B (exploratory; CPU) | CPU (20 cores) | qos_cpu-dev | ~00:04 | — | ~1.5 core-h | COMPLETED |
| 2026-11-13 | 151289 | amendment 7/9: C1-only probe, Qwen-7B (confirmatory; CPU) | CPU (20 cores) | qos_cpu-dev | ~00:02 | — | ~1 core-h | COMPLETED |
| 2026-11-13 | 144909 | A04 Qwen-7B behave F1 test + F1 sel parts (72,000 prompts) | A100 | a100-dev | 00:35:58 | 1 | 0.599 | COMPLETED |
| 2026-11-13 | 144908 | A03b Llama F1 patching (AtP 96 sel + 96 test; verify 160 test items, 51 head sets) | A100 | a100-dev | 00:08:35 | 1 | 0.143 | COMPLETED |

**Block A GPU total: 9.65 A100-h** (smokes 0.50 + chain A01–A11 9.15) of the 25 h cap; CPU ~16 core-h.

## Block B (budget ≤ 15 A100-h)
| date | job id | what | GPU | QoS | elapsed | GPUs | A100-h | state |
|---|---|---|---|---|---|---|---|---|
| 2026-11-13 | 165344 | B0 induction check — FAILED (CUDA OOM: no `no_grad`); cancelled, no results | A100 | a100-dev | ~00:05 | 1 | ~0.08 | CANCELLED |
| 2026-11-13 | 165448 | B0 induction check, 3 models (rerun) | A100 | a100-dev | 00:04:45 | 1 | 0.079 | COMPLETED |
| 2026-11-13 | 165426 | save C1-only probe directions for B2 (CPU refit, 3 models) | CPU (20 cores) | qos_cpu-dev | 00:05:41 | — | ~2 core-h | COMPLETED |
| 2026-11-14 | 165397 | B3a select (Qwen; H_F1, H_F3C7, U grids; 100 sel pairs) | A100 | a100-dev | 00:47:48 | 1 | 0.797 | COMPLETED |
| 2026-11-14 | 165951 | B3a2 select Feeders γ grid | A100 | a100-dev | 00:11:41 | 1 | 0.195 | COMPLETED |
| 2026-11-14 | 165398 | B3b H_F1 fix on TEST + MMLU/WikiText | A100 | a100-dev | 00:31:35 | 1 | 0.526 | COMPLETED |
| 2026-11-14 | 165399 | B3c H_F3C7 fix on TEST + capability | A100 | a100-dev | 00:33:50 | 1 | 0.564 | COMPLETED |
| 2026-11-14 | 165540 | B3d union fix U on TEST + capability | A100 | a100-dev | 00:31:36 | 1 | 0.527 | COMPLETED |
| 2026-11-14 | 165952 | B3d2 Feeders fix on TEST + capability | A100 | a100-dev | 00:32:03 | 1 | 0.534 | COMPLETED |
| 2026-11-14 | 165953 | B3e test γ-curves (U, H_F3C7, Feeders; F3) | A100 | a100-dev | 00:39:58 | 1 | 0.666 | COMPLETED |
| 2026-11-14 | 165440 | B2 steering diagnostic (Qwen; L18 + L28) | A100 | a100-dev | 01:49:25 | 1 | 1.824 | COMPLETED |
| 2026-11-14 | 170008 | B3R1 Llama select + test + capability | A100 | a100-dev | 00:53:41 | 1 | 0.895 | COMPLETED |
| 2026-11-14 | 170009 | B3R2 Gemma select | A100 | a100-dev | 00:26:27 | 1 | 0.441 | COMPLETED |
| 2026-11-14 | 170010 | B3R3 Gemma test + capability | A100 | a100-dev | 01:00:27 | 1 | 1.007 | COMPLETED |
| 2026-11-14 | 174087 | B1 path patching (Qwen; F1/C2k4 + F3/C7k4) | A100 | a100-dev | 00:25:11 | 1 | 0.420 | COMPLETED |

**Block B GPU total: ≈ 8.56 A100-h** (cap 15). Chain: Slurm `afterany` dependencies, one GPU job at a time (`logs/chain.log`).

Slurm dependency chain (`submit_chain.sh 144328`, 14:13 CET): 144906 A02 Llama F2+F3 → 144907 A03 Llama F1 sel + merge + probes + extras → 144908 A03b **Llama F1 patching (H1)** → 144909 A04 Qwen-7B F1 → 144910 A05 Qwen-7B F2+F3 + merge + extras → 144911 A06 Gemma F1 test → 144912 A07 Gemma F2+F3 → 144913 A08 Gemma F1 sel + merge + extras (its start queues A10/A11) → 144915 A09 Qwen-32B (2 h cap) → A10 Qwen-7B F1 patching → A11 Gemma F1 patching + F3 for the best model. Verified 18:14: A08's start step (`chain_tail.sh`) queued A10 = 159247 (after 144915) and A11 = 159248 (after 159247).

## Block D (budget ≤ 4 A100-h)
| date | job id | what | GPU | QoS | elapsed | GPUs | A100-h | state |
|---|---|---|---|---|---|---|---|---|
| 2026-11-14 | 181555 | D0 smoke: LoRA train on 32 items (1 epoch + refinement) + eval on 3 test pairs (F1); a t3 copy 182006 was cancelled before starting | A100 | a100-dev | ~00:04 | 1 | ~0.07 | COMPLETED |
| 2026-11-14 | 182217/182218 → 182219/182220 → 182221/182222 | D1 train lr 2e-4 → D2 train lr 1e-4 → D3 pick + test + capability. Each stage submitted to dev AND t3; the copy that starts first runs, the other is cancelled while pending (Amr, 14 Nov). D1 started on dev (182217); its t3 copy 182218 was cancelled. *Watcher bug:* a zsh word-splitting error made the first watcher cancel D3's t3 copy 182222 instead of 182218 (neither had run); fixed by hand, and D3's t3 copy was resubmitted as **182384**. | A100 | dev / t3 | | 1 | | RUNNING |
| 2026-11-14 | 182217 | D1 LoRA train, lr 2e-4 (val loss: e1 0.482, e2 0.342, refine 0.453) | A100 | a100-dev | 00:40:29 | 1 | 0.675 | COMPLETED |
| 2026-11-14 | 182219 | D2 LoRA train, lr 1e-4 (val loss: e1 0.290, **e2 0.228**, refine 0.296); its t3 copy 182220 was cancelled | A100 | a100-dev | 00:39:43 | 1 | 0.662 | COMPLETED |
| 2026-11-14 | 182221 | D3 pick (lr1e-4/ckpt_e2 by the validation rule) + test F1/F3/F2 + capability; its t3 copy 182384 was cancelled | A100 | a100-dev | 00:35:50 | 1 | 0.597 | COMPLETED |

**Block D GPU total: ≈ 2.00 A100-h** (cap 4): smoke 0.07 + D1 0.675 + D2 0.662 + D3 0.597.

## Block F (budget ≤ 3 A100-h, A100 only)
| date | job id | what | GPU | QoS | elapsed | GPUs | A100-h | state |
|---|---|---|---|---|---|---|---|---|
| 2026-11-14 | 188697 (t3 copy 188699 cancelled pending) | FJ1 F1-main behaviour: Qwen-32B + Qwen-7B, F3 | A100 | a100-dev | 00:35:53 | 1 | 0.60 | COMPLETED |
| 2026-11-14 | 188701 (t3 copy 188702 cancelled pending) | FJ2 F2 fixed-head patching (Llama, Qwen-7B, Gemma; 4 contrasts) + F3 behaviour (Llama, Gemma) | A100 | a100-dev | 01:15:34 | 1 | 1.26 | COMPLETED |
| 2026-11-14 | 188704 (t3 copy 188705 cancelled pending) | FJ3 F1-tables behaviour (Qwen-7B, Llama, Gemma) | A100 | a100-dev | 00:30:00 | 1 | 0.50 | COMPLETED |
| 2026-11-14 | 188716 | F3: C1-only probe refit + chosen-answer AUCs (3 models; CPU). Refit probe files byte-identical to before | CPU (20 cores) | qos_cpu-dev | 00:05:29 | — | ~2 core-h | COMPLETED |

Block F total: **2.36 A100-h** (cap 3) + ~2 CPU core-h. Note: another session's `rlhead` jobs ran on the same allocation in parallel with FJ2.
