#!/bin/bash
# SPRINT Block A: run at the START of job A08. Submits A10 (Qwen-7B F1 patch) after A09, then A11 (Gemma F1 patch + F3 best) after A10.
# Slurm-only chain, no login-node process. PRE is unset so the new jobs do not re-run this script.
unset PRE
cd $WORK_ROOT/sprint_A
if squeue -h -u $USER -n sA_A10 -o %i | grep -q .; then echo "chain_tail: A10 already queued" >> logs/chain.log; exit 0; fi
a09=$(squeue -h -u $USER -n sA_A09 -o %i | head -1)
dep=${a09:+--dependency=afterany:$a09}
[ -z "$a09" ] && dep="--dependency=afterany:$SLURM_JOB_ID"
j10=$(CMDS=cmds/A10_patch_llama_qwen.txt sbatch --parsable $dep --qos=qos_gpu_a100-dev --time=01:58:00 -J sA_A10 --export=ALL sprintA.slurm)
j11=$(CMDS=cmds/A11_patch_gemma_f3best.txt sbatch --parsable --dependency=afterany:$j10 --qos=qos_gpu_a100-dev --time=01:58:00 -J sA_A11 --export=ALL sprintA.slurm)
echo "$(date +%T) chain_tail (from job $SLURM_JOB_ID): A10=$j10 (after ${a09:-$SLURM_JOB_ID}) A11=$j11" >> logs/chain.log
