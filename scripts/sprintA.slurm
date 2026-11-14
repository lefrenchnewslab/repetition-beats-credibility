#!/bin/bash
# SPRINT Block A (13 Nov 2026). Pilot-derived (pilot_illusory/scripts/illusory.slurm): runs a list of commands from $CMDS (one per line).
# Usage (on Jean Zay, in $W/sprint_A): CMDS=cmds/<name>.txt sbatch --qos=qos_gpu_a100-dev --time=01:00:00 -J <name> sprintA.slurm
#SBATCH --account=<account>@a100
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
module purge
module load arch/a100 pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK_ROOT/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
cd $WORK_ROOT/sprint_A
export OUT=runs/$SLURM_JOB_ID; mkdir -p $OUT; cp $CMDS $OUT/cmds.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > logs/gpu_$SLURM_JOB_ID.csv &
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $OUT/gpu_name.txt
[ -n "$PRE" ] && echo "=== PRE $PRE" && bash "$PRE"   # e.g. chain_tail.sh: submits the last chain jobs (dev QoS caps submitted jobs at 10)
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "=== $(date +%T) $line"
  eval "srun $line" < /dev/null || echo "FAILED: $line"   # </dev/null: srun would otherwise eat the command list (job 140845)
done < $CMDS
echo ALL_DONE
