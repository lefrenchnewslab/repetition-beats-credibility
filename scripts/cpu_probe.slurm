#!/bin/bash
# SPRINT Block A, amendment 6: C1-only probe on saved states. CPU node (runs in parallel with the GPU chain, per Amr).
# Usage: TAGS="Llama-3.1-8B-Instruct" sbatch cpu_probe.slurm
#SBATCH --account=<account>@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --job-name=sA_c1probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
module purge
module load pytorch-gpu/py3/2.8.0
export OMP_NUM_THREADS=20
cd $WORK_ROOT/sprint_A
for t in $TAGS; do echo "=== $t $PARGS"; python probe_c1only.py runs/A $t $PARGS || echo "FAILED $t"; done
echo ALL_DONE
