#!/bin/bash
# SPRINT Block A: Slurm dependency chain (replaces feeder.sh, 13 Nov). Each job starts only after the previous one ends
# (afterany), so exactly one GPU job runs at a time, independent of any login-node process or ssh session.
# Order (Amr, 13 Nov): A02 A03 A03b(Llama F1 patch, H1) A04 A05 A06 A07 A08 A09, then A10 A11 via chain_tail.sh at the start of A08.
# Usage: bash submit_chain.sh <job id currently running, e.g. A01>
cd $WORK_ROOT/sprint_A
prev=$1
for name in A02 A03 A03b A04 A05 A06 A07 A08 A09; do
  f=$(ls cmds/${name}_*.txt); t=01:58:00; extra=""
  [ "$name" = A09 ] && t=02:00:00                       # Qwen-32B hard cap: 2 A100-h
  [ "$name" = A08 ] && extra="PRE=chain_tail.sh"
  jid=$(env $extra CMDS=$f sbatch --parsable --dependency=afterany:$prev --qos=qos_gpu_a100-dev --time=$t -J sA_$name --export=ALL sprintA.slurm)
  echo "$(date +%T) $name $f -> $jid (after $prev)" | tee -a logs/chain.log
  prev=$jid
done
