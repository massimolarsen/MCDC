#!/bin/bash -l
# Submit the efficiency sweep for the problems given (default: the two 0D problems),
# from examples/rmc/efficiency on the submit node:
#   ./submit.sh                      absorber and o16
#   ./submit.sh fuel_thermal         one fuel rod, after the 0D results look right
# For each problem:
#   1. a warmup job per RMC configuration that computes and caches the transfer
#      moments and the Numba kernels;
#   2. one job array over its tasks (configurations, levels, seeds), started after the
#      warmups succeed, with at most MAX_RUNNING tasks at once;
#   3. its high-statistics SMC reference (not for the absorber, which is analytic).
# Finished tasks are skipped on resubmission (set SWEEP_FORCE=1 to rerun them).
set -euo pipefail
export SWEEP_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SWEEP_DIR/env.sh"
cd "$SWEEP_DIR"
mkdir -p logs

# Per problem: task group, ranks, time limit per task, array tasks running at once,
# time limit of the reference. 16 ranks fit every share node.
declare -A GROUP=([absorber]=0d [o16]=0d [fuel_thermal]=fuel [fuel_fast]=fuel)
declare -A RANKS=([absorber]=8 [o16]=8 [fuel_thermal]=16 [fuel_fast]=16)
declare -A LIMIT=([absorber]=02:00:00 [o16]=04:00:00 [fuel_thermal]=12:00:00 [fuel_fast]=12:00:00)
declare -A MAX_RUNNING=([absorber]=6 [o16]=6 [fuel_thermal]=2 [fuel_fast]=2)
declare -A REFERENCE_LIMIT=([o16]=12:00:00 [fuel_thermal]=2-00:00:00 [fuel_fast]=2-00:00:00)

problems=("$@")
[ ${#problems[@]} -eq 0 ] && problems=(absorber o16)

for problem in "${problems[@]}"; do
    [ -n "${GROUP[$problem]:-}" ] || { echo "unknown problem: $problem"; exit 1; }
    submit=(sbatch --parsable --partition="$SWEEP_PARTITION"
        --ntasks="${RANKS[$problem]}" --export=ALL,SWEEP_DIR="$SWEEP_DIR")
    warmups=()
    for config in dissertation current; do
        warmups+=("$("${submit[@]}" --time="${LIMIT[$problem]}" \
            --job-name="warmup-$problem-$config" \
            --export=ALL,SWEEP_DIR="$SWEEP_DIR",SWEEP_MODE=warmup,PROBLEM="$problem",CONFIG="$config" \
            sweep.slurm)")
    done
    indices=$(python sweep.py list --group "${GROUP[$problem]}" --problem "$problem" | tail -1)
    dependency=$(IFS=:; echo "${warmups[*]}")
    "${submit[@]}" --time="${LIMIT[$problem]}" \
        --array="${indices}%${MAX_RUNNING[$problem]}" \
        --dependency=afterok:"$dependency" \
        --job-name="sweep-$problem" sweep.slurm
    if [ "$problem" != absorber ]; then
        index=$(python sweep.py list --group reference --problem "$problem" | tail -1)
        "${submit[@]}" --time="${REFERENCE_LIMIT[$problem]}" --array="$index" \
            --job-name="reference-$problem" sweep.slurm
    fi
    echo "$problem: submitted"
done
echo "monitor with: squeue -u $USER"
