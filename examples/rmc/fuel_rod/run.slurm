#!/bin/bash -l
# One fuel rod job, submitted by submit.sh: SCRIPT (input.py, linear.py or merge.py)
# with FUEL_ROD_CASES, FUEL_ROD_RUNS and FUEL_ROD_OUTPUT from --export.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
source "$FUEL_ROD_DIR/../efficiency/env.sh"
cd "$FUEL_ROD_DIR"
git -C "$MCDC_DIR" rev-parse HEAD > "logs/revision-$SLURM_JOB_ID.txt"
echo "start $(date -Iseconds): $SCRIPT cases=${FUEL_ROD_CASES:-} runs=${FUEL_ROD_RUNS:-} output=${FUEL_ROD_OUTPUT:-}"

if [ "$SCRIPT" = merge.py ]; then
    python -u merge.py
else
    mpirun -n "$SLURM_NTASKS" -bind-to none \
        python -u -m mpi4py "$SCRIPT" --mode=numba --no-progress_bar
fi
echo "end $(date -Iseconds)"
