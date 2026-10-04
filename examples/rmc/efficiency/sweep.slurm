#!/bin/bash -l
# One sweep job: a warmup (SWEEP_MODE=warmup, PROBLEM, CONFIG) or one array task
# (SLURM_ARRAY_TASK_ID = task index of sweep.py). Submitted by submit.sh, which sets
# --partition, --ntasks, --time, --array and the dependencies.
#SBATCH --job-name=rmc-sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%A_%a.out

set -eo pipefail
source "$SWEEP_DIR/env.sh"
cd "$SWEEP_DIR"
git -C "$MCDC_DIR" rev-parse HEAD > "logs/revision-$SLURM_JOB_ID.txt"

if [ "${SWEEP_MODE:-run}" = "warmup" ]; then
    mpirun -n "$SLURM_NTASKS" -bind-to none \
        python -u -m mpi4py sweep.py warmup "$PROBLEM" "$CONFIG" --mode=numba --no-progress_bar
else
    mpirun -n "$SLURM_NTASKS" -bind-to none \
        python -u -m mpi4py sweep.py run "$SLURM_ARRAY_TASK_ID" --mode=numba --no-progress_bar
fi
