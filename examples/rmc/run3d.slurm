#!/bin/bash -l
# One job of the 3D fuel examples, submitted by submit_3d.sh: runs SCRIPT in
# EXAMPLE_DIR (input.py with the settings from --export, or merge_3d.py).
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
source "$SLURM_SUBMIT_DIR/efficiency/env.sh"
cd "$EXAMPLE_DIR"
git -C "$MCDC_DIR" rev-parse HEAD > "$SLURM_SUBMIT_DIR/logs/revision-$SLURM_JOB_ID.txt"
echo "start $(date -Iseconds): $EXAMPLE_DIR/$SCRIPT"
if [ "$SCRIPT" = merge_3d.py ]; then
    python -u merge_3d.py
else
    mpirun -n "$SLURM_NTASKS" -bind-to none \
        python -u -m mpi4py "$SCRIPT" --mode=numba --no-progress_bar
fi
echo "end $(date -Iseconds)"
