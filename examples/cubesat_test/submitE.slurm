#!/bin/bash -l
#SBATCH --job-name=cubesatE
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=cubesatE-%j.out

set -eo pipefail

source /nfs/hpc/share/larsemas/MCDC/hpc/env.sh
cd "$MCDC_ROOT/examples/cubesat_test"

archive="$PWD/runs/$SLURM_JOB_ID"
mkdir -p "$archive"
export MCDC_RUN_DIR="$archive"

finish() {
    status=$?
    trap - EXIT
    log="$SLURM_SUBMIT_DIR/cubesatE-$SLURM_JOB_ID.out"
    if [[ -f "$log" ]]; then
        mv "$log" "$archive/slurm.out" || status=1
    fi
    echo "Job outputs: $archive"
    exit "$status"
}
trap finish EXIT

cp inputE_SUBMIT.py cubesat_G4_devices.py "$archive/"
python -m pip freeze > "$archive/python-packages.txt"
git rev-parse HEAD > "$archive/mcdc-revision.txt"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mpirun -n "$SLURM_NTASKS" -bind-to none \
    python -u inputE_SUBMIT.py --mode=numba
