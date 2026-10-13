#!/bin/bash -l
#SBATCH --job-name=cubesat-e
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=cubesat-e-%j.out

# Electron-only CubeSat (AE9 trapped electrons). Submit from this directory:
#   cd <checkout>/examples/electron_cubesat && sbatch submit.slurm
# The Python, MPI, and module stack come from the g4-hpc environment; MC/DC is
# imported from the checkout this script is submitted from.

set -eo pipefail

source /nfs/hpc/share/larsemas/MCDC-g4-hpc/hpc/env.sh

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f input_CubeSat_electron.py ]]; then
    echo "Submit from examples/electron_cubesat (input_CubeSat_electron.py not found)" >&2
    exit 1
fi
ELECTRON_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="$ELECTRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Electron element files (H, Li, C, O, Mg, Al, Si, Co, Cu, Zn) from e-dataset
export MCDC_LIB="${MCDC_ELECTRON_LIB:-$ELECTRON_ROOT/hdf5lib-electrons}"
if [[ ! -d "$MCDC_LIB" ]]; then
    echo "Missing electron data directory: $MCDC_LIB" >&2
    exit 1
fi

archive="$PWD/runs/$SLURM_JOB_ID"
mkdir -p "$archive"
export MCDC_RUN_DIR="$archive"

finish() {
    status=$?
    trap - EXIT
    log="$SLURM_SUBMIT_DIR/cubesat-e-$SLURM_JOB_ID.out"
    if [[ -f "$log" ]]; then
        mv "$log" "$archive/slurm.out" || status=1
    fi
    echo "Job outputs: $archive"
    exit "$status"
}
trap finish EXIT

cp input_CubeSat_electron.py ae9_10MeV.csv "$archive/"
python -c "import mcdc; print(mcdc.__file__)" > "$archive/mcdc-module.txt"
python -m pip freeze > "$archive/python-packages.txt"
git -C "$ELECTRON_ROOT" rev-parse HEAD > "$archive/mcdc-revision.txt"
ls -l "$MCDC_LIB" > "$archive/electron-data.txt"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mpirun -n "$SLURM_NTASKS" -bind-to none \
    python -u input_CubeSat_electron.py --mode=numba --no-progress_bar
