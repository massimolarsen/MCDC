#!/bin/bash -l
#SBATCH --job-name=cubesat-np
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cubesat-np-%j.out
# cn-gpu3/4 lack libnsl.so.2, which Geant4's GDML library (via Xerces-C) needs
#SBATCH --exclude=cn-gpu[3-4]

set -eo pipefail

source /nfs/hpc/share/larsemas/MCDC-g4-hpc/hpc/env.sh
cd "$MCDC_ROOT/examples/cubesat_test"

# Neutron + proton nuclide files to 200 MeV at 300 K: JENDL-5 neutrons
# (FENDL-3.0 Li7) merged with TENDL-2021 protons
export MCDC_LIB="$MCDC_ROOT/hdf5libProton/with_neutron_200MeV"

archive="$PWD/runs/$SLURM_JOB_ID"
mkdir -p "$archive"
export MCDC_RUN_DIR="$archive"

finish() {
    status=$?
    trap - EXIT
    log="$SLURM_SUBMIT_DIR/cubesat-np-$SLURM_JOB_ID.out"
    if [[ -f "$log" ]]; then
        mv "$log" "$archive/slurm.out" || status=1
    fi
    echo "Job outputs: $archive"
    exit "$status"
}
trap finish EXIT

cp inputCE_G4NP_SUBMIT.py cubesat_G4_devices_1um.py ecss_200MeV.csv ap9_200MeV.csv "$archive/"
python -m pip freeze > "$archive/python-packages.txt"
git rev-parse HEAD > "$archive/mcdc-revision.txt"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mpirun -n "$SLURM_NTASKS" -bind-to none \
    python -u inputCE_G4NP_SUBMIT.py --mode=numba
