#!/bin/bash -l
#SBATCH --job-name=cubesat-g4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=cubesat-g4-%j.out
# cn-gpu3/4 lack libnsl.so.2, which Geant4's GDML library (via Xerces-C) needs
#SBATCH --exclude=cn-gpu[3-4]

# Rerun only the Geant4 stage of a coupled run from its saved payloads, e.g.
# after MC/DC finished but the Geant4 workers failed:
#   sbatch submitG4_from_payloads.slurm runs/<job>
# Each payload names its bridge build and output file, so results land in the
# run directory's geant4_h5/ exactly as in the coupled job.

set -eo pipefail

source /nfs/hpc/share/larsemas/MCDC-g4-hpc/hpc/env.sh
cd "$MCDC_ROOT/examples/cubesat_test"

run_dir="$(realpath "${1:?usage: sbatch submitG4_from_payloads.slurm runs/<job>}")"
[[ -d "$run_dir/geant4_payloads" ]] || { echo "No geant4_payloads in $run_dir" >&2; exit 1; }

finish() {
    status=$?
    trap - EXIT
    log="$SLURM_SUBMIT_DIR/cubesat-g4-$SLURM_JOB_ID.out"
    if [[ -f "$log" ]]; then
        mv "$log" "$run_dir/slurm_g4.out" || status=1
    fi
    echo "Geant4 outputs: $run_dir/geant4_h5"
    exit "$status"
}
trap finish EXIT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# regions in parallel, each worker running its payload's Geant4 thread count
pids=()
regions=()
for payload in "$run_dir"/geant4_payloads/*_payload.h5; do
    region="$(basename "$payload" _payload.h5)"
    python -u "$MCDC_ROOT/mcdc/coupling/geant4_worker.py" "$payload" \
        > "$run_dir/geant4_$region.log" 2>&1 &
    pids+=("$!")
    regions+=("$region")
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Geant4 region '${regions[$i]}': finished"
    else
        echo "Geant4 region '${regions[$i]}': FAILED, see geant4_${regions[$i]}.log" >&2
        failed=1
    fi
done
exit "$failed"
