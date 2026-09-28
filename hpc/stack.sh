# Shared cluster configuration; sourced by env.sh and bootstrap.sh.
# Keep module choices identical at build time and run time.
if ! type module >/dev/null 2>&1; then
    echo 'The module command is unavailable. Use a cluster login shell.' >&2
    return 1
fi
if declare -F deactivate >/dev/null; then
    deactivate
fi
module purge || return 1
module load slurm/current python/3.13 mpich/4.0h_gcc-10 cmake/3.27.9 || return 1
# MPICH loads GCC 10.3; select the GCC runtime required by Geant4 11.3.2.
module load gcc/12.5 || return 1

export MCDC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MCDC_G4_VENV="${MCDC_G4_VENV:-${MCDC_ROOT}/../venvs/mcdc-g4}"
export MCDC_G4_SOURCE="${MCDC_G4_SOURCE:-${MCDC_ROOT}/../couple-mcdc-g4}"
export MCDC_G4_BUILD="${MCDC_G4_BUILD:-${MCDC_G4_SOURCE}/build}"
if [[ ${MCDC_G4_SETUP:-} == *geant4-11.2.1*/bin/geant4.sh ]]; then
    unset MCDC_G4_SETUP
fi
export MCDC_G4_SETUP="${MCDC_G4_SETUP:-/usr/local/apps/geant4/11.3.2/bin/geant4.sh}"
if [[ ! -r "$MCDC_G4_SETUP" ]]; then
    echo "Missing Geant4 setup: $MCDC_G4_SETUP" >&2
    return 1
fi
source "$MCDC_G4_SETUP" || return 1
export Geant4_DIR="${MCDC_G4_SETUP%/bin/geant4.sh}/lib64/cmake/Geant4"
export MCDC_LIB="${MCDC_LIB:-${MCDC_ROOT}/../nuclear-data/mcdc-hdf5}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MCDC_ROOT}/../.cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${MCDC_ROOT}/../.cache/numba}"
