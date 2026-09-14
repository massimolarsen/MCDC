#!/usr/bin/env bash
# Explicit provisioning only: never called by env.sh.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/stack.sh"
module load cmake/3.27.9
if [[ -e "$MCDC_G4_BUILD" ]]; then
    echo "Refusing to reuse $MCDC_G4_BUILD. Choose a new MCDC_G4_BUILD." >&2
    exit 1
fi
if [[ -e "$MCDC_G4_VENV" ]]; then
    echo "Refusing to overwrite $MCDC_G4_VENV. Choose a new MCDC_G4_VENV." >&2
    exit 1
fi
[[ -f "$MCDC_G4_SOURCE/CMakeLists.txt" ]] || { echo "Missing bridge source: $MCDC_G4_SOURCE" >&2; exit 1; }
[[ -f "$Geant4_DIR/Geant4Config.cmake" ]] || { echo "Invalid Geant4_DIR: $Geant4_DIR" >&2; exit 1; }
command -v cmake >/dev/null
command -v mpicc >/dev/null
python -c 'import sys; assert sys.version_info[:3] == (3, 13, 13), sys.version'
python -m venv "$MCDC_G4_VENV"
source "$MCDC_G4_VENV/bin/activate"
# Compile mpi4py against the loaded cluster MPI, bypassing cached MPI wheels.
MPICC="$(command -v mpicc)" python -m pip install --no-cache-dir --no-binary=mpi4py \
    -r "$MCDC_ROOT/hpc/requirements.lock.txt"
python -m pip install --no-deps -e "$MCDC_ROOT"
python -m pip check
cmake -S "$MCDC_G4_SOURCE" -B "$MCDC_G4_BUILD" \
    -DGeant4_DIR="$Geant4_DIR" -DPython_ROOT_DIR="$VIRTUAL_ENV" \
    -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
    -DCMAKE_CXX_COMPILER="$(command -v c++)" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build "$MCDC_G4_BUILD" --target geant4_bridge -j "${BUILD_JOBS:-4}"
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'
PYTHONPATH="$MCDC_G4_BUILD${PYTHONPATH:+:$PYTHONPATH}" python -c 'import geant4_bridge; print(geant4_bridge.__file__)'
{
    module -t list 2>&1
    python --version
    c++ --version
    cmake --version
    python -m pip freeze --all
    git -C "$MCDC_ROOT" rev-parse HEAD
    git -C "$MCDC_ROOT" status --short
    git -C "$MCDC_G4_SOURCE" rev-parse HEAD
    git -C "$MCDC_G4_SOURCE" status --short
    echo "Geant4_DIR=$Geant4_DIR"
    echo "MCDC_LIB=$MCDC_LIB"
} > "$MCDC_G4_VENV/build-manifest.txt"
echo "Ready. Activate with: source $MCDC_ROOT/hpc/env.sh"
