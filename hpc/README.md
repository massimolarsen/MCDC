# HPC environment

On an allocated interactive node, use Bash:

```bash
source /nfs/stak/users/larsemas/hpc-share/MCDC/hpc/env.sh
cd "$MCDC_ROOT/examples/cubesat_test"
python inputCE_G4.py --mode=numba
```

`env.sh` works from any directory and can also be sourced by batch scripts.
It deactivates an existing Python virtualenv, resets modules, activates the
shared environment, sources Geant4, and configures data/cache paths. It does
not install packages, launch simulations, change directory, or set shell
options. Start with a fresh Bash shell if Conda is active.

`stack.sh` is the common build/runtime configuration: GCC 10.3, Python 3.13,
MPICH 4.0h (runtime reports 4.0.2), and the existing Geant4 11.2.1 installation.
The captured interpreter is Python 3.13.13; bootstrap checks this patch version.
Module aliases, especially `slurm/current`, can change. This setup depends on
the cluster retaining its installations.

Defaults use sibling `venvs/mcdc-g4`, `couple-mcdc-g4`, and
`nuclear-data/mcdc-hdf5` directories. Export `MCDC_G4_VENV`, `MCDC_G4_SOURCE`,
`MCDC_G4_BUILD`, `MCDC_G4_SETUP`, `Geant4_DIR`, or `MCDC_LIB` to override paths.
When switching Geant4, set both setup and CMake paths to matching locations.

## Rebuild explicitly

The existing environment needs no installation. Create a replacement at new
paths to preserve the working environment and bridge:

```bash
export MCDC_G4_VENV=/nfs/stak/users/larsemas/hpc-share/venvs/mcdc-g4-new
export MCDC_G4_BUILD=/nfs/stak/users/larsemas/hpc-share/couple-mcdc-g4/build-new
bash /nfs/stak/users/larsemas/hpc-share/MCDC/hpc/bootstrap.sh
source /nfs/stak/users/larsemas/hpc-share/MCDC/hpc/env.sh
```

Bootstrap refuses existing virtualenv/build directories. It loads CMake 3.27.9,
installs the pinned Python packages, builds mpi4py against the cluster MPI,
installs this checkout editable, and builds/import-checks the bridge. It needs
package-index access, bridge source, and the existing Geant4 installation.
Failed attempts can leave partial directories; inspect them and use fresh
paths to retry. `BUILD_JOBS` defaults to 4.

The CubeSat input currently locates the bridge at `couple-mcdc-g4/build`.
Update its `BRIDGE_BUILD_DIR` before running with a replacement build.
`MCDC_G4_BUILD` controls bootstrap and the check below, not the input file.

`requirements.lock.txt` captures installed package versions, including
transitive dependencies. It is not a hash-verified artifact archive; isolated
build dependencies are not pinned. Bootstrap records module/compiler/CMake
versions, packages, Git revisions/status, and Geant4/data paths in the new
virtualenv's `build-manifest.txt`. Preserve dirty source changes separately.
The inspected bridge revision was
`66a26df1047d8bb181e9d130cc689cbcc8116663`, with local changes.
Preserve simulation inputs, seeds, and nuclear/Geant4 dataset versions or
checksums alongside results. Bootstrap does not download or version data.

## Check without running a simulation

```bash
python -m pip check
python - <<'PYTHON'
import os
import mcdc
from mpi4py import MPI
from mcdc.coupling.geant4_worker import load_bridge
print(mcdc.__file__)
print(MPI.Get_library_version())
print(load_bridge(os.environ['MCDC_G4_BUILD']).__file__)
PYTHON
```

Use the cluster-supported MPI launcher explicitly for multiple ranks within
Slurm allocations. Activation does not allocate CPUs or choose rank/thread counts.
