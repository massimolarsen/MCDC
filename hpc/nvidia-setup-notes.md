# NVIDIA GPU validation setup

Status: environment installation and CUDA smoke tests passed; the first MC/DC
GPU regression failed during compilation. This is a reproducible investigation
log, not yet a recipe for a passing MC/DC GPU run.

## Initial setup (2026-09-14)

- Branch: `feature/nvidia-gpu-testing`.
- Node: `dgx2-1`; allocated GPU: Tesla V100-SXM3-32GB; driver: 580.173.02.
- Temporary workspace: `/tmp/mcdc-nvidia.NeJMqL` (node-local, not durable).
- Modules: `module load cuda/12.2 python/3.11 openmpi/4.1_gcc-12`.
  Open MPI loads GCC 12.2. Python reports 3.11.15.
- Do not select the cluster-default CUDA 13 for the V100.
- Environment: `python -m venv /tmp/mcdc-nvidia.NeJMqL/venv`.
- Harmonize source: https://github.com/CEMeNT-PSAAP/harmonize.git,
  revision `4de55c55e3ec1c116766fb8c1de85b07108cb5a2`.
- Initial compiler package selection: `numba==0.60.0`, `numpy==2.0.2`,
  `llvmlite==0.43.0`; these retain built-in CUDA support. No separate
  `numba-cuda` package is selected for this initial attempt.
- Validation uses an isolated copy of tracked MC/DC sources, with outputs and
  compiler caches outside the working checkout.

Installation results, exact resolved dependencies, and validation outcomes will
be added after the initial attempt.

## Alignment with the recommended MC/DC installation

The source guide (`docs/source/install.rst`, NVIDIA section) specifies CUDA 11.8
and Numba >=0.60.0. The existing `.github/workflows/regression_test-gpu.yml`
uses CUDA 11.8, GCC 10.3, and MPICH 4.0h. Before validation, switched to those
modules to follow that recipe; CUDA 12.2/Open MPI were used only during initial
environment creation and package installation, not for validation.

```sh
module load cuda/11.8 python/3.11 gcc/10.3 mpich/4.0h_gcc-10
source /tmp/mcdc-nvidia.NeJMqL/venv/bin/activate
```

The CUDA module sets `CUDA_HOME`, `CUDA_PATH`, compiler `PATH`, and library paths.
No manual CUDA path override or source patch has been applied.
MC/DC source snapshot: `8e2e6844656116c0ad9e641b09fb3e5c2c4d6e55`.

Installation completed successfully with:

```sh
python -m pip install --no-cache-dir numba==0.60.0 numpy==2.0.2 \
  llvmlite==0.43.0 mpi4py==4.1.2 \
  -e /tmp/mcdc-nvidia.NeJMqL/mcdc \
  -e /tmp/mcdc-nvidia.NeJMqL/harmonize
```

This is the recommended venv + editable pip installation, with explicit versions
to record the compiler baseline. Python 3.11 also follows the installation
guide's recommendation. The MPI wheel must be verified against the loaded MPI.

## First validation findings

- `python -m pip check`: passed (before adding cffi).
- CUDA 11.8 `nvcc -arch=sm_70` successfully compiled a minimal kernel, which
  executed on the V100 and returned the expected value.
- Harmonize import failed with `ModuleNotFoundError: No module named 'cffi'`.
  Its package metadata does not declare this import dependency. Installed with
  `python -m pip install --no-cache-dir cffi`, resolving `cffi==2.1.1` and
  `pycparser==3.0`. No source change was needed for this issue.
- MPICH initialization emitted `ibv_exp_query_device` errors on this node.
  For the single-process validation, use `export UCX_TLS=self,sm` to select
  local transports. This is not a configuration validated for multi-node or
  GPU-aware MPI transfers; revisit it before scaling beyond this test.

- The MPI warnings persisted with `UCX_TLS=self,sm`; this setting did not
  eliminate device enumeration messages. MPI nevertheless initialized.
- Numba 0.60's default ctypes binding crashed in `cuCtxGetDevice` with the
  580-series driver, including when MPI/Harmonize were omitted. This matches
  [Numba issue 10188](https://github.com/numba/numba/issues/10188).
  Installed `cuda-python==11.8.7` (also installs `cuda-bindings==11.8.7`) and
  set `export NUMBA_CUDA_USE_NVIDIA_BINDING=1`. This uses Numba's supported
  NVIDIA bindings backend rather than modifying Numba source.
- With that binding selected, a Numba CUDA kernel compiled and ran correctly
  on the V100, including with MPI and Harmonize imported. MPI identified
  itself as MPICH 4.0.2 (`ch4:ucx`); Harmonize selected CUDA.

```sh
python -m pip install --no-cache-dir cuda-python==11.8.7
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
cd /tmp/mcdc-nvidia.NeJMqL/mcdc/test/regression
python run.py --name=slab_reed --mode=numba --target=gpu
```

The regression command above used the unchanged runner, default event scheduler,
and original answer file. It completed with exit status 1, reporting 0/1 passed.

## Regression result and next investigation

The regression data repository downloaded successfully at revision
`c80cb6e1702f51d15761f558cd20ded5df6338ab`.

Compilation failed before transport or numerical comparison:

```text
numba.core.errors.TypingError: Failed in cuda mode pipeline (step: nopython frontend)
No implementation of function array_from_ptr found for signature:
    array_from_ptr(void*, UniTuple(int64 x 1), class(float64))
TypingError: At least one dimension in the input shape is non-literal
```

MC/DC's `mcdc/code_factory/gpu/program_builder.py:185` builds a captured shape
tuple, then passes it to `harmonize.array_from_ptr` at line 194 (and again at
line 226). Harmonize's CUDA typing implementation in `harmonize/python/array.py`
requires literal integer dimensions; its lowering also uses `.literal_value`.
This identifies the first interface incompatibility in this tested combination.
It does not prove that other Numba/Harmonize versions fail, or that this is the
only remaining compiler issue. A focused compatibility investigation should
precede any source fix; changing the tally tolerance would not address this error.

No MC/DC or Harmonize source patches were applied. The persistent checkout
change made for this task is this Markdown log; all installs, generated code,
test output, and diagnostic scripts are in the temporary workspace.

## Repeat the current attempt on a fresh allocated V100 node

This follows the [MC/DC installation guide](https://mcdc.readthedocs.io/en/latest/install.html)
and existing GPU workflow, with the observed dependency/binding additions above.
Module names are specific to OSU. Run from the MC/DC repository; the source
snapshot below deliberately excludes unrelated local untracked files.

```sh
module load cuda/11.8 python/3.11 gcc/10.3 mpich/4.0h_gcc-10
NVIDIA_WORK=$(mktemp -d /tmp/mcdc-nvidia.XXXXXX)
python -m venv "$NVIDIA_WORK/venv"
source "$NVIDIA_WORK/venv/bin/activate"
mkdir "$NVIDIA_WORK/mcdc"
git archive 8e2e6844656116c0ad9e641b09fb3e5c2c4d6e55 | tar -x -C "$NVIDIA_WORK/mcdc"
git clone https://github.com/CEMeNT-PSAAP/harmonize.git "$NVIDIA_WORK/harmonize"
git -C "$NVIDIA_WORK/harmonize" switch --detach 4de55c55e3ec1c116766fb8c1de85b07108cb5a2
python -m pip install --no-cache-dir numba==0.60.0 numpy==2.0.2 \
  llvmlite==0.43.0 mpi4py==4.1.2 cffi==2.1.1 cuda-python==11.8.7 \
  -e "$NVIDIA_WORK/mcdc" -e "$NVIDIA_WORK/harmonize"
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export UCX_TLS=self,sm
python -m pip check
cd "$NVIDIA_WORK/mcdc/test/regression"
python run.py --name=slab_reed --mode=numba --target=gpu
```

The commands pin the critical compiler/runtime choices; for an exact package
reproduction also constrain the transitive dependencies to the snapshot below.
The unchanged runner clones or updates regression data automatically, so its
data revision can change on future attempts. `slab_reed` uses its checked-in
multigroup input and answer, not the external continuous-energy library.

The successful Python GPU check was a float64 `numba.cuda.jit` add-one kernel
over 256 elements, launched with `[2, 128]`, followed by synchronization and an
exact comparison with the expected host array. Both plain CUDA C++ (`sm_70`)
and Numba actually executed kernels on the allocated V100. This does not
establish MC/DC kernel compilation, multi-GPU operation, or benchmark performance.

## Resolved packages

Python 3.11.15; pip 24.0. Editable installs: MC/DC 0.12.0 and Harmonize 0.0.1
at the source revisions recorded above. Final `pip check` passed.

```text
black==26.5.1
cffi==2.1.1
click==8.5.0
colorama==0.4.6
contourpy==1.3.3
cuda-bindings==11.8.7
cuda-python==11.8.7
cycler==0.12.1
fonttools==4.65.0
h5py==3.16.0
kiwisolver==1.5.1
llvmlite==0.43.0
matplotlib==3.11.2
mpi4py==4.1.2
mpmath==1.3.0
mypy_extensions==1.1.0
numba==0.60.0
numpy==2.0.2
packaging==26.3
pathspec==1.1.1
pillow==12.3.0
platformdirs==4.11.8
pycparser==3.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytokens==0.4.1
scipy==1.17.1
six==1.17.0
sympy==1.14.0
```

## Temporary diagnostic artifacts

Under `/tmp/mcdc-nvidia.NeJMqL/`:

- `install.log`: initial dependency installation.
- `cuda_smoke.py`, `nvcc_smoke.cu`: minimal validation programs.
- `cuda-smoke.log`, `cuda-only.log`: failed ctypes-binding attempts.
- `cuda-nvidia-binding.log`: passing Python CUDA check with MPI/Harmonize.
- `slab-reed-gpu.log`: full failed regression log.

These paths are node-local and may disappear after the allocation ends. The
revisions, dependency snapshot, commands, and essential failure are preserved
here so the temporary environment is not required for reproduction.
