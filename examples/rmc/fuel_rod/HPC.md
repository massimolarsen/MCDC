# Running the fuel rod problems on the HPC

Runbook for the 1D fuel rod problems ([D] Sec. 4.4.2, see `input.py`), on the
piecewise-constant trial space (the dissertation scheme) and on the all-linear trial
space (linear in energy, polar cosine and z).

## What to run

| Step | Script | Writes | Contents |
|---|---|---|---|
| 1 | `input.py` | `output.h5` | RMC constant basis + SMC reference, thermal and fast |
| 2 | `linear.py` | `output_linear.h5` | RMC all-linear (`energy_basis`, `spatial_basis`, `angular_basis` = `"linear"`), thermal and fast |
| 3 | `plot.py` | `flux_space_<case>.png`, `flux_energy_<case>.png` | dissertation Figs. 4.5-4.8 |
| 4 | `plot_linear.py` | `flux_space_linear_<case>.png`, `flux_energy_linear_<case>.png`, `convergence_linear.png` | constant vs linear vs SMC |
| 5 | `../convergence/plot.py` | `../convergence/epsilon_*.png` | Figs. 4.9-4.10, with the linear runs overlaid |

Steps 1 and 2 are independent and can run as separate jobs. Plotting (3-5) is quick and
can be done locally after copying the `.h5` files back.

`linear.py` and `plot_linear.py` (added 2026-10-04) have not been run end to end yet.

## Prerequisites

- **Branch:** `feature/RMC`. The RMC driver is in `mcdc/rmc/`.
- **Python:** MC/DC installed from this checkout, with `numba`, `mpi4py`, `h5py`,
  `matplotlib`, and an MPI launcher.
- **Nuclear data:** `MCDC_LIB` must point at the library regenerated with the MT-5 fix
  (locally `MCDC/hdf5lib_mt5fix/`; copy it to the cluster if the cluster copy is older).
  - Needs `H1`, `O16`, `U235`, `U238` at `0.1K` and `293.6K`.
  - The older `hdf5lib/` fails on this branch with
    `KeyError: ... 'mass_number' doesn't exist`.
- The thermal case writes synthetic nuclides to `data/` from the 0.1 K files on every
  run; `cache/` and `data/` are gitignored and created as needed.

## Cost (measured 2026-10-04)

The expensive part is the transfer-moment precompute, done once per nuclide and cached
in `cache/`. Single process, all-linear bases, Apple M-series laptop:

| Case | H-1 | O-16 | U-235 | U-238 |
|---|---|---|---|---|
| fast (40 bins) | 25 s | 30 s | > 28 min (stopped, unfinished) | not reached |
| thermal (100 bins) | 2 min 45 s | ~2 min | > 24 min (stopped, unfinished) | not reached |

- The precompute and the transport are both MPI-parallel. The precompute deals the
  table rows round-robin over ranks (`mcdc/rmc/kernel.py`), so use a full node.
- The constant and linear runs need separate moment tables: the cache key includes the
  energy order and the angular basis.
- Earlier local runs of `input.py` were killed during the U-235 precompute, so there are
  no complete constant-basis results yet; `output.h5` must be produced here.
- Histories per iteration: thermal 10^5 (constant) and 4 x 10^5 (linear); fast 4 x 10^5
  and 1.6 x 10^6; 10 iterations each. SMC: 10^7 histories per case; the fast SMC uses
  full ENDF physics with fission.
- Memory is small (moment tables of a few MB per nuclide).

## Settings

- `FUEL_ROD_CASES=thermal` or `fast` runs one case (default both).
- `FUEL_ROD_OUTPUT` names the output file (defaults `output.h5`, `output_linear.h5`).
- `FUEL_ROD_LINEAR_FACTOR` (default 4) multiplies the dissertation's histories per bin
  per iteration for the linear runs. Slope coefficients are tallied with ~sqrt(3) more
  noise, and with 8 coefficients per bin the iteration can diverge at the constant-basis
  counts (`mcdc/rmc/NOTES.md`, sections 7, 9, 10). Set it to 1 for the dissertation's
  exact counts.

`common.Output` truncates its file when a run starts. If cases run as separate jobs,
give each its own `FUEL_ROD_OUTPUT` and merge afterwards (below).

## Constant vs linear vs SMC on the OSU HPC: submit.sh

`./submit.sh` (on the submit node, in this directory) submits each missing piece as its
own job: 16 ranks on `share`, 12 h limit, using the environment of
`../efficiency/env.sh`. Pieces whose output file already holds the result are skipped.

| Job | Runs | Output |
|---|---|---|
| `fuel-constant-thermal` | `input.py`, thermal, RMC + SMC | `output.h5` |
| `fuel-constant-fast` | `input.py`, fast, RMC | `output_fast.h5` |
| `fuel-smc-fast` | `input.py`, fast, SMC (`FUEL_ROD_RUNS=smc`) | `output_smc_fast.h5` |
| `fuel-linear-thermal` | `linear.py`, thermal | `output_linear_thermal.h5` |
| `fuel-linear-fast` | `linear.py`, fast | `output_linear_fast.h5` |

When they end, a 1-core `fuel-merge` job runs `merge.py`. It merges the pieces into
`output.h5` and `output_linear.h5`; the first merge keeps the original `output.h5` as
`output_constant_thermal.h5`. It then runs `plot.py` and `plot_linear.py`. Logs go to
`logs/`. `FUEL_ROD_RANKS` and `FUEL_ROD_LIMIT` override the ranks and the time limit.

## Example SLURM job

```bash
#!/bin/bash -l
#SBATCH --job-name=fuel-rod-rmc
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=fuel-rod-%j.out

set -eo pipefail
# Activate the Python environment with MC/DC (feature/RMC) installed, then:
export MCDC_LIB=/path/to/hdf5lib_mt5fix
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /path/to/MCDC/examples/rmc/fuel_rod
git rev-parse HEAD > mcdc-revision.txt

mpirun -n "$SLURM_NTASKS" python -u input.py --mode=numba > run.log 2>&1
mpirun -n "$SLURM_NTASKS" python -u linear.py --mode=numba > run_linear.log 2>&1
```

To split by case, submit one job per case with, e.g.,
`--export=ALL,FUEL_ROD_CASES=fast,FUEL_ROD_OUTPUT=output_fast.h5` (and
`output_linear_fast.h5` for `linear.py`), then merge into the default names:

```python
import h5py
for target, parts in (("output.h5", ("output_thermal.h5", "output_fast.h5")),
                      ("output_linear.h5", ("output_linear_thermal.h5", "output_linear_fast.h5"))):
    with h5py.File(target, "w") as out:
        for part in parts:
            with h5py.File(part, "r") as f:
                for kind in f:  # "rmc", "smc"
                    for name in f[kind]:
                        f.copy(f[kind][name], out.require_group(kind), name)
```

## Checks after a run

- `output.h5` has `rmc/thermal`, `rmc/fast`, `smc/thermal`, `smc/fast`;
  `output_linear.h5` has `rmc/thermal`, `rmc/fast` with attribute
  `spatial_basis = "linear"`.
- `epsilon_norm` (10 values per run) should fall by orders of magnitude in the first
  iterations and then level off at a noise plateau. Growth instead means the linear
  iteration is unstable: rerun with a larger `FUEL_ROD_LINEAR_FACTOR`.
- With the linear z basis, `psi` holds nodal values on the 25 z edges and
  `psi_cell`/`scalar_flux` the cell averages. The continuous-linear projection does not
  preserve cell averages, so their difference from SMC's cell tallies is larger than
  noise without being an error (`mcdc/rmc/NOTES.md`, section 9).
