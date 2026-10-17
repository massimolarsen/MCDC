# Problem (A): the fuel rod slab in 3D

The 1D fuel rod (`../fuel_rod`) is H2O | UO2 | H2O along z in [0, 3] cm, with 24 z
cells. Here it is built in 3D: one x-y cell of unit width, reflective on all six
faces. The 3D solver should reproduce the slab results.

| Case | Window | Bins | Data | Source |
|---|---|---|---|---|
| thermal | 1e-4 to 1e-3 eV (scaled by 1e4) | 100 | synthetic target-at-rest H1, O16, U235, U238 | uniform over the slab |
| fast | 3 to 10 MeV | 40 | ENDF/B-VIII.1, 293.6 K, all fission prompt | uniform in the UO2 |

## Runs

**RMC**, for every spatial basis and patch layout:
- **Spatial bases:** `dg1`, `dg2` (discontinuous, degree 1 or 2) and `cg1`, `cg2`
  (continuous trilinear or triquadratic). See
  `mcdc/rmc/writeups/cartesian_3d/spatial_bases.tex`.
- **Patch layouts:** J = 8 (octants) and J = 32 (4 polar × 8 azimuthal).
- **Schedule:** 10 collision-only iterations, then 1 full-residual correction pass.
- **Histories:** 100 per spatial unknown, energy bin and patch per iteration.

**SMC:** 10^7 histories with history statistics, tallying the scalar flux per
(cell, energy bin). Every RMC run is compared on that same quantity: last iterate
plus the tallied correction, with the history error of the pass.

## Running

- **HPC:** `../submit_3d.sh` runs both problems, every basis and layout.
- **By hand:**
  `mpiexec -n 16 python -m mpi4py input.py --mode=numba --no-progress_bar`.
- **Settings:**
  - `FUEL_ROD_3D_CASES`, `FUEL_ROD_3D_RUNS`, `FUEL_ROD_3D_BASES`,
    `FUEL_ROD_3D_LAYOUTS` and `FUEL_ROD_3D_OUTPUT`;
  - `FUEL_ROD_3D_SMOKE=1` runs a small version (G = 10, 3 iterations).

## Outputs

`python plot.py` (via `common3d.plot_problem`) writes:
- `convergence_<case>_<basis>-J<J>.png`:
  - left: ||eps~||/||psi~|| per iteration with the fitted rate and the stall; the
    rms relative error of the scalar flux against SMC per iteration and after the
    pass; SMC's rms relative standard error;
  - right: ||r||_1 per term (volume and faces) in the iterations and the pass.
- `flux_<case>_<basis>-J<J>.png`: the spectrum in the H2O and the UO2, and the
  energy-integrated flux per z cell.
- `compare_<case>.png`: the rms difference to SMC, and FOM_iter and FOM_total
  relative to SMC, per basis and layout.
- `fom.md`, the table of figures of merit and agreement:
  - FOM = 1/(sigma^2 T);
  - sigma is the rms over the bins SMC resolves of the standard error relative to
    the SMC mean;
  - FOM_iter counts the iterations and the pass in T; FOM_total adds the precompute
    and the in-scatter tables.

RMC's first iteration includes numba compilation. SMC's compilation is excluded by
its cold pass.

## Results

Not run yet.
