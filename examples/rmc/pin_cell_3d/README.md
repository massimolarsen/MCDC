# Problem (B): a C5G7-style fuel pin in 3D

## Geometry

- Pitch 1.26 cm, with a square UO2 region of side 0.957 cm at the centre and no
  cladding.
- 4 cm tall.
- The trial mesh follows the materials: 3 x 3 in x-y (water 0.1515 | fuel 0.957 |
  water 0.1515 cm) and 4 cells of 1 cm in z, 36 cells in all.

| Variant | Boundaries |
|---|---|
| lattice | reflective in x and y, vacuum in z |
| isolated | vacuum on all six faces |

## Cases, runs and outputs

As in `../fuel_rod_3d/README.md`:
- thermal: 100 bins, source over the whole pin;
- fast: 40 bins, source in the UO2;
- RMC for every basis and layout, SMC 10^7 histories, and the same outputs.

Labels are `<case>-<variant>/<basis>-J<J>`.

| Basis | Unknowns per energy bin and patch | Thermal histories / iteration (J = 8) |
|---|---|---|
| dg1 | 144 | 1.15 x 10^7 |
| dg2 | 360 | 2.9 x 10^7 |
| cg1 | 80 nodes | 6.4 x 10^6 |
| cg2 | 441 nodes | 3.5 x 10^7 |

## Running

- **HPC:** `../submit_3d.sh`. The pin jobs wait for the rod jobs and reuse their
  transfer moments.
- **By hand:**
  `mpiexec -n 16 python -m mpi4py input.py --mode=numba --no-progress_bar`.
- **Settings:**
  - `PIN3D_CASES`, `PIN3D_VARIANTS`, `PIN3D_RUNS`, `PIN3D_BASES`, `PIN3D_LAYOUTS` and
    `PIN3D_OUTPUT`;
  - `PIN3D_SMOKE=1` runs a small version (G = 10, 3 iterations).

## Results

Not run at full size yet.

**Constant basis, local runs** (G = 10, thermal, lattice; `mcdc/rmc/NOTES.md`,
section 13):
- at 5.2 histories per bin the iteration does not contract;
- at 50 or more per bin it plateaus after one iteration, with a structural residual
  of ‖r‖_1 ≈ 1.1 × the source.

This is why the spatial bases and the J = 32 layout are being swept.
