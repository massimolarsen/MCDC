# 3D RMC verification: RMC vs standard Monte Carlo vs analytic

Verification of the 3D Cartesian RMC solver (`mcdc.rmc.driver3d.run3d`) on three small
problems. Each is solved three ways, and all three appear on the same plot:
- RMC with the piecewise-constant trial space and collision-only iterations;
- standard Monte Carlo (SMC) with MC/DC on the same model, 10^6 histories;
- an analytic reference, where one exists.

The math is in `mcdc/rmc/writeups/cartesian_3d/cartesian_3d.tex`.

```
python input.py --mode=numba --no-progress_bar      # or mpiexec -n 4 ...; writes output.h5
python plot.py                                      # the figures below
```

## Convergence rates

`convergence_study.py` measures the L-infinity error against the analytic solution for
RMC and SMC, as functions of iterations and of particle histories. It writes
`convergence.h5`, and `plot_convergence.py` draws the figures:

```
mpiexec -n 4 python convergence_study.py --mode=numba --no-progress_bar
python plot_convergence.py
```

- **3D infinite medium:** A = 1, sigma_s / sigma_t = 1/3, isotropic source at E0 = 101,
  cutoff 1, in a reflective box. psi~ is isotropic with no face jumps, so the
  collision-only iteration converges **exponentially**, to a floor set by G (the binned
  fixed point).
  - RMC: 10 and 100 P/B/I x G = 50, 100, 200.
- **3D pure-absorber box** (the `absorber` problem below): the residual is exact, but
  the constant trial space cannot represent the in-cell and face shape of the
  streaming residual.
  - After one iteration psi~ sits on a noise plateau about the bin averages. The mean of
    the iterates then converges like 1/sqrt(N), with a smaller constant than SMC.
  - RMC: 100 and 1000 histories per (cell, patch) bin.
- **SMC:** 10^3 to 10^7 histories, plus runs at the same cumulative histories as one
  RMC case after 1, 2, 3, 5, 8, 12, 20 and 30 iterations.

| Figure | Shows |
|---|---|
| `convergence_3d.png` | One panel per problem (infinite medium, absorber box): L-infinity error against particle histories, for the RMC run and SMC at the same cumulative histories, with the N^-1/2 rate fitted to the SMC runs. For the absorber, also the mean of the RMC iterates. |
| `flux_comparison.png` | Infinite medium, G = 100: phi(E), analytic, RMC (10 P/B/I, 20 iterations), and SMC with 10^5 and 10^6 particles |

Results (4 ranks, 135 s in total):
- **Infinite medium** (10,100 histories per iteration):
  - RMC's L-infinity error falls from 1.0e-2 at a rate of rho = 0.51 per iteration,
    and reaches its bin-shape floor of 1.5e-6 by about iteration 13.
  - SMC with the same cumulative histories reaches 8.8e-4 at 3.0e5 histories,
    following N^-1/2. That is 600 times RMC's error.
- **Absorber box** (64,000 histories per iteration):
  - The RMC iterate stays at about 1e-2. The mean of the iterates falls roughly as
    N^-1/2, to 2.6e-3 at 1.9e6 histories.
  - SMC reaches 1.4e-3 at the same histories, so here **SMC is more efficient**.
  - With the constant trial space, the streaming residual (the face jumps) never
    shrinks, so collision-only RMC gains nothing on a pure streaming problem. This
    is the 1D finding again: a piecewise-constant spatial trial space made RMC less
    efficient than SMC (`continuous_linear_space.tex`). Linear spatial and angular
    bases are the remedy to test in 3D.

## Problems

| Problem | Setup | Reference |
|---|---|---|
| `infinite` | A = 1, sigma_s = sigma_c = 1 b, monoenergetic isotropic source at 101 eV, reflective box (an infinite medium). G = 41, octant patches. | Analytic flux phi(E) = c S0 / (Sigma_t E0) (E0 / E)^c, as bin averages |
| `absorber` | Pure absorber, Sigma_t = 1 /cm, in [0, 2]^3 on a 2 x 2 x 2 mesh with octant patches. Source in the corner cell [0, 1]^3. Reflective on x = 0, vacuum elsewhere. | Exact ray integral psi = q (e^{-Sigma s1} - e^{-Sigma s2}) / Sigma over the part of the backward ray inside the source (reflection unfolded), averaged over each cell and patch with scrambled Sobol points |
| `slab` | The two-material slab of the 1D tests, built in 3D: one cell in x and y, reflective there. | The slab solver (`mcdc.rmc.run`) and SMC; there is no analytic solution |

## What each problem tests

- **`absorber`: the 3D streaming residual.** It exercises the face jumps on x, y and z
  faces, vacuum and reflective boundaries, the octant patches, the direction samplers
  and the tally normalization.
  - Without scattering the residual is exact, so the iteration is unbiased: RMC must
    agree with the analytic reference and with SMC within statistics, in every
    (cell, patch) bin.
  - 37 of the 64 bins cannot be reached (for example, directions pointing back toward
    the source from the far corner). They must be exactly zero in all three.
- **`infinite`: the patch-to-patch scattering transfer and the energy kernels.**
  - psi~ is isotropic and the face residual vanishes, so the iteration converges
    geometrically to the binned fixed point.
  - That fixed point carries the in-bin bias of a constant energy basis, which the
    analytic comparison shows. It is a few 1e-3 at G = 40.
- **`slab`: consistency with the slab solver.**
  - For a slab-symmetric problem, the 3D constant-basis system has the same fixed point
    as the slab solver: the same polar bins, isotropic in phi, and no x or y jumps.
  - Their plateau means must agree. Both carry the same collision-only bias relative
    to SMC.

RMC is shown as the mean of the iterates after the first two, with its standard error,
where the iteration has reached its noise plateau (absorber, slab); for the infinite
medium it is the last iterate.

## Figures

- `verification_infinite.png`: E phi(E): analytic, SMC (2 sigma) and RMC; below, the
  difference relative to the analytic flux.
- `verification_absorber.png`: psi in each reachable (cell, patch) bin, sorted by the
  analytic value: analytic, SMC and RMC, and their differences relative to the
  analytic value; right, the scalar flux per cell.
- `verification_slab.png`: energy-integrated flux per z cell (SMC, slab RMC, 3D RMC),
  and the spectra in the first and last cells relative to SMC.
- `convergence.png`: ||eps~|| per iteration.

## Results

Run on 4 ranks, 2026-10-16. SMC used 10^6 histories in each problem; RMC ran 10
collision-only iterations, and the plateau mean is over iterations 3-10.

| Problem | Comparison | Result |
|---|---|---|
| `infinite` | RMC vs analytic | largest relative difference 5e-4 over the 40 collided bins (the binned fixed point's in-bin bias) |
| `infinite` | SMC vs analytic | rms z 0.80 |
| `absorber` | RMC vs analytic, 27 reachable bins | rms z 1.24 |
| `absorber` | SMC vs analytic, 27 reachable bins | rms z 1.13 |
| `absorber` | per cell | all 8 cells agree (right panel of the figure) |
| `slab` | 3D RMC vs SMC, 36 (cell, energy) bins | rms z 1.15 |
| `slab` | slab RMC vs SMC | rms z 1.51 |
| `slab` | 3D RMC vs slab RMC | rms z 1.13 |

- **`infinite`: ||eps~|| falls geometrically,** by about 10x per iteration, to 2e-8 of its first value.
- **`absorber` and `slab`: the iteration sits at the noise of one iteration** after the first step, about 2e-2 of the first ||eps~||.
- **`slab`: the far cell is noisy.** Few histories reach cell 3, so RMC's standard error there is 10-46% per energy bin.
- **Unreachable bins (fixed 2026-10-16, not rerun):**
  - In the runs above, RMC had values of about 1e-4 in 5 of the 37 unreachable absorber bins in its last two iterates.
  - Cause: MC/DC's tally filter assigns a direction within 1e-5 of a polar or azimuthal bin edge to the lower bin (`COINCIDENCE_TOLERANCE_DIRECTION`). Residual particles drawn in an upward patch with 0 <= mu < 1e-5 were scored in the downward patch above the source layer.
  - Fix: `mesh3d` now keeps sampled directions 2e-5 inside their patch edges, a distortion of about 1e-5 of the angular distribution.