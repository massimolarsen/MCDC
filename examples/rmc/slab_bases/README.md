# Two-material slab: spatial and angular trial spaces of RMC

## What this is

A small 1D problem with streaming, two materials and leakage through vacuum
boundaries. It is solved with RMC on four trial spaces and compared with one standard
Monte Carlo (SMC) reference on the same MC/DC model:

|  | angle: piecewise constant | angle: linear discontinuous |
|---|---|---|
| **z: piecewise constant** | `z-constant_mu-constant` (the original scheme) | `z-constant_mu-linear` |
| **z: continuous linear** | `z-linear_mu-constant` | `z-linear_mu-linear` |

The energy trial space is piecewise constant in all four. The math is in
`mcdc/rmc/writeups/continuous_linear_space.tex` and
`mcdc/rmc/writeups/linear_discontinuous_angle.tex`.

## The problem

- **Geometry:** a slab from z = 0 to 4 cm, vacuum at both ends and reflective in x and y
  (one-dimensional).
- **Materials:** a synthetic nuclide with A = 1 and sigma_s = sigma_c = 1 b, constant in
  energy, isotropic scattering in the COM frame.
  - **a**, density 1.0 (Sigma_t = 2 /cm), in [0, 2]: each 1 cm cell is 2 mean free paths.
  - **b**, density 0.4, in [2, 4].
- **Source:** isotropic, uniform in z over [0, 1] (cell 0 only), uniform in energy over
  [9.9, 10] eV.
- **Trial space:** 4 cells of 1 cm; 8 log-spaced energy bins over [1, 9.9] eV plus the
  source bin; 2 polar bins, mu in [-1, 0] and [0, 1].
- **RMC:** 8 collision-only iterations (phase 1), then 8 averaged full-residual
  correction passes (phase 2), with 200 histories per bin per iteration.
- **SMC:** 10^6 histories. It tallies, on the same cells, energy and polar bins, the
  flux and its two slope moments: in z within the cell, and in the polar cosine within
  the polar bin.

## What it tests

1. **Unbiasedness of every basis combination.** The correction passes sample the full
   residual, so RMC + corrections must agree with SMC within statistics. A bug in a
   residual term, a sampler, the transfer moments, the tallies or the projection shows
   up as bias. The comparison uses the quantity each spatial projection preserves:
   - **piecewise-constant z:** the cell averages;
   - **continuous linear z:** the hat-function moments of the interior nodes,
     int phi h_i dz. The continuous-linear projection is global and does **not**
     preserve cell averages, so comparing those with SMC cell tallies would show a
     difference that is not an error. SMC's hat moments come from its flux and z-slope
     cell tallies.
2. **The phase-1 plateau.** With a piecewise-constant z the face residuals (pairs of
   equal and opposite face sources) keep the collision-only iteration at a noise
   plateau. A continuous linear z removes them. With only two polar bins, though, the
   in-bin shape in mu of the streaming term, mu dpsi/dz, is about as large as the face
   residual it replaces. The linear angular basis is meant to absorb that term.
   `convergence.png` shows ||eps~|| per iteration for all four.
3. **The angular slopes themselves.** For the linear angular basis, RMC's
   polar-cosine slope coefficients are compared with SMC's (`angular_slopes.png`).

## Figures

- `convergence.png`: ||eps~|| of the collision-only iterations for the four bases.
- `flux_profile.png`: energy-integrated scalar flux in z. SMC is shown as cell averages
  with its in-cell slope; the continuous-linear runs are drawn through their nodal
  values.
- `zscores.png`: z = (RMC + corrections - SMC) / sqrt(sigma_SMC^2 + sigma_RMC^2) per
  energy bin, with sigma_RMC from the spread of the correction passes. The SMC error
  uses the fully correlated sum over the polar bins, an upper bound.
- `angular_slopes.png`: polar-cosine slope coefficients (constant z, linear angle)
  against SMC's, with SMC's 2-sigma bars.

**Pass criterion:** rms z of about 1 for each basis, with no |z| above about 4.

## Results (from the unit tests, 2026-10-04)

These numbers come from `test_slab_against_smc` and `test_slab_against_smc_linear_z`
in `test/unit/rmc/test_driver.py`, each parametrized over the angular basis. The tests
solve this exact problem, with an SMC reference of 2 x 10^5 histories instead of 10^6.
`input.py` has not been run, so `output.h5` and the figures of `plot.py` do not exist
yet. `plot_test_results.py` plots the convergence the tests printed
(`convergence_tests.png`).

**Unbiasedness: all four pass.** For every basis, RMC + corrections agrees with SMC
within the combined error (rms z < 2 and every |z| < 5): cell averages for
piecewise-constant z, interior hat moments for continuous linear z.

**Phase 1 (||eps~||, collision-only iterations 1-6):**

| Basis | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| z constant, mu constant | 2.75 | 0.026 | 0.051 | 0.020 | 0.088 | 0.048 |
| z constant, mu linear | 2.81 | 0.112 | 0.060 | 0.074 | 0.119 | 0.086 |
| z linear, mu constant | 3.83 | 0.054 | 0.064 | 0.080 | 0.049 | 0.045 |
| z linear, mu linear | 3.85 | 0.097 | 0.130 | 0.093 | 0.084 | 0.086 |

All four drop about 50x in the first iteration and then sit at a plateau, so neither
linear basis lowers it at 200 histories per bin. The linear angular basis plateaus
slightly higher (about 0.08-0.13, against 0.02-0.09).

**Why:**
- The plateau here is set by the noise of one iteration, not by the trial space.
  Each slope coefficient is tallied as 3 x (track length x the Legendre variable),
  with about sqrt(3) times the relative noise of an average.
- A linear basis adds those noisier unknowns and raises the iteration's noise gain.
  This is the same effect as the linear energy basis (`mcdc/rmc/NOTES.md` section 7).
- The norm includes the slope coefficients (it is the function norm), so the linear
  bases also carry their extra noise in the plotted quantity.
- To see the trial-space gain, the iteration needs enough histories per bin that its
  noise floor sits below the residual each basis removes: several times 200 per bin.
  A finer z mesh, or more polar bins, also lowers the trial-space error at which the
  noise has to be compared.

## Running

```
python plot_test_results.py                 # convergence_tests.png from the test results
mpiexec -n 4 python input.py --mode=numba   # writes output.h5 (not yet run)
python plot.py                              # figures + rms z per basis
```
