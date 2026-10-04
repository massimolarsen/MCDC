# C-12, infinite medium (0D): end-to-end test of the inverted kernels

## What this is

An infinite medium of pure C-12 (0.1 atoms/b-cm, ENDF/B-VIII.1 at 293.6 K, from the
regenerated library `hdf5lib_mt5fix`). A uniform, isotropic source fills each energy
window, and neutrons leaving the window are killed. Each window is solved twice on the
same MC/DC model:

- **RMC** (`mcdc.rmc.run`): 10 collision-only iterations (phase 1), then a few averaged
  full-residual correction passes (phase 2);
- **SMC** (`simulation.run`): standard Monte Carlo with 10^6 histories, tallied on the
  same energy and angle bins.

## What it tests

Each inverted scattering law was already checked on its own: MC/DC's sampler against
the inverted kernel, with a chi-squared test per law (`mcdc/rmc/writeups/`). This run
checks the same kernels **inside a full RMC solve**:

- **The transfer moments** (bin-integrated kernels) set the phase-1 fixed point.
- **The correction passes** evaluate the kernels pointwise, in the in-scatter density of
  the scattering correction.

The correction passes are unbiased, so **RMC + corrections must agree with SMC within
statistics** if every kernel in the window is right.

| Window | Reactions exercised | Inverted kernel under test |
|---|---|---|
| **fast**, 1-30 MeV, G = 100 | MT-2 elastic | anisotropic COM elastic (tabulated angles, relativistic kinematics) |
| | MT-51 and up, from 4.81 MeV | level inelastic (Law 3) with tabulated angles |
| | MT-91 from 7.89 MeV, MT-28 from 17.3 MeV | **evaporation spectrum (Law 9)**; C-12 is the only nuclide in the 293.6 K library that uses it |
| | MT-5 from 20 MeV | Kalbach-Mann (Law 44) with a **tabulated yield** (multiplicity table, N = floor(nu + xi)); needs the regenerated library |
| **thermal**, 0.01-10 eV, G = 60 | MT-2 elastic below 400 kT = 10.1 eV | **free-gas kernel**: analytic S(alpha, beta) with upscatter, the D(a) normalization, and the nuclide temperature (the 293.6 K fix) |

Laws that cannot be tested on real data: no nuclide in the 293.6 K library uses the
Maxwellian spectrum (Law 7) or several spectra with a tabulated yield. Both were
verified only with the synthetic cases in the write-ups.

## How to read the results

`plot.py` writes `flux_fast.png` and `flux_thermal.png`:

- **top**: flux from SMC, the RMC phase-1 fixed point, and RMC + corrections;
- **middle**: ratio to SMC, with the SMC ±2-sigma band. The fixed point may sit
  outside the band (the piecewise-constant trial-space bias in bins where Sigma_t
  varies). RMC + corrections should not;
- **bottom**: per-bin z = (RMC + corrections - SMC) / sqrt(sigma_SMC^2 + sigma_corr^2).

**Pass criterion: rms z of about 1, with no structured run of |z| > 2 in an energy
range.** A run of large z where one reaction turns on (4.81, 7.89, 17.3 or 20 MeV, or
the thermal range) points at that reaction's kernel.

This checks consistency with MC/DC's samplers, not the physics itself. A sampler bug
would appear identically in both RMC and SMC.

## Things found while setting it up

- C-12's elastic angular grid repeats 20.0 MeV (a data discontinuity). The inverted
  kernels divided by the zero-width interval when a quadrature node or support
  evaluation landed exactly on it. Fixed with `grid_fraction` in
  `mcdc/rmc/evaluate.py`, which takes the one-sided limit. MC/DC's own sampler has the
  same zero-width interval, but a sampled energy never hits 20.0 MeV exactly.
- The thermal window starts at 0.01 eV, not lower. MC/DC scores a particle within
  1e-5 eV above a bin edge in the bin below, which would misplace a large share of the
  flux in very narrow low-energy bins.

## Running

```
mpiexec -n 4 python input.py --mode=numba   # writes output.h5
python plot.py                              # figures + rms z per window
```

## Results (phase 1 only, 2026-10-03)

| Window | Result |
|---|---|
| fast, 1-30 MeV | Pass in flat-Sigma_t bins (rms z = 1.25 with the fully correlated SMC error), no structure at the 4.81, 7.89, 17.3 or 20 MeV thresholds. Resonance bins (2.08, 6.3 MeV, ...) are low by 3-12%: the known phase-1 bias of the constant basis. **Open:** the top bin (29-30 MeV) is +4.4% (z = +6) in a flat bin. MT-5 cannot feed it, and the top bins of H-2 and O-16 are clean. |
| thermal, 0.01-10 eV | The 0D test is not decisive: with c = 0.9993 the transport inverse amplifies the phase-1 operator mismatch by ~1/(1-c) = 1400. The iteration has not converged after 10 iterations (eps at 5% of its start), and RMC's spectrum is a few percent hotter than SMC's. The **free-gas kernel itself was checked directly and is correct**: its transfer moments match histograms of MC/DC's free-gas sampler (2e5 collisions per incident bin, 0.03-3.5 eV) with per-bin rms z of 0.8-1.1, upscatter fractions equal within 3e-4, totals within 1e-4, and particle conservation within 2e-7 in the window. A decisive 0D thermal test needs an absorber (e.g. B-10) to bring c down to ~0.5-0.9. |
