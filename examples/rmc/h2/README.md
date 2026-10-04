# H-2, infinite medium (0D): end-to-end test of the inverted N-body kernel

## What this is

An infinite medium of pure H-2 (0.05 atoms/b-cm, ENDF/B-VIII.1 at 293.6 K, from the
regenerated library `hdf5lib_mt5fix`). A uniform, isotropic source fills 1-20 MeV
(G = 100 bins), and neutrons leaving the window are killed. As in `../c12`, the window
is solved with RMC (10 collision-only iterations, then averaged correction passes) and
with SMC (10^6 histories) on the same MC/DC model.

## What it tests

| Reaction | Inverted kernel under test |
|---|---|
| MT-16 (n,2n), from 3.34 MeV | **N-body phase-space distribution (ENDF Law 6 / ACE Law 66)**: P(E_cm) proportional to sqrt(E_cm)(E_max - E_cm)^(3n/2 - 4) with n = 3, isotropic in the COM frame, two neutrons per reaction |
| MT-2 elastic | anisotropic COM elastic for A = 2, the largest energy transfer of any nuclide after H-1 |

H-2 MT-16 is the only N-body reaction in the library, and it is present only in the
regenerated library. This run also checks two fixes end to end, in addition to the
per-law chi-squared test:

- **E_max scaling**: the data generator now stores the particle count n = 3 and the
  total mass ratio A_p, and the sampler scales the tabulated reduced variable by
  E_max(E) = (A_p - 1)/A_p (A/(A+1) E + Q). Before the fix it returned T x 1 MeV
  whatever the incident energy.
- **The yield of 2**: counted in both the moments and the correction kernel.

The N-body density has square-root behaviour at both ends of its support. Its bin
quadrature therefore converges more slowly than for the other laws (see
`mcdc/rmc/writeups/n_body.tex`), which this run exercises in every bin above 3.34 MeV.

## How to read the results

`plot.py` writes `flux_fast.png`, with the same three panels as `../c12`: flux; ratio to
SMC with the ±2-sigma band; and per-bin z of RMC + corrections against SMC.

**Pass criterion: rms z of about 1, with no structured excursion of |z| > 2 above
3.34 MeV**, where (n,2n) turns on. This checks consistency with MC/DC's sampler; a
sampler bug would appear identically in both RMC and SMC.

## Running

```
mpiexec -n 4 python input.py --mode=numba   # writes output.h5
python plot.py                              # figure + rms z
```
