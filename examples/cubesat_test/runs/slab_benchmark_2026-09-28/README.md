# Electron slab benchmark, doubled statistics (2026-09-28)

Rerun of `../slab_benchmark_2026-09-24` with 2x particles: MC/DC 1000 e- per case,
Geant4 40000 e- per case (new seeds 201-208). Geometry, materials, physics settings
and scripts are identical to the 2026-09-24 run; see that README for the full setup
and the cross-section equivalence check (xs_compare.txt).

- MC/DC revision: see mcdc-revision.txt. Electron physics and data are the same as the
  2026-09-24 run (the elastic-sampler change that was uncommitted then is committed now;
  hdf5lib electron files unchanged).
- The 10 MeV MC/DC case was cancelled at ~31% (too slow); only its partial log is kept.
  Geant4 10 MeV result is in geant4/g4_results.txt.

## Results
| E (MeV) | Al (cm) | MC/DC det. entries / e- | Geant4 det. entries / e- | MC/DC / G4 | MC/DC backscatter | G4 backscatter |
|---|---|---|---|---|---|---|
| 0.1 | 0.3 | 0 (< 0.001) | 0 | - | 0.172 +- 0.013 | 0.146 +- 0.002 |
| 0.255 | 0.3 | 0 (< 0.001) | 0 | - | 0.155 +- 0.012 | 0.134 +- 0.002 |
| 1 | 0.3 | 0 (< 0.001) | 0 | - | 0.372 +- 0.019 | 0.098 +- 0.002 |
| 4 | 0.3 | 0.204 +- 0.014 | 0.930 +- 0.005 | 0.22 +- 0.02 | 0.478 +- 0.022 | 0.036 +- 0.001 |
| 6 | 0.3 | 0.551 +- 0.024 | 1.038 +- 0.005 | 0.53 +- 0.02 | 0.374 +- 0.019 | 0.021 +- 0.001 |
| 10 | 0.3 | (cancelled) | 1.057 +- 0.005 | - | - | 0.014 +- 0.001 |
| 0.1 | 0.003 | 0.497 +- 0.022 | 0.475 +- 0.003 | 1.05 +- 0.05 | 0.171 +- 0.013 | 0.143 +- 0.002 |
| 0.255 | 0.012 | 0.570 +- 0.024 | 0.592 +- 0.004 | 0.96 +- 0.04 | 0.155 +- 0.012 | 0.134 +- 0.002 |

1-sigma Poisson uncertainties. MC/DC wall time (4 cases in parallel, 8-core laptop):
0.1-0.255 MeV 5.5-9.5 min, 1 MeV 21 min, 4 MeV 50 min, 6 MeV 59 min.
