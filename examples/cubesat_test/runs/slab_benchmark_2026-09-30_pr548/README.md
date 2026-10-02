# Electron slab benchmark against PR #548 (2026-09-30)

Rerun of `../slab_benchmark_2026-09-28` with the head of mcdc-project/mcdc PR #548
("Refactor and debug the Electron Transport Implementation", revision in
mcdc-revision.txt). Same cases, N = 1000 e- per case, same Geant4 reference
(`g4_results.txt` copied from the 09-28 run, 40000 e- per case).

Differences from 09-28:
- `slab_mcdc.py` ported to the upstream-dev API (`mcdc.Simulation`). Geometry,
  materials, source and energy bins unchanged.
- dev has no `surface_mesh`, so the slab x-min (backscatter) and x-max (transmitted)
  faces are separate cell+surface tallies (`slab_back`, `slab_trans`).
- Workaround for a dev bug: a cell-only surface tally reads `cell.surfaces` before the
  cell is compiled, so the detector cell's bounding surfaces are set in the input.
- 10 MeV omitted (cancelled for runtime on 09-28).
- Data: existing `hdf5lib` (EPRDATA14). The PR generator writes the same 16 elastic
  angular energies (0.256 MeV -> 10 MeV gap is in EPRDATA14 itself), so regenerating
  would not change the elastic tables.

## Results
| E (MeV) | Al (cm) | MC/DC det. entries / e- | Geant4 det. entries / e- | MC/DC / G4 | MC/DC backscatter | G4 backscatter |
|---|---|---|---|---|---|---|
| 0.1 | 0.3 | 0 (< 0.001) | 0 | - | 0.144 +- 0.012 | 0.146 +- 0.002 |
| 0.255 | 0.3 | 0 (< 0.001) | 0 | - | 0.159 +- 0.013 | 0.134 +- 0.002 |
| 1 | 0.3 | 0 (< 0.001) | 0 | - | 0.385 +- 0.020 | 0.098 +- 0.002 |
| 4 | 0.3 | 0.197 +- 0.014 | 0.930 +- 0.005 | 0.21 +- 0.02 | 0.486 +- 0.022 | 0.036 +- 0.001 |
| 6 | 0.3 | 0.607 +- 0.025 | 1.038 +- 0.005 | 0.58 +- 0.02 | 0.371 +- 0.019 | 0.021 +- 0.001 |
| 0.1 | 0.003 | 0.484 +- 0.022 | 0.475 +- 0.003 | 1.02 +- 0.05 | 0.144 +- 0.012 | 0.143 +- 0.002 |
| 0.255 | 0.012 | 0.542 +- 0.023 | 0.592 +- 0.004 | 0.92 +- 0.04 | 0.157 +- 0.013 | 0.134 +- 0.002 |

Slab transmission (out of x-max face, per e-): 4 MeV 0.222 vs G4 0.978; 6 MeV 0.677 vs 1.064.

MC/DC wall time (4 cases in parallel, 8-core laptop): 0.1-0.255 MeV 3-4 min,
1 MeV 6.7 min, 4 MeV 13.8 min, 6 MeV 16.6 min (09-28: 21 / 50 / 59 min for 1 / 4 / 6 MeV).

## Conclusion
PR #548 does not fix the 1-6 MeV discrepancy: backscatter and detector entries match the
09-28 results within statistics. Below 0.256 MeV (thin foils) MC/DC still agrees with Geant4.
The PR is ~3-3.5x faster at 1-6 MeV.

## Reproduce
`run_mcdc.sh` (uses a `git archive` of the PR head on PYTHONPATH), then
`python analyze.py` in this directory.
