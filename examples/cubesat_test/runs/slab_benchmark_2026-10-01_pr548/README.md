# Electron slab benchmark against PR #548 head 8984ccf1 (2026-10-01)

Rerun of `../slab_benchmark_2026-09-30_pr548` after new commits on PR #548
(settings redesign, low-energy-first swap moved into `sample_ionization`, collision
tallies store the incident particle). Same cases, N = 1000, same Geant4 reference.

Only input change: `simulation.settings.set_transported_particles(["electron"])` removed
(the API was removed; electrons are activated from the source).

## Result
All seven tally outputs (detector current-in, slab back/transmitted current-out, means
and sdevs) are bit-identical to the 09-30 run. See results.md; conclusion unchanged:
1-6 MeV backscatter is 4-20x Geant4, detector entries 0.21x (4 MeV) and 0.58x (6 MeV);
cases below 0.256 MeV agree with Geant4.

Wall time: 1 MeV 9.1 min, 4 MeV 13.7 min, 6 MeV 16.5 min (4 cases in parallel).
