# Electron slab benchmark: MC/DC vs Geant4 (2026-09-24)

Tests whether MC/DC electron transport matches Geant4 below and above the
0.256 MeV gap in the EPRDATA14 elastic angular tables (tables jump 0.256 -> 10 MeV).

## Setup (identical in both codes)
- Pencil beam, normal incidence, starts at x = -0.5 cm heading +x.
- Slab: Al6061, x in [0, T], y,z in [-10, 10] cm. T = 0.3 cm (CubeSat shear panel).
  Atom densities (atoms/b-cm): Al 0.059057360465045, Mg 0.00081349602416576,
  Si 0.00046494828605733 (same as inputE_SUBMIT.py). Geant4 material built from
  these atom densities (verified to 9 digits in g4_calc.txt; rho = 2.7005 g/cc).
- Detector: Si (0.050132618436459 atoms/b-cm, rho = 2.3380 g/cc), ADCS-SV sized
  4.5 x 4.5 x 1.8 cm, 0.7 cm behind the slab, on axis.
- Void elsewhere, vacuum outer boundary x in [-1, T+3.5], y,z in [-11, 11].
- Electrons only: Geant4 kills photons (MC/DC does not transport them).
- Geant4 11.4.0, G4EmStandardPhysics_option4, 1 um production cut, 20000 e- per case.
- MC/DC at the local revision (numba mode), 500 e- per case.
- Scored: electron entries into the detector (all electrons incl. secondaries,
  counting every entry) and electrons leaving the slab's front face (backscatter).
- Thin-foil supplement (T = 0.003 cm at 0.1 MeV, 0.012 cm at 0.255 MeV) because
  electrons below 0.256 MeV cannot cross 0.3 cm Al in either code.

## Physics equivalence (xs_compare.txt, MC/DC data / Geant4)
- Radiative stopping power: 0.99-1.00.
- Collision stopping power: 0.96-0.99 for 50 keV-10 MeV (0.91 at 10 keV).
- Elastic transport xs in the EPRDATA14 data: 0.89-0.93 at all energies (model difference).
- Elastic transport xs as MC/DC samples it: ~1.0 below 0.256 MeV, 1.7-10x inside
  0.256-10 MeV (linear interpolation across the angular-table gap), 0.90 at 10 MeV.

## Results (results.md)
| E (MeV) | Al (cm) | MC/DC det. entries / e- | Geant4 det. entries / e- | MC/DC / G4 | MC/DC backscatter | G4 backscatter |
|---|---|---|---|---|---|---|
| 0.1 | 0.3 | 0 (< 0.002) | 0 | - | 0.162 +- 0.018 | 0.146 +- 0.003 |
| 0.255 | 0.3 | 0 (< 0.002) | 0 | - | 0.154 +- 0.018 | 0.129 +- 0.003 |
| 1 | 0.3 | 0 (< 0.002) | 0 | - | 0.372 +- 0.027 | 0.101 +- 0.002 |
| 4 | 0.3 | 0.202 +- 0.020 | 0.930 +- 0.007 | 0.22 +- 0.02 | 0.498 +- 0.032 | 0.036 +- 0.001 |
| 6 | 0.3 | 0.568 +- 0.034 | 1.035 +- 0.007 | 0.55 +- 0.03 | 0.344 +- 0.026 | 0.023 +- 0.001 |
| 10 | 0.3 | 1.118 +- 0.047 | 1.057 +- 0.007 | 1.06 +- 0.05 | 0.102 +- 0.014 | 0.014 +- 0.001 |
| 0.1 | 0.003 | 0.468 +- 0.031 | 0.475 +- 0.005 | 0.98 +- 0.07 | 0.162 +- 0.018 | 0.145 +- 0.003 |
| 0.255 | 0.012 | 0.564 +- 0.034 | 0.594 +- 0.005 | 0.95 +- 0.06 | 0.154 +- 0.018 | 0.133 +- 0.003 |

Uncertainties are 1-sigma Poisson. Entries per e- can exceed 1 (secondaries, re-entries).

## Reproduce
- MC/DC: `run_mcdc.sh` (reads cases.txt; env E [eV], T [cm], N, OUT; slab_mcdc.py --mode=numba).
- Geant4: build geant4/ with CMake against geant4-v11.4.0; `slab_e <E_MeV> <T_cm> <N> <threads> <seed>`;
  `slab_e calc` dumps stopping powers and msc transport xs (g4_calc.txt).
- analyze.py expects mcdc_*.h5 and g4_results.txt in the working directory.
- xs_compare.py uses sp.py (EPRDATA14 reader) and g4_calc.txt.
