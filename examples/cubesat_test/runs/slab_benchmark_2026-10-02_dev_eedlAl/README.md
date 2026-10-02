# Why the Lockwood regression passes: EEDL Al data test on merged dev (2026-10-02)

Code: merged upstream dev a0ddec22 (PR #548). Same slab cases and N = 1000 as
`../slab_benchmark_2026-10-01_pr548`, but Al electron data replaced by the Al.h5 the
Lockwood regression uses (mcdc-regression_test_data, EEDL-based, older format; copied
here as `regression_Al_EEDL.h5`). Mg/Si stay EPRDATA14. `PURE_AL=1` runs drop Mg/Si.

## Lockwood regression
- Harness compares output.h5 to answer.h5 at rtol 1e-6: reproducibility only.
- answer.h5 was regenerated in PR #548 with N = 10 electrons. Backscatter 2/10; depth-dose
  0-1.5x Lockwood experiment (mean |dev| 36%).
- It runs on the EEDL Al.h5 (31 elastic angle tables incl. 0.47, 0.87, 1.6, 2.9, 5.4 MeV),
  not EPRDATA14, so the 0.256-10 MeV gap is never exercised.
- For reference, MCDCe-VV (April, 1e5 e-, EEDL library, older code) matched Lockwood 1 MeV Al
  depth-dose within ~3% (mean |dev| 4%) with backscatter 0.094.

## Slab results (Geant4 from 09-28)
| Case | 1 MeV back | 4 MeV det ratio | 4 MeV back | 6 MeV det ratio | 6 MeV back |
|---|---|---|---|---|---|
| Geant4 | 0.098 | 1 | 0.036 | 1 | 0.021 |
| EPRDATA14 (10-01) | 0.385 | 0.21 | 0.486 | 0.58 | 0.371 |
| EEDL Al + EPRDATA14 Mg,Si | 0.171 | 0.83 | 0.130 | 0.98 | 0.192 |
| EEDL pure Al | 0.158 | 0.90 | 0.117 | 1.01 | 0.201 |

Excess backscatter is near-primary energy (0.5-1 MeV bin at 1 MeV; 1-4 MeV at 6 MeV):
hard elastic scattering, not secondaries. Full tables: results.md (Al6061 cases).

## Elastic transport xs as MC/DC samples it (Al), ratio to EPRDATA14 transport xs
sigma_large * <1-mu>_table + (sigma_tot - sigma_large) * <1-mu>_screened-Rutherford, MU_CUTOFF = 0.999999
| E (MeV) | 0.1 | 0.255 | 0.47 | 1 | 2.9 | 4 | 6 | 10 |
|---|---|---|---|---|---|---|---|---|
| EEDL (large/tot = 1.000 everywhere) | 1.10 | 1.03 | 1.21 | 1.47 | 1.74 | 2.09 | 2.31 | 2.84 |
| EPRDATA14 (large/tot 0.34 at 10 MeV) | 1.12 | 1.04 | 1.85 | 3.88 | 9.61 | 11.1 | 10.5 | 1.00 |

- EPRDATA14: correct on its table energies (1.00 at 10 MeV), 2-11x in the 0.256-10 MeV gap.
- EEDL file: at 0.064-0.256 MeV and 10 MeV its tables are the same as EPRDATA14's (same point
  counts, <1-mu> within 2%). The gap-filling tables (0.47, 0.87, 1.6, 2.9, 5.4 MeV; 127 points
  each) have <1-mu> 1.21, 1.37, 1.52, 1.50, 1.30x what the EPRDATA14 transport xs implies.
  On top of that, `large_angle/xs` equals the total elastic xs everywhere, while EPRDATA14's
  large-angle fraction drops to 0.88 / 0.76 / 0.57 at 2.9 / 4 / 6 MeV. Together: transport xs
  1.2-2.8x too high above 0.3 MeV, which is the residual backscatter excess.

## Detector entry spectra (per e-, MeV bins <0.5 / 0.5-1 / 1-2 / 2-4 / 4-6)
| 4 MeV | <0.5 | 0.5-1 | 1-2 | 2-4 | total |
|---|---|---|---|---|---|
| Geant4 | 0.045 | 0.033 | 0.106 | 0.746 | 0.930 |
| EEDL pure Al | 0.063 | 0.054 | 0.138 | 0.583 | 0.838 |
| EEDL Al6061 | 0.048 | 0.049 | 0.132 | 0.545 | 0.774 |

| 6 MeV | <0.5 | 0.5-1 | 1-2 | 2-4 | 4-6 | total |
|---|---|---|---|---|---|---|
| Geant4 | 0.035 | 0.016 | 0.025 | 0.113 | 0.850 | 1.038 |
| EEDL pure Al | 0.046 | 0.036 | 0.049 | 0.133 | 0.789 | 1.053 |
| EEDL Al6061 | 0.047 | 0.028 | 0.044 | 0.151 | 0.743 | 1.013 |

Totals look close because a deficit near full energy is offset by excess degraded electrons.
