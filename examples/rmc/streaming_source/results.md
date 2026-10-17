# Held-out 3D one-group matrix results

All ratios include source construction, pilot, baseline and correction time. JIT compilation is warmed for the selected variant and recorded separately. Intervals are pointwise 95% independent-run bootstrap intervals for the empirical variance ratio at observed mean wall times; they do not include timing-system variability or simultaneous-case multiplicity.

Each resolution compares RMC and SMC on identical cell responses. The refined variant uses more tally cells, so comparisons across resolutions are not a controlled change of trial space alone.

## antipodal

Settings: `{"antithetic": true, "cells": 2, "forward": false, "histories": 8000, "inner": [4], "linear": false, "output": "examples/rmc/streaming_source/results/antipodal.json", "pilot": 64, "power": 10, "repeats": 24, "seed": 20261101, "smc_histories": 200000, "split": false, "targeted": false}`.

| case | FOM RMC/SMC | bootstrap interval | reported-variance FOM | max agreement z | empirical/reported variance |
|---|---:|---:|---:|---:|---:|
| thin-c0.2-uniform | 47.8 | 35.4–63.5 | 47.4 | 1.96 | 1.03 |
| thin-c0.2-corner | 0.799 | 0.534–1.26 | 0.778 | 1.35 | 1.15 |
| thin-c0.8-uniform | 18.6 | 13–26.5 | 17.6 | 2.47 | 0.90 |
| thin-c0.8-corner | 0.515 | 0.362–0.741 | 0.619 | 1.88 | 1.11 |
| thin-c0.97-uniform | 11.6 | 8.88–15.2 | 12.7 | 1.66 | 1.01 |
| thin-c0.97-corner | 0.456 | 0.305–0.709 | 0.514 | 1.74 | 1.03 |
| thick-c0.2-uniform | 50.1 | 37.6–67 | 68.8 | 1.26 | 1.04 |
| thick-c0.2-corner | 1.04 | 0.616–1.67 | 1.08 | 1.46 | 0.92 |
| thick-c0.8-uniform | 1.53 | 1.1–2.04 | 1.44 | 2.42 | 0.97 |
| thick-c0.8-corner | 0.0618 | 0.0401–0.097 | 0.0754 | 1.16 | 1.36 |
| thick-c0.97-uniform | 0.5 | 0.367–0.684 | 0.513 | 1.58 | 0.98 |
| thick-c0.97-corner | 0.0528 | 0.035–0.0753 | 0.051 | 1.92 | 1.10 |
| contrast-c0.2-uniform | 9.78 | 7.42–12.9 | 10.2 | 2.79 | 0.95 |
| contrast-c0.2-corner | 0.205 | 0.131–0.311 | 0.251 | 1.42 | 1.16 |
| contrast-c0.8-uniform | 2.18 | 1.57–3.01 | 1.84 | 1.68 | 0.89 |
| contrast-c0.8-corner | 0.0569 | 0.0391–0.0815 | 0.071 | 2.06 | 1.00 |
| contrast-c0.97-uniform | 0.749 | 0.56–1.01 | 0.833 | 1.99 | 1.06 |
| contrast-c0.97-corner | 0.0542 | 0.0367–0.0792 | 0.0542 | 2.60 | 0.89 |

Pointwise intervals above one: 7; below one: 8. These counts describe this matrix, not a universal efficiency claim.

## flat

Settings: `{"antithetic": false, "cells": 2, "forward": false, "histories": 8000, "inner": [4], "linear": false, "output": "examples/rmc/streaming_source/results/flat.json", "pilot": 64, "power": 10, "repeats": 24, "seed": 20261101, "smc_histories": 200000, "split": false, "targeted": false}`.

| case | FOM RMC/SMC | bootstrap interval | reported-variance FOM | max agreement z | empirical/reported variance |
|---|---:|---:|---:|---:|---:|
| thin-c0.2-uniform | 51.1 | 38.3–67.1 | 49.5 | 1.78 | 1.00 |
| thin-c0.2-corner | 0.829 | 0.545–1.31 | 0.827 | 1.25 | 1.18 |
| thin-c0.8-uniform | 15 | 10.8–20.8 | 17.3 | 2.19 | 1.09 |
| thin-c0.8-corner | 0.443 | 0.3–0.694 | 0.582 | 1.73 | 1.21 |
| thin-c0.97-uniform | 11.6 | 8.66–15.9 | 12.1 | 1.87 | 0.96 |
| thin-c0.97-corner | 0.469 | 0.319–0.696 | 0.58 | 1.52 | 1.13 |
| thick-c0.2-uniform | 52.3 | 39.4–69.4 | 65.8 | 1.18 | 0.95 |
| thick-c0.2-corner | 0.941 | 0.502–1.8 | 1.04 | 1.25 | 0.98 |
| thick-c0.8-uniform | 1.37 | 0.978–1.92 | 1.39 | 1.92 | 1.04 |
| thick-c0.8-corner | 0.081 | 0.0572–0.115 | 0.0826 | 1.96 | 1.13 |
| thick-c0.97-uniform | 0.468 | 0.352–0.627 | 0.489 | 2.98 | 1.00 |
| thick-c0.97-corner | 0.0635 | 0.0402–0.0949 | 0.054 | 1.36 | 0.97 |
| contrast-c0.2-uniform | 10.2 | 7.78–13.2 | 10.5 | 2.64 | 0.95 |
| contrast-c0.2-corner | 0.187 | 0.115–0.304 | 0.251 | 1.58 | 1.28 |
| contrast-c0.8-uniform | 2.21 | 1.68–2.87 | 1.86 | 3.14 | 0.89 |
| contrast-c0.8-corner | 0.0641 | 0.0434–0.0981 | 0.0753 | 2.12 | 0.94 |
| contrast-c0.97-uniform | 0.743 | 0.546–1 | 0.812 | 0.93 | 1.04 |
| contrast-c0.97-corner | 0.0525 | 0.0373–0.0752 | 0.0576 | 1.00 | 0.98 |

Pointwise intervals above one: 6; below one: 8. These counts describe this matrix, not a universal efficiency claim.

## linear

Settings: `{"antithetic": false, "cells": 2, "forward": false, "histories": 8000, "inner": [4], "linear": true, "output": "examples/rmc/streaming_source/results/linear.json", "pilot": 64, "power": 10, "repeats": 24, "seed": 20261101, "smc_histories": 200000, "split": false, "targeted": false}`.

| case | FOM RMC/SMC | bootstrap interval | reported-variance FOM | max agreement z | empirical/reported variance |
|---|---:|---:|---:|---:|---:|
| thin-c0.2-uniform | 41.9 | 31.5–54.8 | 40.7 | 1.79 | 1.01 |
| thin-c0.2-corner | 0.757 | 0.506–1.2 | 0.776 | 1.03 | 1.21 |
| thin-c0.8-uniform | 14 | 10–19.7 | 16.4 | 2.22 | 1.11 |
| thin-c0.8-corner | 0.536 | 0.36–0.83 | 0.647 | 1.64 | 1.11 |
| thin-c0.97-uniform | 11.2 | 8.44–15.2 | 11.5 | 1.85 | 0.95 |
| thin-c0.97-corner | 0.549 | 0.393–0.777 | 0.576 | 2.36 | 0.96 |
| thick-c0.2-uniform | 49.2 | 36.7–65.5 | 62.2 | 1.17 | 0.96 |
| thick-c0.2-corner | 1.5 | 0.896–2.39 | 1.28 | 2.03 | 0.76 |
| thick-c0.8-uniform | 1.29 | 0.925–1.79 | 1.27 | 2.33 | 1.01 |
| thick-c0.8-corner | 0.102 | 0.0657–0.16 | 0.114 | 1.04 | 1.25 |
| thick-c0.97-uniform | 0.45 | 0.335–0.608 | 0.461 | 2.86 | 0.98 |
| thick-c0.97-corner | 0.105 | 0.0672–0.161 | 0.089 | 1.37 | 0.96 |
| contrast-c0.2-uniform | 10.3 | 7.74–13.6 | 12 | 2.79 | 1.06 |
| contrast-c0.2-corner | 0.208 | 0.132–0.309 | 0.249 | 2.27 | 1.14 |
| contrast-c0.8-uniform | 3.21 | 2.36–4.34 | 2.55 | 1.53 | 0.84 |
| contrast-c0.8-corner | 0.07 | 0.0458–0.109 | 0.0883 | 2.61 | 1.01 |
| contrast-c0.97-uniform | 0.629 | 0.474–0.821 | 0.723 | 0.78 | 1.10 |
| contrast-c0.97-corner | 0.0552 | 0.034–0.0939 | 0.0697 | 2.37 | 1.12 |

Pointwise intervals above one: 6; below one: 9. These counts describe this matrix, not a universal efficiency claim.

## refined

Settings: `{"antithetic": false, "cells": 4, "forward": false, "histories": 16000, "inner": [4], "linear": false, "output": "examples/rmc/streaming_source/results/refined.json", "pilot": 16, "power": 8, "repeats": 24, "seed": 20261101, "smc_histories": 200000, "split": false, "targeted": false}`.

| case | FOM RMC/SMC | bootstrap interval | reported-variance FOM | max agreement z | empirical/reported variance |
|---|---:|---:|---:|---:|---:|
| thin-c0.2-uniform | 42.7 | 37.2–48.7 | 39.1 | 2.89 | 0.92 |
| thin-c0.2-corner | 0.663 | 0.568–0.77 | 0.549 | 2.11 | 0.89 |
| thin-c0.8-uniform | 22.9 | 20.2–26.1 | 20.3 | 2.37 | 0.95 |
| thin-c0.8-corner | 0.55 | 0.479–0.631 | 0.499 | 2.51 | 0.94 |
| thin-c0.97-uniform | 16.7 | 14.5–19.2 | 15.6 | 3.01 | 0.96 |
| thin-c0.97-corner | 0.464 | 0.406–0.529 | 0.458 | 2.32 | 0.97 |
| thick-c0.2-uniform | 76.4 | 67.3–87.3 | 81 | 3.19 | 1.00 |
| thick-c0.2-corner | 1.43 | 1.04–2.04 | 1.7 | 3.26 | 1.07 |
| thick-c0.8-uniform | 1.68 | 1.48–1.92 | 1.58 | 2.67 | 0.96 |
| thick-c0.8-corner | 0.0732 | 0.0583–0.0928 | 0.0892 | 3.38 | 1.17 |
| thick-c0.97-uniform | 0.446 | 0.385–0.513 | 0.469 | 2.59 | 1.06 |
| thick-c0.97-corner | 0.0544 | 0.0469–0.0634 | 0.0552 | 2.34 | 0.99 |
| contrast-c0.2-uniform | 14.7 | 13–16.5 | 13.8 | 2.88 | 0.95 |
| contrast-c0.2-corner | 0.146 | 0.0536–0.658 | 0.128 | 3.49 | 0.97 |
| contrast-c0.8-uniform | 2.16 | 1.89–2.48 | 2.19 | 2.35 | 1.01 |
| contrast-c0.8-corner | 0.0994 | 0.0723–0.132 | 0.0791 | 2.79 | 0.93 |
| contrast-c0.97-uniform | 0.857 | 0.743–0.991 | 0.796 | 2.86 | 0.99 |
| contrast-c0.97-corner | 0.0533 | 0.0386–0.0731 | 0.0524 | 3.55 | 1.03 |

Pointwise intervals above one: 8; below one: 10. These counts describe this matrix, not a universal efficiency claim.

## unfitted

Settings: `{"antithetic": false, "cells": 2, "forward": false, "histories": 8000, "inner": [4], "linear": false, "output": "examples/rmc/streaming_source/results/unfitted.json", "pilot": 64, "power": 10, "repeats": 24, "seed": 20261101, "smc_histories": 200000, "split": false, "targeted": false, "unfitted": true}`.

| case | FOM RMC/SMC | bootstrap interval | reported-variance FOM | max agreement z | empirical/reported variance |
|---|---:|---:|---:|---:|---:|
| thin-c0.2-uniform | 27.2 | 20–35.4 | 25.5 | 1.85 | 0.97 |
| thin-c0.2-corner | 0.83 | 0.562–1.27 | 0.848 | 1.33 | 1.21 |
| thin-c0.8-uniform | 2.69 | 1.96–3.66 | 2.77 | 2.41 | 0.97 |
| thin-c0.8-corner | 0.451 | 0.314–0.636 | 0.541 | 2.18 | 1.10 |
| thin-c0.97-uniform | 1.94 | 1.51–2.53 | 1.94 | 1.71 | 0.92 |
| thin-c0.97-corner | 0.374 | 0.245–0.576 | 0.489 | 1.57 | 1.19 |
| thick-c0.2-uniform | 2.91 | 2.15–4 | 3.71 | 0.98 | 0.96 |
| thick-c0.2-corner | 0.674 | 0.389–1.14 | 0.73 | 2.50 | 0.96 |
| thick-c0.8-uniform | 0.265 | 0.196–0.352 | 0.297 | 1.82 | 1.15 |
| thick-c0.8-corner | 0.113 | 0.0773–0.159 | 0.106 | 1.35 | 1.04 |
| thick-c0.97-uniform | 0.229 | 0.168–0.321 | 0.265 | 1.88 | 1.10 |
| thick-c0.97-corner | 0.105 | 0.0633–0.171 | 0.107 | 1.72 | 1.17 |
| contrast-c0.2-uniform | 3.68 | 2.87–4.78 | 4.08 | 2.72 | 1.01 |
| contrast-c0.2-corner | 0.189 | 0.118–0.297 | 0.243 | 2.04 | 1.22 |
| contrast-c0.8-uniform | 0.418 | 0.32–0.539 | 0.388 | 3.08 | 0.98 |
| contrast-c0.8-corner | 0.0522 | 0.0354–0.0767 | 0.0611 | 2.46 | 0.93 |
| contrast-c0.97-uniform | 0.262 | 0.197–0.345 | 0.27 | 1.51 | 0.98 |
| contrast-c0.97-corner | 0.0354 | 0.0228–0.0626 | 0.0392 | 1.52 | 0.99 |

Pointwise intervals above one: 5; below one: 11. These counts describe this matrix, not a universal efficiency claim.

