# 0D residual-sampling prototype

Standalone (numpy + numba, no MC/DC) comparison of trial spaces, split/unsplit residual
sampling and schedules for continuous-energy RMC. See
`mcdc/rmc/writeups/residual_sampling.tex` for the analysis.

- `verify_reference.py`: reference grid convergence and SMC against the reference.
- `run.py [G N N_ITER]`: per-iteration histories for each method.
- `compare.py [G N N_ITER]`: final errors over 3 repeats (table in the write-up).
- `adapt.py`: residual-driven energy refinement.

Recorded outputs: `results_*.txt` (defaults: G = 30, 20000 histories x 40 iterations).
- `split_vs_unsplit.py`, `split_vs_unsplit_coll.py`: per-pass variance of split
  (current 90/10, mass-proportional) vs one-term residual sampling, and convergence
  with the full residual every iteration. `RMC0D_RES=0` runs the smooth problem.
