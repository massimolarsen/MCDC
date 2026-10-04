# Efficiency sweep: RMC against SMC, dissertation against current

Error against cost for three methods on four problems, over the number of histories
per trial-space bin. It answers how much cheaper RMC is than SMC for
a given error, and how much the current method gains over the dissertation's.

| Configuration | Trial space | Schedule |
|---|---|---|
| `smc` | (tallies on the same bins) | standard Monte Carlo, histories = level x bins x 10 |
| `dissertation` | constant in energy, z and angle | 10 collision-only iterations (phase 1 only) |
| `current` | linear energy; for the fuel rod also linear angle and continuous linear z | 10 collision-only iterations + 4 correction passes |

| Problem | Type | Reference |
|---|---|---|
| `absorber` | 0D, A = 1, G = 100 | analytic |
| `o16` | 0D, O-16 1-10 MeV, G = 100 | SMC, 10^8 histories |
| `fuel_thermal`, `fuel_fast` | 1D fuel rod (`../fuel_rod`) | SMC, 10^8 histories |

Levels (histories per (cell, energy, polar) bin per iteration): 100, 400 and 1600 for
the 0D problems, 100 and 400 for the fuel rod (about 4800 bins); one seed. That is
9 runs per 0D problem and 6 per fuel rod (30 in all) plus 3 references. More seeds can
be added later (`SWEEP_SEEDS=1,2`; finished tasks are skipped).

## Files

- `sweep.py`: the task list, one task per invocation (`run INDEX`), and the warmups.
  Each task writes `results/<problem>/<config>/n<level>_s<seed>.h5`, with a `timing`
  group (see "Timings" below).
- `sweep.slurm`: the job template, for one warmup or one array task.
- `submit.sh [PROBLEM ...]`: submits the given problems (default: `absorber o16`),
  each with:
  1. warmups that compute and cache the transfer moments once per configuration;
  2. one job array over its tasks, after the warmups;
  3. its reference (not for the absorber).
- `env.sh`: site settings (paths, Python environment, partition). **Edit this first.**
- `analyze.py`: `efficiency_<problem>.png` and `summary.csv`, including the figure of
  merit 1/(error^2 x cost); `timing.csv` and `timing_warmup.csv`.

Finished tasks are skipped when resubmitted (`SWEEP_FORCE=1` reruns them). The levels,
seeds, correction passes and reference size can be overridden with `SWEEP_LEVELS`,
`SWEEP_SEEDS`, `SWEEP_N_CORRECTION` and `SWEEP_N_REFERENCE`.

## Settings for the OSU College of Engineering HPC

Checked on the cluster (submit-a, 2026-10-04) and in the CoE HPC documentation
([Slurm howto](https://it.engineering.oregonstate.edu/hpc/slurm-howto),
[FAQs](https://it.engineering.oregonstate.edu/hpc/faqs)):

- **Partition:** `share`, open to all users with no account (up to 14 days; default
  12 h). Its 62 nodes have 16-64 cores, most of them 16-24. No `--account` is
  requested, so jobs run under your default account. `preempt` is avoided: preempted
  jobs are cancelled.
- **Ranks:** single-node jobs with 8 ranks for the 0D problems and 16 for the fuel
  rod, which fits every `share` node.
- **Python/MPI:** the module stack and virtualenv of the `MCDC-g4-hpc` jobs:
  `module load slurm/current python/3.13 mpich/4.0h_gcc-10`, `venvs/mcdc-g4`
  (mpi4py built against that MPICH; mcdc installed editable from `hpc-share/MCDC`),
  launched with `mpirun -bind-to none`.
  - The conda environments `mcdc-env*` are not used: their mpi4py has no MPI library
    of its own.
- **Data:** `MCDC/hdf5lib_mt5fix` (199 files, no symlinks).
- **Storage:** everything under `/nfs/hpc/share/$USER`; home is limited to 25 GB.
- **Limits:** 1000 submitted and 400 running jobs per user, and arrays up to 1001 tasks.
  The sweep is kept far below them: at most 6 tasks of each 0D array and 2 of each
  fuel rod array run at once (`MAX_RUNNING` in `submit.sh`).

| Problem | Tasks | Ranks | Limit per task | At once | Reference |
|---|---|---|---|---|---|
| `absorber` | 9 | 8 | 2 h | 6 | analytic |
| `o16` | 9 | 8 | 4 h | 6 | 8 ranks, 12 h |
| `fuel_thermal` | 6 | 16 | 12 h | 2 | 16 ranks, 2 days |
| `fuel_fast` | 6 | 16 | 12 h | 2 | 16 ranks, 2 days |

Plus two warmups per problem with the problem's ranks and limit. At most about 100 cores
are in use at once, and the whole sweep is capped at about 5000 core-hours if every job
ran to its limit (the 0D tasks should take minutes).

## Running on the cluster

The checkout `/nfs/hpc/share/$USER/MCDC` must be on `feature/RMC` with this directory
pulled. Then, on a submit node:

```bash
cd /nfs/hpc/share/$USER/MCDC/examples/rmc/efficiency
./submit.sh                   # absorber and o16
squeue -u $USER
./submit.sh fuel_thermal      # after the 0D results look right; then fuel_fast
```

When the jobs are done, run `python analyze.py` there, or copy `results/` back and run
it locally.

## Cost expectations

- The 0D problems are cheap: minutes per task.
- The fuel rod's transfer moments are expensive: the U-235/U-238 precompute took more
  than 28 min on one process (`../fuel_rod/HPC.md`). That is why the warmups compute
  them once per configuration before the arrays start.
- At 400 histories per bin the thermal fuel rod has about 1.9 x 10^6 histories per
  iteration. Its 12 h limit is untested: check the first fuel rod task's timing before
  submitting the second fuel rod.
- The `current` configuration's correction passes use the integrated sampler, the
  most expensive part per history (`mcdc/rmc/writeups/scattering_correction_samplers.tex`).

## Timings

Each task runs the solve twice in the same process:
1. a **cold** pass on a small problem (RMC: 1 iteration and 1 correction pass at the
   same level; SMC: 100 histories), which pays the Numba compilation, or the loading
   of the compiled code from the Numba cache;
2. the **measured** (warm) pass, the full run.

Wall times are taken between MPI barriers. CPU times are `time.process_time` summed
over the ranks. All are in seconds.

| Attribute (`timing` group) | Meaning |
|---|---|
| `wall`, `cpu` | the measured run: what a run costs once the code is compiled and the transfer moments are cached |
| `precompute` | RMC: transfer-moment setup in the measured run (a cache load) |
| `iterations`, `corrections` | RMC: the phase-1 iterations and the correction passes, summed; `timing.csv` also gives them per iteration and per pass |
| `cold_wall`, `cold_cpu` | the cold pass |
| `cold_precompute`, `cold_iteration`, `cold_correction` | RMC: the cold pass by stage |
| `jit_estimate` | the cold pass minus the warm cost of the same work: the compilation, or the cache load in a fresh process. RMC: per stage; SMC: `cold_wall - wall x 100 / N` |

The root attributes `wall` and `ranks` give the cost of the efficiency plots,
wall x ranks of the measured run. `wall_x_ranks` against `cpu` in `timing.csv` shows
the time ranks spend waiting (load imbalance, MPI).

The warmups write `results/<problem>/<config>/warmup.h5` with the one-time costs:
`precompute` (the transfer moments computed from scratch, `moments_computed` true
when the warmup did it rather than finding a cache), the whole warmup's `wall` and
`cpu` (also the first compilation into an empty Numba cache), and `ranks`.

A local check (absorber, 5 histories per bin, 1 process, Numba cache already filled):

| | cold pass | measured run | of which precompute / iterations / corrections |
|---|---|---|---|
| RMC current | 32.8 s | 1.8 s | 0.13 / 0.51 / 1.14 s |
| SMC | 30.5 s | 0.5 s | |

The cold pass in a fresh process takes about 30 s even with the Numba cache filled.
The first transfer-moment computation (warmup) took 44.6 s of an 84 s warmup.

## Reading the results

- **Error** is the rms relative error of the bin-averaged scalar flux against the
  reference; the reference's own noise sets a floor.
- For the fuel rod, a second panel uses the interior hat-function moments, which the
  continuous linear z projection preserves. The cell averages of the `current` runs
  include the projection's in-cell error.
- **Cost** is wall time x ranks of the measured run, including the transfer-moment
  load but not its first computation or the compilation (both are in the timing
  tables).
- The `current` configuration needs more histories per bin to iterate stably: each
  slope coefficient is tallied with about sqrt(3) more noise
  (`mcdc/rmc/NOTES.md`, sections 7, 9 and 10). Its lowest levels may show large errors;
  that minimum level is part of the result.
