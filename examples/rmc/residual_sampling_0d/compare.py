"""Final rms error (over 3 independent repeats) of each sampler / trial space /
schedule at the same total history count. Usage: python compare.py [G N N_ITER]."""
import numpy as np

import run as R
from rmc0d import Setup, transport

N_ITER, n1 = R.N_ITER, R.AVG_START
cases = [
    ("flat P0, split, every iteration", "flat", 1, ["split"] * N_ITER),
    ("flat P0, unsplit, every iteration", "flat", 1, ["unsplit"] * N_ITER),
    ("flat P0, phase 1 + phase 2 (split)", "flat", 1, ["phase1"] * n1 + ["avg:split"] * (N_ITER - n1)),
    ("coll P1, split, every iteration", "coll", 2, ["split"] * N_ITER),
    ("coll P1, unsplit, every iteration", "coll", 2, ["unsplit"] * N_ITER),
    ("coll P1, phase 1 + phase 2 (unsplit)", "coll", 2, ["phase1"] * n1 + ["avg:unsplit"] * (N_ITER - n1)),
    ("coll P2, split, every iteration", "coll", 3, ["split"] * N_ITER),
    ("coll P2, unsplit, every iteration", "coll", 3, ["unsplit"] * N_ITER),
    ("coll P2, phase 1 + phase 2 (unsplit)", "coll", 3, ["phase1"] * n1 + ["avg:unsplit"] * (N_ITER - n1)),
]
S = Setup(R.G, 1, "flat")
Ntot = R.N * N_ITER
e = []
for rep in range(3):
    rng = np.random.default_rng(7 + rep)
    E0 = S.edges[-2] + rng.random(Ntot) * (S.edges[-1] - S.edges[-2])
    t = transport(E0, np.full(Ntot, 1.0 / Ntot), S.edges, 1, 7 + rep)
    e.append(R.err(t[:, 0] / np.diff(S.edges))[0])
print(f"G={R.G}, {N_ITER} iterations x {R.N} histories")
print(f"{'SMC, same histories':40s} rms err {np.mean(e):.2e} (+-{np.std(e):.1e})")
for label, basis, P, schedule in cases:
    h = [R.run(label, basis, P, schedule, rep, verbose=False)[-1] for rep in range(3)]
    err = np.array([x[4] for x in h])
    print(f"{label:40s} rms err {err.mean():.2e} (+-{err.std():.1e})   |r|_1 {np.mean([x[2] for x in h]):.3f}")
