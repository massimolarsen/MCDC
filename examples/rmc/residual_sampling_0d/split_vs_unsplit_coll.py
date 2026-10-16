"""Per-pass variance at the phase-1 fixed point for the collision-density basis
(same measurement as split_vs_unsplit.py)."""
import os

import numpy as np

import run as R
from split_vs_unsplit import passes, smc_var, norms, PASSES, N
from rmc0d import Setup

label = "resonance" if float(os.environ.get("RMC0D_RES", "1")) else "smooth"
print(f"problem: {label}; collision-density basis; {PASSES} passes x {N} histories")
for P in (1, 2, 3):
    S = Setup(R.G, P, "coll")
    v_smc, _ = smc_var(S)
    rng = np.random.default_rng(0)
    c = np.zeros((R.G, P))
    for n in range(30):
        dc, _, _ = R.iterate(S, c, "phase1", rng, n)
        c = c + dc
    full, rc, rs = norms(S, c)
    print(f"\n=== coll P{P-1}  |r|_1 = {full:.4f}   |r_c|_1 + |r_s|_1 = {rc:.4f} + {rs:.4f}")
    for mode, alloc, tag in (("split", "fixed", "split, 90/10 (current)"),
                             ("split", "mass", "split, mass-proportional"),
                             ("unsplit", "mass", "one term (unsplit)")):
        m, v = passes(S, c, mode, alloc, 11)
        print(f"    {tag:26s} sum var / SMC sum var = {v.sum()/v_smc.sum():.5f}")
