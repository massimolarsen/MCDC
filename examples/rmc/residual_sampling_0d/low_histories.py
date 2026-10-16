"""Stability of single-loop full-residual RMC at low histories per bin (3D will have
far fewer per energy bin than the 0D runs). 40 iterations; reports the fraction of
runs that diverged and the median final error of the rest. RMC0D_RES=0: smooth."""
import os

import numpy as np

import run as R
from rmc0d import Setup

REPS = 8
label = "resonance" if float(os.environ.get("RMC0D_RES", "1")) else "smooth"
print(f"problem: {label}; G={R.G}; 40 iterations; cells: diverged/{REPS}, median rms error")
Ns = (300, 1000, 3000, 10000)
print("basis p | " + " | ".join(f"N={n:5d} ({n/R.G:4.0f}/bin)" for n in Ns))
for basis, ps in (("flat", (0, 1, 2)), ("coll", (0, 1, 2, 3, 4))):
    for p in ps:
        S = Setup(R.G, p + 1, basis)
        cells = []
        for N in Ns:
            R.N = N
            errs, div = [], 0
            for rep in range(REPS):
                rng = np.random.default_rng(300 + rep)
                c = np.zeros((S.G, S.P)); avg = []
                for n in range(40):
                    dc, _, _ = R.iterate(S, c, "unsplit", rng, 20000 + 100 * rep + n)
                    c = c + dc
                    if n >= 10:
                        avg.append(c.copy())
                if not np.isfinite(c).all() or np.abs(c).max() > 1e3:
                    div += 1
                else:
                    errs.append(R.err(S.flux_bin_avg(np.mean(avg, 0)))[0])
            med = f"{np.median(errs):.1e}" if errs else "   -   "
            cells.append(f"{div}/{REPS}  {med}       ")
        print(f"{basis:4s} P{p} | " + " | ".join(cells), flush=True)
