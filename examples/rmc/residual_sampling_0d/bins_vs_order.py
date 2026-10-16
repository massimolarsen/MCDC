"""Fewer, wider energy bins at higher order vs many low-order bins, at equal unknowns
per cell (G (p+1) = 30), collision-density basis, single-loop full-residual RMC
(40 iterations, averaging after 10). Error: pointwise flux L1, int|psi~ - psi| / int psi,
which includes the in-bin shape. 6 repeats; 'div' counts diverged runs.
RMC0D_RES=0: smooth."""
import os

import numpy as np

import run as R
from rmc0d import Setup, reference_collision_density

REPS = 6
label = "resonance" if float(os.environ.get("RMC0D_RES", "1")) else "smooth"
configs = [(30, 0), (15, 1), (10, 2), (6, 4), (30, 1), (30, 2)]
Ns = (1000, 3000, 20000)
print(f"problem: {label}; collision-density basis; median pointwise L1 error (div = diverged/{REPS})")
print("  G  p unknowns | deterministic | " + " | ".join(f"N={n:5d}" for n in Ns))
for G, p in configs:
    S = Setup(G, p + 1, "coll")
    psi_ref = reference_collision_density(S.Ef, S.q) / S.St
    L1 = lambda c: np.trapezoid(np.abs(S.psi_fine(c) - psi_ref), S.Ef) / np.trapezoid(psi_ref, S.Ef)
    # deterministic Galerkin floor of this trial space
    n = S.G * S.P
    A = np.zeros((n, n))
    for col in range(n):
        g, b = divmod(col, S.P)
        A[:, col] = S.moments(S.U[col] - S.St * S.phi[g, b]).ravel()
    c_fp = np.linalg.solve(A, -S.moments(S.q).ravel()).reshape(S.G, S.P)
    cells = []
    for N in Ns:
        R.N = N
        errs, div = [], 0
        for rep in range(REPS):
            rng = np.random.default_rng(700 + rep)
            c = np.zeros((S.G, S.P)); avg = []
            for it in range(40):
                dc, _, _ = R.iterate(S, c, "unsplit", rng, 40000 + 100 * rep + it)
                c = c + dc
                if it >= 10:
                    avg.append(c.copy())
            if not np.isfinite(c).all() or np.abs(c).max() > 1e3:
                div += 1
            else:
                errs.append(L1(np.mean(avg, 0)))
        cells.append((f"{np.median(errs):.1e}" if errs else "  -    ") + f" ({div})")
    print(f"{G:3d} P{p} {G*(p+1):5d}    |   {L1(c_fp):.1e}     | " + " | ".join(cells), flush=True)
