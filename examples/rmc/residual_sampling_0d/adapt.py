"""Residual-driven energy refinement, then a single-loop full-residual RMC run.
Error is measured on the 30 original (coarse) bins against the reference."""
import numpy as np
import run as R
from rmc0d import Setup, transport, reference_collision_density
G0, P, N, N_ITER = 30, 3, 20000, 40
coarse = Setup(G0, 1, "flat")
F = reference_collision_density(coarse.Ef, coarse.q)
phi_ref = coarse.moments(F / coarse.St)[:, 0] / np.diff(coarse.edges)

def coarse_flux(S, c):
    psi = S.psi_fine(c)
    out = []
    for a, b in zip(coarse.edges[:-1], coarse.edges[1:]):
        E = np.linspace(a, b, 4001)
        out.append(np.trapezoid(np.interp(E, S.Ef, psi), E) / (b - a))
    return np.array(out)

def fixed_point(S):
    rng = np.random.default_rng(0); c = np.zeros((S.G, S.P))
    for n in range(30):
        dc,_,_ = R.iterate(S, c, "phase1", rng, n); c += dc
    return c

edges = coarse.edges.copy()
R.N = N
for rnd in range(5):
    S = Setup(None, P, "coll", edges=edges)
    c = fixed_point(S)
    _,_,_,Tn,_ = R.evaluator(S, c)
    r = S.q + Tn - S.St * S.psi_fine(c)
    m = np.array([np.trapezoid(np.abs(r[i0:i1+1]), S.Ef[i0:i1+1]) for i0,i1 in S.masks])
    # unbiased single-loop run on this grid
    rng = np.random.default_rng(5); cc = np.zeros_like(c); avg=[]
    for n in range(N_ITER):
        dc,_,_ = R.iterate(S, cc, "unsplit", rng, 500+n); cc += dc
        if n >= 10: avg.append(cc.copy())
    e_run = np.sqrt(np.mean((coarse_flux(S, np.mean(avg,0))/phi_ref-1)**2))
    e_fp = np.sqrt(np.mean((coarse_flux(S, c)/phi_ref-1)**2))
    print(f"round {rnd}: G={S.G:3d}  |r|_1 at FP={m.sum():.4f}  Galerkin FP bias={e_fp:.2e}  "
          f"RMC(40x{N}) rms err={e_run:.2e}")
    # split the bins holding the top 50% of the residual mass (geometric midpoint)
    o = np.argsort(m)[::-1]; k = np.searchsorted(np.cumsum(m[o]), 0.5*m.sum()) + 1
    mids = np.sqrt(edges[o[:k]] * edges[o[:k]+1])
    edges = np.sort(np.concatenate([edges, mids]))
