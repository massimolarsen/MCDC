"""Split (r_c + r_s, as in feature/RMC) vs one-term (unsplit) residual sampling.

1. Per-pass variance at a fixed trial function (no feedback): 200 independent passes
   of N histories each, at (a) an early iterate and (b) the phase-1 fixed point, where
   phase-2 correction passes run. Reports the summed per-bin variance of eps~ (relative
   to SMC's with the same N) and the source-weight second moment N E[w^2].
2. Convergence when the full residual is sampled every iteration (5 repeats).
Set RMC0D_RES=0 for the smooth (no-resonance) problem. Usage: python split_vs_unsplit.py
"""
import os

import numpy as np

import run as R
from rmc0d import Setup, transport, sample_term

N, PASSES = 20000, 200
R.N = N




def passes(S, c, mode, alloc, seed0):
    rng = np.random.default_rng(seed0)
    eps, w2 = [], []
    for p in range(PASSES):
        dc, _, _ = R.iterate(S, c, mode, rng, seed0 * 7919 + p, alloc)
        eps.append(dc)
    eps = np.array(eps)
    # flux bin average of the correction: P0 coefficient for the flat basis
    phi = np.array([S.flux_bin_avg(e) for e in eps])
    return phi.mean(0), phi.var(0, ddof=1)


def smc_var(S):
    rng = np.random.default_rng(3)
    phis = []
    for p in range(PASSES):
        E0 = R.SRC_LO + rng.random(N) * (R.E_MAX - R.SRC_LO)
        t = transport(E0, np.full(N, 1.0 / N), S.edges, 1, 50000 + p)
        phis.append(t[:, 0] / np.diff(S.edges))
    phis = np.array(phis)
    return phis.var(0, ddof=1), phis.mean(0)


def norms(S, c):
    psi, q, T, Tn, legendre_fn = R.evaluator(S, c)
    psin = S.psi_fine(c)
    Sbn = legendre_fn(S.moments(Tn))(S.Ef)
    full = S.q + Tn - S.St * psin
    rc = S.q + Sbn - S.St * psin
    rs = Tn - Sbn
    L1 = lambda f: np.trapezoid(np.abs(f), S.Ef)
    return L1(full), L1(rc), L1(rs)


if __name__ == "__main__":
    label = "resonance" if float(os.environ.get("RMC0D_RES", "1")) else "smooth"
    print(f"problem: {label}; G={R.G}; {PASSES} passes x {N} histories")
    for P in (1, 2):
        S = Setup(R.G, P, "flat")
        v_smc, phi_smc = smc_var(S)
        rng = np.random.default_rng(0)
        c = np.zeros((R.G, P))
        states = {}
        for n in range(40):
            mode = "phase1"
            dc, _, _ = R.iterate(S, c, mode, rng, n)
            c = c + dc
            if n == 1:
                states["early iterate (2 phase-1 its)"] = c.copy()
        states["phase-1 fixed point"] = c.copy()
        print(f"\n=== flat P{P-1}")
        for name, cs in states.items():
            full, rc, rs = norms(S, cs)
            print(f"  [{name}]  |r|_1 = {full:.3f}   |r_c|_1 + |r_s|_1 = {rc:.3f} + {rs:.3f} = {rc+rs:.3f}")
            res = {}
            for mode, alloc, tag in (("split", "fixed", "split, 90/10 (current)"),
                                     ("split", "mass", "split, mass-proportional"),
                                     ("unsplit", "mass", "one term (unsplit)")):
                m, v = passes(S, cs, mode, alloc, 11)
                res[tag] = (m, v)
                print(f"    {tag:26s} sum var / SMC sum var = {v.sum()/v_smc.sum():.4f}")
            d = res["split, 90/10 (current)"][0] - res["one term (unsplit)"][0]
            se = np.sqrt((res["split, 90/10 (current)"][1] + res["one term (unsplit)"][1]) / PASSES)
            print(f"    unbiasedness check: max |mean difference| / s.e. = {np.max(np.abs(d)/se):.2f}")

    print("\nConvergence with the full residual every iteration (40 x 20000, rms error, 5 repeats)")
    for P in (1, 2):
        for mode, alloc, tag in (("split", "fixed", "split 90/10"), ("split", "mass", "split mass-prop"),
                                 ("unsplit", "mass", "one term")):
            errs = []
            for rep in range(5):
                S = Setup(R.G, P, "flat")
                rng = np.random.default_rng(100 + rep)
                c = np.zeros((R.G, P)); avg = []
                for n in range(40):
                    dc, _, _ = R.iterate(S, c, mode, rng, 9000 + 100 * rep + n, alloc)
                    c = c + dc
                    if n >= 10:
                        avg.append(c.copy())
                errs.append((R.err(S.flux_bin_avg(c))[0], R.err(S.flux_bin_avg(np.mean(avg, 0)))[0]))
            e = np.array(errs)
            print(f"  flat P{P-1} {tag:16s} last iterate {e[:,0].mean():.2e}   averaged {e[:,1].mean():.2e} (+-{e[:,1].std():.1e})")
