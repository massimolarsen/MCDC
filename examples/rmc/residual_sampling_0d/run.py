import sys
import time
from collections import defaultdict

import numpy as np

from rmc0d import (Setup, sig_t, legendre, reference_collision_density, sample_term,
                   transport, E_MIN, E_MAX, SRC_LO)

G = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
N_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 40
AVG_START = 10  # iterate/pass averaging starts after this many iterations

# --- reference on the flat setup's fine grid
ref_setup = Setup(G, 1, "flat")
Ef = ref_setup.Ef
F = reference_collision_density(Ef, ref_setup.q)
phi_ref = ref_setup.moments(F / ref_setup.St)[:, 0] / np.diff(ref_setup.edges)


def err(phi):
    rel = phi / phi_ref - 1
    return np.sqrt(np.mean(rel**2)), np.max(np.abs(rel))


def evaluator(S, c):
    """Exact pointwise pieces at arbitrary E."""
    Tn = S.T_fine(c)

    def bin_of(E):
        return np.clip(np.searchsorted(S.edges, E, side="right") - 1, 0, S.G - 1)

    def psi(E):
        g = bin_of(E)
        x = (2 * E - S.edges[g] - S.edges[g + 1]) / (S.edges[g + 1] - S.edges[g])
        v = np.zeros_like(E)
        for b in range(S.P):
            v += c[g, b] * legendre(b, x)
        return v / sig_t(E) if S.basis == "coll" else v

    def q(E):
        return np.where(E > SRC_LO, 1.0 / (E_MAX - SRC_LO), 0.0)

    def T(E):
        return np.interp(E, S.Ef, Tn)

    def legendre_fn(mom):
        """sum_a (2a+1)/dE mom[g,a] P_a(x): a test-space polynomial."""
        def f(E):
            g = bin_of(E)
            dE = S.edges[g + 1] - S.edges[g]
            x = (2 * E - S.edges[g] - S.edges[g + 1]) / dE
            v = np.zeros_like(E)
            for a in range(S.P):
                v += (2 * a + 1) / dE * mom[g, a] * legendre(a, x)
            return v
        return f

    return psi, q, T, Tn, legendre_fn


# wall time per pass component (residual build / sampling / transport), accumulated
TIMES = defaultdict(float)


def iterate(S, c, mode, rng, seed, alloc="mass"):
    t0 = time.perf_counter()
    psi, q, T, Tn, legendre_fn = evaluator(S, c)
    psin = S.psi_fine(c)
    full_n = S.q + Tn - S.St * psin

    def full(E):
        return q(E) + T(E) - sig_t(E) * psi(E)

    parts = []
    if mode == "unsplit":
        parts.append((full_n, full))
    elif mode == "phase1":
        f1 = legendre_fn(S.moments(full_n))  # Galerkin-projected residual
        parts.append((f1(S.Ef), f1))
    elif mode == "split":
        Sbar = legendre_fn(S.moments(Tn))
        Sbn = Sbar(S.Ef)

        def rc(E):
            return q(E) + Sbar(E) - sig_t(E) * psi(E)

        def rs(E):
            return T(E) - Sbar(E)

        parts.append((S.q + Sbn - S.St * psin, rc))
        parts.append((Tn - Sbn, rs))
    masses = np.array([np.trapezoid(np.abs(rn), S.Ef) for rn, _ in parts])
    if alloc == "fixed" and len(parts) == 2:
        # feature/RMC default: collision_fraction = 0.9 of the histories to r_c
        Ns = np.array([int(0.9 * N), N - int(0.9 * N)])
    else:
        Ns = np.maximum(1, np.round(N * masses / masses.sum()).astype(int))
    t1 = time.perf_counter()
    TIMES["residual"] += t1 - t0
    Es, ws = [], []
    for (rn, rf), Ni in zip(parts, Ns):
        E, w = sample_term(rng, S.Ef, rn, rf, Ni)
        Es.append(E)
        ws.append(w)
    E = np.concatenate(Es)
    w = np.concatenate(ws)
    t2 = time.perf_counter()
    TIMES["sample"] += t2 - t1
    t = transport(E, w, S.edges, S.P, seed)
    dc = np.einsum("gab,gb->ga", S.Binv, t)
    TIMES["transport"] += time.perf_counter() - t2
    return dc, masses.sum(), np.trapezoid(np.abs(full_n), S.Ef)


def run(label, basis, P, schedule, rep=0, verbose=True):
    """schedule: list of modes per iteration; 'avg:<mode>' = correction pass averaged at
    the frozen iterate (phase 2); otherwise accumulate."""
    S = Setup(G, P, basis)
    rng = np.random.default_rng(1 + 100 * rep)
    c = np.zeros((G, P))
    hist = []
    frozen, corr = None, []
    t0 = time.time()
    polyak = []
    for n, mode in enumerate(schedule):
        if mode.startswith("avg:"):
            if frozen is None:
                frozen = c.copy()
            dc, m, rfull = iterate(S, frozen, mode[4:], rng, 1000 + n + 100000 * rep)
            corr.append(dc)
            c_rep = frozen + np.mean(corr, axis=0)
        else:
            dc, m, rfull = iterate(S, c, mode, rng, 1000 + n + 100000 * rep)
            c = c + dc
            if mode != "phase1" and n >= AVG_START:
                polyak.append(c.copy())
            c_rep = np.mean(polyak, axis=0) if polyak else c
        e_rms, e_max = err(S.flux_bin_avg(c_rep))
        e_it, _ = err(S.flux_bin_avg(c if frozen is None else frozen + dc))
        hist.append((n, np.abs(dc).sum(), rfull, e_it, e_rms, e_max))
    if not verbose:
        return hist
    print(f"\n== {label}  ({time.time()-t0:.1f}s)")
    print(" it   |dc|_1      |r|_1     err(iterate)  err(reported rms)  max")
    for h in hist:
        if h[0] < 12 or h[0] % 5 == 4 or h[0] == len(schedule) - 1:
            print(f"{h[0]:3d}  {h[1]:.3e}  {h[2]:.3e}   {h[3]:.3e}     {h[4]:.3e}     {h[5]:.3e}")
    return hist


if __name__ == "__main__":
    # --- SMC at the same total budget
    S = Setup(G, 1, "flat")
    rng = np.random.default_rng(7)
    Ntot = N * N_ITER
    E0 = S.edges[-2] + rng.random(Ntot) * (S.edges[-1] - S.edges[-2])
    t = transport(E0, np.full(Ntot, 1.0 / Ntot), S.edges, 1, 7)
    phi_smc = t[:, 0] / np.diff(S.edges)
    print(f"G={G}, N/iter={N}, iterations={N_ITER}; SMC with {Ntot} histories: "
          f"rms {err(phi_smc)[0]:.3e} max {err(phi_smc)[1]:.3e}")

    n1 = AVG_START
    run("flat P0, phase 1 only (binned Galerkin fixed point)", "flat", 1, ["phase1"] * N_ITER)
    run("flat P0, split full residual every iteration (current r_c + integrated r_s)",
        "flat", 1, ["split"] * N_ITER)
    run("flat P0, phase 1 then phase 2 (averaged split correction passes)", "flat", 1,
        ["phase1"] * n1 + ["avg:split"] * (N_ITER - n1))
    run("flat P1, phase 1 then phase 2", "flat", 2,
        ["phase1"] * n1 + ["avg:split"] * (N_ITER - n1))
    run("collision-density P0, phase 1 only", "coll", 1, ["phase1"] * N_ITER)
    run("collision-density P1, phase 1 only", "coll", 2, ["phase1"] * N_ITER)
    run("collision-density P1, UNSPLIT full residual every iteration (+ Polyak avg)",
        "coll", 2, ["unsplit"] * N_ITER)
    run("collision-density P2, UNSPLIT full residual every iteration (+ Polyak avg)",
        "coll", 3, ["unsplit"] * N_ITER)
