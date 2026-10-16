"""Energy polynomial order sweep (P0..P4), flat and collision-density bases, with
figure of merit FOM = 1/(error^2 x time) relative to SMC.

Per order:
  - deterministic Galerkin fixed point (direct solve): bin-average bias, pointwise
    shape error, residual |r|_1 left there;
  - one-term residual passes at that fixed point: variance and time per pass, and the
    per-pass FOM relative to an SMC pass with the same histories;
  - single-loop full-residual RMC (40 x 20000, averaging after iteration 10): error,
    total time, FOM relative to SMC with the same 8e5 histories;
  - setup time (fine grid, bin-to-point table U), paid once per material.
RMC0D_RES=0: smooth problem.
"""
import os
import time

import numpy as np

import run as R
from rmc0d import Setup, reference_collision_density, transport, SRC_LO, E_MAX

N, N_ITER, PASSES, REPS = 20000, 40, 50, 2


def galerkin(S):
    """Solve int_g P_a (q + T[c] - Sigma_t psi~[c]) dE = 0 for all (g, a)."""
    n = S.G * S.P
    A = np.zeros((n, n))
    for col in range(n):
        g, b = divmod(col, S.P)
        A[:, col] = S.moments(S.U[col] - S.St * S.phi[g, b]).ravel()
    return np.linalg.solve(A, -S.moments(S.q).ravel()).reshape(S.G, S.P)


def smc(S, Nh, seed):
    rng = np.random.default_rng(seed)
    E0 = SRC_LO + rng.random(Nh) * (E_MAX - SRC_LO)
    t0 = time.perf_counter()
    t = transport(E0, np.full(Nh, 1.0 / Nh), S.edges, 1, seed)
    return t[:, 0] / np.diff(S.edges), time.perf_counter() - t0


def single_loop(S, rep):
    R.N = N
    rng = np.random.default_rng(100 + rep)
    c = np.zeros((S.G, S.P))
    avg = []
    t0 = time.perf_counter()
    for n in range(N_ITER):
        dc, _, _ = R.iterate(S, c, "unsplit", rng, 9000 + 100 * rep + n)
        c = c + dc
        if n >= 10:
            avg.append(c.copy())
    dt = time.perf_counter() - t0
    if not np.isfinite(c).all() or np.abs(dc).sum() > np.abs(c).sum():
        return np.inf, dt
    return R.err(S.flux_bin_avg(np.mean(avg, 0)))[0], dt


label = "resonance" if float(os.environ.get("RMC0D_RES", "1")) else "smooth"
S0 = Setup(R.G, 1, "flat")
smc(S0, 1000, 0)  # compile
# SMC references: per-pass variance/time at N, and full-budget error/time
pp = [smc(S0, N, 50000 + p) for p in range(PASSES)]
v_smc = np.var([x[0] for x in pp], axis=0, ddof=1).sum()
t_smc_pass = np.mean([x[1] for x in pp])
full = [smc(S0, N * N_ITER, 7 + r) for r in range(REPS)]
e_smc = np.mean([R.err(x[0])[0] for x in full])
t_smc = np.mean([x[1] for x in full])
print(f"problem: {label}; G={R.G}; SMC {N*N_ITER} histories: rms {e_smc:.2e} in {t_smc:.2f}s")
print("basis p | FP bias  shape L1  |r|_1  | pass var/SMC  pass time/SMC  pass FOM/SMC | "
      "loop err  loop time/SMC  loop FOM/SMC | setup s")
for basis in ("flat", "coll"):
    for p in range(5):
        t0 = time.perf_counter()
        S = Setup(R.G, p + 1, basis)
        t_setup = time.perf_counter() - t0
        psi_ref = reference_collision_density(S.Ef, S.q) / S.St
        c = galerkin(S)
        bias = R.err(S.flux_bin_avg(c))[0]
        shape = np.trapezoid(np.abs(S.psi_fine(c) - psi_ref), S.Ef) / np.trapezoid(psi_ref, S.Ef)
        rnorm = np.trapezoid(np.abs(S.q + S.T_fine(c) - S.St * S.psi_fine(c)), S.Ef)
        R.N = N
        rng = np.random.default_rng(11)
        R.iterate(S, c, "unsplit", rng, 1)  # warm up
        t0 = time.perf_counter()
        phis = [S.flux_bin_avg(R.iterate(S, c, "unsplit", rng, 7000 + k)[0]) for k in range(PASSES)]
        t_pass = (time.perf_counter() - t0) / PASSES
        pv = np.var(phis, axis=0, ddof=1).sum() / v_smc
        pt = t_pass / t_smc_pass
        loops = [single_loop(S, r) for r in range(REPS)]
        el = np.mean([x[0] for x in loops])
        tl = np.mean([x[1] for x in loops]) / t_smc
        fom = (e_smc / el) ** 2 / tl if np.isfinite(el) else 0.0
        es = "diverged" if not np.isfinite(el) else f"{el:.2e}"
        print(f"{basis:4s} P{p} | {bias:.1e}  {shape:.1e}  {rnorm:.4f} | {pv:10.4f}  {pt:10.2f}  "
              f"{1/(pv*pt):11.1f} | {es:8s}  {tl:8.2f}  {fom:10.1f} | {t_setup:.1f}", flush=True)
