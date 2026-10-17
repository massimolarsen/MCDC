"""
How expensive is the pointwise residual of a streaming-exact trial function?

For psi~ = T^-1 (q + Q~) with Q~ flat and isotropic per coarse cell and fine group (the
"SX-P0 dg0" trial space of project.streaming_exact, Q~ = cell averages of the reference
scattering source), the residual at a point is r = S psi~ - Q~. A correction pass that
samples r pointwise needs S psi~(x, Omega, g), i.e. the angular moments of psi~ at x,
from backward rays of psi~ in many directions Omega'. This script ray-traces psi~
exactly (piecewise-constant source and cross sections on the coarse cells, reflective
or vacuum walls) at random points, and reports, against the number of ray directions
K, the error of S psi~ relative to |r| at the same points: the cost of estimating r to
the accuracy its smallness requires.
    python inner_cost.py [problem]      (thermal-lattice by default)
"""

import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import run as study_run  # noqa: E402
import sn3d  # noqa: E402


@njit(cache=True)
def _trace(x, y, z, ox, oy, oz, xe, ye, ze, source, Sigma, bc, max_tau):
    """
    psi~(x, -omega direction backward) for all groups: integrate the flat sources
    source[cell, g] (per steradian) attenuated by Sigma[cell, g] along the ray from
    (x, y, z) in direction -(ox, oy, oz), reflecting on reflective walls.
    Returns psi[g].
    """
    G = source.shape[1]
    psi = np.zeros(G)
    tau = np.zeros(G)
    dx, dy, dz = -ox, -oy, -oz
    nx, ny, nz = len(xe) - 1, len(ye) - 1, len(ze) - 1
    for _ in range(100000):
        i = min(max(np.searchsorted(xe, x) - 1, 0), nx - 1)
        j = min(max(np.searchsorted(ye, y) - 1, 0), ny - 1)
        k = min(max(np.searchsorted(ze, z) - 1, 0), nz - 1)
        # Distance to the cell boundary
        tx = ((xe[i + 1] if dx > 0 else xe[i]) - x) / dx if dx != 0 else 1e30
        ty = ((ye[j + 1] if dy > 0 else ye[j]) - y) / dy if dy != 0 else 1e30
        tz = ((ze[k + 1] if dz > 0 else ze[k]) - z) / dz if dz != 0 else 1e30
        s = max(min(tx, ty, tz), 0.0)
        c = (i * ny + j) * nz + k
        done = True
        for g in range(G):
            sig = Sigma[c, g]
            att = math.exp(-tau[g])
            if sig * s > 1e-12:
                psi[g] += att * source[c, g] / sig * (1.0 - math.exp(-sig * s))
            else:
                psi[g] += att * source[c, g] * s
            tau[g] += sig * s
            if tau[g] < max_tau:
                done = False
        if done:
            break
        eps = 1e-10
        x += (s + eps) * dx
        y += (s + eps) * dy
        z += (s + eps) * dz
        # Walls
        for axis in range(3):
            pos = x if axis == 0 else (y if axis == 1 else z)
            e = xe if axis == 0 else (ye if axis == 1 else ze)
            if pos < e[0] or pos > e[-1]:
                side = 0 if pos < e[0] else 1
                if bc[axis, side] == 0:
                    return psi
                wall = e[0] if side == 0 else e[-1]
                pos = 2.0 * wall - pos
                if axis == 0:
                    x = pos
                    dx = -dx
                elif axis == 1:
                    y = pos
                    dy = -dy
                else:
                    z = pos
                    dz = -dz
    return psi


@njit(cache=True, parallel=True)
def scalar_moments(points, directions, weights, Y, xe, ye, ze, source, Sigma, bc):
    """Angular moments of psi~ at each point: out[p, lm, g] = sum_d w_d Y_lm(d) psi~."""
    P = points.shape[0]
    n_lm = Y.shape[0]
    G = source.shape[1]
    out = np.zeros((P, n_lm, G))
    for p in prange(P):
        for d in range(directions.shape[0]):
            psi = _trace(points[p, 0], points[p, 1], points[p, 2], directions[d, 0],
                         directions[d, 1], directions[d, 2], xe, ye, ze, source, Sigma,
                         bc, 30.0)
            for lm in range(n_lm):
                for g in range(G):
                    out[p, lm, g] += weights[d] * Y[lm, d] * psi[g]
    return out


def directions_quadrature(K, rng):
    """K randomized (shifted) Fibonacci-lattice directions on the sphere, equal weights."""
    i = np.arange(K) + 0.5
    u, v = (i / K + rng.random()) % 1.0, (i * 0.6180339887498949 + rng.random()) % 1.0
    mu = 2.0 * u - 1.0
    phi = 2.0 * np.pi * v - np.pi
    s = np.sqrt(1.0 - mu**2)
    return np.stack([s * np.cos(phi), s * np.sin(phi), mu], axis=1), mu, phi, np.full(K, 4 * np.pi / K)


def main(name):
    p = study_run.PROBLEMS[name]()
    L = study_run.L_SCATTER
    quadrature = sn3d.product_quadrature(*study_run.QUADRATURE)
    start = time.time()
    psi, _ = sn3d.solve(
        dict(x=p["fine"][0], y=p["fine"][1], z=p["fine"][2], material=p["material"],
             Sigma_t=p["Sigma_t"], T=p["T"], E_fine=p["E_fine"], q=p["q"], bc=p["bc"]),
        quadrature, L, verbose=False,
    )
    print(f"reference {time.time() - start:.0f} s", flush=True)
    mu, phi, w = quadrature
    Y = sn3d.real_harmonics(L, mu, phi)
    degree = sn3d.harmonic_degree(L)
    G = len(p["E_fine"]) - 1
    dE = np.diff(p["E_fine"])
    T, mat = p["T"], p["material"]
    moments = np.stack([psi[g].astype(np.float64) @ (Y * w).T for g in range(G)])  # (G, cell, lm)
    S00 = np.zeros((G, moments.shape[1]))
    for g in range(G):
        for gp in range(g, G):
            S00[g] += T[mat, 0, gp, g] * moments[gp][:, 0]
        S00[g] /= dE[g]
    # Coarse cells: flat isotropic Q~ = volume average of S psi's isotropic part
    n = p["n"]
    coarse = p["coarse"]
    K3 = tuple(len(e) - 1 for e in coarse)
    fine_shape = tuple(n * k for k in K3)
    hx, hy, hz = (np.diff(f) for f in p["fine"])
    V = (hx[:, None, None] * hy[None, :, None] * hz[None, None, :]).ravel()
    ix, iy, iz = np.meshgrid(*(np.arange(f) for f in fine_shape), indexing="ij")
    coarse_of = ((ix // n) * K3[1] + iy // n) * K3[2] + iz // n
    coarse_of = coarse_of.ravel()
    Nc = int(np.prod(K3))
    Vc = np.bincount(coarse_of, V, Nc)
    Y00 = 1.0 / math.sqrt(4 * math.pi)
    Qc = np.stack([np.bincount(coarse_of, S00[g] * V, Nc) / Vc for g in range(G)], axis=1) * Y00
    # Match the E0 coarse-bin source used by project.streaming_exact, rather
    # than retaining three independent fine-group values per coarse bin.
    for g0 in range(0, G, p["n_fine"]):
        sl = slice(g0, g0 + p["n_fine"])
        Qc[:, sl] = (Qc[:, sl] @ dE[sl] / dE[sl].sum())[:, None]
    qc = np.stack([np.bincount(coarse_of, p["q"][:, g] * V, Nc) / Vc for g in range(G)], axis=1)
    mat_c = np.bincount(coarse_of, p["material"] * V, Nc) / Vc
    mat_c = np.rint(mat_c).astype(np.int64)
    Sigma_c = p["Sigma_t"][mat_c]  # (cell, G)
    source = qc + Qc  # per steradian (q isotropic per steradian already)
    bc = np.asarray(p["bc"], dtype=np.int64)
    xe, ye, ze = (np.asarray(e, dtype=float) for e in coarse)

    rng = np.random.default_rng(1)
    P = 400
    lo = np.array([xe[0], ye[0], ze[0]])
    hi = np.array([xe[-1], ye[-1], ze[-1]])
    points = lo + rng.random((P, 3)) * (hi - lo)
    # Reference moments with many directions
    ref_dirs, rmu, rphi, rw = directions_quadrature(16384, rng)
    Yr = sn3d.real_harmonics(L, rmu, rphi)
    start = time.time()
    ref = scalar_moments(points, ref_dirs, rw, Yr, xe, ye, ze, source, Sigma_c, bc)
    print(f"reference moments: {time.time() - start:.0f} s for {P} points x 16384 rays", flush=True)

    # Point values: S psi~ for a random outgoing direction and group; r = S psi~ - Q~
    out_mu = 2.0 * rng.random(P) - 1.0
    out_phi = 2.0 * np.pi * rng.random(P) - np.pi
    Yout = sn3d.real_harmonics(L, out_mu, out_phi)  # (lm, P)
    groups = rng.integers(0, G, P)
    cell_of = []
    for pt in points:
        i = min(np.searchsorted(xe, pt[0]) - 1, K3[0] - 1)
        j = min(np.searchsorted(ye, pt[1]) - 1, K3[1] - 1)
        k = min(np.searchsorted(ze, pt[2]) - 1, K3[2] - 1)
        cell_of.append((i * K3[1] + j) * K3[2] + k)
    cell_of = np.array(cell_of)
    mat_p = mat_c[cell_of]

    def S_point(mom):
        out = np.zeros(P)
        for pi in range(P):
            g = groups[pi]
            for gp in range(g, G):
                Tl = T[mat_p[pi], :, gp, g][degree] / dE[g]
                out[pi] += np.sum(Tl * Yout[:, pi] * mom[pi, :, gp])
        return out

    S_ref = S_point(ref)
    r = S_ref - Qc[cell_of, groups]
    print(f"mean |S psi~| = {np.mean(np.abs(S_ref)):.4g}, mean |r| = {np.mean(np.abs(r)):.4g}, "
          f"ratio {np.mean(np.abs(r)) / np.mean(np.abs(S_ref)):.4f}", flush=True)
    for K in (16, 64, 256, 1024, 4096):
        errors = []
        for trial in range(3):
            dirs, kmu, kphi, kw = directions_quadrature(K, rng)
            Yk = sn3d.real_harmonics(L, kmu, kphi)
            start = time.time()
            mom = scalar_moments(points, dirs, kw, Yk, xe, ye, ze, source, Sigma_c, bc)
            elapsed = time.time() - start
            errors.append(S_point(mom) - S_ref)
        errors = np.array(errors)
        rms = np.sqrt(np.mean(errors**2))
        print(f"K = {K:5d}: rms error of S psi~ / mean |r| = {rms / np.mean(np.abs(r)):.3f}, "
              f"time per point {elapsed / P * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thermal-lattice")
