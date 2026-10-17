"""
3D Cartesian multigroup discrete-ordinates (S_N) solver: the fine reference of the
basis study.

  - Diamond difference with set-to-zero negative-flux fixup and rebalance [1, Ch. 4].
  - Product quadrature: piecewise Gauss-Legendre in mu = Omega_z and in
    phi = atan2(Omega_y, Omega_x), aligned with the edges of the finest patch layout,
    so dOmega = dmu dphi is integrated by the weights and every patch holds whole
    quadrature cells.
  - Scattering in real spherical harmonics up to order L (addition theorem), from the
    Legendre transfer moments of data.py; downscatter only, so groups are solved from
    the highest energy down, each with a within-group source iteration.
  - Vacuum or reflective faces; reflective incoming fluxes are lagged one iteration.

[1] E. E. Lewis and W. F. Miller, Computational Methods of Neutron Transport, Wiley
    (1984).
"""

import math

import numpy as np
from numba import njit, prange
from scipy.special import lpmv, factorial

VACUUM, REFLECTIVE = 0, 1


# ======================================================================================
# Quadrature and spherical harmonics
# ======================================================================================


def product_quadrature(n_mu_intervals, n_phi_intervals, points=3):
    """(mu, phi, w): piecewise Gauss on equal mu and phi intervals; sum w = 4 pi."""
    x, w = np.polynomial.legendre.leggauss(points)
    mu_edges = np.linspace(-1.0, 1.0, n_mu_intervals + 1)
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_intervals + 1)
    mu = (0.5 * (mu_edges[1:] + mu_edges[:-1])[:, None]
          + 0.5 * np.diff(mu_edges)[:, None] * x[None, :]).ravel()
    w_mu = (0.5 * np.diff(mu_edges)[:, None] * w[None, :]).ravel()
    phi = (0.5 * (phi_edges[1:] + phi_edges[:-1])[:, None]
           + 0.5 * np.diff(phi_edges)[:, None] * x[None, :]).ravel()
    w_phi = (0.5 * np.diff(phi_edges)[:, None] * w[None, :]).ravel()
    MU, PHI = np.meshgrid(mu, phi, indexing="ij")
    W = w_mu[:, None] * w_phi[None, :]
    return MU.ravel(), PHI.ravel(), W.ravel()


def real_harmonics(L, mu, phi):
    """Y[(l, m), d], orthonormal on the sphere, l <= L, m = -l..l."""
    out = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            k = abs(m)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * factorial(l - k) / factorial(l + k))
            P = lpmv(k, l, mu)
            if m == 0:
                out.append(norm * P)
            elif m > 0:
                out.append(math.sqrt(2.0) * norm * P * np.cos(k * phi))
            else:
                out.append(math.sqrt(2.0) * norm * P * np.sin(k * phi))
    return np.array(out)


def harmonic_degree(L):
    return np.array([l for l in range(L + 1) for _ in range(2 * l + 1)])


def mirror_directions(mu, phi):
    """Index of the reflected direction for each axis (x, y, z)."""
    def find(m, p):
        p = (p + np.pi) % (2 * np.pi) - np.pi
        d = (mu[:, None] - m[None, :]) ** 2 + (np.angle(np.exp(1j * (phi[:, None] - p[None, :])))) ** 2
        return np.argmin(d, axis=0)

    return np.stack([find(mu, np.pi - phi), find(mu, -phi), find(-mu, phi)])


# ======================================================================================
# Sweep
# ======================================================================================


@njit(cache=True, parallel=True)
def _sweep(source, Sigma, hx, hy, hz, ox, oy, oz, bc, mirror, in_x, in_y, in_z, psi,
           out_x, out_y, out_z, fixup):
    """
    One transport sweep of all directions for one group.
    source[cell, d] per unit volume and steradian; Sigma[cell]; psi[cell, d] cell
    averages (output). in_a / out_a [side, ., ., d]: per axis, the low (0) and high
    (1) face fluxes of the previous / this iteration (reflective boundaries).
    """
    nx, ny, nz = len(hx), len(hy), len(hz)
    N = len(ox)
    for d in prange(N):
        ax, ay, az = abs(ox[d]), abs(oy[d]), abs(oz[d])
        fx = np.zeros((ny, nz))
        fy = np.zeros((nx, nz))
        fz = np.zeros((nx, ny))
        # Incoming boundary values
        sx = 0 if ox[d] > 0 else 1
        sy = 0 if oy[d] > 0 else 1
        sz = 0 if oz[d] > 0 else 1
        if bc[0, sx] == REFLECTIVE:
            fx[:, :] = in_x[sx, :, :, mirror[0, d]]
        if bc[1, sy] == REFLECTIVE:
            fy[:, :] = in_y[sy, :, :, mirror[1, d]]
        if bc[2, sz] == REFLECTIVE:
            fz[:, :] = in_z[sz, :, :, mirror[2, d]]
        for kk in range(nz):
            k = kk if oz[d] > 0 else nz - 1 - kk
            for jj in range(ny):
                j = jj if oy[d] > 0 else ny - 1 - jj
                for ii in range(nx):
                    i = ii if ox[d] > 0 else nx - 1 - ii
                    c = (i * ny + j) * nz + k
                    cx = 2.0 * ax / hx[i]
                    cy = 2.0 * ay / hy[j]
                    cz = 2.0 * az / hz[k]
                    inx = fx[j, k]
                    iny = fy[i, k]
                    inz = fz[i, j]
                    value = (source[c, d] + cx * inx + cy * iny + cz * inz) / (
                        Sigma[c] + cx + cy + cz
                    )
                    outx = 2.0 * value - inx
                    outy = 2.0 * value - iny
                    outz = 2.0 * value - inz
                    if fixup and (outx < 0.0 or outy < 0.0 or outz < 0.0):
                        # Set-to-zero fixup with rebalance: fixed faces leak nothing
                        fixx = outx < 0.0
                        fixy = outy < 0.0
                        fixz = outz < 0.0
                        for _ in range(3):
                            num = source[c, d]
                            den = Sigma[c]
                            if fixx:
                                num += 0.5 * cx * inx
                            else:
                                num += cx * inx
                                den += cx
                            if fixy:
                                num += 0.5 * cy * iny
                            else:
                                num += cy * iny
                                den += cy
                            if fixz:
                                num += 0.5 * cz * inz
                            else:
                                num += cz * inz
                                den += cz
                            value = num / den
                            outx = 0.0 if fixx else 2.0 * value - inx
                            outy = 0.0 if fixy else 2.0 * value - iny
                            outz = 0.0 if fixz else 2.0 * value - inz
                            if outx < 0.0:
                                fixx = True
                            if outy < 0.0:
                                fixy = True
                            if outz < 0.0:
                                fixz = True
                        value = max(value, 0.0)
                    psi[c, d] = value
                    fx[j, k] = outx
                    fy[i, k] = outy
                    fz[i, j] = outz
                    # Outgoing boundary values, for reflective lagging
                    if ii == nx - 1:
                        out_x[1 - sx, j, k, d] = outx
                    if jj == ny - 1:
                        out_y[1 - sy, i, k, d] = outy
                    if kk == nz - 1:
                        out_z[1 - sz, i, j, d] = outz


def solve(problem, quadrature, L, tol=1e-6, max_iter=1000, verbose=True):
    """
    problem: dict with x, y, z (fine edges), material[cell] (index), Sigma_t[m, G],
    T[m, L+1, G, G] (data.group_constants), E_fine, q[cell, G] (isotropic source per
    unit volume, energy and steradian), bc[3, 2].
    Returns psi[G, cell, d] (float32, per unit energy) and the solver statistics.
    """
    hx, hy, hz = (np.diff(problem[a]) for a in "xyz")
    nx, ny, nz = len(hx), len(hy), len(hz)
    mu, phi, w = quadrature
    s = np.sqrt(1.0 - mu**2)
    ox, oy, oz = s * np.cos(phi), s * np.sin(phi), mu
    mirror = mirror_directions(mu, phi)
    Y = real_harmonics(L, mu, phi)  # (n_lm, D)
    degree = harmonic_degree(L)
    material = problem["material"]
    Sigma_t, T = problem["Sigma_t"], problem["T"]
    dE = np.diff(problem["E_fine"])
    G = len(dE)
    n_cell, D = nx * ny * nz, len(mu)
    bc = np.asarray(problem["bc"], dtype=np.int64)
    psi_all = np.zeros((G, n_cell, D), dtype=np.float32)
    moments = np.zeros((G, n_cell, len(degree)))
    Yw = Y * w  # moments = psi @ Yw.T
    stats = []
    for g in range(G - 1, -1, -1):
        # Downscatter source from higher groups (and the external source)
        s_lm = np.zeros((n_cell, len(degree)))
        for gp in range(g + 1, G):
            Tl = T[material, :, gp, g][:, degree]  # (cell, n_lm)
            s_lm += Tl * moments[gp]
        s_lm /= dE[g]
        fixed = s_lm @ Y + problem["q"][:, g][:, None]
        Tself = T[material, :, g, g][:, degree] / dE[g]
        Sigma = Sigma_t[material, g]
        psi = np.zeros((n_cell, D))
        b_in = [np.zeros((2, ny, nz, D)), np.zeros((2, nx, nz, D)), np.zeros((2, nx, ny, D))]
        b_out = [np.zeros((2, ny, nz, D)), np.zeros((2, nx, nz, D)), np.zeros((2, nx, ny, D))]
        m = np.zeros((n_cell, len(degree)))
        for it in range(max_iter):
            source = fixed + (Tself * m) @ Y
            _sweep(source, Sigma, hx, hy, hz, ox, oy, oz, bc, mirror,
                   *b_in, psi, *b_out, True)
            new = psi @ Yw.T
            change = np.max(np.abs(new - m)) / max(np.max(np.abs(new)), 1e-300)
            m = new
            b_in, b_out = b_out, b_in
            if change < tol:
                break
        moments[g] = m
        psi_all[g] = psi
        stats.append((g, it + 1, change))
        if change >= tol:
            raise RuntimeError(f"S_N group {g} did not converge in {max_iter} iterations")
        if verbose and g % 5 == 0:
            print(f"  group {g}: {it + 1} iterations, change {change:.1e}", flush=True)
    return psi_all, stats


def transport_only(problem, quadrature, source, Sigma, tol=1e-7, max_iter=500):
    """
    psi = (Omega . grad + Sigma_t)^-1 source for one group (no scattering), with the
    problem's boundary conditions (reflective ones iterated to convergence).
    source[cell, d]; Sigma[cell]. Returns psi[cell, d].
    """
    hx, hy, hz = (np.diff(problem[a]) for a in "xyz")
    nx, ny, nz = len(hx), len(hy), len(hz)
    mu, phi, w = quadrature
    s = np.sqrt(1.0 - mu**2)
    ox, oy, oz = s * np.cos(phi), s * np.sin(phi), mu
    mirror = mirror_directions(mu, phi)
    bc = np.asarray(problem["bc"], dtype=np.int64)
    D = len(mu)
    psi = np.zeros((nx * ny * nz, D))
    b_in = [np.zeros((2, ny, nz, D)), np.zeros((2, nx, nz, D)), np.zeros((2, nx, ny, D))]
    b_out = [np.zeros((2, ny, nz, D)), np.zeros((2, nx, nz, D)), np.zeros((2, nx, ny, D))]
    reflective = np.any(bc == REFLECTIVE)
    previous = None
    for it in range(max_iter if reflective else 1):
        _sweep(source, Sigma, hx, hy, hz, ox, oy, oz, bc, mirror, *b_in, psi, *b_out,
               False)
        b_in, b_out = b_out, b_in
        # A conserved scalar total can conceal unconverged signed spatial and
        # angular modes. Compare the full iterate instead of that total.
        if previous is not None:
            change = np.max(np.abs(psi - previous))
            if change <= tol * max(np.max(np.abs(psi)), 1e-300):
                break
        previous = psi.copy()
    else:
        if reflective:
            raise RuntimeError(f"Streaming inverse did not converge in {max_iter} iterations")
    return psi
