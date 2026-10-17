"""
Fine-group data for the basis study's discrete-ordinates reference (sn3d.py).

Physics: elastic scattering on a target at rest, isotropic in the centre-of-mass frame
(the synthetic thermal nuclides of ../fuel_rod are exactly this), and every other
reaction counted as removal. For an incident energy E the outgoing energy is uniform
on [alpha E, E], alpha = ((A - 1)/(A + 1))^2, with lab cosine

    mu0(E, E') = ((A + 1) sqrt(E'/E) - (A - 1) sqrt(E/E')) / 2.

Multigroup constants with a flat flux within each fine group:

    Sigma_t[g]      = (1/dE_g) int_g sum_n N_n sigma_t,n(E) dE,
    T_l[g' -> g]    = int_{g'} dE sum_n N_n sigma_s,n(E) int_{g cap [alpha E, E]} dE'
                        P_l(mu0(E, E')) / ((1 - alpha) E),

so the scattering source per unit energy in group g is sum_{g'} T_l[g'->g] / dE_g
times the l-th angular moments of psi_{g'}. Neutrons scattered below the window are
lost (the window's boundary condition).
"""

import numpy as np
from scipy.special import eval_legendre

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def _nodes(a, b, pieces=16):
    """Composite 8-point Gauss nodes and weights on [a, b] (resolves resonances)."""
    edges = np.linspace(a, b, pieces + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    x = (mid[:, None] + half[:, None] * _GL_X[None, :]).ravel()
    w = (half[:, None] * _GL_W[None, :]).ravel()
    return x, w


def fine_groups(E_edges, n_fine):
    """Fine group edges: each coarse bin split into n_fine equal-lethargy groups."""
    out = [E_edges[0]]
    for a, b in zip(E_edges[:-1], E_edges[1:]):
        out.extend(np.geomspace(a, b, n_fine + 1)[1:])
    return np.array(out)


def group_constants(nuclides, E_fine, L):
    """
    nuclides: [(A, N, (energy, sigma_t), (energy, sigma_s)), ...] with N in /b-cm.
    Returns Sigma_t[G] and T[L + 1, G', G] (see module doc).
    """
    G = len(E_fine) - 1
    Sigma_t = np.zeros(G)
    T = np.zeros((L + 1, G, G))
    for A, N, (E_t, s_t), (E_s, s_s) in nuclides:
        alpha = ((A - 1.0) / (A + 1.0)) ** 2
        for gp in range(G):
            E, w = _nodes(E_fine[gp], E_fine[gp + 1])
            Sigma_t[gp] += N * np.sum(w * np.interp(E, E_t, s_t)) / (
                E_fine[gp + 1] - E_fine[gp]
            )
            sigma = N * np.interp(E, E_s, s_s)
            for g in range(gp + 1):
                lo = np.maximum(E_fine[g], alpha * E)
                hi = np.minimum(E_fine[g + 1], E)
                ok = hi > lo
                if not np.any(ok):
                    continue
                # Inner integral over E' with 8-point Gauss, per incident node
                half = 0.5 * (hi - lo)
                mid = 0.5 * (hi + lo)
                Ep = mid[:, None] + half[:, None] * _GL_X[None, :]
                mu0 = 0.5 * (
                    (A + 1.0) * np.sqrt(Ep / E[:, None])
                    - (A - 1.0) * np.sqrt(E[:, None] / Ep)
                )
                mu0 = np.clip(mu0, -1.0, 1.0)
                density = 1.0 / ((1.0 - alpha) * E)
                for l in range(L + 1):
                    inner = np.sum(
                        half[:, None] * _GL_W[None, :] * eval_legendre(l, mu0), axis=1
                    )
                    T[l, gp, g] += np.sum((w * sigma * density * inner)[ok])
    return Sigma_t, T
