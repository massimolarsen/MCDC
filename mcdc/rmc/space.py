"""
Continuous piecewise-linear spatial trial space in z (see
writeups/continuous_linear_space.tex).

psi~(z, E, mu) = sum_i h_i(z) Phi_i(E, mu) with hat functions h_i on the nodes
z_0..z_K; on cell k, psi~ = (1 - s) Phi_k + s Phi_{k+1}, s = (z - z_k) / h_k. The
nodal coefficients Phi[i, g, j, a] use the same energy-angle trial space as the
piecewise-constant scheme (polar bins j, energy polynomials a).

Boundary conditions are imposed on the trial space (exactly satisfied by the solution):
  - vacuum: incoming nodal values are zero (mu > 0 at z_0, mu < 0 at z_K);
  - reflective: the boundary node's value of a polar bin equals that of its mirror bin
    (for the angular slope coefficients, minus it: y_j*(-mu) = -y_j(mu)).
With them, and with continuity inside, the residual has no face terms.

The projection of a function f is the L2 projection onto that constrained space. With
t[i, g, j, a] = int f h_i P_a dz dE dmu (the tallies) it solves, per (g, a),

    T^T (D_mu (x) M) T u = T^T t (2a + 1) / dE_g,     Phi = T u,

where M_il = int h_i h_l dz (tridiagonal), D_mu = diag(dmu_j), and T maps the reduced
unknowns u to all nodal values (zero for vacuum-incoming, shared for mirrored pairs).
"""

import numpy as np

####

from mcdc.rmc.residual import (
    BOUNDARY_REFLECTIVE,
    BOUNDARY_VACUUM,
    coefficient_norms,
    mirror_index,
)


def mass_matrix(z_edges):
    """M_il = int h_i h_l dz on the nodes of z_edges (tridiagonal, dense array)."""
    h = np.diff(z_edges)
    K = len(h)
    M = np.zeros((K + 1, K + 1))
    for k in range(K):
        M[k, k] += h[k] / 3.0
        M[k + 1, k + 1] += h[k] / 3.0
        M[k, k + 1] += h[k] / 6.0
        M[k + 1, k] += h[k] / 6.0
    return M


def constraint_map(K, mu_edges, boundary_low, boundary_high, odd=False):
    """
    T[(j, i), u]: nodal values from the reduced unknowns, with the full index
    j * (K + 1) + i (polar bin j, node i). odd: the coefficients of an odd angular
    polynomial, tied to minus their mirror at a reflective boundary.
    """
    J = len(mu_edges) - 1
    mu_center = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mirror = None
    if BOUNDARY_REFLECTIVE in (boundary_low, boundary_high):
        mirror = mirror_index(mu_edges)

    # reduced[j, i]: reduced unknown of nodal value (j, i), or -1 if fixed to zero;
    #   sign[j, i]: -1 for an odd coefficient tied to its mirror
    reduced = -np.ones((J, K + 1), dtype=np.int64)
    sign = np.ones((J, K + 1))
    N = 0
    for j in range(J):
        for i in range(K + 1):
            if i == 0 and boundary_low == BOUNDARY_VACUUM and mu_center[j] > 0.0:
                continue
            if i == K and boundary_high == BOUNDARY_VACUUM and mu_center[j] < 0.0:
                continue
            reflective = (i == 0 and boundary_low == BOUNDARY_REFLECTIVE) or (
                i == K and boundary_high == BOUNDARY_REFLECTIVE
            )
            if reflective and mirror[j] < j:
                reduced[j, i] = reduced[mirror[j], i]
                sign[j, i] = -1.0 if odd else 1.0
                continue
            reduced[j, i] = N
            N += 1

    T = np.zeros((J * (K + 1), N))
    for j in range(J):
        for i in range(K + 1):
            if reduced[j, i] >= 0:
                T[j * (K + 1) + i, reduced[j, i]] = sign[j, i]
    return T


class LinearZProjection:
    """L2 projection onto the constrained continuous piecewise-linear space in z."""

    def __init__(self, z_edges, E_edges, mu_edges, boundary_low, boundary_high):
        z_edges = np.asarray(z_edges, dtype=float)
        self.K = len(z_edges) - 1
        self.J = len(mu_edges) - 1
        self.dE = np.diff(E_edges)
        dmu = np.diff(mu_edges)
        full = np.kron(np.diag(dmu), mass_matrix(z_edges))
        # Even and odd angular polynomials (different reflective ties)
        self.T = [
            constraint_map(self.K, mu_edges, boundary_low, boundary_high, odd)
            for odd in (False, True)
        ]
        self.A = [T.T @ full @ T for T in self.T]

    def project(self, t, P=None):
        """
        Nodal coefficients Phi[i, g, j, c] from moments t[i, g, j, c] =
        int f h_i P_a P_b, c = b P + a (P energy polynomials; default: all).
        """
        K1, G, J, C = t.shape
        P = P or C
        norms = coefficient_norms(C, P)
        Phi = np.empty_like(t)
        for b in range(C // P):
            cs = slice(b * P, (b + 1) * P)
            T, A = self.T[b % 2], self.A[b % 2]
            scale = norms[cs][None, :] / self.dE[:, None]  # (G, P)
            rhs = np.transpose(t[..., cs], (2, 0, 1, 3)) * scale[None, None]
            rhs = rhs.reshape(J * K1, G * P)
            u = np.linalg.solve(A, T.T @ rhs)
            Phi[..., cs] = np.transpose((T @ u).reshape(J, K1, G, P), (1, 2, 0, 3))
        return np.ascontiguousarray(Phi)


def node_moments(m0, m1):
    """
    Hat-function moments t[i] = int f h_i from the cell moments m0[k] = int_k f and
    m1[k] = int_k f x_k (x_k = 2s - 1): t_k += (m0 - m1)/2, t_{k+1} += (m0 + m1)/2.
    """
    K = m0.shape[0]
    t = np.zeros((K + 1,) + m0.shape[1:])
    t[:K] += 0.5 * (m0 - m1)
    t[1:] += 0.5 * (m0 + m1)
    return t


def cell_average(Phi):
    """Cell averages (Phi_k + Phi_{k+1}) / 2 of a nodal array."""
    return 0.5 * (Phi[:-1] + Phi[1:])
