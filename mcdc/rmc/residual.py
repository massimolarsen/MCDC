"""
Residual of the piecewise-constant trial-space solution psi~[k, g, j].

The residual r = Q - mu d(psi~)/dz - Sigma_t(E) psi~ + S[psi~] + F[psi~] is split, after
adding and subtracting the binned in-scatter S_bar + F_bar (weight cancellation), into
  - r_c (collision): c[k, g, j] - Sigma_t,k(E) psi~[k, g, j] inside each bin, with
    c = Q + S_bar + F_bar constant per bin and Sigma_t continuous in E;
  - r_e (edge): -mu (psi~_R - psi~_L) delta(z - z_f) at slab faces;
  - r_s, r_f (scattering/fission): (S + F)[psi~] - (S_bar + F_bar), sampled pointwise
    (see `mcdc.rmc.source`).
This module computes the deterministic pieces: c, the bin masses of |r_c| and |r_e|.
"""

import math

import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get

BOUNDARY_VACUUM = 0
BOUNDARY_REFLECTIVE = 1

# ======================================================================================
# Material total cross section (piecewise linear on the union grid)
# ======================================================================================


def material_total_xs(simulation, data, material, E_low, E_high):
    """Union energy grid in [E_low, E_high] and Sigma_t on it (exact: lin-lin)."""
    grids = []
    for i in range(material["N_nuclide"]):
        nuclide = simulation["nuclides"][
            mcdc_get.material.nuclide_IDs(i, material, data)
        ]
        grids.append(mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data))
    union = np.unique(np.concatenate(grids + [np.array([E_low, E_high])]))
    union = union[(union >= E_low) & (union <= E_high)]

    Sigma = np.zeros_like(union)
    for i in range(material["N_nuclide"]):
        nuclide = simulation["nuclides"][
            mcdc_get.material.nuclide_IDs(i, material, data)
        ]
        density = mcdc_get.material.nuclide_densities(i, material, data)
        grid = mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data)
        total = mcdc_get.nuclide.neutron_total_xs_all(nuclide, data)
        Sigma += density * np.interp(union, grid, total)
    return union, Sigma


@njit
def _abs_linear_integral(x0, x1, y0, y1):
    """int_{x0}^{x1} |y| dx for y linear between (x0, y0) and (x1, y1)."""
    if y0 * y1 >= 0.0:
        return 0.5 * (abs(y0) + abs(y1)) * (x1 - x0)
    x_root = x0 + (x1 - x0) * y0 / (y0 - y1)
    return 0.5 * (abs(y0) * (x_root - x0) + abs(y1) * (x1 - x_root))


@njit
def collision_residual_mass(c, psi, E_a, E_b, E_grid, Sigma):
    """int_{E_a}^{E_b} |c - psi Sigma_t(E)| dE, exact on the lin-lin union grid."""
    i0 = np.searchsorted(E_grid, E_a, side="right")
    i1 = np.searchsorted(E_grid, E_b, side="left")
    x_prev = E_a
    y_prev = c - psi * np.interp(E_a, E_grid, Sigma)
    total = 0.0
    for i in range(i0, i1 + 1):
        x = E_grid[i] if i < i1 else E_b
        y = c - psi * (Sigma[i] if i < i1 else np.interp(E_b, E_grid, Sigma))
        total += _abs_linear_integral(x_prev, x, y_prev, y)
        x_prev = x
        y_prev = y
    return total


# ======================================================================================
# Residual pieces for one iteration
# ======================================================================================


def binned_source(M, psi, E_edges, mu_edges):
    """S_bar[g, j] = 1/(dE_g dmu_j) sum_{g', j'} M[g', j', g, j] psi[g', j']."""
    dE = np.diff(E_edges)
    dmu = np.diff(mu_edges)
    return np.einsum("pqgj,pq->gj", M, psi) / np.outer(dE, dmu)


def mirror_index(mu_edges):
    """Index of the bin mirrored about mu = 0 (requires a symmetric grid)."""
    if not np.allclose(mu_edges, -mu_edges[::-1]):
        raise ValueError("RMC: reflective boundaries need a mu grid symmetric about 0")
    J = len(mu_edges) - 1
    return J - 1 - np.arange(J)


def face_jumps(psi, mu_edges, boundary_low, boundary_high):
    """
    Jump D[f, g, j] = psi~_R - psi~_L across faces f = 0..K of the slab, where the
    edge residual is -mu D delta(z - z_f). Only incoming directions carry a jump at
    the boundary faces (mu > 0 at f = 0, mu < 0 at f = K).
    """
    K, G, J = psi.shape
    D = np.zeros((K + 1, G, J))
    D[1:K] = psi[1:] - psi[:-1]

    mu_center = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    incoming_low = mu_center > 0.0
    incoming_high = mu_center < 0.0
    mirror = (
        mirror_index(mu_edges)
        if BOUNDARY_REFLECTIVE
        in (
            boundary_low,
            boundary_high,
        )
        else None
    )

    # z_low boundary: outside value is the boundary condition
    outside = psi[0][:, mirror] if boundary_low == BOUNDARY_REFLECTIVE else 0.0
    D[0][:, incoming_low] = (psi[0] - outside)[:, incoming_low]
    # z_high boundary
    outside = psi[-1][:, mirror] if boundary_high == BOUNDARY_REFLECTIVE else 0.0
    D[K][:, incoming_high] = (outside - psi[-1])[:, incoming_high]
    return D


def abs_mu_integral(mu_edges):
    """int_{bin j} |mu| dmu (bins must not straddle 0)."""
    a = mu_edges[:-1]
    b = mu_edges[1:]
    if np.any((a < 0.0) & (b > 0.0)):
        raise ValueError("RMC: mu = 0 must be a bin edge")
    return np.abs(b * np.abs(b) - a * np.abs(a)) / 2.0


class Residual:
    """
    Residual of one iteration on a slab trial space.

    Parameters
    ----------
    psi : ndarray (K, G, J)
        Trial-space angular flux psi~.
    Q : ndarray (K, G, J)
        Binned external source.
    M : list of ndarray (G, J, G, J)
        Material transfer moments (scattering + fission) per trial cell.
    xs : list of (E_grid, Sigma_t)
        Material total cross section per trial cell.
    """

    def __init__(
        self, psi, Q, M, xs, z_edges, E_edges, mu_edges, boundary_low, boundary_high
    ):
        K, G, J = psi.shape
        h = np.diff(z_edges)
        dE = np.diff(E_edges)
        dmu = np.diff(mu_edges)

        # Collision part: c = Q + S_bar, and |r_c| bin masses
        self.c = np.zeros((K, G, J))
        self.collision_mass = np.zeros((K, G, J))
        for k in range(K):
            self.c[k] = Q[k] + binned_source(M[k], psi[k], E_edges, mu_edges)
            E_grid, Sigma = xs[k]
            for g in range(G):
                for j in range(J):
                    self.collision_mass[k, g, j] = (
                        h[k]
                        * dmu[j]
                        * collision_residual_mass(
                            self.c[k, g, j],
                            psi[k, g, j],
                            E_edges[g],
                            E_edges[g + 1],
                            E_grid,
                            Sigma,
                        )
                    )

        # Edge part: jumps and |r_e| face masses
        self.jump = face_jumps(psi, mu_edges, boundary_low, boundary_high)
        self.edge_mass = (
            np.abs(self.jump)
            * dE[None, :, None]
            * abs_mu_integral(mu_edges)[None, None, :]
        )

    @property
    def collision_edge_norm(self):
        """L1 norm of r_c + r_e."""
        return self.collision_mass.sum() + self.edge_mass.sum()
