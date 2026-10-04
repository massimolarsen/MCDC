"""
Residual of the trial-space solution psi~[k, g, j, a]: piecewise constant in z and mu,
and in energy sum_a psi~[k, g, j, a] P_a(x_g(E)) with P_0 = 1, P_1 = x (constant or
linear discontinuous; x_g = (2E - E_g - E_{g+1}) / dE_g, see `mcdc.rmc.kernel`).

The residual r = Q - mu d(psi~)/dz - Sigma_t(E) psi~ + S[psi~] + F[psi~] is split, after
adding and subtracting the projected in-scatter S_bar + F_bar (weight cancellation), into
  - r_c (collision): c(E) - Sigma_t,k(E) psi~(E) inside each bin, with
    c = Q + S_bar + F_bar in the trial space and Sigma_t continuous in E (or, for the
    collision-only iterations, c(E) - P[Sigma_t psi~](E) with the projected removal);
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


@njit(cache=True)
def _abs_linear_integral(x0, x1, y0, y1):
    """int_{x0}^{x1} |y| dx for y linear between (x0, y0) and (x1, y1)."""
    if y0 * y1 >= 0.0:
        return 0.5 * (abs(y0) + abs(y1)) * (x1 - x0)
    x_root = x0 + (x1 - x0) * y0 / (y0 - y1)
    return 0.5 * (abs(y0) * (x_root - x0) + abs(y1) * (x1 - x_root))


@njit(cache=True)
def _quadratic(t, y0, b, a):
    return y0 + t * (b + t * a)


@njit(cache=True)
def _abs_quadratic_integral(x0, x1, y0, ym, y1):
    """
    int_{x0}^{x1} |y| dx for y quadratic through (x0, y0), (midpoint, ym), (x1, y1):
    split at its roots, then Simpson (exact) on each piece.
    """
    # y(t) = y0 + b t + a t^2 on t in [0, 1]
    a = 2.0 * y0 - 4.0 * ym + 2.0 * y1
    b = -3.0 * y0 + 4.0 * ym - y1
    roots = np.empty(4)
    roots[0] = 0.0
    N = 1
    scale = abs(y0) + abs(ym) + abs(y1)
    if abs(a) <= 1e-14 * scale:
        if b != 0.0:
            t = -y0 / b
            if 0.0 < t < 1.0:
                roots[N] = t
                N += 1
    else:
        discriminant = b * b - 4.0 * a * y0
        if discriminant > 0.0:
            q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
            for t in (q / a, y0 / q if q != 0.0 else -1.0):
                if 0.0 < t < 1.0:
                    roots[N] = t
                    N += 1
    roots[N] = 1.0
    N += 1
    roots[:N] = np.sort(roots[:N])
    total = 0.0
    for i in range(N - 1):
        ta = roots[i]
        tb = roots[i + 1]
        total += (tb - ta) * (
            abs(_quadratic(ta, y0, b, a))
            + 4.0 * abs(_quadratic(0.5 * (ta + tb), y0, b, a))
            + abs(_quadratic(tb, y0, b, a))
        )
    return total * (x1 - x0) / 6.0


@njit(cache=True)
def _trial_value(coefficients, E, E_a, E_b):
    """sum_a coefficients[a] P_a(x(E)) on the bin [E_a, E_b]."""
    value = coefficients[0]
    if len(coefficients) > 1:
        value += coefficients[1] * (2.0 * E - E_a - E_b) / (E_b - E_a)
    return value


@njit(cache=True)
def collision_residual_mass(c, psi, E_a, E_b, E_grid, Sigma):
    """
    int_{E_a}^{E_b} |c(E) - psi~(E) Sigma_t(E)| dE with c, psi~ the energy coefficients
    of one bin; exact on the lin-lin union grid (a quadratic on each grid interval).
    """
    i0 = np.searchsorted(E_grid, E_a, side="right")
    i1 = np.searchsorted(E_grid, E_b, side="left")
    x_prev = E_a
    total = 0.0
    for i in range(i0, i1 + 1):
        x = E_grid[i] if i < i1 else E_b
        if x > x_prev:
            y = np.empty(3)
            for n in range(3):
                E = x_prev + 0.5 * n * (x - x_prev)
                y[n] = _trial_value(c, E, E_a, E_b) - _trial_value(
                    psi, E, E_a, E_b
                ) * np.interp(E, E_grid, Sigma)
            total += _abs_quadratic_integral(x_prev, x, y[0], y[1], y[2])
        x_prev = x
    return total


# ======================================================================================
# Residual pieces for one iteration
# ======================================================================================


def binned_source(M, psi, E_edges, mu_edges):
    """
    S_bar[g, j, a] = (2a + 1)/(dE_g dmu_j)
                     sum_{a', g', j'} M[a', g', j', a, g, j] psi[g', j', a'].
    """
    dE = np.diff(E_edges)
    dmu = np.diff(mu_edges)
    P = psi.shape[-1]
    return (
        np.einsum("bpqagj,pqb->gja", M, psi)
        * (2.0 * np.arange(P) + 1.0)
        / np.outer(dE, dmu)[:, :, None]
    )


def projected_removal(table, E_edges, P):
    """
    R[g, a, b] = (2a + 1)/dE_g int_g Sigma_t P_a P_b dE: the projection of Sigma_t psi~
    onto the trial space is sum_b R[g, a, b] psi~[g, b] (exact: Simpson on the lin-lin
    grid, the integrand is at most cubic).
    """
    energy, Sigma = table
    G = len(E_edges) - 1
    R = np.zeros((G, P, P))
    for g in range(G):
        E_a, E_b = E_edges[g], E_edges[g + 1]
        x = np.concatenate(([E_a], energy[(energy > E_a) & (energy < E_b)], [E_b]))
        mid = 0.5 * (x[:-1] + x[1:])
        width = np.diff(x)
        for points, weight in ((x[:-1], 1.0), (mid, 4.0), (x[1:], 1.0)):
            s = (2.0 * points - E_a - E_b) / (E_b - E_a)
            polynomials = [np.ones_like(s), s][:P]
            values = np.interp(points, energy, Sigma) * weight * width / 6.0
            for a in range(P):
                for b in range(P):
                    R[g, a, b] += np.sum(values * polynomials[a] * polynomials[b])
        R[g] *= (2.0 * np.arange(P) + 1.0)[:, None] / (E_b - E_a)
    return R


def mirror_index(mu_edges):
    """Index of the bin mirrored about mu = 0 (requires a symmetric grid)."""
    if not np.allclose(mu_edges, -mu_edges[::-1]):
        raise ValueError("RMC: reflective boundaries need a mu grid symmetric about 0")
    J = len(mu_edges) - 1
    return J - 1 - np.arange(J)


def face_jumps(psi, mu_edges, boundary_low, boundary_high):
    """
    Jump D[f, g, j, a] = psi~_R - psi~_L across faces f = 0..K of the slab, where the
    edge residual is -mu D delta(z - z_f). Only incoming directions carry a jump at
    the boundary faces (mu > 0 at f = 0, mu < 0 at f = K).
    """
    K = psi.shape[0]
    D = np.zeros((K + 1,) + psi.shape[1:])
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
    psi : ndarray (K, G, J, P)
        Trial-space angular flux psi~ (P energy polynomials per bin).
    Q : ndarray (K, G, J, P)
        Projected external source.
    M : list of ndarray (P, G, J, P, G, J)
        Material transfer moments (scattering + fission) per trial cell.
    xs : list of (E_grid, Sigma_t)
        Material total cross section per trial cell.
    removal : list of ndarray (G, P, P), optional
        Projected removal per trial cell (`projected_removal`). If given, r_c uses
        P[Sigma_t psi~] (folded into c; then Sigma_t = 0 in r_c) instead of the
        pointwise Sigma_t(E) psi~(E).
    """

    def __init__(
        self,
        psi,
        Q,
        M,
        xs,
        z_edges,
        E_edges,
        mu_edges,
        boundary_low,
        boundary_high,
        removal=None,
    ):
        K, G, J, P = psi.shape
        h = np.diff(z_edges)
        dE = np.diff(E_edges)
        dmu = np.diff(mu_edges)
        no_xs = (np.array([E_edges[0], E_edges[-1]]), np.zeros(2))

        # Collision part: c = Q + S_bar (- P[Sigma_t psi~]), and |r_c| bin masses
        self.c = np.zeros((K, G, J, P))
        self.collision_mass = np.zeros((K, G, J))
        for k in range(K):
            self.c[k] = Q[k] + binned_source(M[k], psi[k], E_edges, mu_edges)
            E_grid, Sigma = xs[k]
            if removal is not None:
                self.c[k] -= np.einsum("gab,gjb->gja", removal[k], psi[k])
                E_grid, Sigma = no_xs
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

        # Edge part: jumps and |r_e| face masses, int_g |D(E)| dE int_j |mu| dmu
        self.jump = face_jumps(psi, mu_edges, boundary_low, boundary_high)
        D0 = self.jump[..., 0]
        D1 = self.jump[..., 1] if P > 1 else np.zeros_like(D0)
        mean_abs = np.where(
            np.abs(D1) <= np.abs(D0),
            np.abs(D0),
            0.5 * (D0 * D0 + D1 * D1) / np.maximum(np.abs(D1), 1e-300),
        )
        self.edge_mass = (
            mean_abs * dE[None, :, None] * abs_mu_integral(mu_edges)[None, None, :]
        )

    @property
    def collision_edge_norm(self):
        """L1 norm of r_c + r_e."""
        return self.collision_mass.sum() + self.edge_mass.sum()


# ======================================================================================
# Continuous piecewise-linear trial space in z (mcdc.rmc.space)
# ======================================================================================

_MU_X, _MU_W = np.polynomial.legendre.leggauss(8)


@njit(cache=True)
def collision_residual_mass_linear_z(
    c_left, c_right, psi_left, psi_right, delta, h, E_a, E_b, mu_a, mu_b, E_grid, Sigma
):
    """
    Approximately int_cell int_bin int_bin |r_c| dz dE dmu for
        r_c = c(s, E) - Sigma_t(E) psi~(s, E) - mu delta(E),
    with c and psi~ linear in s between the cell's left and right nodal coefficients
    and delta = (Phi_{k+1} - Phi_k) / h. Exact in s for each (E, mu), Gauss-Legendre
    in mu, Simpson on the union-grid intervals in E. Only shapes the proposal.
    """
    i0 = np.searchsorted(E_grid, E_a, side="right")
    i1 = np.searchsorted(E_grid, E_b, side="left")
    x_prev = E_a
    total = 0.0
    for i in range(i0, i1 + 1):
        x = E_grid[i] if i < i1 else E_b
        if x > x_prev:
            panel = 0.0
            for n in range(3):
                E = x_prev + 0.5 * n * (x - x_prev)
                S = np.interp(E, E_grid, Sigma)
                left = _trial_value(c_left, E, E_a, E_b) - S * _trial_value(
                    psi_left, E, E_a, E_b
                )
                right = _trial_value(c_right, E, E_a, E_b) - S * _trial_value(
                    psi_right, E, E_a, E_b
                )
                stream = _trial_value(delta, E, E_a, E_b)
                inner = 0.0
                for q in range(len(_MU_X)):
                    mu = 0.5 * (mu_b - mu_a) * _MU_X[q] + 0.5 * (mu_b + mu_a)
                    inner += (
                        0.5
                        * (mu_b - mu_a)
                        * _MU_W[q]
                        * _abs_linear_integral(
                            0.0, 1.0, left - mu * stream, right - mu * stream
                        )
                    )
                panel += (4.0 if n == 1 else 1.0) * inner
            total += panel * (x - x_prev) / 6.0
        x_prev = x
    return h * total


class ResidualLinearZ:
    """
    Residual of one iteration on the continuous piecewise-linear trial space in z.
    psi are nodal coefficients (K + 1, G, J, P); per cell k the residual uses the
    nodal values Phi_k (left) and Phi_{k+1} (right) with the cell's material:

        r_c = Q_k + (1 - s) S_bar[Phi_k] + s S_bar[Phi_{k+1}]
              - Sigma_t(E) ((1 - s) Phi_k + s Phi_{k+1}) - mu (Phi_{k+1} - Phi_k) / h_k

    (S acts on energy and angle only, so the in-scatter of a linear-in-z psi~ is the
    linear interpolation of the nodal in-scatter: no projection in z is needed). The
    boundary conditions are in the trial space, so there is no face residual.
    With `removal`, the projected removal replaces Sigma_t psi~ in c (as in Residual).
    """

    def __init__(
        self, psi, Q, M, xs, z_edges, E_edges, mu_edges, cell_material, removal=None
    ):
        K1, G, J, P = psi.shape
        K = K1 - 1
        h = np.diff(z_edges)
        self.psi_left = np.ascontiguousarray(psi[:-1])
        self.psi_right = np.ascontiguousarray(psi[1:])
        self.c_left = np.zeros((K, G, J, P))
        self.c_right = np.zeros((K, G, J, P))
        self.delta = (self.psi_right - self.psi_left) / h[:, None, None, None]
        self.collision_mass = np.zeros((K, G, J))
        no_xs = (np.array([E_edges[0], E_edges[-1]]), np.zeros(2))
        for k in range(K):
            m = cell_material[k]
            for side, nodal, c in (
                (0, self.psi_left[k], self.c_left),
                (1, self.psi_right[k], self.c_right),
            ):
                c[k] = Q[k] + binned_source(M[m], nodal, E_edges, mu_edges)
                if removal is not None:
                    c[k] -= np.einsum("gab,gjb->gja", removal[m], nodal)
            E_grid, Sigma = xs[m] if removal is None else no_xs
            for g in range(G):
                for j in range(J):
                    self.collision_mass[k, g, j] = collision_residual_mass_linear_z(
                        self.c_left[k, g, j],
                        self.c_right[k, g, j],
                        self.psi_left[k, g, j],
                        self.psi_right[k, g, j],
                        self.delta[k, g, j],
                        h[k],
                        E_edges[g],
                        E_edges[g + 1],
                        mu_edges[j],
                        mu_edges[j + 1],
                        E_grid,
                        Sigma,
                    )

    @property
    def collision_edge_norm(self):
        return self.collision_mass.sum()
