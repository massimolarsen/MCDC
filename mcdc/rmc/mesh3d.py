"""
Residual of the 3D Cartesian, piecewise-constant RMC trial space
(writeups/cartesian_3d/cartesian_3d.tex).

psi~[kx, ky, kz, g, j] is constant on each cell x energy bin x angular patch
(`mcdc.rmc.patches`). Its residual r = q - L psi~ has two parts:

  - collision, inside every cell:  r_c = Q + S[psi~] - Sigma_t psi~, sampled with the
    binned in-scatter S_bar (collision-only iterations) and either the bin-averaged
    or the pointwise Sigma_t;
  - faces, on every face of every cell:  r_e = -(Omega . e_a) D delta(x_a - x_f), with
    D = psi~(downstream side) - psi~(upstream side) along +e_a, and at the boundary
    the jump from the boundary condition (vacuum: psi = 0; reflective: the mirrored
    patch), for incoming directions only.

Both are sampled in proportion to |r| (bins by mass, then uniform position and
energy; directions uniform on the patch for r_c and proportional to |Omega_a| for
r_e), with signed weights w = (N / N_term) r / q.
"""

import math

import numpy as np

from numba import njit, uint64

####

import mcdc.numba_types as type_
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
from mcdc.constant import COINCIDENCE_TOLERANCE_DIRECTION, PARTICLE_NEUTRON
from mcdc.rmc.patches import AXIS_X, AXIS_Y, AXIS_Z, sample_direction_abs
from mcdc.rmc.residual import (
    BOUNDARY_REFLECTIVE,
    BOUNDARY_VACUUM,
    collision_residual_mass,
)
from mcdc.rmc.source import FACE_NUDGE


class CartesianMesh:
    """Structured x-y-z cells; widths, volumes and face areas."""

    def __init__(self, x_edges, y_edges, z_edges):
        self.edges = tuple(
            np.asarray(e, dtype=float) for e in (x_edges, y_edges, z_edges)
        )
        for e in self.edges:
            if len(e) < 2 or np.any(np.diff(e) <= 0.0):
                raise ValueError("RMC: mesh edges must increase")
        self.shape = tuple(len(e) - 1 for e in self.edges)
        self.width = tuple(np.diff(e) for e in self.edges)
        hx, hy, hz = self.width
        self.volume = hx[:, None, None] * hy[None, :, None] * hz[None, None, :]
        # Face area per axis, broadcastable over the face index grid of that axis
        self.face_area = (
            (hy[:, None] * hz[None, :])[None, :, :],
            (hx[:, None] * hz[None, :])[:, None, :],
            (hx[:, None] * hy[None, :])[:, :, None],
        )

    @property
    def N_cell(self):
        return int(np.prod(self.shape))


def face_jumps_3d(psi, axis, boundary, patches, mirror=None):
    """
    Jump D across the faces of `axis` (shape of psi with N_a + 1 along that axis):
    D = psi~(+ side) - psi~(- side) at interior faces; at the low boundary
    D = psi~(first cell) - psi_bc for Omega_a > 0, and at the high boundary
    D = psi_bc - psi~(last cell) for Omega_a < 0 (0 for outgoing directions).
    psi_bc = 0 (vacuum) or the mirrored patch of the adjacent cell (reflective).
    """
    p = np.moveaxis(psi, axis, 0)
    D = np.zeros((p.shape[0] + 1,) + p.shape[1:])
    D[1:-1] = p[1:] - p[:-1]
    sign = patches.direction_sign[axis]
    low, high = boundary
    outside = p[0][..., mirror] if low == BOUNDARY_REFLECTIVE else 0.0
    D[0][..., sign > 0] = (p[0] - outside)[..., sign > 0]
    outside = p[-1][..., mirror] if high == BOUNDARY_REFLECTIVE else 0.0
    D[-1][..., sign < 0] = (outside - p[-1])[..., sign < 0]
    return np.moveaxis(D, 0, axis)


class Residual3D:
    """
    Residual of one iteration on the 3D trial space.

    Parameters
    ----------
    psi, Q : ndarray (Kx, Ky, Kz, G, J)
        Trial-space angular flux and projected source (per unit volume, energy and
        solid angle).
    S_bar : ndarray (Kx, Ky, Kz, G, J)
        Binned in-scatter (+ fission) of psi~.
    removal : ndarray (Kx, Ky, Kz, G) or None
        Bin-averaged Sigma_t per cell; if given, r_c = Q + S_bar - Sigma_bar psi~ is
        constant in each bin, else r_c uses the pointwise Sigma_t(E) of `xs`.
    xs : list of (E_grid, Sigma_t) per cell (flattened cell order)
    """

    def __init__(
        self, psi, Q, S_bar, mesh, E_edges, patches, boundary, mirrors, removal, xs
    ):
        G, J = psi.shape[3:]
        dE = np.diff(E_edges)
        self.c = Q + S_bar
        cell_measure = (
            mesh.volume[..., None, None] * patches.area[None, None, None, None, :]
        )
        if removal is not None:
            self.c = self.c - removal[..., None] * psi
            self.collision_mass = cell_measure * dE[:, None] * np.abs(self.c)
        else:
            self.collision_mass = np.zeros(psi.shape)
            c_flat = self.c.reshape(-1, G, J)
            psi_flat = psi.reshape(-1, G, J)
            mass = self.collision_mass.reshape(-1, G, J)
            for n in range(c_flat.shape[0]):
                E_grid, Sigma = xs[n]
                for g in range(G):
                    for j in range(J):
                        mass[n, g, j] = collision_residual_mass(
                            c_flat[n, g, j : j + 1],
                            psi_flat[n, g, j : j + 1],
                            E_edges[g],
                            E_edges[g + 1],
                            E_grid,
                            Sigma,
                        )
            self.collision_mass *= cell_measure

        self.jump = []
        self.face_mass = []
        for axis in (AXIS_X, AXIS_Y, AXIS_Z):
            D = face_jumps_3d(psi, axis, boundary[axis], patches, mirrors[axis])
            self.jump.append(D)
            self.face_mass.append(
                mesh.face_area[axis][..., None, None]
                * dE[:, None]
                * np.abs(D)
                * patches.abs_direction_integral[axis]
            )

    @property
    def norm(self):
        """L1 norm of r_c + r_e."""
        return self.collision_mass.sum() + sum(m.sum() for m in self.face_mass)


# ======================================================================================
# Sampling
# ======================================================================================


@njit(cache=True)
def _new_container(seed):
    container = np.zeros(1, type_.particle_data)
    container[0]["rng_seed"] = seed
    return container


@njit(cache=True)
def _bank(container, x, y, z, E, mu, phi, w, simulation):
    particle = container[0]
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    particle["x"] = x
    particle["y"] = y
    particle["z"] = z
    particle["t"] = 0.0
    particle["ux"] = s * math.cos(phi)
    particle["uy"] = s * math.sin(phi)
    particle["uz"] = mu
    particle["E"] = E
    particle["w"] = w
    particle["particle_type"] = PARTICLE_NEUTRON
    particle_bank_module.bank_source_particle(container, simulation)


@njit(cache=True)
def _sample_cdf(cdf, xi):
    idx = np.searchsorted(cdf, xi * cdf[-1], side="right") - 1
    return min(max(idx, 0), len(cdf) - 2)


@njit(cache=True)
def _uniform(edges, k, container):
    return edges[k] + rng.lcg(container) * (edges[k + 1] - edges[k])


# MC/DC's tally filter assigns a direction within COINCIDENCE_TOLERANCE_DIRECTION of a
#   polar or azimuthal bin edge to the lower bin. Residual particles are kept this far
#   inside their patch, so each is scored in the patch it was sampled in.
_PATCH_MARGIN = 2.0 * COINCIDENCE_TOLERANCE_DIRECTION


@njit(cache=True)
def _inside(value, low, high):
    return min(max(value, low + _PATCH_MARGIN), high - _PATCH_MARGIN)


@njit(cache=True)
def _face_particle(
    idx,
    Kx,
    Ky,
    Kz,
    G,
    J,
    container,
    x_edges,
    y_edges,
    z_edges,
    E_edges,
    mu_low,
    mu_high,
    phi_low,
    phi_high,
    abs_integral,
    jump_x,
    jump_y,
    jump_z,
):
    """
    A particle of the face residual from face bin idx (x, then y, then z face grids,
    each (face, g, j) flattened): uniform on the face and in energy, direction with
    density |Omega_a| on the patch. Returns its phase-space point and int r over the
    bin, so that its weight is that times (total mass / bin mass).
    """
    N_fx = (Kx + 1) * Ky * Kz * G * J
    N_fy = Kx * (Ky + 1) * Kz * G * J
    if idx < N_fx:
        axis = AXIS_X
        n_x, n_y, n_z = Kx + 1, Ky, Kz
    elif idx < N_fx + N_fy:
        axis = AXIS_Y
        idx -= N_fx
        n_x, n_y, n_z = Kx, Ky + 1, Kz
    else:
        axis = AXIS_Z
        idx -= N_fx + N_fy
        n_x, n_y, n_z = Kx, Ky, Kz + 1
    rest, j = divmod(idx, J)
    rest, g = divmod(rest, G)
    rest, kz = divmod(rest, n_z)
    kx, ky = divmod(rest, n_y)
    E = _uniform(E_edges, g, container)
    mu, phi = sample_direction_abs(
        axis,
        mu_low[j],
        mu_high[j],
        phi_low[j],
        phi_high[j],
        rng.lcg(container),
        rng.lcg(container),
    )
    mu = _inside(mu, mu_low[j], mu_high[j])
    phi = _inside(phi, phi_low[j], phi_high[j])
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    if axis == AXIS_X:
        omega = s * math.cos(phi)
        D = jump_x[kx, ky, kz, g, j]
        x = x_edges[kx] + math.copysign(FACE_NUDGE, omega)
        y = _uniform(y_edges, ky, container)
        z = _uniform(z_edges, kz, container)
        face = (y_edges[ky + 1] - y_edges[ky]) * (z_edges[kz + 1] - z_edges[kz])
    elif axis == AXIS_Y:
        omega = s * math.sin(phi)
        D = jump_y[kx, ky, kz, g, j]
        x = _uniform(x_edges, kx, container)
        y = y_edges[ky] + math.copysign(FACE_NUDGE, omega)
        z = _uniform(z_edges, kz, container)
        face = (x_edges[kx + 1] - x_edges[kx]) * (z_edges[kz + 1] - z_edges[kz])
    else:
        omega = mu
        D = jump_z[kx, ky, kz, g, j]
        x = _uniform(x_edges, kx, container)
        y = _uniform(y_edges, ky, container)
        z = z_edges[kz] + math.copysign(FACE_NUDGE, omega)
        face = (x_edges[kx + 1] - x_edges[kx]) * (y_edges[ky + 1] - y_edges[ky])
    # q = mass / total / (face dE) |Omega_a| / int_patch |Omega_a|
    value = (
        math.copysign(1.0, -omega)
        * D
        * face
        * (E_edges[g + 1] - E_edges[g])
        * abs_integral[axis, j]
    )
    return x, y, z, E, mu, phi, value


@njit(cache=True)
def sample_residual_3d(
    i_start,
    i_end,
    seed,
    x_edges,
    y_edges,
    z_edges,
    E_edges,
    mu_low,
    mu_high,
    phi_low,
    phi_high,
    area,
    abs_integral,
    psi,
    c,
    jump_x,
    jump_y,
    jump_z,
    mass,
    xs_offsets,
    xs_energy,
    xs_total,
    cell_material,
    simulation,
):
    """
    Bank sample indices [i_start, i_end) of the particles drawn from r_c + r_e.
    mass is the flattened concatenation of the collision masses (kx, ky, kz, g, j) and
    the face masses of x, y and z (face grids (Kx+1, Ky, Kz), (Kx, Ky+1, Kz),
    (Kx, Ky, Kz+1), times (g, j)); psi and c are (Kx, Ky, Kz, G, J), the jumps are on
    the face grids. xs_* hold each material's union grid and the Sigma_t used in r_c
    (zero when the removal is binned into c).
    """
    Kx, Ky, Kz, G, J = psi.shape
    N_c = Kx * Ky * Kz * G * J
    cdf = np.zeros(len(mass) + 1)
    cdf[1:] = np.cumsum(mass)
    total = cdf[-1]

    for i in range(i_start, i_end):
        container = _new_container(rng.split_seed(uint64(i), seed))
        if total == 0.0:
            _bank(container, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, simulation)
            continue
        n = _sample_cdf(cdf, rng.lcg(container))
        if n < N_c:
            # Collision residual: uniform in cell, energy bin and patch
            rest, j = divmod(n, J)
            rest, g = divmod(rest, G)
            rest, kz = divmod(rest, Kz)
            kx, ky = divmod(rest, Ky)
            x = _uniform(x_edges, kx, container)
            y = _uniform(y_edges, ky, container)
            z = _uniform(z_edges, kz, container)
            E = _uniform(E_edges, g, container)
            mu = mu_low[j] + rng.lcg(container) * (mu_high[j] - mu_low[j])
            phi = phi_low[j] + rng.lcg(container) * (phi_high[j] - phi_low[j])
            mu = _inside(mu, mu_low[j], mu_high[j])
            phi = _inside(phi, phi_low[j], phi_high[j])
            m = cell_material[(kx * Ky + ky) * Kz + kz]
            start = xs_offsets[m]
            end = xs_offsets[m + 1]
            Sigma = np.interp(E, xs_energy[start:end], xs_total[start:end])
            r = c[kx, ky, kz, g, j] - Sigma * psi[kx, ky, kz, g, j]
            measure = (
                (x_edges[kx + 1] - x_edges[kx])
                * (y_edges[ky + 1] - y_edges[ky])
                * (z_edges[kz + 1] - z_edges[kz])
                * (E_edges[g + 1] - E_edges[g])
                * area[j]
            )
            w = r * measure * total / mass[n]
        else:
            x, y, z, E, mu, phi, value = _face_particle(
                n - N_c,
                Kx,
                Ky,
                Kz,
                G,
                J,
                container,
                x_edges,
                y_edges,
                z_edges,
                E_edges,
                mu_low,
                mu_high,
                phi_low,
                phi_high,
                abs_integral,
                jump_x,
                jump_y,
                jump_z,
            )
            w = value * total / mass[n]
        # MC/DC divides the tallies by N_total, so w = r / q estimates int r
        _bank(container, x, y, z, E, mu, phi, w, simulation)
