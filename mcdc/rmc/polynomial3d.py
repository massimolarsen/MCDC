"""
Residual of the 3D Cartesian RMC trial space with a polynomial spatial basis
(`mcdc.rmc.space3d`, writeups/cartesian_3d/spatial_bases.tex), and its sampling.

Inside cell k, for energy bin g and patch j, with t the local coordinates,

    r(t, E, Omega) = sum_m P_m(t) [Q_m + T_m(E, Omega) - Sigma_t(E) C_m - Omega . G_m],

with C_m the Legendre coefficients of psi~, G_m those of its gradient, and T_m the
in-scatter of mode m. In the collision-only iterations T_m is the binned S_bar_m and
the removal is binned into it (Sigma_t = 0 in the sampler); in a correction pass T_m
is the interpolated tabulated in-scatter (`mcdc.rmc.correction3d`) and Sigma_t is
pointwise. Both use the same node grid in (E, mu, phi) (one interval per bin and patch
for the iterations), so one sampler serves both. On the faces, r_e = -Omega_n D, with
D the jump of psi~ (a polynomial on the face).

Sampling: volume sub-cells (cell, energy interval, mu and phi interval of a patch) and
face bins (face, energy bin, patch) are drawn from one CDF over their masses, then the
point is uniform inside (the direction proportional to |Omega_n| on faces) and the
weight is r / q with r evaluated at the point. Masses use the root mean square of the
spatial polynomial (exact through orthogonality, sum_m c_m^2 / norm_m), which bounds
its mean absolute value; the largest value over the sub-cell's (E, Omega) corners is
taken, plus the departure of Sigma_t from linear between energy nodes.
The non-bilinear angular streaming departure is bounded separately by the
direction-component ranges times the RMS of each gradient component.
"""

import math

import numpy as np

from numba import njit, uint64

####

import mcdc.transport.rng as rng

from mcdc.rmc.mesh3d import _bank, _inside, _new_container, _sample_cdf, _uniform
from mcdc.rmc.patches import AXIS_X, AXIS_Y, sample_direction_abs
from mcdc.rmc.source import FACE_NUDGE


def direction_nodes(nodes, patches):
    """Omega at the direction nodes of every patch: (J, n_mu, n_phi, 3)."""
    t_mu = np.linspace(0.0, 1.0, nodes.n_mu)
    t_phi = np.linspace(0.0, 1.0, nodes.n_phi)
    mu = patches.mu_low[:, None] + t_mu[None, :] * (patches.mu_high - patches.mu_low)[:, None]
    phi = (
        patches.phi_low[:, None]
        + t_phi[None, :] * (patches.phi_high - patches.phi_low)[:, None]
    )
    s = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    out = np.empty((patches.J, nodes.n_mu, nodes.n_phi, 3))
    out[..., 0] = s[:, :, None] * np.cos(phi)[:, None, :]
    out[..., 1] = s[:, :, None] * np.sin(phi)[:, None, :]
    out[..., 2] = mu[:, :, None]
    return out


def volume_masses(basis, C, Q, Tn, gradient, Sigma_nodes, cell_material, nodes,
                  patches, departure):
    """
    Mass of each volume sub-cell (k, e, j, a, b), flattened in that order.
    Tn[k, m, i, j, a, b]: the in-scatter at the nodes (binned S_bar_m - Sigma_bar C_m
    for the iterations); Sigma_nodes[material, i] the pointwise Sigma_t at the energy
    nodes (zero for the iterations); departure[material, e] as in correction3d.
    """
    mesh = basis.mesh
    K = Tn.shape[0]
    G, J, M = C.shape[-3:]
    g = nodes.node_bin
    C_flat = C.reshape(K, G, J, M)
    Q_flat = Q.reshape(K, G, J, M)
    grad = gradient.reshape(3, K, G, J, M)
    omega = direction_nodes(nodes, patches)  # (J, n_mu, n_phi, 3)
    # Coefficients of r at every node: (K, M, N_nodes, J, n_mu, n_phi)
    base = Q_flat[:, g] - Sigma_nodes[cell_material][:, :, None, None] * C_flat[:, g]
    r = Tn + np.transpose(base, (0, 3, 1, 2))[..., None, None]
    streaming = np.einsum("jabd,dkgjm->kmgjab", omega, grad)  # (K, M, G, J, a, b)
    r -= streaming[:, :, g]
    rms = np.sqrt(np.einsum("kmijab,m->kijab", r**2, 1.0 / basis.norm))
    left = nodes.interval_left
    rms = np.maximum(rms[:, left], rms[:, left + 1])
    rms = np.maximum(rms[..., :-1, :], rms[..., 1:, :])
    rms = np.maximum(rms[..., :-1], rms[..., 1:])  # (K, N_int, J, A, B)
    # Omega_x and Omega_y are not bilinear in (mu, phi). The bilinear
    # interpolant lies within each component's corner range, as does the true
    # component on an octant sub-patch. Their difference is bounded by that
    # range. Omega_z = mu is interpolated exactly. Retain this departure even
    # when the residual cancels at every corner (otherwise support is lost).
    component_range = []
    for axis in (AXIS_X, AXIS_Y):
        values = omega[..., axis]
        low = np.minimum(values[:, :-1], values[:, 1:])
        high = np.maximum(values[:, :-1], values[:, 1:])
        low = np.minimum(low[..., :-1], low[..., 1:])
        high = np.maximum(high[..., :-1], high[..., 1:])
        component_range.append(high - low)
    for axis in (AXIS_X, AXIS_Y):
        grad_rms = np.sqrt(basis.mean_square(grad[axis]))
        rms += (
            grad_rms[:, nodes.interval_bin, :, None, None]
            * component_range[axis][None, None]
        )
    psi_rms = np.sqrt(basis.mean_square(C_flat))  # (K, G, J)
    rms = rms + (
        psi_rms[:, nodes.interval_bin] * departure[cell_material][:, :, None]
    )[..., None, None]
    dE = nodes.E_nodes[left + 1] - nodes.E_nodes[left]
    sub_area = patches.area / ((nodes.n_mu - 1) * (nodes.n_phi - 1))
    measure = mesh.volume.reshape(K)[:, None, None] * dE[None, :, None] * sub_area
    return (rms * measure[..., None, None]).ravel()


def face_masses(basis, jumps, E_edges, patches):
    """Mass of each face bin: area dE int_patch |Omega_n| sqrt(mean D^2)."""
    dE = np.diff(E_edges)
    out = []
    for axis in range(3):
        D = jumps[axis]
        rms = np.sqrt(np.sum(D**2 / basis.face_norm(axis), axis=-1))
        out.append(
            (
                basis.mesh.face_area[axis][..., None, None]
                * dE[:, None]
                * rms
                * patches.abs_direction_integral[axis]
            ).ravel()
        )
    return out


@njit(cache=True)
def _legendre(n, t):
    if n == 0:
        return 1.0
    if n == 1:
        return t
    return 1.5 * t * t - 0.5


@njit(cache=True)
def _face_value(D, face_modes, t1, t2):
    total = 0.0
    for f in range(len(D)):
        total += D[f] * _legendre(face_modes[f, 0], t1) * _legendre(face_modes[f, 1], t2)
    return total


@njit(cache=True)
def sample_polynomial_3d(
    i_start,
    i_end,
    seed,
    x_edges,
    y_edges,
    z_edges,
    E_edges,
    E_nodes,
    interval_left,
    interval_bin,
    n_mu,
    n_phi,
    mu_low,
    mu_high,
    phi_low,
    phi_high,
    abs_integral,
    modes,
    face_modes,
    C,
    Q,
    gradient,
    Tn,
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
    Bank sample indices [i_start, i_end) of the particles drawn from the residual (see
    module doc). C, Q: (Kx, Ky, Kz, G, J, M); gradient: (3, Kx, Ky, Kz, G, J, M);
    Tn: (K, M, N_nodes, J, n_mu, n_phi); jumps: face grids x (G, J, M_face); mass:
    volume sub-cells, then the x, y and z face bins.
    """
    Kx, Ky, Kz, G, J, M = C.shape
    N_int = len(interval_left)
    A = n_mu - 1
    B = n_phi - 1
    N_v = Kx * Ky * Kz * N_int * J * A * B
    N_fx = (Kx + 1) * Ky * Kz * G * J
    N_fy = Kx * (Ky + 1) * Kz * G * J
    cdf = np.zeros(len(mass) + 1)
    cdf[1:] = np.cumsum(mass)
    total = cdf[-1]

    for i in range(i_start, i_end):
        container = _new_container(rng.split_seed(uint64(i), seed))
        if total == 0.0:
            _bank(container, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, simulation)
            continue
        n = _sample_cdf(cdf, rng.lcg(container))
        if n < N_v:
            rest, b = divmod(n, B)
            rest, a = divmod(rest, A)
            rest, j = divmod(rest, J)
            k, e = divmod(rest, N_int)
            rest, kz = divmod(k, Kz)
            kx, ky = divmod(rest, Ky)
            i0 = interval_left[e]
            g = interval_bin[e]
            tx = 2.0 * rng.lcg(container) - 1.0
            ty = 2.0 * rng.lcg(container) - 1.0
            tz = 2.0 * rng.lcg(container) - 1.0
            hx = x_edges[kx + 1] - x_edges[kx]
            hy = y_edges[ky + 1] - y_edges[ky]
            hz = z_edges[kz + 1] - z_edges[kz]
            x = x_edges[kx] + 0.5 * (tx + 1.0) * hx
            y = y_edges[ky] + 0.5 * (ty + 1.0) * hy
            z = z_edges[kz] + 0.5 * (tz + 1.0) * hz
            E0 = E_nodes[i0]
            E1 = E_nodes[i0 + 1]
            E = E0 + rng.lcg(container) * (E1 - E0)
            d_mu = (mu_high[j] - mu_low[j]) / A
            d_phi = (phi_high[j] - phi_low[j]) / B
            mu = mu_low[j] + (a + rng.lcg(container)) * d_mu
            phi = phi_low[j] + (b + rng.lcg(container)) * d_phi
            mu = _inside(mu, mu_low[j], mu_high[j])
            phi = _inside(phi, phi_low[j], phi_high[j])
            s = math.sqrt(max(0.0, 1.0 - mu * mu))
            ox = s * math.cos(phi)
            oy = s * math.sin(phi)
            oz = mu
            tE = (E - E0) / (E1 - E0)
            tm = min(1.0, max(0.0, (mu - mu_low[j]) / d_mu - a))
            tp = min(1.0, max(0.0, (phi - phi_low[j]) / d_phi - b))
            m_ID = cell_material[k]
            start = xs_offsets[m_ID]
            end = xs_offsets[m_ID + 1]
            Sigma = np.interp(E, xs_energy[start:end], xs_total[start:end])
            r = 0.0
            for m in range(M):
                P = (
                    _legendre(modes[m, 0], tx)
                    * _legendre(modes[m, 1], ty)
                    * _legendre(modes[m, 2], tz)
                )
                T = 0.0
                for p in range(2):
                    wE = tE if p == 1 else 1.0 - tE
                    for q in range(2):
                        wm = tm if q == 1 else 1.0 - tm
                        for u in range(2):
                            wp = tp if u == 1 else 1.0 - tp
                            T += wE * wm * wp * Tn[k, m, i0 + p, j, a + q, b + u]
                coefficient = (
                    Q[kx, ky, kz, g, j, m]
                    + T
                    - Sigma * C[kx, ky, kz, g, j, m]
                    - ox * gradient[0, kx, ky, kz, g, j, m]
                    - oy * gradient[1, kx, ky, kz, g, j, m]
                    - oz * gradient[2, kx, ky, kz, g, j, m]
                )
                r += P * coefficient
            measure = hx * hy * hz * (E1 - E0) * d_mu * d_phi
            w = r * measure * total / mass[n]
        else:
            idx = n - N_v
            if idx < N_fx:
                axis = AXIS_X
                n_y, n_z = Ky, Kz
            elif idx < N_fx + N_fy:
                axis = AXIS_Y
                idx -= N_fx
                n_y, n_z = Ky + 1, Kz
            else:
                axis = 2
                idx -= N_fx + N_fy
                n_y, n_z = Ky, Kz + 1
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
            t1 = 2.0 * rng.lcg(container) - 1.0
            t2 = 2.0 * rng.lcg(container) - 1.0
            if axis == AXIS_X:
                omega = s * math.cos(phi)
                D = _face_value(jump_x[kx, ky, kz, g, j], face_modes[0], t1, t2)
                x = x_edges[kx] + math.copysign(FACE_NUDGE, omega)
                y = y_edges[ky] + 0.5 * (t1 + 1.0) * (y_edges[ky + 1] - y_edges[ky])
                z = z_edges[kz] + 0.5 * (t2 + 1.0) * (z_edges[kz + 1] - z_edges[kz])
                face = (y_edges[ky + 1] - y_edges[ky]) * (z_edges[kz + 1] - z_edges[kz])
            elif axis == AXIS_Y:
                omega = s * math.sin(phi)
                D = _face_value(jump_y[kx, ky, kz, g, j], face_modes[1], t1, t2)
                x = x_edges[kx] + 0.5 * (t1 + 1.0) * (x_edges[kx + 1] - x_edges[kx])
                y = y_edges[ky] + math.copysign(FACE_NUDGE, omega)
                z = z_edges[kz] + 0.5 * (t2 + 1.0) * (z_edges[kz + 1] - z_edges[kz])
                face = (x_edges[kx + 1] - x_edges[kx]) * (z_edges[kz + 1] - z_edges[kz])
            else:
                omega = mu
                D = _face_value(jump_z[kx, ky, kz, g, j], face_modes[2], t1, t2)
                x = x_edges[kx] + 0.5 * (t1 + 1.0) * (x_edges[kx + 1] - x_edges[kx])
                y = y_edges[ky] + 0.5 * (t2 + 1.0) * (y_edges[ky + 1] - y_edges[ky])
                z = z_edges[kz] + math.copysign(FACE_NUDGE, omega)
                face = (x_edges[kx + 1] - x_edges[kx]) * (y_edges[ky + 1] - y_edges[ky])
            w = (
                math.copysign(1.0, -omega)
                * D
                * face
                * (E_edges[g + 1] - E_edges[g])
                * abs_integral[axis, j]
                * total
                / mass[n]
            )
        _bank(container, x, y, z, E, mu, phi, w, simulation)
