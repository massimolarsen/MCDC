"""
Full-residual correction passes of the 3D Cartesian RMC solver, with a tabulated
in-scatter kernel (writeups/cartesian_3d/tabulated_correction.tex).

The collision-only iterations of `driver3d` converge to a fixed point that carries the
in-bin shape bias of the binned in-scatter S_bar. A correction pass samples the full
residual of psi~,

    r(x, E, Omega) = Q + T[psi~](E, Omega) - Sigma_t(E) psi~    inside each cell,
                   + r_e                                         on the faces,

with T the continuous-energy scattering and fission emission density of psi~. Since
psi~ is constant on each bin, inside a cell of material m

    T(E, Omega) = sum_{g', j'} psi~[g', j'] U_m[g', j'](E, Omega),
    U_m[g', j'](E, Omega) = int_{g'} dE' sum_r N_r sigma_r(E')
                            int dmu0 f_r(E' -> E, mu0) F_j'(Omega, mu0),

where F_j' is the fraction of the circle of directions at cosine mu0 to Omega that lies
in patch j' (`patches.arc_fractions`). U_m depends only on the data and the grids: it
is tabulated once per material at nodes in energy and direction, only for directions
in the patches that the layout's symmetries do not determine, and T is interpolated
trilinearly in (E, mu, phi) between the nodes. A constant per bin makes the bin
average of the interpolant equal to S_bar, which the transfer moments give exactly.

The volume residual is sampled as a single term: sub-cells of the node grid, with mass
equal to their measure times the largest |r| at their corners plus |psi~| times the
mean departure of Sigma_t from linear between the energy nodes; uniform inside; weight
r / q with r evaluated at the sampled point. The face residual is sampled as in the
iterations (`mesh3d`). Both share one CDF, so each term receives histories in
proportion to its mass.
"""

import hashlib
import math
import os
import uuid

import h5py
import numpy as np

from mpi4py import MPI
from numba import njit, uint64

####

import mcdc.transport.rng as rng

from mcdc.constant import (
    ANGLE_ISOTROPIC,
    NEUTRON_REACTION_ELASTIC_SCATTERING,
    NEUTRON_REACTION_FISSION,
    PI,
    REFERENCE_FRAME_COM,
)
from mcdc.print_ import print_msg
from mcdc.rmc.kernel import GK_WG, GK_WK, GK_X
from mcdc.rmc.mesh3d import (
    _bank,
    _face_particle,
    _inside,
    _new_container,
    _sample_cdf,
    _uniform,
)
from mcdc.rmc.patches import _symmetries, arc_fractions
from mcdc.rmc.reaction import (
    KERNEL_CONTINUOUS,
    KERNEL_ELASTIC,
    _com_energy_range,
    continuous_lab_density,
    delta_lab_line,
    free_gas_threshold,
    is_free_gas,
    lab_energy_support,
)
from mcdc.rmc.residual import _abs_linear_integral
from mcdc.transport.physics.neutron.native import reaction_micro_xs

CORRECTION_TABLE_FORMAT_VERSION = 1
# Relative tolerance and depth of the adaptive incident-energy integral of U
TABLE_TOLERANCE = 1e-6
TABLE_MAX_DEPTH = 30
# Gauss-Legendre points per piece of the mu0 integral of continuous laws
_GL16_X, _GL16_W = np.polynomial.legendre.leggauss(16)
# Energy nodes closer than this fraction of the bin width are merged
_NODE_MERGE = 0.02


# ======================================================================================
# Nodes and symmetries
# ======================================================================================


class CorrectionNodes:
    """
    Interpolation nodes of T.

    Energy: each bin has its own nodes (both edges, n_energy uniform intervals, and
    the kinks alpha E_edge of elastic scattering on a target at rest, alpha =
    ((A - 1)/(A + 1))^2, where an incident bin edge enters the kinematic range),
    stored flat with offsets per bin. Direction: an n_mu x n_phi grid on every patch,
    edges included, with dOmega = dmu dphi.

    Symmetries: rotation by the azimuthal patch width (uniform phi edges) and the
    reflection mu -> -mu (symmetric mu edges) map patches to patches and leave U
    invariant (it depends on directions only through Omega' . Omega). Patch j is the
    image sigma_j(f_j) of a fundamental patch f_j; T on patch j is evaluated from the
    fundamental nodes with psi~ permuted by sigma_j.
    """

    def __init__(self, E_edges, patches, alphas, n_energy=4, n_mu=3, n_phi=3):
        if n_energy < 1 or n_mu < 2 or n_phi < 2:
            raise ValueError("RMC: correction nodes need n_energy >= 1, n_mu, n_phi >= 2")
        G = len(E_edges) - 1
        nodes, offsets = [], [0]
        for g in range(G):
            a, b = E_edges[g], E_edges[g + 1]
            merge = _NODE_MERGE * (b - a)
            kept = [a, b]
            kinks = sorted(
                E for alpha in alphas for E in alpha * E_edges if a + merge < E < b - merge
            )
            uniform = np.linspace(a, b, n_energy + 1)[1:-1]
            for E in list(kinks) + list(uniform):
                if min(abs(E - k) for k in kept) > merge:
                    kept.append(E)
            kept = np.sort(kept)
            nodes.append(kept)
            offsets.append(offsets[-1] + len(kept))
        self.E_nodes = np.concatenate(nodes)
        self.E_offsets = np.array(offsets, dtype=np.int64)
        self.node_bin = np.repeat(np.arange(G), np.diff(self.E_offsets))
        # Intervals between consecutive nodes of the same bin: left node, bin
        last = self.E_offsets[1:] - 1
        left = np.setdiff1d(np.arange(len(self.E_nodes)), last)
        self.interval_left = left.astype(np.int64)
        self.interval_bin = self.node_bin[left].astype(np.int64)
        # Unique energies (a shared bin edge is tabulated once)
        self.E_unique, self.node_unique = np.unique(self.E_nodes, return_inverse=True)

        self.n_mu, self.n_phi = n_mu, n_phi
        J_mu, J_phi = patches.J_mu, patches.J_phi
        rotation, mirror = _symmetries(patches)
        j_mu_max = (J_mu + 1) // 2 if mirror else J_mu
        j_phi_max = 1 if rotation else J_phi
        fundamental = [jm * J_phi + jf for jm in range(j_mu_max) for jf in range(j_phi_max)]
        self.fundamental = np.array(fundamental, dtype=np.int64)
        # Patch j = sigma_j(f_j): sigma_j = rotation^k_j after mirror^m_j
        J = patches.J
        self.fund_of = np.zeros(J, dtype=np.int64)
        self.flip = np.zeros(J, dtype=bool)
        self.permutation = np.zeros((J, J), dtype=np.int64)
        for j in range(J):
            jm, jf = divmod(j, J_phi)
            m = mirror and jm >= j_mu_max
            jm0 = J_mu - 1 - jm if m else jm
            k = jf if rotation else 0
            jf0 = 0 if rotation else jf
            self.fund_of[j] = fundamental.index(jm0 * J_phi + jf0)
            self.flip[j] = m
            for jpp in range(J):
                am, af = divmod(jpp, J_phi)
                am = J_mu - 1 - am if m else am
                self.permutation[j, jpp] = am * J_phi + (af + k) % J_phi
        # Direction nodes of the fundamental patches, d = (f n_mu + a) n_phi + b
        t_mu = np.linspace(0.0, 1.0, n_mu)
        t_phi = np.linspace(0.0, 1.0, n_phi)
        mu, phi = [], []
        for jf in fundamental:
            for a in range(n_mu):
                for b in range(n_phi):
                    mu.append(
                        patches.mu_low[jf] + t_mu[a] * (patches.mu_high[jf] - patches.mu_low[jf])
                    )
                    phi.append(
                        patches.phi_low[jf]
                        + t_phi[b] * (patches.phi_high[jf] - patches.phi_low[jf])
                    )
        self.dir_mu, self.dir_phi = np.array(mu), np.array(phi)

    @property
    def key(self):
        """Array that identifies the nodes (for cache keys)."""
        return np.concatenate(
            (self.E_nodes, [-1.0], self.dir_mu, [-1.0], self.dir_phi)
        )


# ======================================================================================
# Tabulation of U
# ======================================================================================


@njit(cache=True)
def _emission(
    E_in, E_out, reaction, ktype, nuclide, dir_mu, dir_phi, mu_edges, phi_edges, area,
    simulation, data, out,
):
    """
    out[d, j'] += sigma_r(E_in) int dmu0 f_r(E_in -> E_out, mu0) F_j'(Omega_d, mu0);
    returns sigma_r(E_in) int dmu0 f_r (the emission density at E_out per incident
    flux, summed over j').
    """
    sigma = reaction_micro_xs(E_in, reaction, nuclide, data)
    if sigma <= 0.0:
        return 0.0
    low, high = lab_energy_support(E_in, reaction, nuclide, simulation, data)
    if E_out < low or E_out > high:
        return 0.0
    N_D = len(dir_mu)
    J = len(area)
    free_gas = is_free_gas(E_in, reaction, nuclide)

    # Delta laws (elastic on a target at rest, level): one cosine mu0*
    if ktype != KERNEL_CONTINUOUS and not free_gas:
        g, mu0 = delta_lab_line(E_in, E_out, reaction, nuclide, simulation, data)
        if g == 0.0:
            return 0.0
        mu0 = min(1.0, max(-1.0, mu0))
        c = sigma * g
        for d in range(N_D):
            arc_fractions(dir_mu[d], dir_phi[d], mu0, mu_edges, phi_edges, c, out[d])
        return c

    # Isotropic lab emission (fission): int F_j' dmu0 = |A_j'| / (2 pi)
    if reaction["sub_type"] == NEUTRON_REACTION_FISSION:
        fission = simulation["neutron_fission_reactions"][reaction["sub_ID"]]
        if (
            fission["angle_type"] == ANGLE_ISOTROPIC
            and reaction["reference_frame"] != REFERENCE_FRAME_COM
        ):
            f = continuous_lab_density(E_in, E_out, 0.0, reaction, nuclide, simulation, data)
            c = sigma * f
            for d in range(N_D):
                for jp in range(J):
                    out[d, jp] += c * area[jp] / (2.0 * PI)
            return 2.0 * c

    # Continuous laws: Gauss-Legendre over the supported mu0 range (as in
    #   reaction.emission_kernel_bin_moments), cosine-mapped on each piece
    mu0_low, mu0_high = -1.0, 1.0
    if reaction["reference_frame"] == REFERENCE_FRAME_COM and not free_gas:
        E_cm_low, E_cm_high = _com_energy_range(E_in, reaction, simulation, data)
        s_A = math.sqrt(E_in) / (nuclide["atomic_weight_ratio"] + 1)
        denominator = 2.0 * s_A * math.sqrt(E_out)
        mu0_low = max(mu0_low, (E_out + s_A * s_A - E_cm_high) / denominator)
        mu0_high = min(mu0_high, (E_out + s_A * s_A - E_cm_low) / denominator)
    if mu0_high <= mu0_low:
        return 0.0
    points = np.empty(4)
    points[0] = mu0_low
    N = 1
    if free_gas:
        # The free-gas kernel peaks toward mu0 = 1 near E_out = E_in
        for p0 in (1.0 - 1e-2, 1.0 - 1e-4):
            if mu0_low < p0 < mu0_high:
                points[N] = p0
                N += 1
    points[N] = mu0_high
    N += 1
    total = 0.0
    for p in range(N - 1):
        c0 = points[p]
        c1 = points[p + 1]
        for q in range(len(_GL16_X)):
            t = 0.5 * (_GL16_X[q] + 1.0)
            mu0 = c0 + 0.5 * (c1 - c0) * (1.0 - math.cos(PI * t))
            w = 0.5 * _GL16_W[q] * 0.5 * (c1 - c0) * PI * math.sin(PI * t)
            f = continuous_lab_density(E_in, E_out, mu0, reaction, nuclide, simulation, data)
            if f == 0.0:
                continue
            c = sigma * f * w
            total += c
            for d in range(N_D):
                arc_fractions(dir_mu[d], dir_phi[d], mu0, mu_edges, phi_edges, c, out[d])
    return total


@njit(cache=True)
def _integrate_incident(
    E_out, a, b, reaction, ktype, nuclide, dir_mu, dir_phi, mu_edges, phi_edges, area,
    simulation, data, out,
):
    """
    out += int_a^b dE' (emission of `_emission`), by adaptive Gauss-Kronrod (G7-K15)
    on the summed emission, from four initial pieces.
    """
    N_D, J = out.shape
    node = np.zeros((N_D, J))
    interval = np.zeros((N_D, J))
    size = 4 + 2 * TABLE_MAX_DEPTH + 2
    stack_a = np.empty(size)
    stack_b = np.empty(size)
    stack_d = np.empty(size, dtype=np.int64)
    top = 0
    for p in range(4):
        stack_a[top] = a + (b - a) * (3 - p) / 4.0
        stack_b[top] = a + (b - a) * (4 - p) / 4.0
        stack_d[top] = 0
        top += 1
    while top > 0:
        top -= 1
        lo = stack_a[top]
        hi = stack_b[top]
        depth = stack_d[top]
        half = 0.5 * (hi - lo)
        center = 0.5 * (hi + lo)
        K = 0.0
        Gs = 0.0
        interval[:] = 0.0
        for q in range(15):
            node[:] = 0.0
            s = _emission(
                center + half * GK_X[q], E_out, reaction, ktype, nuclide, dir_mu,
                dir_phi, mu_edges, phi_edges, area, simulation, data, node,
            )
            if s == 0.0:
                continue
            K += half * GK_WK[q] * s
            Gs += half * GK_WG[q] * s
            interval += (half * GK_WK[q]) * node
        if (
            (K != 0.0 and abs(K - Gs) <= TABLE_TOLERANCE * abs(K))
            or depth >= TABLE_MAX_DEPTH
            or (K == 0.0 and depth >= 3)
        ):
            out += interval
        else:
            stack_a[top] = lo
            stack_b[top] = center
            stack_d[top] = depth + 1
            stack_a[top + 1] = center
            stack_b[top + 1] = hi
            stack_d[top + 1] = depth + 1
            top += 2


@njit(cache=True)
def _reachable(E_out, a, b, reaction, nuclide, simulation, data):
    """False if no incident energy in [a, b] can emit at E_out (support sampled at 9 points)."""
    low = math.inf
    high = -math.inf
    for i in range(9):
        E_in = a + (b - a) * i / 8.0
        lo, hi = lab_energy_support(E_in, reaction, nuclide, simulation, data)
        if lo <= hi:
            low = min(low, lo)
            high = max(high, hi)
    return low * (1.0 - 1e-6) <= E_out <= high * (1.0 + 1e-6)


@njit(cache=True)
def _tabulate(
    E_values, i_start, i_step, E_edges, dir_mu, dir_phi, mu_edges, phi_edges, area,
    rx_ID, rx_nuclide, rx_density, rx_type, simulation, data, U,
):
    """U[i, d, g', j'] for the energies E_values[i_start::i_step] (see module doc)."""
    G = len(E_edges) - 1
    N_D = len(dir_mu)
    J = len(area)
    buffer = np.zeros((N_D, J))
    for i in range(i_start, len(E_values), i_step):
        E_out = E_values[i]
        for r in range(len(rx_ID)):
            reaction = simulation["neutron_reactions"][rx_ID[r]]
            nuclide = simulation["nuclides"][rx_nuclide[r]]
            ktype = rx_type[r]
            for gp in range(G):
                pieces = np.empty(4)
                pieces[0] = E_edges[gp]
                pieces[1] = E_edges[gp + 1]
                N = 2
                if (
                    ktype == KERNEL_ELASTIC
                    and reaction["sub_type"] == NEUTRON_REACTION_ELASTIC_SCATTERING
                ):
                    # Target at rest: E' in [E_out, E_out / alpha] (above free gas)
                    threshold = free_gas_threshold(nuclide)
                    A = nuclide["atomic_weight_ratio"]
                    alpha = ((A - 1.0) / (A + 1.0)) ** 2
                    E_low = max(E_edges[gp], threshold)
                    E_high = E_edges[gp + 1]
                    if E_low < E_high:
                        pieces[0] = max(E_low, E_out)
                        pieces[1] = E_high if alpha <= 0.0 else min(E_high, E_out / alpha)
                        if E_edges[gp] < threshold:
                            # Free-gas part below the threshold, unclipped
                            pieces[2] = E_edges[gp]
                            pieces[3] = threshold
                            N = 4
                    else:
                        pieces[0] = E_edges[gp]
                        pieces[1] = E_edges[gp + 1]
                elif not _reachable(
                    E_out, E_edges[gp], E_edges[gp + 1], reaction, nuclide, simulation, data
                ):
                    continue
                for p in range(0, N, 2):
                    if pieces[p + 1] <= pieces[p]:
                        continue
                    buffer[:] = 0.0
                    _integrate_incident(
                        E_out, pieces[p], pieces[p + 1], reaction, ktype, nuclide,
                        dir_mu, dir_phi, mu_edges, phi_edges, area, simulation, data,
                        buffer,
                    )
                    U[i, :, gp, :] += rx_density[r] * buffer


def material_table(
    simulation, data, rx_tables, nuclides, nodes, E_edges, patches, cache_dir, label
):
    """
    U[i, d, g', j'] of one material at the unique node energies and the fundamental
    direction nodes, from `cache_dir` if a matching file exists. rx_tables: the
    material's (rx_ID, rx_nuclide, rx_density, rx_type) over its neutron-emitting
    reactions (scattering and fission); nuclides: (name, data file) per nuclide.
    """
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    path = None
    if cache_dir is not None:
        from mcdc.rmc.driver import data_fingerprint

        text = f"format={CORRECTION_TABLE_FORMAT_VERSION}|tol={TABLE_TOLERANCE}|"
        text += f"depth={TABLE_MAX_DEPTH}|GL={len(_GL16_X)}|"
        for name, file in nuclides:
            text += f"{name}:{data_fingerprint(file)}|"
        for array in (*rx_tables, E_edges, patches.key, nodes.key):
            text += np.asarray(array, dtype=float).tobytes().hex() + "|"
        key = hashlib.sha256(text.encode()).hexdigest()[:20]
        path = os.path.join(cache_dir, f"correction-table-{key}.h5")
        found = os.path.exists(path) if rank == 0 else None
        if comm.bcast(found, root=0):
            with h5py.File(path, "r") as f:
                return f["U"][()]
    print_msg(f"RMC: in-scatter table of {label}")
    G, J = len(E_edges) - 1, patches.J
    U = np.zeros((len(nodes.E_unique), len(nodes.dir_mu), G, J))
    _tabulate(
        nodes.E_unique,
        rank,
        size,
        np.asarray(E_edges, dtype=float),
        nodes.dir_mu,
        nodes.dir_phi,
        patches.mu_edges,
        patches.phi_edges,
        patches.area,
        *rx_tables,
        simulation,
        data,
        U,
    )
    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, U, op=MPI.SUM)
    if path is not None and rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        temporary = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with h5py.File(temporary, "w") as f:
            f.create_dataset("U", data=U)
            f.create_dataset("E_nodes", data=nodes.E_unique)
            f.create_dataset("dir_mu", data=nodes.dir_mu)
            f.create_dataset("dir_phi", data=nodes.dir_phi)
            f.attrs["format_version"] = CORRECTION_TABLE_FORMAT_VERSION
        os.replace(temporary, path)
    return U


# ======================================================================================
# Residual at the nodes
# ======================================================================================


def node_in_scatter(psi, S_bar, U, cell_material, nodes, patches):
    """
    T~ at the nodes of every cell, Tn[k, i, j, a, b] (k flat cell, i flat energy node,
    a and b the direction node of patch j): the trilinear interpolant of T, shifted per
    (cell, bin, patch) so that its bin average equals S_bar.
    """
    G, J = psi.shape[-2:]
    n_mu, n_phi = nodes.n_mu, nodes.n_phi
    n_fund = len(nodes.fundamental)
    psi_flat = psi.reshape(-1, G, J)
    K = psi_flat.shape[0]
    N_u = len(nodes.E_unique)
    Tu = np.zeros((K, N_u, J, n_mu, n_phi))
    for m, U_m in enumerate(U):
        cells = np.flatnonzero(cell_material == m)
        if len(cells) == 0:
            continue
        # Columns: psi~ of each cell, permuted by sigma_j, for each patch j
        Psi = psi_flat[cells][:, :, nodes.permutation]  # (cells, G, J, J'')
        Psi = np.transpose(Psi, (1, 3, 0, 2)).reshape(G * J, len(cells) * J)
        T = U_m.reshape(N_u * n_fund * n_mu * n_phi, G * J) @ Psi
        T = T.reshape(N_u, n_fund, n_mu, n_phi, len(cells), J)
        for j in range(J):
            block = T[:, nodes.fund_of[j], :, :, :, j]  # (N_u, n_mu, n_phi, cells)
            if nodes.flip[j]:
                block = block[:, ::-1]
            Tu[cells, :, j] = np.transpose(block, (3, 0, 1, 2))
    Tn = Tu[:, nodes.node_unique]  # (K, N_nodes, J, n_mu, n_phi)

    # Shift: bin average of the interpolant (trapezoid weights) -> S_bar
    w_E = _trapezoid_weights(nodes.E_nodes, nodes.E_offsets)
    w_mu = _trapezoid_weights(np.linspace(0.0, 1.0, n_mu), [0, n_mu])
    w_phi = _trapezoid_weights(np.linspace(0.0, 1.0, n_phi), [0, n_phi])
    angular = np.einsum("knjab,a,b->knj", Tn, w_mu, w_phi)
    average = np.zeros((K, G, J))
    np.add.at(average, (slice(None), nodes.node_bin), (angular * w_E[None, :, None]))
    shift = S_bar.reshape(K, G, J) - average
    Tn += shift[:, nodes.node_bin][..., None, None]
    return Tn


def _trapezoid_weights(x, offsets):
    """Weights of the trapezoid average of a linear interpolant on each segment."""
    w = np.zeros(len(x))
    for s in range(len(offsets) - 1):
        lo, hi = offsets[s], offsets[s + 1]
        h = np.diff(x[lo:hi])
        w[lo : hi - 1] += 0.5 * h
        w[lo + 1 : hi] += 0.5 * h
        w[lo:hi] /= x[hi - 1] - x[lo]
    return w


def linear_departure(nodes, xs):
    """
    dev[m, e] = mean over energy interval e of |Sigma_t(E) - Sigma_lin(E)|, with
    Sigma_t linear between the material's union-grid points (as the samplers evaluate
    it) and Sigma_lin linear between the interval's end nodes.
    """
    E_lo = nodes.E_nodes[nodes.interval_left]
    E_hi = nodes.E_nodes[nodes.interval_left + 1]
    dev = np.zeros((len(xs), len(E_lo)))
    for m, (grid, Sigma) in enumerate(xs):
        S_lo = np.interp(E_lo, grid, Sigma)
        S_hi = np.interp(E_hi, grid, Sigma)
        for e in range(len(E_lo)):
            inside = grid[(grid > E_lo[e]) & (grid < E_hi[e])]
            if len(inside) == 0:
                continue
            E = np.concatenate(([E_lo[e]], inside, [E_hi[e]]))
            t = (E - E_lo[e]) / (E_hi[e] - E_lo[e])
            delta = np.interp(E, grid, Sigma) - (S_lo[e] + t * (S_hi[e] - S_lo[e]))
            total = sum(
                _abs_linear_integral(E[n], E[n + 1], delta[n], delta[n + 1])
                for n in range(len(E) - 1)
            )
            dev[m, e] = total / (E_hi[e] - E_lo[e])
    return dev


def volume_masses(psi, Q, Tn, cell_material, nodes, mesh, patches, xs, departure):
    """
    Sampling mass of each volume sub-cell (k, e, j, a, b): its measure times
    (max |r| over its 8 corner nodes + |psi~| dev[m, e]). r = Q + T~ - Sigma_t psi~ is
    trilinear up to Sigma_t's departure from linear in E, so this bounds |r| up to that
    departure, which enters through its interval mean.
    """
    G, J = psi.shape[-2:]
    K = Tn.shape[0]
    psi_flat = psi.reshape(K, G, J)
    Q_flat = Q.reshape(K, G, J)
    g = nodes.node_bin
    Sigma = np.array([np.interp(nodes.E_nodes, grid, S) for grid, S in xs])
    removal = Sigma[cell_material][:, :, None] * psi_flat[:, g, :]  # (K, N_nodes, J)
    r = np.abs(Tn + (Q_flat[:, g, :] - removal)[..., None, None])
    left = nodes.interval_left
    r = np.maximum(r[:, left], r[:, left + 1])
    r = np.maximum(r[..., :-1, :], r[..., 1:, :])
    r = np.maximum(r[..., :-1], r[..., 1:])  # (K, N_int, J, n_mu - 1, n_phi - 1)
    e_bin = nodes.interval_bin
    r = r + (
        np.abs(psi_flat[:, e_bin, :]) * departure[cell_material][:, :, None]
    )[..., None, None]
    dE = nodes.E_nodes[left + 1] - nodes.E_nodes[left]
    sub_area = patches.area / ((nodes.n_mu - 1) * (nodes.n_phi - 1))
    measure = (
        mesh.volume.reshape(K)[:, None, None] * dE[None, :, None] * sub_area[None, None, :]
    )
    return r * measure[..., None, None]


# ======================================================================================
# Sampling
# ======================================================================================


@njit(cache=True)
def sample_correction_3d(
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
    psi,
    Q,
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
    Bank sample indices [i_start, i_end) of the particles drawn from the full residual.
    mass: the volume sub-cell masses (k, e, j, a, b), then the face masses of x, y and
    z as in `mesh3d.sample_residual_3d`. The volume residual is evaluated at the
    sampled point: Q + T~ (trilinear in the node values Tn) - Sigma_t(E) psi~.
    """
    Kx, Ky, Kz, G, J = psi.shape
    N_int = len(interval_left)
    A = n_mu - 1
    B = n_phi - 1
    N_v = Kx * Ky * Kz * N_int * J * A * B
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
            x = _uniform(x_edges, kx, container)
            y = _uniform(y_edges, ky, container)
            z = _uniform(z_edges, kz, container)
            E0 = E_nodes[i0]
            E1 = E_nodes[i0 + 1]
            E = E0 + rng.lcg(container) * (E1 - E0)
            d_mu = (mu_high[j] - mu_low[j]) / A
            d_phi = (phi_high[j] - phi_low[j]) / B
            mu = mu_low[j] + (a + rng.lcg(container)) * d_mu
            phi = phi_low[j] + (b + rng.lcg(container)) * d_phi
            mu = _inside(mu, mu_low[j], mu_high[j])
            phi = _inside(phi, phi_low[j], phi_high[j])
            tE = (E - E0) / (E1 - E0)
            tm = min(1.0, max(0.0, (mu - mu_low[j]) / d_mu - a))
            tp = min(1.0, max(0.0, (phi - phi_low[j]) / d_phi - b))
            T = 0.0
            for p in range(2):
                wE = tE if p == 1 else 1.0 - tE
                for q in range(2):
                    wm = tm if q == 1 else 1.0 - tm
                    for s in range(2):
                        wp = tp if s == 1 else 1.0 - tp
                        T += wE * wm * wp * Tn[k, i0 + p, j, a + q, b + s]
            m = cell_material[k]
            start = xs_offsets[m]
            end = xs_offsets[m + 1]
            Sigma = np.interp(E, xs_energy[start:end], xs_total[start:end])
            r = Q[kx, ky, kz, g, j] + T - Sigma * psi[kx, ky, kz, g, j]
            measure = (
                (x_edges[kx + 1] - x_edges[kx])
                * (y_edges[ky + 1] - y_edges[ky])
                * (z_edges[kz + 1] - z_edges[kz])
                * (E1 - E0)
                * d_mu
                * d_phi
            )
            w = r * measure * total / mass[n]
        else:
            x, y, z, E, mu, phi, value = _face_particle(
                n - N_v,
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
        _bank(container, x, y, z, E, mu, phi, w, simulation)
