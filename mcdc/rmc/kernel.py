"""
Binned transfer moments of the in-scattering (and fission) operator.

The trial space is piecewise constant in the polar cosine (bins j about the slab axis)
and, in energy, piecewise constant or linear discontinuous on each bin g:
psi~(E) = sum_a psi~[g, j, a] P_a(x_g(E)), with the Legendre polynomials P_0 = 1,
P_1 = x and x_g = (2E - E_g - E_{g+1}) / dE_g in [-1, 1]. The projection of the
in-scatter source of psi~ onto the same space is exactly a matrix product:

    S_bar[g, j, a] = (2a + 1) / (dE_g dmu_j)
                     sum_{a', g', j'} M[a', g', j', a, g, j] psi~[g', j', a'],

    M[a', g', j', a, g, j] = sum_r int_{g'} dE_in sigma_r(E_in) P_a'(x_g'(E_in))
        int_g dE_out P_a(x_g(E_out)) int dmu0 f_r(E_in -> E_out, mu0) A_{j j'}(mu0),

where f_r is the lab emission kernel of reaction r (yield included, see
`mcdc.rmc.reaction`) and A_{j j'} the angular transfer (`mcdc.rmc.angular`). M depends
only on nuclear data and the grids.

Integration is done in the data's native variables so delta laws need no special
treatment and no frame Jacobian appears:
  - Elastic/level (delta in E_cm): g(E_out) dE_out = yield p(mu_cm) dmu_cm, with the
    mu_cm interval of each outgoing bin in closed form.
  - Continuous COM laws: f_C(E_cm, mu_cm) dE_cm dmu_cm; for fixed E_cm, E_out is affine
    in mu_cm, so each outgoing bin is a closed-form mu_cm interval.
  - Continuous lab laws: directly in (E_out, mu0).
  - Free-gas elastic (E <= 400 kT): the analytic lab kernel in (E_out, mu0).
Inner integrals use Gauss-Legendre on pieces split at every kink/jump of the sampled
densities. The outer E_in integral uses adaptive Gauss-Kronrod (G7-K15), split at
cross-section grid points, the reaction threshold, and where kinematic support edges
cross bin edges. Kinematically impossible bins are never evaluated.
"""

import math

import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import (
    ANGLE_DISTRIBUTED,
    BOLTZMANN_K,
    ANGLE_ENERGY_CORRELATED,
    ANGLE_ISOTROPIC,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
    NEUTRON_REACTION_ELASTIC_SCATTERING,
    NEUTRON_REACTION_FISSION,
    REFERENCE_FRAME_COM,
)
from mcdc.rmc.angular import angular_transfer
from mcdc.rmc.evaluate import (
    correlated_distribution_support,
    correlated_energy_breakpoints,
    distribution_breakpoints,
    distribution_support,
    evaluate_correlated_distribution,
    evaluate_distribution,
    tabulated_energy_angle_cosine_breakpoints,
)
from mcdc.rmc.kinematics import (
    com_to_lab,
    elastic_E_out,
    elastic_mu_cm,
    elastic_mu_lab,
    level_mu_cm,
)
from mcdc.rmc.reaction import (
    KERNEL_CONTINUOUS,
    KERNEL_ELASTIC,
    _inelastic_spectrum,
    _spectrum_weight,
    fission_yield,
    free_gas_density,
    free_gas_support,
    free_gas_threshold,
    inelastic_yield,
    is_free_gas,
    kernel_type,
    lab_energy_support,
)
from mcdc.transport.physics.neutron.native import reaction_micro_xs

# ======================================================================================
# Quadrature rules
# ======================================================================================

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)

# Gauss-Kronrod 15 (with embedded Gauss 7) on [-1, 1]
_XGK = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ]
)
_WGK = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_WG7 = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)
GK_X = np.concatenate((-_XGK[:-1], _XGK[::-1]))
GK_WK = np.concatenate((_WGK[:-1], _WGK[::-1]))
GK_WG = np.zeros(15)
for _i, _w in zip((1, 3, 5), _WG7[:3]):
    GK_WG[_i] = _w
    GK_WG[14 - _i] = _w
GK_WG[7] = _WG7[3]

MAX_DEPTH = 20
# Tightest usable relative tolerance. The angular transfer table is linear in theta
# between N_THETA_DEFAULT nodes and the inner angular integrals are accurate to
# TAU_TOLERANCE, so below ~1e-8 every unlisted table node becomes a kink to resolve:
# an O-16 elastic row takes ~5 ms at 1e-8 and ~130 s at 1e-9 for the same 7 digits.
TOLERANCE_FLOOR = 1e-8

# ======================================================================================
# Helpers
# ======================================================================================


@njit(cache=True)
def _pieces(low, high, candidates):
    """Sorted unique breakpoints of [low, high] including candidates inside it."""
    points = np.empty(len(candidates) + 2)
    points[0] = low
    points[1] = high
    N = 2
    for x in candidates:
        if low < x < high:
            points[N] = x
            N += 1
    points = np.sort(points[:N])
    # Drop duplicates
    out = np.empty(N)
    out[0] = points[0]
    M = 1
    for i in range(1, N):
        if points[i] > out[M - 1]:
            out[M] = points[i]
            M += 1
    return out[:M]


@njit(cache=True)
def _bin_range(E_edges, low, high):
    """Outgoing bins [g0, g1] overlapping (low, high); empty if g0 > g1."""
    G = len(E_edges) - 1
    g0 = 0
    while g0 < G and E_edges[g0 + 1] <= low:
        g0 += 1
    g1 = G - 1
    while g1 >= 0 and E_edges[g1] >= high:
        g1 -= 1
    return g0, g1


@njit(cache=True)
def _add_outgoing(out, g, E_out, E_edges, weight, A):
    """out[a, g] += weight P_a(x_g(E_out)) A for the energy polynomials a < len(out)."""
    out[0, g] += weight * A
    if out.shape[0] > 1:
        x = (2.0 * E_out - E_edges[g] - E_edges[g + 1]) / (E_edges[g + 1] - E_edges[g])
        out[1, g] += (weight * x) * A


@njit(cache=True)
def _add_incoming(K, f, weight, E_in, E_center, E_half):
    """K[a'] += weight P_a'(x(E_in)) f on the incident bin (center, half width)."""
    K[0] += weight * f
    if K.shape[0] > 1:
        K[1] += (weight * (E_in - E_center) / E_half) * f


@njit(cache=True)
def _angle_density(E_in, mu, angle_type, mu_ID, simulation, data):
    if angle_type == ANGLE_ISOTROPIC:
        return 0.5
    return evaluate_distribution(
        E_in, mu, simulation["distributions"][mu_ID], simulation, data, False
    )


@njit(cache=True)
def _angle_breakpoints(E_in, angle_type, mu_ID, simulation, data):
    if angle_type != ANGLE_DISTRIBUTED:
        return np.empty(0)
    return distribution_breakpoints(
        E_in, simulation["distributions"][mu_ID], simulation, data, False
    )


# ======================================================================================
# Inner integrals at fixed E_in: out[a, g, j, j'] += emission into (g, j) from j',
#   weighted by the outgoing energy polynomial P_a
# ======================================================================================


@njit(cache=True)
def _delta_inner(E_in, reaction, nuclide, simulation, data, E_edges, table, out, A):
    """Elastic or level scattering: integrate yield * p(mu_cm) over mu_cm intervals."""
    awr = nuclide["atomic_weight_ratio"]
    elastic = reaction["sub_type"] == NEUTRON_REACTION_ELASTIC_SCATTERING
    E_cm = 0.0
    if elastic:
        record = simulation["neutron_elastic_scattering_reactions"][reaction["sub_ID"]]
        angle_type = ANGLE_DISTRIBUTED
        mu_ID = record["mu_table_ID"]
        Y = 1.0
    else:
        inelastic = simulation["neutron_inelastic_scattering_reactions"][
            reaction["sub_ID"]
        ]
        spectrum = _inelastic_spectrum(inelastic, 0, simulation, data)
        level = simulation["level_scattering_distributions"][spectrum["sub_ID"]]
        E_cm = level["C2"] * (E_in - level["C1"])
        if E_cm <= 0.0:
            return
        angle_type = inelastic["angle_type"]
        mu_ID = inelastic["mu_ID"]
        Y = inelastic_yield(E_in, inelastic, simulation, data)

    low, high = lab_energy_support(E_in, reaction, nuclide, simulation, data)
    g0, g1 = _bin_range(E_edges, low, high)
    mu_points = _angle_breakpoints(E_in, angle_type, mu_ID, simulation, data)

    for g in range(g0, g1 + 1):
        Ea = max(E_edges[g], low)
        Eb = min(E_edges[g + 1], high)
        if Eb <= Ea:
            continue
        if elastic:
            ca = elastic_mu_cm(E_in, Ea, awr)
            cb = elastic_mu_cm(E_in, Eb, awr)
        else:
            ca = level_mu_cm(E_in, Ea, E_cm, awr)
            cb = level_mu_cm(E_in, Eb, E_cm, awr)
        ca = max(-1.0, min(1.0, ca))
        cb = max(-1.0, min(1.0, cb))
        if cb <= ca:
            continue

        pieces = _pieces(ca, cb, mu_points)
        for p in range(len(pieces) - 1):
            c = pieces[p]
            d = pieces[p + 1]
            for q in range(len(_GL_X)):
                mu_cm = 0.5 * (d - c) * _GL_X[q] + 0.5 * (d + c)
                p_mu = _angle_density(E_in, mu_cm, angle_type, mu_ID, simulation, data)
                if p_mu == 0.0:
                    continue
                if elastic:
                    E_out = elastic_E_out(E_in, mu_cm, awr)
                    mu_lab = elastic_mu_lab(mu_cm, awr)
                else:
                    E_out, mu_lab = com_to_lab(E_in, E_cm, mu_cm, awr)
                angular_transfer(mu_lab, table, A)
                _add_outgoing(
                    out, g, E_out, E_edges, 0.5 * (d - c) * _GL_W[q] * Y * p_mu, A
                )


@njit(cache=True)
def _com_spectrum_inner(
    E_in,
    weight,
    spectrum,
    angle_type,
    mu_ID,
    awr,
    simulation,
    data,
    E_edges,
    table,
    out,
    A,
):
    """Continuous COM spectrum: integrate f_C(E_cm, mu_cm) over (E_cm, mu_cm) cells."""
    correlated = angle_type == ANGLE_ENERGY_CORRELATED
    law61 = correlated and spectrum["sub_type"] == DISTRIBUTION_TABULATED_ENERGY_ANGLE
    if correlated:
        low, high = correlated_distribution_support(E_in, spectrum, simulation, data)
        E_points = correlated_energy_breakpoints(E_in, spectrum, simulation, data)
    else:
        low, high = distribution_support(E_in, spectrum, simulation, data, True)
        E_points = distribution_breakpoints(E_in, spectrum, simulation, data, True)
    low = max(low, 0.0)
    if high <= low:
        return

    # E_cm where a bin edge leaves the mu_cm range at mu_cm = +-1
    s = math.sqrt(E_in) / (awr + 1)
    G = len(E_edges) - 1
    candidates = np.empty(len(E_points) + 2 * (G + 1))
    candidates[: len(E_points)] = E_points
    N = len(E_points)
    for E_edge in E_edges:
        candidates[N] = (math.sqrt(E_edge) - s) ** 2
        candidates[N + 1] = (math.sqrt(E_edge) + s) ** 2
        N += 2
    E_pieces = _pieces(low, high, candidates[:N])
    mu_points = _angle_breakpoints(E_in, angle_type, mu_ID, simulation, data)

    for e in range(len(E_pieces) - 1):
        a = E_pieces[e]
        b = E_pieces[e + 1]
        for qe in range(len(_GL_X)):
            E_cm = 0.5 * (b - a) * _GL_X[qe] + 0.5 * (b + a)
            wE = 0.5 * (b - a) * _GL_W[qe] * weight
            if E_cm <= 0.0:
                continue
            p_E = 1.0
            if not correlated:
                p_E = evaluate_distribution(
                    E_in, E_cm, spectrum, simulation, data, True
                )
                if p_E == 0.0:
                    continue
            if law61:
                mu_points = tabulated_energy_angle_cosine_breakpoints(
                    E_in, E_cm, spectrum, simulation, data
                )

            # Outgoing bins reachable at this E_cm; mu_cm(E_out) is affine
            denominator = 2.0 * (awr + 1) * math.sqrt(E_in * E_cm)
            g0, g1 = _bin_range(
                E_edges, (math.sqrt(E_cm) - s) ** 2, (math.sqrt(E_cm) + s) ** 2
            )
            for g in range(g0, g1 + 1):
                ca = ((awr + 1) ** 2 * (E_edges[g] - E_cm) - E_in) / denominator
                cb = ((awr + 1) ** 2 * (E_edges[g + 1] - E_cm) - E_in) / denominator
                ca = max(-1.0, ca)
                cb = min(1.0, cb)
                if cb <= ca:
                    continue
                mu_pieces = _pieces(ca, cb, mu_points)
                for p in range(len(mu_pieces) - 1):
                    c = mu_pieces[p]
                    d = mu_pieces[p + 1]
                    for qm in range(len(_GL_X)):
                        mu_cm = 0.5 * (d - c) * _GL_X[qm] + 0.5 * (d + c)
                        if correlated:
                            f = evaluate_correlated_distribution(
                                E_in, E_cm, mu_cm, spectrum, simulation, data
                            )
                        else:
                            f = p_E * _angle_density(
                                E_in, mu_cm, angle_type, mu_ID, simulation, data
                            )
                        if f == 0.0:
                            continue
                        E_out, mu_lab = com_to_lab(E_in, E_cm, mu_cm, awr)
                        angular_transfer(mu_lab, table, A)
                        _add_outgoing(
                            out,
                            g,
                            E_out,
                            E_edges,
                            wE * 0.5 * (d - c) * _GL_W[qm] * f,
                            A,
                        )


@njit(cache=True)
def _lab_spectrum_inner(
    E_in,
    weight,
    spectrum,
    angle_type,
    mu_ID,
    simulation,
    data,
    E_edges,
    mu_edges,
    table,
    out,
    A,
):
    """Continuous lab-frame spectrum: integrate f(E_out, mu0) over outgoing bins."""
    J = len(mu_edges) - 1
    correlated = angle_type == ANGLE_ENERGY_CORRELATED
    if correlated:
        low, high = correlated_distribution_support(E_in, spectrum, simulation, data)
        E_points = correlated_energy_breakpoints(E_in, spectrum, simulation, data)
    else:
        low, high = distribution_support(E_in, spectrum, simulation, data, True)
        E_points = distribution_breakpoints(E_in, spectrum, simulation, data, True)
    if high <= low:
        return

    # Uncorrelated: energy and angle integrals factorize
    angle_factor = np.zeros((J, J))
    if not correlated:
        if angle_type == ANGLE_ISOTROPIC:
            for j in range(J):
                for jp in range(J):
                    dmu_j = mu_edges[j + 1] - mu_edges[j]
                    dmu_jp = mu_edges[jp + 1] - mu_edges[jp]
                    angle_factor[j, jp] = 0.5 * dmu_j * dmu_jp
        else:
            mu_pieces = _pieces(
                -1.0, 1.0, _angle_breakpoints(E_in, angle_type, mu_ID, simulation, data)
            )
            for p in range(len(mu_pieces) - 1):
                c = mu_pieces[p]
                d = mu_pieces[p + 1]
                for q in range(len(_GL_X)):
                    mu0 = 0.5 * (d - c) * _GL_X[q] + 0.5 * (d + c)
                    p_mu = _angle_density(
                        E_in, mu0, angle_type, mu_ID, simulation, data
                    )
                    angular_transfer(mu0, table, A)
                    angle_factor += (0.5 * (d - c) * _GL_W[q] * p_mu) * A

    g0, g1 = _bin_range(E_edges, low, high)
    for g in range(g0, g1 + 1):
        Ea = max(E_edges[g], low)
        Eb = min(E_edges[g + 1], high)
        if Eb <= Ea:
            continue
        E_pieces = _pieces(Ea, Eb, E_points)
        for e in range(len(E_pieces) - 1):
            a = E_pieces[e]
            b = E_pieces[e + 1]
            for qe in range(len(_GL_X)):
                E_out = 0.5 * (b - a) * _GL_X[qe] + 0.5 * (b + a)
                wE = 0.5 * (b - a) * _GL_W[qe] * weight
                if not correlated:
                    p_E = evaluate_distribution(
                        E_in, E_out, spectrum, simulation, data, True
                    )
                    _add_outgoing(out, g, E_out, E_edges, wE * p_E, angle_factor)
                    continue
                if spectrum["sub_type"] == DISTRIBUTION_TABULATED_ENERGY_ANGLE:
                    mu_points = tabulated_energy_angle_cosine_breakpoints(
                        E_in, E_out, spectrum, simulation, data
                    )
                else:
                    mu_points = np.empty(0)
                mu_pieces = _pieces(-1.0, 1.0, mu_points)
                for p in range(len(mu_pieces) - 1):
                    c = mu_pieces[p]
                    d = mu_pieces[p + 1]
                    for q in range(len(_GL_X)):
                        mu0 = 0.5 * (d - c) * _GL_X[q] + 0.5 * (d + c)
                        f = evaluate_correlated_distribution(
                            E_in, E_out, mu0, spectrum, simulation, data
                        )
                        if f == 0.0:
                            continue
                        angular_transfer(mu0, table, A)
                        _add_outgoing(
                            out, g, E_out, E_edges, wE * 0.5 * (d - c) * _GL_W[q] * f, A
                        )


@njit(cache=True)
def _free_gas_inner(E_in, nuclide, E_edges, table, out, A):
    """
    Free-gas elastic kernel in (E_out, mu0): pieces in E_out at E_in and at multiples
    of the kernel width sqrt(4 E kT / A) around it, and in mu0 toward the forward
    peak at mu0 = 1.
    """
    awr = nuclide["atomic_weight_ratio"]
    kT = BOLTZMANN_K * nuclide["temperature"]
    low, high = free_gas_support(E_in, awr, kT)
    width = math.sqrt(4.0 * E_in * kT / awr) + kT
    E_points = np.empty(13)
    E_points[0] = E_in
    for k in range(6):
        E_points[1 + 2 * k] = E_in - (0.25 * 2.0**k) * width
        E_points[2 + 2 * k] = E_in + (0.25 * 2.0**k) * width
    mu_points = np.array([0.0, 0.9, 0.99, 0.999, 0.9999, 0.99999])
    mu_pieces = _pieces(-1.0, 1.0, mu_points)
    g0, g1 = _bin_range(E_edges, low, high)
    for g in range(g0, g1 + 1):
        Ea = max(E_edges[g], low)
        Eb = min(E_edges[g + 1], high)
        if Eb <= Ea:
            continue
        E_pieces = _pieces(Ea, Eb, E_points)
        for e in range(len(E_pieces) - 1):
            a = E_pieces[e]
            b = E_pieces[e + 1]
            for qe in range(len(_GL_X)):
                E_out = 0.5 * (b - a) * _GL_X[qe] + 0.5 * (b + a)
                wE = 0.5 * (b - a) * _GL_W[qe]
                for p in range(len(mu_pieces) - 1):
                    c = mu_pieces[p]
                    d = mu_pieces[p + 1]
                    for q in range(len(_GL_X)):
                        mu0 = 0.5 * (d - c) * _GL_X[q] + 0.5 * (d + c)
                        f = free_gas_density(E_in, E_out, mu0, awr, kT)
                        if f == 0.0:
                            continue
                        angular_transfer(mu0, table, A)
                        _add_outgoing(
                            out, g, E_out, E_edges, wE * 0.5 * (d - c) * _GL_W[q] * f, A
                        )


@njit(cache=True)
def _continuous_inner(
    E_in, reaction, nuclide, simulation, data, E_edges, mu_edges, table, out, A
):
    awr = nuclide["atomic_weight_ratio"]
    com = reaction["reference_frame"] == REFERENCE_FRAME_COM

    if reaction["sub_type"] == NEUTRON_REACTION_FISSION:
        fission = simulation["neutron_fission_reactions"][reaction["sub_ID"]]
        spectrum = simulation["distributions"][fission["spectrum_ID"]]
        weight = fission_yield(E_in, nuclide, simulation, data)
        if com:
            _com_spectrum_inner(
                E_in,
                weight,
                spectrum,
                fission["angle_type"],
                fission["mu_ID"],
                awr,
                simulation,
                data,
                E_edges,
                table,
                out,
                A,
            )
        else:
            _lab_spectrum_inner(
                E_in,
                weight,
                spectrum,
                fission["angle_type"],
                fission["mu_ID"],
                simulation,
                data,
                E_edges,
                mu_edges,
                table,
                out,
                A,
            )
        return

    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    for n in range(inelastic["N_spectrum"]):
        weight = _spectrum_weight(E_in, inelastic, n, simulation, data)
        if weight == 0.0:
            continue
        spectrum = _inelastic_spectrum(inelastic, n, simulation, data)
        if com:
            _com_spectrum_inner(
                E_in,
                weight,
                spectrum,
                inelastic["angle_type"],
                inelastic["mu_ID"],
                awr,
                simulation,
                data,
                E_edges,
                table,
                out,
                A,
            )
        else:
            _lab_spectrum_inner(
                E_in,
                weight,
                spectrum,
                inelastic["angle_type"],
                inelastic["mu_ID"],
                simulation,
                data,
                E_edges,
                mu_edges,
                table,
                out,
                A,
            )


@njit(cache=True)
def _integrand(
    E_in, reaction, ktype, nuclide, simulation, data, E_edges, mu_edges, table, out, A
):
    """out[a, g, j, j'] = sigma_r(E_in) * inner(E_in)."""
    out[:] = 0.0
    sigma = reaction_micro_xs(E_in, reaction, nuclide, data)
    if sigma <= 0.0:
        return
    if ktype == KERNEL_CONTINUOUS:
        _continuous_inner(
            E_in, reaction, nuclide, simulation, data, E_edges, mu_edges, table, out, A
        )
    elif is_free_gas(E_in, reaction, nuclide):
        _free_gas_inner(E_in, nuclide, E_edges, table, out, A)
    else:
        _delta_inner(E_in, reaction, nuclide, simulation, data, E_edges, table, out, A)
    out *= sigma


# ======================================================================================
# Outer E_in integral (adaptive Gauss-Kronrod) for one incident bin
# ======================================================================================


@njit(cache=True)
def _support_crossings(points, reaction, nuclide, simulation, data, E_edges):
    """Incident energies between points where a support edge crosses a bin edge."""
    N = len(points)
    lows = np.empty(N)
    highs = np.empty(N)
    for i in range(N):
        lows[i], highs[i] = lab_energy_support(
            points[i], reaction, nuclide, simulation, data
        )

    extra = np.empty(4 * len(E_edges) + 4 * N)
    M = 0
    for i in range(N - 1):
        for side in range(2):
            v0 = lows[i] if side == 0 else highs[i]
            v1 = lows[i + 1] if side == 0 else highs[i + 1]
            if not (math.isfinite(v0) and math.isfinite(v1)):
                continue
            for E_edge in E_edges:
                if (v0 - E_edge) * (v1 - E_edge) >= 0.0:
                    continue
                # Bisection on the support edge
                a = points[i]
                b = points[i + 1]
                fa = v0 - E_edge
                for _ in range(60):
                    m = 0.5 * (a + b)
                    lo, hi = lab_energy_support(m, reaction, nuclide, simulation, data)
                    fm = (lo if side == 0 else hi) - E_edge
                    if fa * fm <= 0.0:
                        b = m
                    else:
                        a = m
                        fa = fm
                if M < len(extra):
                    extra[M] = 0.5 * (a + b)
                    M += 1
    return extra[:M]


@njit(cache=True)
def reaction_transfer_row(
    E_a, E_b, reaction, nuclide, simulation, data, E_edges, mu_edges, table, tol, out
):
    """
    out[a', a, g, j, j'] += int_{E_a}^{E_b} dE_in sigma_r(E_in) P_a'(x(E_in))
                            inner_r(E_in)[a, g, j, j'],
    with x(E_in) relative to [E_a, E_b] (the incident bin).

    Adaptive G7-K15 on pieces. A piece [a, b] is accepted when its Kronrod-Gauss
    difference is below tol times its own magnitude, or below its width share of tol
    times the row magnitude S (from one G7-K15 pass per piece): the row's error is then
    <= tol S. The width share stops refinement of negligible pieces (kinematic tails,
    near-zero kernels) where a relative tolerance cannot be met.
    """
    ktype = kernel_type(reaction, simulation, data)
    tol = max(tol, TOLERANCE_FLOOR)
    E_center = 0.5 * (E_a + E_b)
    E_half = 0.5 * (E_b - E_a)

    # Threshold
    xs_grid = mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data)
    E_threshold = xs_grid[reaction["xs_offset_"]]
    E_a = max(E_a, E_threshold)
    if E_a >= E_b:
        return 0

    # Breakpoints: cross-section grid, then support/bin-edge crossings
    i0 = np.searchsorted(xs_grid, E_a, side="right")
    i1 = np.searchsorted(xs_grid, E_b, side="left")
    points = _pieces(E_a, E_b, xs_grid[i0:i1])
    if reaction["sub_type"] == NEUTRON_REACTION_ELASTIC_SCATTERING:
        # The kernel switches from free gas to target at rest above 400 kT
        points = _pieces(
            E_a, E_b, np.concatenate((points, np.array([free_gas_threshold(nuclide)])))
        )
    crossings = _support_crossings(points, reaction, nuclide, simulation, data, E_edges)
    points = _pieces(E_a, E_b, np.concatenate((points, crossings)))

    shape = out.shape
    J = shape[3]
    A = np.zeros((J, J))
    f = np.zeros(shape[1:])
    K = np.zeros(shape)
    Gs = np.zeros(shape)

    # Interval stack
    stack_a = np.empty(MAX_DEPTH * 2 + len(points))
    stack_b = np.empty(MAX_DEPTH * 2 + len(points))
    stack_d = np.empty(MAX_DEPTH * 2 + len(points), dtype=np.int64)
    N_evaluation = 0

    # Row magnitude: one G7-K15 pass per piece
    S = 0.0
    for p in range(len(points) - 1):
        half = 0.5 * (points[p + 1] - points[p])
        center = 0.5 * (points[p + 1] + points[p])
        K[:] = 0.0
        for q in range(15):
            E_in = center + half * GK_X[q]
            _integrand(
                E_in,
                reaction,
                ktype,
                nuclide,
                simulation,
                data,
                E_edges,
                mu_edges,
                table,
                f,
                A,
            )
            N_evaluation += 1
            _add_incoming(K, f, half * GK_WK[q], E_in, E_center, E_half)
        S += np.sum(np.abs(K))
    width = points[-1] - points[0]

    for p in range(len(points) - 1):
        top = 0
        stack_a[0] = points[p]
        stack_b[0] = points[p + 1]
        stack_d[0] = 0
        top = 1
        while top > 0:
            top -= 1
            a = stack_a[top]
            b = stack_b[top]
            depth = stack_d[top]
            half = 0.5 * (b - a)
            center = 0.5 * (b + a)
            K[:] = 0.0
            Gs[:] = 0.0
            for q in range(15):
                E_in = center + half * GK_X[q]
                _integrand(
                    E_in,
                    reaction,
                    ktype,
                    nuclide,
                    simulation,
                    data,
                    E_edges,
                    mu_edges,
                    table,
                    f,
                    A,
                )
                N_evaluation += 1
                _add_incoming(K, f, half * GK_WK[q], E_in, E_center, E_half)
                if GK_WG[q] != 0.0:
                    _add_incoming(Gs, f, half * GK_WG[q], E_in, E_center, E_half)
            error = np.max(np.abs(K - Gs))
            magnitude = np.sum(np.abs(K))
            if (
                error <= tol * magnitude
                or error <= tol * S * (b - a) / width
                or depth >= MAX_DEPTH
                or magnitude == 0.0
            ):
                out += K
            else:
                stack_a[top] = a
                stack_b[top] = center
                stack_d[top] = depth + 1
                stack_a[top + 1] = center
                stack_b[top + 1] = b
                stack_d[top + 1] = depth + 1
                top += 2
    return N_evaluation


# ======================================================================================
# Nuclide transfer moments
# ======================================================================================


def nuclide_reactions(simulation, data, nuclide):
    """(base reaction, is_fission) for every neutron-emitting reaction of a nuclide."""
    reactions = []
    groups = [
        (
            "N_neutron_elastic_scattering_reaction",
            mcdc_get.nuclide.neutron_elastic_scattering_reaction_IDs,
            False,
        ),
        (
            "N_neutron_inelastic_scattering_reaction",
            mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs,
            False,
        ),
        (
            "N_neutron_fission_reaction",
            mcdc_get.nuclide.neutron_fission_reaction_IDs,
            True,
        ),
    ]
    for count, getter, is_fission in groups:
        for i in range(nuclide[count]):
            reactions.append(
                (simulation["neutron_reactions"][getter(i, nuclide, data)], is_fission)
            )
    return reactions


def nuclide_transfer_moments(
    simulation,
    data,
    nuclide,
    E_edges,
    mu_edges,
    table,
    tol=1e-8,
    tol_low=1e-8,
    energy_order=0,
):
    """
    Transfer moments M[a', g', j', a, g, j] of one nuclide (per unit atom density) for
    energy polynomials a', a <= energy_order, split into scattering (elastic +
    inelastic) and fission. The lowest energy decade of the
    grid uses the tighter tolerance tol_low. Rows are computed in parallel over MPI
    ranks and summed on all of them.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    G = len(E_edges) - 1
    J = len(mu_edges) - 1
    P = energy_order + 1
    M_scatter = np.zeros((P, G, J, P, G, J))
    M_fission = np.zeros((P, G, J, P, G, J))
    row = np.zeros((P, P, G, J, J))
    task = -1
    for reaction, is_fission in nuclide_reactions(simulation, data, nuclide):
        target = M_fission if is_fission else M_scatter
        for gp in range(G):
            # Rows are distributed round-robin over MPI ranks
            task += 1
            if task % comm.Get_size() != comm.Get_rank():
                continue
            row[:] = 0.0
            tolerance = tol_low if E_edges[gp + 1] <= 10.0 * E_edges[0] else tol
            reaction_transfer_row(
                E_edges[gp],
                E_edges[gp + 1],
                reaction,
                nuclide,
                simulation,
                data,
                E_edges,
                mu_edges,
                table,
                tolerance,
                row,
            )
            target[:, gp] += np.transpose(row, (0, 4, 1, 2, 3))
    for M in (M_scatter, M_fission):
        comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
    return M_scatter, M_fission
