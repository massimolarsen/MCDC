"""
Probability density evaluators for the MC/DC distribution samplers.

Each evaluator returns the density that the matching sampler in
`mcdc.transport.distribution` actually produces (table selection by interpolation
factor, unit-base scaling, linear inversion of tabulated PDFs, ...). Residual Monte
Carlo needs these "inverted" scattering laws: given incident energy E and an outgoing
point (E_out[, mu]), return the probability density of producing it.

Supported:
  - Multi-table (tabulated angle and energy spectra, scaled or not; ENDF Law 1/4)
  - Evaporation (ENDF Law 9) and simple Maxwellian fission spectrum (ENDF Law 7)
  - Kalbach-Mann (ENDF Law 44)
  - Tabulated energy-angle (ENDF Law 61)
  - N-body phase space (ENDF Law 6, ACE Law 66)
Level scattering (Law 3), elastic scattering and the free-gas thermal kernel are
handled at the reaction level. The derivations are in mcdc/rmc/writeups.
"""

import math

import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import (
    DISTRIBUTION_EVAPORATION,
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_MAXWELLIAN,
    DISTRIBUTION_MULTITABLE,
    DISTRIBUTION_N_BODY,
    DISTRIBUTION_TABULATED,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
    INTERPOLATION_HISTOGRAM,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.util import find_bin

# ======================================================================================
# General distribution evaluators
# ======================================================================================


@njit(cache=True)
def evaluate_distribution(E, x, distribution, simulation, data, scale):
    """Density of `sample_distribution[_with_scale]` producing x at incident energy E."""
    distribution_type = distribution["sub_type"]
    ID = distribution["sub_ID"]

    if distribution_type == DISTRIBUTION_TABULATED:
        table = simulation["tabulated_distributions"][ID]
        return evaluate_tabulated(x, table, simulation, data)

    elif distribution_type == DISTRIBUTION_MULTITABLE:
        multi_table = simulation["multi_table_distributions"][ID]
        return evaluate_multi_table(E, x, multi_table, simulation, data, scale)

    elif distribution_type == DISTRIBUTION_EVAPORATION:
        evaporation = simulation["evaporation_distributions"][ID]
        return evaluate_evaporation(E, x, evaporation, simulation, data)

    elif distribution_type == DISTRIBUTION_MAXWELLIAN:
        maxwellian = simulation["maxwellian_distributions"][ID]
        return evaluate_maxwellian(E, x, maxwellian, simulation, data)

    else:
        raise ValueError("RMC: unsupported distribution type for evaluation")


@njit(cache=True)
def distribution_support(E, distribution, simulation, data, scale):
    """Range [x_low, x_high] of values the sampler can produce at incident energy E."""
    distribution_type = distribution["sub_type"]
    ID = distribution["sub_ID"]

    if distribution_type == DISTRIBUTION_TABULATED:
        table = simulation["tabulated_distributions"][ID]
        return tabulated_support(table, simulation, data)

    elif distribution_type == DISTRIBUTION_MULTITABLE:
        multi_table = simulation["multi_table_distributions"][ID]
        return multi_table_support(E, multi_table, simulation, data, scale)

    elif distribution_type == DISTRIBUTION_EVAPORATION:
        evaporation = simulation["evaporation_distributions"][ID]
        return 0.0, max(0.0, E - evaporation["restriction_energy"])

    elif distribution_type == DISTRIBUTION_MAXWELLIAN:
        maxwellian = simulation["maxwellian_distributions"][ID]
        return 0.0, max(0.0, E - maxwellian["restriction_energy"])

    else:
        raise ValueError("RMC: unsupported distribution type for evaluation")


@njit(cache=True)
def evaluate_correlated_distribution(E, E_out, mu, distribution, simulation, data):
    """Joint density of `sample_correlated_distribution_with_scale` producing (E_out, mu)."""
    distribution_type = distribution["sub_type"]
    ID = distribution["sub_ID"]

    if distribution_type == DISTRIBUTION_KALBACH_MANN:
        kalbach_mann = simulation["kalbach_mann_distributions"][ID]
        return evaluate_kalbach_mann(E, E_out, mu, kalbach_mann, data)

    elif distribution_type == DISTRIBUTION_TABULATED_ENERGY_ANGLE:
        table = simulation["tabulated_energy_angle_distributions"][ID]
        return evaluate_tabulated_energy_angle(E, E_out, mu, table, data)

    elif distribution_type == DISTRIBUTION_N_BODY:
        nbody = simulation["nbody_distributions"][ID]
        return evaluate_nbody(E, E_out, mu, nbody, simulation, data)

    else:
        raise ValueError("RMC: unsupported correlated distribution for evaluation")


@njit(cache=True)
def correlated_distribution_support(E, distribution, simulation, data):
    """Outgoing-energy range [E_low, E_high] of a correlated distribution at E."""
    distribution_type = distribution["sub_type"]
    ID = distribution["sub_ID"]

    if distribution_type == DISTRIBUTION_KALBACH_MANN:
        kalbach_mann = simulation["kalbach_mann_distributions"][ID]
        grid = mcdc_get.kalbach_mann_distribution.energy_all(kalbach_mann, data)
        offsets = mcdc_get.kalbach_mann_distribution.offset_all(kalbach_mann, data)
        energy_out = mcdc_get.kalbach_mann_distribution.energy_out_all(
            kalbach_mann, data
        )
        return _correlated_energy_range(E, grid, offsets, energy_out)

    elif distribution_type == DISTRIBUTION_TABULATED_ENERGY_ANGLE:
        table = simulation["tabulated_energy_angle_distributions"][ID]
        grid = mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)
        offsets = mcdc_get.tabulated_energy_angle_distribution.offset_all(table, data)
        energy_out = mcdc_get.tabulated_energy_angle_distribution.energy_out_all(
            table, data
        )
        return _correlated_energy_range(E, grid, offsets, energy_out)

    elif distribution_type == DISTRIBUTION_N_BODY:
        nbody = simulation["nbody_distributions"][ID]
        return 0.0, max(0.0, nbody_energy_max(E, nbody))

    else:
        raise ValueError("RMC: unsupported correlated distribution for evaluation")


# ======================================================================================
# Tabulated
# ======================================================================================


@njit(cache=True)
def _pdf_table(table, simulation, data):
    pdf_data = simulation["data"][table["pdf_ID"]]
    return simulation["table_data"][pdf_data["sub_ID"]]


@njit(cache=True)
def evaluate_tabulated(x, table, simulation, data):
    """Density of `sample_tabulated`: the histogram or piecewise-linear table PDF."""
    pdf_table = _pdf_table(table, simulation, data)
    values = mcdc_get.table_data.x_all(pdf_table, data)
    pdf = mcdc_get.table_data.y_all(pdf_table, data)
    interpolation = mcdc_get.table_data.interpolations(0, pdf_table, data)
    return _piecewise_pdf(x, values, pdf, interpolation == INTERPOLATION_HISTOGRAM)


@njit(cache=True)
def tabulated_support(table, simulation, data):
    pdf_table = _pdf_table(table, simulation, data)
    return (
        mcdc_get.table_data.x(0, pdf_table, data),
        mcdc_get.table_data.x_last(pdf_table, data),
    )


@njit(cache=True)
def _piecewise_pdf(x, values, pdf, histogram):
    """Histogram or linear PDF on a value grid; zero outside the grid."""
    if x < values[0] or x > values[-1]:
        return 0.0
    k = find_bin(x, values)
    if k == -1:
        # Exactly on the first point (find_bin treats it as outside)
        k = 0
    if histogram:
        return pdf[k]
    return pdf[k] + (pdf[k + 1] - pdf[k]) * (x - values[k]) / (
        values[k + 1] - values[k]
    )


# ======================================================================================
# Multi-table
# ======================================================================================


@njit(cache=True)
def _multi_table_tabulated(idx, multi_table, simulation, data):
    ID = mcdc_get.multi_table_distribution.table_IDs(idx, multi_table, data)
    sub_ID = simulation["distributions"][ID]["sub_ID"]
    return simulation["tabulated_distributions"][sub_ID]


@njit(cache=True)
def grid_fraction(E, grid, idx):
    """
    Interpolation fraction of E in [grid[idx], grid[idx + 1]]; 0 on a zero-width
    interval (a repeated grid energy marking a discontinuity of the data, where the
    samplers' fraction is undefined: only reached by quadrature nodes and support
    evaluations exactly at that energy, never by a sampled energy).
    """
    width = grid[idx + 1] - grid[idx]
    if width <= 0.0:
        return 0.0
    return (E - grid[idx]) / width


@njit(cache=True)
def _multi_table_interval(E, grid):
    """Mirror `_sample_multi_table` table selection: (idx, f, use_scale_allowed)."""
    if E < grid[0]:
        return 0, 0.0, False
    if E > grid[-1]:
        return len(grid) - 1, 0.0, False
    idx = find_bin(E, grid)
    f = grid_fraction(E, grid, idx)
    return idx, f, True


@njit(cache=True)
def evaluate_multi_table(E, x, multi_table, simulation, data, scale):
    """
    Density of `_sample_multi_table`.

    Unscaled: (1 - f) p_i(x) + f p_{i+1}(x).
    Scaled (unit-base interpolation): sum_l w_l p_l(x_l) (x_l,K - x_l,1) / (x_K - x_1),
    with x_l = x_l,1 + (x - x_1)(x_l,K - x_l,1)/(x_K - x_1), w_i = 1 - f, w_{i+1} = f.
    """
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)
    idx, f, in_grid = _multi_table_interval(E, grid)

    # Off grid: single edge table, never scaled
    if not in_grid:
        table = _multi_table_tabulated(idx, multi_table, simulation, data)
        return evaluate_tabulated(x, table, simulation, data)

    table0 = _multi_table_tabulated(idx, multi_table, simulation, data)
    table1 = _multi_table_tabulated(idx + 1, multi_table, simulation, data)

    if not scale:
        return (1.0 - f) * evaluate_tabulated(x, table0, simulation, data) + f * (
            evaluate_tabulated(x, table1, simulation, data)
        )

    low0, high0 = tabulated_support(table0, simulation, data)
    low1, high1 = tabulated_support(table1, simulation, data)
    val_min = low0 + f * (low1 - low0)
    val_max = high0 + f * (high1 - high0)
    if x < val_min or x > val_max:
        return 0.0

    density = 0.0
    if f < 1.0:
        x0 = low0 + (x - val_min) * (high0 - low0) / (val_max - val_min)
        density += (
            (1.0 - f)
            * evaluate_tabulated(x0, table0, simulation, data)
            * (high0 - low0)
            / (val_max - val_min)
        )
    if f > 0.0:
        x1 = low1 + (x - val_min) * (high1 - low1) / (val_max - val_min)
        density += (
            f
            * evaluate_tabulated(x1, table1, simulation, data)
            * (high1 - low1)
            / (val_max - val_min)
        )
    return density


@njit(cache=True)
def multi_table_support(E, multi_table, simulation, data, scale):
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)
    idx, f, in_grid = _multi_table_interval(E, grid)

    if not in_grid:
        table = _multi_table_tabulated(idx, multi_table, simulation, data)
        return tabulated_support(table, simulation, data)

    table0 = _multi_table_tabulated(idx, multi_table, simulation, data)
    table1 = _multi_table_tabulated(idx + 1, multi_table, simulation, data)
    low0, high0 = tabulated_support(table0, simulation, data)
    low1, high1 = tabulated_support(table1, simulation, data)

    if scale:
        return low0 + f * (low1 - low0), high0 + f * (high1 - high0)
    # Unscaled mixture: union of the two tables (f = 0 or 1 selects one)
    if f == 0.0:
        return low0, high0
    return min(low0, low1), max(high0, high1)


# ======================================================================================
# Correlated energy part (shared by Kalbach-Mann and tabulated energy-angle)
# ======================================================================================


@njit(cache=True)
def _outgoing_table_range(l, grid, offsets, N_out):
    start = int(offsets[l])
    if l + 1 == len(grid):
        end = N_out
    else:
        end = int(offsets[l + 1])
    return start, end


@njit(cache=True)
def _correlated_energy_range(E, grid, offsets, energy_out):
    """[E_min, E_max] of the unit-base interpolated outgoing table at E."""
    idx = find_bin(E, grid)
    if idx == -1:
        return 0.0, 0.0
    f = grid_fraction(E, grid, idx)
    N_out = len(energy_out)
    start0, end0 = _outgoing_table_range(idx, grid, offsets, N_out)
    start1, end1 = _outgoing_table_range(idx + 1, grid, offsets, N_out)
    E_min = energy_out[start0] + f * (energy_out[start1] - energy_out[start0])
    E_max = energy_out[end0 - 1] + f * (energy_out[end1 - 1] - energy_out[end0 - 1])
    return E_min, E_max


@njit(cache=True)
def _correlated_energy_component(
    E_out, l, weight, E_min, E_max, grid, offsets, energy_out, pdf
):
    """
    Contribution of outgoing table l to the scaled energy density, mirroring the
    linear CDF inversion and unit-base scaling of the correlated samplers.

    Returns (density, k_global, E_hat); density = 0 if E_out is not reachable via l.
    """
    if weight <= 0.0:
        return 0.0, -1, 0.0
    start, end = _outgoing_table_range(l, grid, offsets, len(energy_out))
    E_low = energy_out[start]
    E_high = energy_out[end - 1]
    E_hat = E_low + (E_out - E_min) * (E_high - E_low) / (E_max - E_min)
    if E_hat < E_low or E_hat > E_high:
        return 0.0, -1, 0.0

    k = find_bin(E_hat, energy_out[start:end])
    if k == -1:
        k = 0
    k += start
    E0 = energy_out[k]
    E1 = energy_out[k + 1]
    p = pdf[k]
    if E1 > E0:
        p += (pdf[k + 1] - pdf[k]) * (E_hat - E0) / (E1 - E0)
    density = weight * p * (E_high - E_low) / (E_max - E_min)
    return density, k, E_hat


# ======================================================================================
# Kalbach-Mann (ENDF Law 44)
# ======================================================================================


@njit(cache=True)
def kalbach_density(mu, R, A):
    """Kalbach angular density p(mu) = A / (2 sinh A) [cosh(A mu) + R sinh(A mu)]."""
    if A < 1e-8:
        # A -> 0 limit: isotropic plus linear precompound term
        return 0.5 * (1.0 + R * A * mu)
    return 0.5 * A / math.sinh(A) * (math.cosh(A * mu) + R * math.sinh(A * mu))


@njit(cache=True)
def evaluate_kalbach_mann(E, E_out, mu, kalbach_mann, data):
    """Joint (E_out, mu) density of `sample_kalbach_mann` (COM or lab per reaction)."""
    if mu < -1.0 or mu > 1.0:
        return 0.0
    grid = mcdc_get.kalbach_mann_distribution.energy_all(kalbach_mann, data)
    offsets = mcdc_get.kalbach_mann_distribution.offset_all(kalbach_mann, data)
    energy_out = mcdc_get.kalbach_mann_distribution.energy_out_all(kalbach_mann, data)
    pdf = mcdc_get.kalbach_mann_distribution.pdf_all(kalbach_mann, data)
    R_all = mcdc_get.kalbach_mann_distribution.precompound_factor_all(
        kalbach_mann, data
    )
    A_all = mcdc_get.kalbach_mann_distribution.angular_slope_all(kalbach_mann, data)

    idx = find_bin(E, grid)
    if idx == -1:
        return 0.0
    f = grid_fraction(E, grid, idx)
    E_min, E_max = _correlated_energy_range(E, grid, offsets, energy_out)
    if E_out < E_min or E_out > E_max:
        return 0.0

    density = 0.0
    for l, weight in ((idx, 1.0 - f), (idx + 1, f)):
        p_E, k, E_hat = _correlated_energy_component(
            E_out, l, weight, E_min, E_max, grid, offsets, energy_out, pdf
        )
        if p_E == 0.0:
            continue
        mE = (E_hat - energy_out[k]) / (energy_out[k + 1] - energy_out[k])
        R = R_all[k] + mE * (R_all[k + 1] - R_all[k])
        A = A_all[k] + mE * (A_all[k + 1] - A_all[k])
        density += p_E * kalbach_density(mu, R, A)
    return density


# ======================================================================================
# Tabulated energy-angle (ENDF Law 61)
# ======================================================================================


@njit(cache=True)
def evaluate_tabulated_energy_angle(E, E_out, mu, table, data):
    """Joint (E_out, mu) density of `sample_tabulated_energy_angle`."""
    grid = mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)
    offsets = mcdc_get.tabulated_energy_angle_distribution.offset_all(table, data)
    energy_out = mcdc_get.tabulated_energy_angle_distribution.energy_out_all(
        table, data
    )
    pdf = mcdc_get.tabulated_energy_angle_distribution.pdf_all(table, data)
    cdf = mcdc_get.tabulated_energy_angle_distribution.cdf_all(table, data)
    cosine_offsets = mcdc_get.tabulated_energy_angle_distribution.cosine_offset__all(
        table, data
    )
    cosine = mcdc_get.tabulated_energy_angle_distribution.cosine_all(table, data)
    cosine_pdf = mcdc_get.tabulated_energy_angle_distribution.cosine_pdf_all(
        table, data
    )
    N_out = len(energy_out)

    idx = find_bin(E, grid)
    if idx == -1:
        return 0.0
    f = grid_fraction(E, grid, idx)
    E_min, E_max = _correlated_energy_range(E, grid, offsets, energy_out)
    if E_out < E_min or E_out > E_max:
        return 0.0

    density = 0.0
    for l, weight in ((idx, 1.0 - f), (idx + 1, f)):
        p_E, k, E_hat = _correlated_energy_component(
            E_out, l, weight, E_min, E_max, grid, offsets, energy_out, pdf
        )
        if p_E == 0.0:
            continue

        # Angular table: nearest outgoing point in CDF, as in the sampler
        #   (xi2 - c_k > c_{k+1} - xi2, where xi2 = c(E_hat))
        dE = E_hat - energy_out[k]
        width = energy_out[k + 1] - energy_out[k]
        m = (pdf[k + 1] - pdf[k]) / width if width > 0.0 else 0.0
        c_hat = cdf[k] + pdf[k] * dE + 0.5 * m * dE * dE
        j = k + 1 if c_hat - cdf[k] > cdf[k + 1] - c_hat else k

        start = int(cosine_offsets[j])
        end = len(cosine) if j + 1 == N_out else int(cosine_offsets[j + 1])
        p_mu = _piecewise_pdf(mu, cosine[start:end], cosine_pdf[start:end], False)
        density += p_E * p_mu
    return density


# ======================================================================================
# Evaporation (Law 9) and simple Maxwellian (Law 7)
# ======================================================================================
#
# Both samplers draw from the untruncated law and reject samples outside
# [0, E - U] (U: restriction energy), so the density is the law renormalized on
# [0, E - U], with the nuclear temperature theta(E) interpolated as the sampler does.


@njit(cache=True)
def _nuclear_temperature(E, distribution, simulation, data):
    table = simulation["data"][distribution["nuclear_temperature_ID"]]
    return evaluate_data(E, table, simulation, data)


@njit(cache=True)
def evaporation_normalization(w):
    """int_0^w s e^{-s} ds = 1 - e^{-w} (1 + w), cancellation-free for small w."""
    if w < 1e-3:
        return w * w * (0.5 - w / 3.0 + w * w / 8.0 - w * w * w / 30.0)
    return -math.expm1(-w) - w * math.exp(-w)


@njit(cache=True)
def maxwellian_normalization(w):
    """int_0^w sqrt(s) e^{-s} ds = sqrt(pi)/2 erf(sqrt(w)) - sqrt(w) e^{-w}."""
    if w < 1e-3:
        return w * math.sqrt(w) * (2.0 / 3.0 - 0.4 * w + w * w / 7.0)
    r = math.sqrt(w)
    return 0.5 * math.sqrt(math.pi) * math.erf(r) - r * math.exp(-w)


@njit(cache=True)
def evaluate_evaporation(E, E_out, evaporation, simulation, data):
    """f(E_out | E) = E_out e^{-E_out/theta} / (theta^2 I(w)), 0 <= E_out <= E - U."""
    limit = E - evaporation["restriction_energy"]
    if limit <= 0.0 or E_out < 0.0 or E_out > limit:
        return 0.0
    theta = _nuclear_temperature(E, evaporation, simulation, data)
    if theta <= 0.0:
        return 0.0
    norm = theta * theta * evaporation_normalization(limit / theta)
    return E_out * math.exp(-E_out / theta) / norm


@njit(cache=True)
def evaluate_maxwellian(E, E_out, maxwellian, simulation, data):
    """f(E_out | E) = sqrt(E_out) e^{-E_out/theta} / (theta^{3/2} I(w)), 0 <= E_out <= E - U."""
    limit = E - maxwellian["restriction_energy"]
    if limit <= 0.0 or E_out < 0.0 or E_out > limit:
        return 0.0
    theta = _nuclear_temperature(E, maxwellian, simulation, data)
    if theta <= 0.0:
        return 0.0
    norm = theta * math.sqrt(theta) * maxwellian_normalization(limit / theta)
    return math.sqrt(E_out) * math.exp(-E_out / theta) / norm


# ======================================================================================
# N-body phase space (ENDF Law 6, ACE Law 66)
# ======================================================================================


@njit(cache=True)
def nbody_energy_max(E, nbody):
    """E_max(E) = (Ap - 1)/Ap (A/(A + 1) E + Q), stored as slope and offset."""
    return nbody["energy_max_slope"] * E + nbody["energy_max_offset"]


@njit(cache=True)
def evaluate_nbody(E, E_cm, mu_cm, nbody, simulation, data):
    """
    f(E_cm, mu_cm | E) = p(T) / E_max / 2 with T = E_cm / E_max: the sampler draws T
    from the tabulated reduced distribution and an isotropic COM cosine.
    """
    E_max = nbody_energy_max(E, nbody)
    if E_max <= 0.0 or E_cm < 0.0 or E_cm > E_max or abs(mu_cm) > 1.0:
        return 0.0
    return 0.5 * evaluate_tabulated(E_cm / E_max, nbody, simulation, data) / E_max


# ======================================================================================
# Breakpoints (kinks of the sampled densities), for piecewise-smooth quadrature
# ======================================================================================


@njit(cache=True)
def _append_points(out, N, values):
    for v in values:
        out[N] = v
        N += 1
    return N


@njit(cache=True)
def distribution_breakpoints(E, distribution, simulation, data, scale):
    """Values x where the density of `evaluate_distribution` has kinks (unsorted)."""
    distribution_type = distribution["sub_type"]
    ID = distribution["sub_ID"]

    if distribution_type == DISTRIBUTION_TABULATED:
        table = simulation["tabulated_distributions"][ID]
        return mcdc_get.table_data.x_all(
            _pdf_table(table, simulation, data), data
        ).copy()

    if (
        distribution_type == DISTRIBUTION_EVAPORATION
        or distribution_type == DISTRIBUTION_MAXWELLIAN
    ):
        # Smooth on the support; its end points come from distribution_support
        return np.empty(0)

    multi_table = simulation["multi_table_distributions"][ID]
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)
    idx, f, in_grid = _multi_table_interval(E, grid)
    table0 = _multi_table_tabulated(idx, multi_table, simulation, data)
    x0 = mcdc_get.table_data.x_all(_pdf_table(table0, simulation, data), data)
    if not in_grid:
        return x0.copy()

    table1 = _multi_table_tabulated(idx + 1, multi_table, simulation, data)
    x1 = mcdc_get.table_data.x_all(_pdf_table(table1, simulation, data), data)
    out = np.empty(len(x0) + len(x1))
    if not scale:
        N = _append_points(out, 0, x0)
        _append_points(out, N, x1)
        return out

    # Map each table's points through the unit-base scaling
    val_min = x0[0] + f * (x1[0] - x0[0])
    val_max = x0[-1] + f * (x1[-1] - x0[-1])
    N = 0
    for x_table in (x0, x1):
        span = x_table[-1] - x_table[0]
        for v in x_table:
            out[N] = val_min + (v - x_table[0]) * (val_max - val_min) / span
            N += 1
    return out


@njit(cache=True)
def correlated_energy_breakpoints(E, distribution, simulation, data):
    """Outgoing energies where a correlated distribution's energy density has kinks."""
    if distribution["sub_type"] == DISTRIBUTION_N_BODY:
        # Table points of the reduced variable, scaled by E_max(E)
        nbody = simulation["nbody_distributions"][distribution["sub_ID"]]
        values = mcdc_get.table_data.x_all(_pdf_table(nbody, simulation, data), data)
        return values * nbody_energy_max(E, nbody)

    if distribution["sub_type"] == DISTRIBUTION_KALBACH_MANN:
        kalbach_mann = simulation["kalbach_mann_distributions"][distribution["sub_ID"]]
        grid = mcdc_get.kalbach_mann_distribution.energy_all(kalbach_mann, data)
        offsets = mcdc_get.kalbach_mann_distribution.offset_all(kalbach_mann, data)
        energy_out = mcdc_get.kalbach_mann_distribution.energy_out_all(
            kalbach_mann, data
        )
    else:
        table = simulation["tabulated_energy_angle_distributions"][
            distribution["sub_ID"]
        ]
        grid = mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)
        offsets = mcdc_get.tabulated_energy_angle_distribution.offset_all(table, data)
        energy_out = mcdc_get.tabulated_energy_angle_distribution.energy_out_all(
            table, data
        )

    law61 = distribution["sub_type"] == DISTRIBUTION_TABULATED_ENERGY_ANGLE
    pdf = energy_out  # placeholders (only used for Law 61)
    cdf = energy_out
    if law61:
        table = simulation["tabulated_energy_angle_distributions"][
            distribution["sub_ID"]
        ]
        pdf = mcdc_get.tabulated_energy_angle_distribution.pdf_all(table, data)
        cdf = mcdc_get.tabulated_energy_angle_distribution.cdf_all(table, data)

    idx = find_bin(E, grid)
    if idx == -1:
        return np.empty(0)
    E_min, E_max = _correlated_energy_range(E, grid, offsets, energy_out)
    N_out = len(energy_out)
    start0, end0 = _outgoing_table_range(idx, grid, offsets, N_out)
    start1, end1 = _outgoing_table_range(idx + 1, grid, offsets, N_out)
    out = np.empty(2 * ((end0 - start0) + (end1 - start1)))
    N = 0
    for start, end in ((start0, end0), (start1, end1)):
        E_low = energy_out[start]
        E_high = energy_out[end - 1]
        scale = (E_max - E_min) / (E_high - E_low)
        for k in range(start, end):
            out[N] = E_min + (energy_out[k] - E_low) * scale
            N += 1
            # Law 61: the angular table switches where the CDF crosses the
            #   midpoint of the outgoing bin (a jump in the angular density)
            if law61 and k < end - 1:
                target = 0.5 * (cdf[k + 1] - cdf[k])
                width = energy_out[k + 1] - energy_out[k]
                m = (pdf[k + 1] - pdf[k]) / width if width > 0.0 else 0.0
                if abs(m) < 1e-300:
                    dE = target / pdf[k] if pdf[k] > 0.0 else 0.0
                else:
                    dE = (
                        math.sqrt(max(0.0, pdf[k] ** 2 + 2.0 * m * target)) - pdf[k]
                    ) / m
                out[N] = E_min + (energy_out[k] + dE - E_low) * scale
                N += 1
    return out[:N]


@njit(cache=True)
def tabulated_energy_angle_cosine_breakpoints(E, E_out, distribution, simulation, data):
    """Cosine-grid points of the Law-61 angular tables selected at (E, E_out)."""
    table = simulation["tabulated_energy_angle_distributions"][distribution["sub_ID"]]
    grid = mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)
    offsets = mcdc_get.tabulated_energy_angle_distribution.offset_all(table, data)
    energy_out = mcdc_get.tabulated_energy_angle_distribution.energy_out_all(
        table, data
    )
    pdf = mcdc_get.tabulated_energy_angle_distribution.pdf_all(table, data)
    cdf = mcdc_get.tabulated_energy_angle_distribution.cdf_all(table, data)
    cosine_offsets = mcdc_get.tabulated_energy_angle_distribution.cosine_offset__all(
        table, data
    )
    cosine = mcdc_get.tabulated_energy_angle_distribution.cosine_all(table, data)
    N_out = len(energy_out)

    out = np.empty(2 * len(cosine))
    N = 0
    idx = find_bin(E, grid)
    if idx == -1:
        return out[:0]
    f = grid_fraction(E, grid, idx)
    E_min, E_max = _correlated_energy_range(E, grid, offsets, energy_out)
    for l, weight in ((idx, 1.0 - f), (idx + 1, f)):
        p_E, k, E_hat = _correlated_energy_component(
            E_out, l, weight, E_min, E_max, grid, offsets, energy_out, pdf
        )
        if p_E == 0.0:
            continue
        dE = E_hat - energy_out[k]
        width = energy_out[k + 1] - energy_out[k]
        m = (pdf[k + 1] - pdf[k]) / width if width > 0.0 else 0.0
        c_hat = cdf[k] + pdf[k] * dE + 0.5 * m * dE * dE
        j = k + 1 if c_hat - cdf[k] > cdf[k + 1] - c_hat else k
        start = int(cosine_offsets[j])
        end = len(cosine) if j + 1 == N_out else int(cosine_offsets[j + 1])
        N = _append_points(out, N, cosine[start:end])
    return out[:N]
