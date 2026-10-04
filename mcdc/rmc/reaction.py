"""
Reaction-level lab-frame emission kernels for Residual Monte Carlo.

For a neutron reaction r at incident energy E_in, the kernel is the density of emitted
neutrons per reaction (yield included) in lab outgoing energy E_out and lab scattering
cosine mu_lab, exactly as produced by the MC/DC reaction samplers in
`mcdc.transport.physics.neutron.native`:

  - KERNEL_CONTINUOUS: a joint density f_L(E_out, mu_lab | E_in).
  - KERNEL_ELASTIC, KERNEL_LEVEL: a delta in the COM outgoing energy. mu_lab is a
    function of E_out, so the kernel is a line g(E_out | E_in) delta(mu_lab - mu_lab*).

Elastic scattering is target-at-rest only (RMC requires energies above the free-gas
threshold). Fission is all prompt, with yield nu_total(E_in) / k_eff.
"""

import math

import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import (
    ANGLE_ENERGY_CORRELATED,
    ANGLE_ISOTROPIC,
    BOLTZMANN_K,
    DISTRIBUTION_LEVEL_SCATTERING,
    PI,
    THERMAL_THRESHOLD_FACTOR,
    NEUTRON_REACTION_ELASTIC_SCATTERING,
    NEUTRON_REACTION_FISSION,
    NEUTRON_REACTION_INELASTIC_SCATTERING,
    REFERENCE_FRAME_COM,
)
from mcdc.rmc.evaluate import (
    correlated_distribution_support,
    distribution_support,
    evaluate_correlated_distribution,
    evaluate_distribution,
)
from mcdc.rmc.kinematics import (
    azimuthal_bin_probability,
    com_to_lab_jacobian,
    elastic_dmu_cm_dE_out,
    elastic_E_out_range,
    elastic_mu_cm,
    elastic_mu_lab,
    lab_to_com,
    level_dmu_cm_dE_out,
    level_E_out_range,
    level_mu_cm,
)
from mcdc.transport.physics.neutron.native import (
    neutron_fission_delayed_multiplicity,
    neutron_fission_prompt_multiplicity,
)
from mcdc.transport.data import evaluate_data
import mcdc.transport.rng as rng
from mcdc.transport.util import find_bin

KERNEL_CONTINUOUS = 0
KERNEL_ELASTIC = 1
KERNEL_LEVEL = 2

# ======================================================================================
# Kernel classification
# ======================================================================================


@njit(cache=True)
def kernel_type(reaction, simulation, data):
    """Kernel type of a base neutron reaction record."""
    reaction_type = reaction["sub_type"]
    if reaction_type == NEUTRON_REACTION_ELASTIC_SCATTERING:
        return KERNEL_ELASTIC
    if reaction_type == NEUTRON_REACTION_INELASTIC_SCATTERING:
        inelastic = simulation["neutron_inelastic_scattering_reactions"][
            reaction["sub_ID"]
        ]
        spectrum = _inelastic_spectrum(inelastic, 0, simulation, data)
        if spectrum["sub_type"] == DISTRIBUTION_LEVEL_SCATTERING:
            if inelastic["N_spectrum"] != 1 or (
                reaction["reference_frame"] != REFERENCE_FRAME_COM
            ):
                raise ValueError("RMC: unsupported level-scattering reaction layout")
            return KERNEL_LEVEL
        return KERNEL_CONTINUOUS
    if reaction_type == NEUTRON_REACTION_FISSION:
        return KERNEL_CONTINUOUS
    raise ValueError("RMC: reaction emits no neutrons")


@njit(cache=True)
def _inelastic_spectrum(inelastic, n, simulation, data):
    ID = mcdc_get.neutron_inelastic_scattering_reaction.energy_spectrum_IDs(
        n, inelastic, data
    )
    return simulation["distributions"][ID]


@njit(cache=True)
def inelastic_yield(E_in, inelastic, simulation, data):
    """Expected secondaries per reaction: integer multiplicity or tabulated yield(E)."""
    if inelastic["multiplicity_tabulated"]:
        table = simulation["data"][inelastic["multiplicity_table_ID"]]
        return evaluate_data(E_in, table, simulation, data)
    return float(inelastic["multiplicity"])


@njit(cache=True)
def _secondaries_through(N, E_in, inelastic, n, data):
    """
    Secondaries emitted through spectrum n when N are emitted: one per spectrum if
    N == N_spectrum, otherwise each picks a spectrum by its tabulated probability.
    """
    if N == inelastic["N_spectrum"]:
        return 1.0
    grid = mcdc_get.neutron_inelastic_scattering_reaction.spectrum_probability_grid_all(
        inelastic, data
    )
    idx = find_bin(E_in, grid)
    return N * mcdc_get.neutron_inelastic_scattering_reaction.spectrum_probability(
        idx, n, inelastic, data
    )


@njit(cache=True)
def _spectrum_weight(E_in, inelastic, n, simulation, data):
    """
    Expected number of secondaries emitted through spectrum n, mirroring
    `sample_inelastic_scattering`. With a tabulated yield nu(E) the sampler emits
    N = floor(nu + xi), i.e. m = floor(nu) or m + 1 with probability nu - m, and
    then applies the integer-multiplicity rule to that N.
    """
    if not inelastic["multiplicity_tabulated"]:
        return _secondaries_through(inelastic["multiplicity"], E_in, inelastic, n, data)
    nu = inelastic_yield(E_in, inelastic, simulation, data)
    m = math.floor(nu)
    f = nu - m
    weight = 0.0
    if f < 1.0:
        weight += (1.0 - f) * _secondaries_through(int(m), E_in, inelastic, n, data)
    if f > 0.0:
        weight += f * _secondaries_through(int(m) + 1, E_in, inelastic, n, data)
    return weight


@njit(cache=True)
def fission_yield(E_in, nuclide, simulation, data):
    """Expected fission neutrons per fission (all prompt), as in `sample_fission`."""
    nu = neutron_fission_prompt_multiplicity(
        E_in, nuclide, simulation, data
    ) + neutron_fission_delayed_multiplicity(E_in, nuclide, simulation, data)
    return nu / simulation["k_eff"]


# ======================================================================================
# Free-gas elastic scattering (thermal target motion)
# ======================================================================================
#
# Below E = 400 kT MC/DC samples the target velocity from a Maxwellian weighted by the
# relative speed (constant cross section) and scatters isotropically in the COM frame
# (`sample_elastic_scattering`, `sample_nucleus_velocity`). The induced lab kernel is
# the free-gas law (derivation in writeups/free_gas.tex):
#
#   f(E', mu0 | E) = ((A + 1)/A)^2 / (2 kT) sqrt(E'/E) S(alpha, beta) / D(a),
#   S = exp(-(alpha + beta)^2 / (4 alpha)) / sqrt(4 pi alpha),
#   alpha = (E' + E - 2 mu0 sqrt(E E')) / (A kT),  beta = (E' - E) / kT,
#   D(a) = (1 + 1/(2 a^2)) erf(a) + exp(-a^2) / (a sqrt(pi)),  a = sqrt(A E / kT).
#
# It requires isotropic COM scattering below the threshold (checked by the driver).

FREE_GAS_EXPONENT_CUTOFF = 50.0


@njit(cache=True)
def free_gas_threshold(nuclide):
    """MC/DC samples target motion for E <= 400 kT (none at T = 0)."""
    return THERMAL_THRESHOLD_FACTOR * BOLTZMANN_K * nuclide["temperature"]


@njit(cache=True)
def is_free_gas(E_in, reaction, nuclide):
    if reaction["sub_type"] != NEUTRON_REACTION_ELASTIC_SCATTERING:
        return False
    return nuclide["temperature"] > 0.0 and E_in <= free_gas_threshold(nuclide)


@njit(cache=True)
def free_gas_density(E_in, E_out, mu0, A, kT):
    """Free-gas lab density per unit E_out and lab cosine mu0 (normalized to 1)."""
    if E_out <= 0.0 or E_in <= 0.0 or mu0 < -1.0 or mu0 > 1.0:
        return 0.0
    alpha = (E_out + E_in - 2.0 * mu0 * math.sqrt(E_in * E_out)) / (A * kT)
    if alpha <= 0.0:
        return 0.0
    beta = (E_out - E_in) / kT
    exponent = (alpha + beta) ** 2 / (4.0 * alpha)
    if exponent > FREE_GAS_EXPONENT_CUTOFF:
        return 0.0
    S = math.exp(-exponent) / math.sqrt(4.0 * PI * alpha)
    a = math.sqrt(A * E_in / kT)
    D = (1.0 + 0.5 / (a * a)) * math.erf(a) + math.exp(-a * a) / (a * math.sqrt(PI))
    return ((A + 1.0) / A) ** 2 / (2.0 * kT) * math.sqrt(E_out / E_in) * S / D


@njit(cache=True)
def _free_gas_min_exponent(E_in, E_out, A, kT):
    """min over mu0 of (alpha + beta)^2 / (4 alpha) at fixed E_out <= E_in."""
    beta = (E_out - E_in) / kT
    root = math.sqrt(E_in)
    a_min = (math.sqrt(E_out) - root) ** 2 / (A * kT)
    a_max = (math.sqrt(E_out) + root) ** 2 / (A * kT)
    if a_min <= -beta <= a_max:
        return 0.0
    a = a_max if -beta > a_max else a_min
    if a <= 0.0:
        return math.inf
    return (a + beta) ** 2 / (4.0 * a)


@njit(cache=True)
def free_gas_support(E_in, A, kT):
    """
    [E_low, E_high] outside which the free-gas exponent exceeds the cutoff for every
    mu0: (alpha + beta)^2/(4 alpha) >= beta gives E_high = E + cutoff kT; E_low by
    bisection on the minimum exponent over mu0 (monotone below E).
    """
    cutoff = FREE_GAS_EXPONENT_CUTOFF
    E_high = E_in + cutoff * kT
    if _free_gas_min_exponent(E_in, 0.0, A, kT) <= cutoff:
        return 0.0, E_high
    low, high = 0.0, E_in
    for _ in range(80):
        mid = 0.5 * (low + high)
        if _free_gas_min_exponent(E_in, mid, A, kT) > cutoff:
            low = mid
        else:
            high = mid
    return low, E_high


@njit(cache=True)
def sample_free_gas(E_in, A, kT, container):
    """
    (E_out, mu0) of one free-gas collision: the target-velocity rejection scheme of
    `sample_nucleus_velocity` (relative-speed acceptance) and isotropic COM
    scattering, in units with E = v^2 (neutron along +z).
    """
    v = math.sqrt(E_in)
    beta = math.sqrt(A / kT)
    y = beta * v
    while True:
        if rng.lcg(container) < 2.0 / (2.0 + math.sqrt(PI) * y):
            x = math.sqrt(-math.log(rng.lcg(container) * rng.lcg(container)))
        else:
            c = math.cos(0.5 * PI * rng.lcg(container))
            x = math.sqrt(
                -math.log(rng.lcg(container)) - math.log(rng.lcg(container)) * c * c
            )
        V = x / beta
        mu_t = 2.0 * rng.lcg(container) - 1.0
        if rng.lcg(container) < math.sqrt(v * v + V * V - 2.0 * v * V * mu_t) / (v + V):
            break
    phi = 2.0 * PI * rng.lcg(container)
    s = math.sqrt(max(0.0, 1.0 - mu_t * mu_t))
    Vx, Vy, Vz = V * s * math.cos(phi), V * s * math.sin(phi), V * mu_t
    # COM velocity and the neutron's COM speed
    cx, cy, cz = A * Vx / (A + 1.0), A * Vy / (A + 1.0), (v + A * Vz) / (A + 1.0)
    w = math.sqrt(cx * cx + cy * cy + (v - cz) ** 2)
    # Isotropic COM direction
    mu = 2.0 * rng.lcg(container) - 1.0
    phi = 2.0 * PI * rng.lcg(container)
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    ox = cx + w * s * math.cos(phi)
    oy = cy + w * s * math.sin(phi)
    oz = cz + w * mu
    E_out = ox * ox + oy * oy + oz * oz
    return E_out, oz / math.sqrt(E_out)


def elastic_isotropic_below(reaction, E_limit, simulation, data, tolerance=1e-4):
    """
    True if the elastic COM angular density that the sampler uses at E <= E_limit is
    isotropic within `tolerance` (max |p(mu) - 1/2|). The tables at or below E_limit
    must be isotropic; the next table enters only through linear interpolation, so its
    deviation is weighted by the largest interpolation fraction it reaches below
    E_limit. (C-12: isotropic up to 25.3 meV, next table at 2 keV with |p - 1/2| =
    5.6e-4, so the density sampled below 10.1 eV deviates by 3e-6.)
    """
    elastic = simulation["neutron_elastic_scattering_reactions"][reaction["sub_ID"]]
    distribution = simulation["distributions"][elastic["mu_table_ID"]]
    multi_table = simulation["multi_table_distributions"][distribution["sub_ID"]]
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)
    last = min(len(grid) - 1, int(np.searchsorted(grid, E_limit, side="right")))
    for idx in range(last + 1):
        ID = mcdc_get.multi_table_distribution.table_IDs(idx, multi_table, data)
        table = simulation["tabulated_distributions"][
            simulation["distributions"][ID]["sub_ID"]
        ]
        pdf_table = simulation["table_data"][
            simulation["data"][table["pdf_ID"]]["sub_ID"]
        ]
        x = mcdc_get.table_data.x_all(pdf_table, data)
        y = mcdc_get.table_data.y_all(pdf_table, data)
        if abs(x[0] + 1.0) > tolerance or abs(x[-1] - 1.0) > tolerance:
            return False
        weight = 1.0
        if grid[idx] > E_limit and idx > 0:
            weight = (E_limit - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
        if weight * np.max(np.abs(y - 0.5)) > tolerance:
            return False
    return True


# ======================================================================================
# Continuous kernels
# ======================================================================================


@njit(cache=True)
def _emission_density(
    E_in, E_out, mu_lab, A, frame, angle_type, mu_ID, spectrum, simulation, data
):
    """Lab density of one emitted neutron through one spectrum (no weight)."""
    if frame == REFERENCE_FRAME_COM:
        E_x, mu_x = lab_to_com(E_in, E_out, mu_lab, A)
        if E_x <= 0.0 or mu_x < -1.0 or mu_x > 1.0:
            return 0.0
        jacobian = com_to_lab_jacobian(E_out, E_x)
    else:
        E_x, mu_x = E_out, mu_lab
        jacobian = 1.0

    if angle_type == ANGLE_ENERGY_CORRELATED:
        return jacobian * evaluate_correlated_distribution(
            E_in, E_x, mu_x, spectrum, simulation, data
        )

    p_E = evaluate_distribution(E_in, E_x, spectrum, simulation, data, True)
    if p_E == 0.0:
        return 0.0
    if angle_type == ANGLE_ISOTROPIC:
        p_mu = 0.5
    else:
        p_mu = evaluate_distribution(
            E_in, mu_x, simulation["distributions"][mu_ID], simulation, data, False
        )
    return jacobian * p_E * p_mu


@njit(cache=True)
def continuous_lab_density(E_in, E_out, mu_lab, reaction, nuclide, simulation, data):
    """
    Emitted-neutron density f_L(E_out, mu_lab | E_in) of a continuous reaction, or of
    elastic scattering in the free-gas range.
    """
    A = nuclide["atomic_weight_ratio"]
    frame = reaction["reference_frame"]

    if reaction["sub_type"] == NEUTRON_REACTION_ELASTIC_SCATTERING:
        kT = BOLTZMANN_K * nuclide["temperature"]
        return free_gas_density(E_in, E_out, mu_lab, A, kT)

    if reaction["sub_type"] == NEUTRON_REACTION_FISSION:
        fission = simulation["neutron_fission_reactions"][reaction["sub_ID"]]
        spectrum = simulation["distributions"][fission["spectrum_ID"]]
        return fission_yield(E_in, nuclide, simulation, data) * _emission_density(
            E_in,
            E_out,
            mu_lab,
            A,
            frame,
            fission["angle_type"],
            fission["mu_ID"],
            spectrum,
            simulation,
            data,
        )

    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    density = 0.0
    for n in range(inelastic["N_spectrum"]):
        weight = _spectrum_weight(E_in, inelastic, n, simulation, data)
        if weight == 0.0:
            continue
        spectrum = _inelastic_spectrum(inelastic, n, simulation, data)
        density += weight * _emission_density(
            E_in,
            E_out,
            mu_lab,
            A,
            frame,
            inelastic["angle_type"],
            inelastic["mu_ID"],
            spectrum,
            simulation,
            data,
        )
    return density


# ======================================================================================
# Delta (line) kernels: elastic and level scattering
# ======================================================================================


@njit(cache=True)
def delta_lab_line(E_in, E_out, reaction, nuclide, simulation, data):
    """
    Line kernel (g, mu_lab*): emitted density in E_out, and the lab cosine it implies.

    g(E_out | E_in) = yield * p(mu_cm*) |d mu_cm / d E_out|, with p(mu_cm) the COM
    angular density (real tabulated data for elastic scattering).
    """
    A = nuclide["atomic_weight_ratio"]

    if reaction["sub_type"] == NEUTRON_REACTION_ELASTIC_SCATTERING:
        mu_cm = elastic_mu_cm(E_in, E_out, A)
        if mu_cm < -1.0 or mu_cm > 1.0:
            return 0.0, 0.0
        elastic = simulation["neutron_elastic_scattering_reactions"][reaction["sub_ID"]]
        mu_distribution = simulation["distributions"][elastic["mu_table_ID"]]
        p_mu = evaluate_distribution(
            E_in, mu_cm, mu_distribution, simulation, data, False
        )
        g = p_mu * elastic_dmu_cm_dE_out(E_in, E_out, A)
        return g, elastic_mu_lab(mu_cm, A)

    # Level scattering
    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    spectrum = _inelastic_spectrum(inelastic, 0, simulation, data)
    level = simulation["level_scattering_distributions"][spectrum["sub_ID"]]
    E_cm = level["C2"] * (E_in - level["C1"])
    if E_cm <= 0.0:
        return 0.0, 0.0
    mu_cm = level_mu_cm(E_in, E_out, E_cm, A)
    if mu_cm < -1.0 or mu_cm > 1.0:
        return 0.0, 0.0
    if inelastic["angle_type"] == ANGLE_ISOTROPIC:
        p_mu = 0.5
    else:
        mu_distribution = simulation["distributions"][inelastic["mu_ID"]]
        p_mu = evaluate_distribution(
            E_in, mu_cm, mu_distribution, simulation, data, False
        )
    g = (
        inelastic_yield(E_in, inelastic, simulation, data)
        * p_mu
        * level_dmu_cm_dE_out(E_in, E_cm, A)
    )
    mu_lab = mu_cm * math.sqrt(E_cm / E_out) + math.sqrt(E_in / E_out) / (A + 1)
    return g, mu_lab


# ======================================================================================
# Kinematic support
# ======================================================================================


@njit(cache=True)
def lab_energy_support(E_in, reaction, nuclide, simulation, data):
    """
    Closed-form lab E_out range [E_low, E_high] reachable at E_in (empty: E_low > E_high).
    Used to skip impossible (E_in, E_out) combinations before any density evaluation.
    """
    A = nuclide["atomic_weight_ratio"]
    reaction_type = reaction["sub_type"]
    frame = reaction["reference_frame"]

    if reaction_type == NEUTRON_REACTION_ELASTIC_SCATTERING:
        if is_free_gas(E_in, reaction, nuclide):
            return free_gas_support(E_in, A, BOLTZMANN_K * nuclide["temperature"])
        return elastic_E_out_range(E_in, A)

    if reaction_type == NEUTRON_REACTION_FISSION:
        fission = simulation["neutron_fission_reactions"][reaction["sub_ID"]]
        spectrum = simulation["distributions"][fission["spectrum_ID"]]
        low, high = _spectrum_support(
            E_in, fission["angle_type"], spectrum, simulation, data
        )
        if frame == REFERENCE_FRAME_COM:
            return level_E_out_range(E_in, low, high, A)
        return low, high

    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    E_low = math.inf
    E_high = -math.inf
    for n in range(inelastic["N_spectrum"]):
        spectrum = _inelastic_spectrum(inelastic, n, simulation, data)
        if spectrum["sub_type"] == DISTRIBUTION_LEVEL_SCATTERING:
            level = simulation["level_scattering_distributions"][spectrum["sub_ID"]]
            E_cm = level["C2"] * (E_in - level["C1"])
            if E_cm <= 0.0:
                continue
            low, high = E_cm, E_cm
        else:
            low, high = _spectrum_support(
                E_in, inelastic["angle_type"], spectrum, simulation, data
            )
        if frame == REFERENCE_FRAME_COM:
            low, high = level_E_out_range(E_in, low, high, A)
        E_low = min(E_low, low)
        E_high = max(E_high, high)
    return E_low, E_high


@njit(cache=True)
def _spectrum_support(E_in, angle_type, spectrum, simulation, data):
    if angle_type == ANGLE_ENERGY_CORRELATED:
        return correlated_distribution_support(E_in, spectrum, simulation, data)
    return distribution_support(E_in, spectrum, simulation, data, True)


# ======================================================================================
# Emission kernel about the slab axis, integrated over an incident polar bin
# ======================================================================================

# Gauss-Kronrod 15 / Gauss 7 on [-1, 1] (as in mcdc.rmc.kernel)
_XK = (
    -0.991455371120812639206854697526329,
    -0.949107912342758524526189684047851,
    -0.864864423359769072789712788640926,
    -0.741531185599394439863864773280788,
    -0.586087235467691130294144845693013,
    -0.405845151377397166906606412076961,
    -0.207784955007898467600689403773245,
    0.0,
    0.207784955007898467600689403773245,
    0.405845151377397166906606412076961,
    0.586087235467691130294144845693013,
    0.741531185599394439863864773280788,
    0.864864423359769072789712788640926,
    0.949107912342758524526189684047851,
    0.991455371120812639206854697526329,
)
_WK = (
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
    0.204432940075298892414161999234649,
    0.190350578064785409913256402421014,
    0.169004726639267902826583426598550,
    0.140653259715525918745189590510238,
    0.104790010322250183839876322541518,
    0.063092092629978553290700663189204,
    0.022935322010529224963732008058970,
)
_WG = (
    0.0,
    0.129484966168869693270611432679082,
    0.0,
    0.279705391489276667901467771423780,
    0.0,
    0.381830050505118944950369775488975,
    0.0,
    0.417959183673469387755102040816327,
    0.0,
    0.381830050505118944950369775488975,
    0.0,
    0.279705391489276667901467771423780,
    0.0,
    0.129484966168869693270611432679082,
    0.0,
)

TAU_TOLERANCE = 1e-10
TAU_MAX_DEPTH = 40


@njit(cache=True)
def emission_kernel_bin(
    E_in, E_out, mu_out, mu_low, mu_high, reaction, ktype, nuclide, simulation, data
):
    """
    tau_bar_r: emitted neutrons per reaction per unit E_out and per unit outgoing polar
    cosine mu_out, integrated over incident polar cosines mu_in in [mu_low, mu_high]
    (azimuths integrated). The azimuthal kernel is symmetric in (mu_in, mu_out), so

        tau_bar = int dmu0 f_L(E_out, mu0 | E_in) Pi(mu_out, mu0; mu_low, mu_high),

    with Pi the analytic bin probability (bounded). Delta laws: g(E_out) Pi(mu0*).
    Continuous laws: adaptive Gauss-Kronrod over the supported mu0 range, split at the
    kinks of Pi and cosine-mapped on each piece.
    """
    low, high = lab_energy_support(E_in, reaction, nuclide, simulation, data)
    if E_out < low or E_out > high:
        return 0.0

    free_gas = is_free_gas(E_in, reaction, nuclide)
    if ktype != KERNEL_CONTINUOUS and not free_gas:
        g, mu0 = delta_lab_line(E_in, E_out, reaction, nuclide, simulation, data)
        if g == 0.0:
            return 0.0
        return g * azimuthal_bin_probability(mu_out, mu0, mu_low, mu_high)

    # Lab-cosine support: for COM laws, E_cm = E_out + s_A^2 - 2 mu0 s_A sqrt(E_out)
    #   (s_A = sqrt(E_in)/(A+1)) must lie in the spectra's E_cm range
    mu0_low, mu0_high = -1.0, 1.0
    if reaction["reference_frame"] == REFERENCE_FRAME_COM and not free_gas:
        E_cm_low, E_cm_high = _com_energy_range(E_in, reaction, simulation, data)
        s_A = math.sqrt(E_in) / (nuclide["atomic_weight_ratio"] + 1)
        denominator = 2.0 * s_A * math.sqrt(E_out)
        mu0_low = max(mu0_low, (E_out + s_A * s_A - E_cm_high) / denominator)
        mu0_high = min(mu0_high, (E_out + s_A * s_A - E_cm_low) / denominator)
    if mu0_high <= mu0_low:
        return 0.0

    # Pieces: kinks of Pi in mu0 at b mu_out +- sqrt(1 - b^2) sqrt(1 - mu_out^2)
    points = np.empty(8)
    points[0] = mu0_low
    points[1] = mu0_high
    N = 2
    if free_gas:
        # The free-gas kernel peaks toward mu0 = 1 near E_out = E_in
        for p0 in (1.0 - 1e-2, 1.0 - 1e-4):
            if mu0_low < p0 < mu0_high:
                points[N] = p0
                N += 1
    s_out = math.sqrt(max(0.0, 1.0 - mu_out * mu_out))
    for edge in (mu_low, mu_high):
        s = math.sqrt(max(0.0, 1.0 - edge * edge)) * s_out
        for kink in (edge * mu_out - s, edge * mu_out + s):
            if mu0_low < kink < mu0_high:
                points[N] = kink
                N += 1
    points = np.sort(points[:N])

    total = 0.0
    stack_a = np.empty(TAU_MAX_DEPTH + 2)
    stack_b = np.empty(TAU_MAX_DEPTH + 2)
    stack_d = np.empty(TAU_MAX_DEPTH + 2, dtype=np.int64)
    for p in range(N - 1):
        c = points[p]
        d = points[p + 1]
        if d <= c:
            continue
        # Adaptive G7-K15 in t on [0, 1], mu0 = c + (d - c)(1 - cos(pi t)) / 2
        stack_a[0] = 0.0
        stack_b[0] = 1.0
        stack_d[0] = 0
        top = 1
        while top > 0:
            top -= 1
            a = stack_a[top]
            b = stack_b[top]
            depth = stack_d[top]
            half = 0.5 * (b - a)
            center = 0.5 * (b + a)
            K = 0.0
            Gs = 0.0
            for q in range(15):
                t = center + half * _XK[q]
                mu0 = c + 0.5 * (d - c) * (1.0 - math.cos(math.pi * t))
                dmu0 = 0.5 * (d - c) * math.pi * math.sin(math.pi * t)
                f = continuous_lab_density(
                    E_in, E_out, mu0, reaction, nuclide, simulation, data
                )
                if f != 0.0:
                    f *= dmu0 * azimuthal_bin_probability(mu_out, mu0, mu_low, mu_high)
                K += half * _WK[q] * f
                Gs += half * _WG[q] * f
            if (
                (K != 0.0 and abs(K - Gs) <= TAU_TOLERANCE * abs(K))
                or depth >= TAU_MAX_DEPTH
                or (K == 0.0 and depth >= 3)
            ):
                total += K
            else:
                stack_a[top] = a
                stack_b[top] = center
                stack_d[top] = depth + 1
                stack_a[top + 1] = center
                stack_b[top + 1] = b
                stack_d[top + 1] = depth + 1
                top += 2
    return total


@njit(cache=True)
def _com_energy_range(E_in, reaction, simulation, data):
    """Union of the COM outgoing-energy ranges of a continuous reaction's spectra."""
    if reaction["sub_type"] == NEUTRON_REACTION_FISSION:
        fission = simulation["neutron_fission_reactions"][reaction["sub_ID"]]
        spectrum = simulation["distributions"][fission["spectrum_ID"]]
        return _spectrum_support(
            E_in, fission["angle_type"], spectrum, simulation, data
        )
    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    low = math.inf
    high = -math.inf
    for n in range(inelastic["N_spectrum"]):
        spectrum = _inelastic_spectrum(inelastic, n, simulation, data)
        lo, hi = _spectrum_support(
            E_in, inelastic["angle_type"], spectrum, simulation, data
        )
        low = min(low, lo)
        high = max(high, hi)
    return low, high
