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
    DISTRIBUTION_LEVEL_SCATTERING,
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
from mcdc.transport.util import find_bin

KERNEL_CONTINUOUS = 0
KERNEL_ELASTIC = 1
KERNEL_LEVEL = 2

# ======================================================================================
# Kernel classification
# ======================================================================================


@njit
def kernel_type(reaction, simulation, data):
    """Kernel type of a base neutron reaction record."""
    reaction_type = reaction["sub_type"]
    if reaction_type == NEUTRON_REACTION_ELASTIC_SCATTERING:
        return KERNEL_ELASTIC
    if reaction_type == NEUTRON_REACTION_INELASTIC_SCATTERING:
        inelastic = simulation["neutron_inelastic_scattering_reactions"][
            reaction["sub_ID"]
        ]
        if inelastic["multiplicity_tabulated"] and inelastic["N_spectrum"] != 1:
            raise ValueError("RMC: unsupported multi-spectrum tabulated-yield reaction")
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


@njit
def _inelastic_spectrum(inelastic, n, simulation, data):
    ID = mcdc_get.neutron_inelastic_scattering_reaction.energy_spectrum_IDs(
        n, inelastic, data
    )
    return simulation["distributions"][ID]


@njit
def inelastic_yield(E_in, inelastic, simulation, data):
    """Expected secondaries per reaction: integer multiplicity or tabulated yield(E)."""
    if inelastic["multiplicity_tabulated"]:
        table = simulation["data"][inelastic["multiplicity_table_ID"]]
        return evaluate_data(E_in, table, simulation, data)
    return float(inelastic["multiplicity"])


@njit
def _spectrum_weight(E_in, inelastic, n, simulation, data):
    """
    Expected number of secondaries emitted through spectrum n, mirroring
    `sample_inelastic_scattering`: one per spectrum if multiplicity == N_spectrum,
    otherwise multiplicity times the tabulated spectrum probability. A tabulated
    yield (single spectrum only) emits yield(E) through spectrum 0.
    """
    if inelastic["multiplicity_tabulated"]:
        return inelastic_yield(E_in, inelastic, simulation, data)
    N = inelastic["multiplicity"]
    if N == inelastic["N_spectrum"]:
        return 1.0
    grid = mcdc_get.neutron_inelastic_scattering_reaction.spectrum_probability_grid_all(
        inelastic, data
    )
    idx = find_bin(E_in, grid)
    return N * mcdc_get.neutron_inelastic_scattering_reaction.spectrum_probability(
        idx, n, inelastic, data
    )


@njit
def fission_yield(E_in, nuclide, simulation, data):
    """Expected fission neutrons per fission (all prompt), as in `sample_fission`."""
    nu = neutron_fission_prompt_multiplicity(
        E_in, nuclide, simulation, data
    ) + neutron_fission_delayed_multiplicity(E_in, nuclide, simulation, data)
    return nu / simulation["k_eff"]


# ======================================================================================
# Continuous kernels
# ======================================================================================


@njit
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


@njit
def continuous_lab_density(E_in, E_out, mu_lab, reaction, nuclide, simulation, data):
    """Emitted-neutron density f_L(E_out, mu_lab | E_in) of a continuous reaction."""
    A = nuclide["atomic_weight_ratio"]
    frame = reaction["reference_frame"]

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


@njit
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


@njit
def lab_energy_support(E_in, reaction, nuclide, simulation, data):
    """
    Closed-form lab E_out range [E_low, E_high] reachable at E_in (empty: E_low > E_high).
    Used to skip impossible (E_in, E_out) combinations before any density evaluation.
    """
    A = nuclide["atomic_weight_ratio"]
    reaction_type = reaction["sub_type"]
    frame = reaction["reference_frame"]

    if reaction_type == NEUTRON_REACTION_ELASTIC_SCATTERING:
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


@njit
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


@njit
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

    if ktype != KERNEL_CONTINUOUS:
        g, mu0 = delta_lab_line(E_in, E_out, reaction, nuclide, simulation, data)
        if g == 0.0:
            return 0.0
        return g * azimuthal_bin_probability(mu_out, mu0, mu_low, mu_high)

    # Lab-cosine support: for COM laws, E_cm = E_out + s_A^2 - 2 mu0 s_A sqrt(E_out)
    #   (s_A = sqrt(E_in)/(A+1)) must lie in the spectra's E_cm range
    mu0_low, mu0_high = -1.0, 1.0
    if reaction["reference_frame"] == REFERENCE_FRAME_COM:
        E_cm_low, E_cm_high = _com_energy_range(E_in, reaction, simulation, data)
        s_A = math.sqrt(E_in) / (nuclide["atomic_weight_ratio"] + 1)
        denominator = 2.0 * s_A * math.sqrt(E_out)
        mu0_low = max(mu0_low, (E_out + s_A * s_A - E_cm_high) / denominator)
        mu0_high = min(mu0_high, (E_out + s_A * s_A - E_cm_low) / denominator)
    if mu0_high <= mu0_low:
        return 0.0

    # Pieces: kinks of Pi in mu0 at b mu_out +- sqrt(1 - b^2) sqrt(1 - mu_out^2)
    points = np.empty(6)
    points[0] = mu0_low
    points[1] = mu0_high
    N = 2
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


@njit
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
