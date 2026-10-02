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
def _spectrum_weight(E_in, inelastic, n, data):
    """
    Expected number of secondaries emitted through spectrum n, mirroring
    `sample_inelastic_scattering`: one per spectrum if multiplicity == N_spectrum,
    otherwise multiplicity times the tabulated spectrum probability.
    """
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
        weight = _spectrum_weight(E_in, inelastic, n, data)
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
    g = inelastic["multiplicity"] * p_mu * level_dmu_cm_dE_out(E_in, E_cm, A)
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
