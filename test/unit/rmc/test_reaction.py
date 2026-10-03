"""
Reaction-level kernels vs the MC/DC reaction samplers: emitted-neutron histograms in
lab (E_out, mu_lab) must match the RMC kernels, including frame transformation,
Jacobian, and yield.
"""

import numpy as np
import pytest

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle_bank as particle_bank_module
from mcdc.constant import (
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
    PARTICLE_NEUTRON,
)
from mcdc.rmc.reaction import (
    KERNEL_CONTINUOUS,
    KERNEL_ELASTIC,
    KERNEL_LEVEL,
    continuous_lab_density,
    delta_lab_line,
    emission_kernel_bin,
    fission_yield,
    inelastic_yield,
    kernel_type,
    lab_energy_support,
)
from mcdc.transport.physics.neutron.native import (
    sample_elastic_scattering,
    sample_fission,
    sample_inelastic_scattering,
)

from .conftest import (
    hashed_seed,
    numba_only,
    bin_integral_1d,
    interior_energy,
    bin_integral_2d,
    chi_square_pvalue,
    inelastic_reactions,
    requires_nuclide,
    spectrum_distribution,
)

N_COLLISION = 20000
P_MIN = 1.0e-4

pytestmark = numba_only


def sample_emissions(sampler, reaction, nuclide, simulation, data, E_in):
    """Lab (E_out, mu_lab) of every neutron emitted by N_COLLISION reactions."""
    emissions = []
    for seed in range(1, N_COLLISION + 1):
        particle_bank_module.set_bank_size(simulation["bank_active"], 0)
        particles = np.zeros(1, type_.particle)
        particles[0]["E"] = E_in
        particles[0]["w"] = 1.0
        particles[0]["alive"] = True
        particles[0]["uz"] = 1.0
        particles[0]["particle_type"] = PARTICLE_NEUTRON
        particles[0]["rng_seed"] = hashed_seed(seed)
        collision_data = np.zeros(1, type_.collision_data)
        sampler(reaction, particles, collision_data, nuclide, simulation, data)

        size = particle_bank_module.get_bank_size(simulation["bank_active"])
        for P in simulation["bank_active"]["particle_data"][:size]:
            emissions.append((P["E"], P["uz"]))
        if particles[0]["alive"]:
            emissions.append((particles[0]["E"], particles[0]["uz"]))
    return np.array(emissions)


def correlated_grid(simulation, data, spectrum):
    if spectrum["sub_type"] == DISTRIBUTION_KALBACH_MANN:
        km = simulation["kalbach_mann_distributions"][spectrum["sub_ID"]]
        return mcdc_get.kalbach_mann_distribution.energy_all(km, data)
    table = simulation["tabulated_energy_angle_distributions"][spectrum["sub_ID"]]
    return mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)


def base_reaction(simulation, data, nuclide, kind, index=0):
    getter = {
        "elastic": mcdc_get.nuclide.neutron_elastic_scattering_reaction_IDs,
        "fission": mcdc_get.nuclide.neutron_fission_reaction_IDs,
    }[kind]
    return simulation["neutron_reactions"][getter(index, nuclide, data)]


def find_inelastic(simulation, data, nuclide, predicate):
    for base, inelastic in inelastic_reactions(simulation, data, nuclide):
        if predicate(base, inelastic):
            return base
    raise AssertionError("no matching reaction")


def check_continuous(reaction, nuclide, simulation, data, E_in, sampler, yield_):
    emissions = sample_emissions(sampler, reaction, nuclide, simulation, data, E_in)
    E_low, E_high = lab_energy_support(E_in, reaction, nuclide, simulation, data)
    assert E_low <= emissions[:, 0].min() and emissions[:, 0].max() <= E_high

    edges_E = np.linspace(E_low, E_high, 9)
    edges_mu = np.linspace(-1.0, 1.0, 6)
    observed, _, _ = np.histogram2d(
        emissions[:, 0], emissions[:, 1], [edges_E, edges_mu]
    )
    integral = bin_integral_2d(
        lambda e, mu: continuous_lab_density(
            E_in, e, mu, reaction, nuclide, simulation, data
        ),
        edges_E,
        edges_mu,
        N_panel_x=48,
        N_panel_y=8,
    )
    assert integral.sum() == pytest.approx(yield_, rel=5e-3)
    assert len(emissions) / N_COLLISION == pytest.approx(yield_, rel=0.03)
    assert chi_square_pvalue(observed, integral / integral.sum()) > P_MIN


def check_line(reaction, nuclide, simulation, data, E_in, sampler, yield_):
    emissions = sample_emissions(sampler, reaction, nuclide, simulation, data, E_in)
    E_low, E_high = lab_energy_support(E_in, reaction, nuclide, simulation, data)
    assert E_low <= emissions[:, 0].min() and emissions[:, 0].max() <= E_high

    # mu_lab is a deterministic function of E_out
    for E_out, mu_lab in emissions[:200]:
        _, mu_star = delta_lab_line(E_in, E_out, reaction, nuclide, simulation, data)
        assert mu_lab == pytest.approx(mu_star, abs=1e-8)

    edges = np.linspace(E_low, E_high, 21)
    observed, _ = np.histogram(emissions[:, 0], edges)
    integral = bin_integral_1d(
        lambda e: delta_lab_line(E_in, e, reaction, nuclide, simulation, data)[0],
        edges,
    )
    assert integral.sum() == pytest.approx(yield_, rel=1e-4)
    assert chi_square_pvalue(observed, integral / integral.sum()) > P_MIN


@requires_nuclide("O16")
def test_elastic_line(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    reaction = base_reaction(simulation, data, nuclide, "elastic")
    assert kernel_type(reaction, simulation, data) == KERNEL_ELASTIC
    check_line(
        reaction, nuclide, simulation, data, 3.7e6, sample_elastic_scattering, 1.0
    )


@requires_nuclide("O16")
def test_level_scattering_line(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    reaction = find_inelastic(
        simulation,
        data,
        nuclide,
        lambda base, inelastic: kernel_type(base, simulation, data) == KERNEL_LEVEL
        and inelastic["angle_type"] != 0,
    )
    check_line(
        reaction, nuclide, simulation, data, 9.0e6, sample_inelastic_scattering, 1.0
    )


@pytest.mark.parametrize(
    "name, sub_type",
    [
        pytest.param("O16", DISTRIBUTION_KALBACH_MANN, marks=requires_nuclide("O16")),
        pytest.param(
            "U235",
            DISTRIBUTION_TABULATED_ENERGY_ANGLE,
            marks=requires_nuclide("U235"),
        ),
    ],
)
def test_correlated_inelastic(nuclide_simulation, name, sub_type):
    simulation, data, nuclide = nuclide_simulation(name)
    reaction = find_inelastic(
        simulation,
        data,
        nuclide,
        lambda base, inelastic: inelastic["multiplicity"] == 1
        and spectrum_distribution(simulation, data, inelastic)["sub_type"] == sub_type,
    )
    # Incident energy inside the spectrum's own incident grid (middle interval)
    spectrum = spectrum_distribution(
        simulation,
        data,
        simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]],
    )
    E_in = interior_energy(correlated_grid(simulation, data, spectrum), None)
    assert kernel_type(reaction, simulation, data) == KERNEL_CONTINUOUS
    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    check_continuous(
        reaction,
        nuclide,
        simulation,
        data,
        E_in,
        sample_inelastic_scattering,
        float(inelastic["multiplicity"]),
    )


@requires_nuclide("U235")
def test_fission(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("U235")
    reaction = base_reaction(simulation, data, nuclide, "fission")
    E_in = 2.3e6
    check_continuous(
        reaction,
        nuclide,
        simulation,
        data,
        E_in,
        sample_fission,
        fission_yield(E_in, nuclide, simulation, data),
    )


@requires_nuclide("U235")
def test_tabulated_yield(tabulated_yield_simulation):
    simulation, data, nuclide = tabulated_yield_simulation
    reaction = find_inelastic(
        simulation, data, nuclide, lambda base, inelastic: base["MT"] == 5
    )
    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    assert inelastic["multiplicity_tabulated"]
    E_in = 14.0e6
    yield_ = inelastic_yield(E_in, inelastic, simulation, data)
    assert yield_ == pytest.approx(0.3 + 14.0 / 30.0, rel=1e-6)
    check_continuous(
        reaction,
        nuclide,
        simulation,
        data,
        E_in,
        sample_inelastic_scattering,
        yield_,
    )


# ======================================================================================
# Pointwise emission kernel tau about the slab axis
# ======================================================================================


def sample_emissions_polar(
    sampler, reaction, nuclide, simulation, data, E_in, mu_low, mu_high, N
):
    """Lab (E_out, u_z) of emitted neutrons for incident mu_in ~ U[mu_low, mu_high]."""
    import math

    emissions = []
    generator = np.random.default_rng(5)
    for seed in range(1, N + 1):
        mu_in = mu_low + (mu_high - mu_low) * generator.random()
        s = math.sqrt(1.0 - mu_in * mu_in)
        phi = 2.0 * math.pi * generator.random()
        particle_bank_module.set_bank_size(simulation["bank_active"], 0)
        particles = np.zeros(1, type_.particle)
        particles[0]["E"] = E_in
        particles[0]["w"] = 1.0
        particles[0]["alive"] = True
        particles[0]["ux"] = s * math.cos(phi)
        particles[0]["uy"] = s * math.sin(phi)
        particles[0]["uz"] = mu_in
        particles[0]["particle_type"] = PARTICLE_NEUTRON
        particles[0]["rng_seed"] = hashed_seed(seed)
        collision_data = np.zeros(1, type_.collision_data)
        sampler(reaction, particles, collision_data, nuclide, simulation, data)
        size = particle_bank_module.get_bank_size(simulation["bank_active"])
        for P in simulation["bank_active"]["particle_data"][:size]:
            emissions.append((P["E"], P["uz"]))
        if particles[0]["alive"]:
            emissions.append((particles[0]["E"], particles[0]["uz"]))
    return np.array(emissions)


def check_emission_kernel(sampler, reaction, nuclide, simulation, data, E_in, yield_):
    """tau_bar_{j'} / dmu_j' is the emission density for mu_in uniform in bin j'."""
    ktype = kernel_type(reaction, simulation, data)
    mu_low, mu_high = 0.0, 1.0
    emissions = sample_emissions_polar(
        sampler, reaction, nuclide, simulation, data, E_in, mu_low, mu_high, N_COLLISION
    )
    E_low, E_high = lab_energy_support(E_in, reaction, nuclide, simulation, data)
    edges_E = np.linspace(E_low, E_high, 7)
    edges_mu = np.linspace(-1.0, 1.0, 5)
    observed, _, _ = np.histogram2d(
        emissions[:, 0], emissions[:, 1], [edges_E, edges_mu]
    )
    integral = bin_integral_2d(
        lambda e, mu: emission_kernel_bin(
            E_in, e, mu, mu_low, mu_high, reaction, ktype, nuclide, simulation, data
        )
        / (mu_high - mu_low),
        edges_E,
        edges_mu,
        N_panel_x=12,
        N_panel_y=6,
        N_point=4,
    )
    assert integral.sum() == pytest.approx(yield_, rel=1e-2)
    assert chi_square_pvalue(observed, integral / integral.sum()) > P_MIN


@requires_nuclide("O16")
def test_emission_kernel_elastic(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    reaction = base_reaction(simulation, data, nuclide, "elastic")
    check_emission_kernel(
        sample_elastic_scattering, reaction, nuclide, simulation, data, 3.7e6, 1.0
    )


@requires_nuclide("O16")
def test_emission_kernel_kalbach(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    reaction = find_inelastic(
        simulation,
        data,
        nuclide,
        lambda base, inelastic: inelastic["multiplicity"] == 1
        and spectrum_distribution(simulation, data, inelastic)["sub_type"]
        == DISTRIBUTION_KALBACH_MANN,
    )
    spectrum = spectrum_distribution(
        simulation,
        data,
        simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]],
    )
    E_in = interior_energy(correlated_grid(simulation, data, spectrum), None)
    check_emission_kernel(
        sample_inelastic_scattering,
        reaction,
        nuclide,
        simulation,
        data,
        E_in,
        1.0,
    )
