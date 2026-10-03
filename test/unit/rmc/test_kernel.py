"""
Transfer moments M[g', j', g, j]: conservation, and agreement with a direct Monte Carlo
estimate built from MC/DC's own reaction samplers.
"""

import math

import numpy as np
import pytest

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
from mcdc.constant import (
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
    NEUTRON_REACTION_ELASTIC_SCATTERING,
    NEUTRON_REACTION_FISSION,
    PARTICLE_NEUTRON,
)
from mcdc.rmc.angular import build_angular_transfer_table
from mcdc.rmc.kernel import reaction_transfer_row
from mcdc.rmc.reaction import KERNEL_LEVEL, kernel_type
from mcdc.transport.physics.neutron.native import (
    reaction_micro_xs,
    sample_elastic_scattering,
    sample_fission,
    sample_inelastic_scattering,
)

from .conftest import (
    hashed_seed,
    inelastic_reactions,
    numba_only,
    requires_nuclide,
    spectrum_distribution,
)

pytestmark = numba_only

MU_EDGES = np.array([-1.0, 0.0, 1.0])
TABLE = build_angular_transfer_table(MU_EDGES, 2049)


def transfer_row(reaction, nuclide, simulation, data, E_edges, gp, tol=1e-8):
    G = len(E_edges) - 1
    J = len(MU_EDGES) - 1
    row = np.zeros((G, J, J))
    reaction_transfer_row(
        E_edges[gp],
        E_edges[gp + 1],
        reaction,
        nuclide,
        simulation,
        data,
        E_edges,
        MU_EDGES,
        TABLE,
        tol,
        row,
    )
    return row


def exact_xs_integral(reaction, nuclide, data, E_a, E_b):
    """int sigma_r dE over [E_a, E_b] (exact: lin-lin on the xs grid)."""
    grid = mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data)
    x = np.concatenate(([E_a], grid[(grid > E_a) & (grid < E_b)], [E_b]))
    y = [reaction_micro_xs(e, reaction, nuclide, data) for e in x]
    return np.trapezoid(y, x)


@requires_nuclide("O16")
def test_elastic_row_conservation(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    ID = mcdc_get.nuclide.neutron_elastic_scattering_reaction_IDs(0, nuclide, data)
    reaction = simulation["neutron_reactions"][ID]
    E_edges = np.logspace(5.0, 7.0, 9)
    for gp in (4, 7):  # all emission stays inside the grid (alpha = 0.78)
        row = transfer_row(reaction, nuclide, simulation, data, E_edges, gp)
        xs_integral = exact_xs_integral(
            reaction, nuclide, data, E_edges[gp], E_edges[gp + 1]
        )
        for jp in range(2):
            dmu = MU_EDGES[jp + 1] - MU_EDGES[jp]
            assert row[:, :, jp].sum() == pytest.approx(dmu * xs_integral, rel=1e-7)
        # Elastic never upscatters (target at rest)
        assert np.all(row[gp + 1 :] == 0.0)


# ======================================================================================
# Direct Monte Carlo estimate of a row with the MC/DC samplers
# ======================================================================================


def monte_carlo_row(sampler, reaction, nuclide, simulation, data, E_edges, gp, jp, N):
    G = len(E_edges) - 1
    J = len(MU_EDGES) - 1
    tally = np.zeros((G, J))
    tally_square = np.zeros((G, J))
    E_a, E_b = E_edges[gp], E_edges[gp + 1]
    mu_a, mu_b = MU_EDGES[jp], MU_EDGES[jp + 1]
    generator = np.random.default_rng(17)
    for seed in range(1, N + 1):
        E_in = E_a + (E_b - E_a) * generator.random()
        mu_in = mu_a + (mu_b - mu_a) * generator.random()
        phi = 2.0 * math.pi * generator.random()
        weight = reaction_micro_xs(E_in, reaction, nuclide, data)
        weight *= (E_b - E_a) * (mu_b - mu_a)
        if weight == 0.0:
            continue

        particle_bank_module.set_bank_size(simulation["bank_active"], 0)
        particles = np.zeros(1, type_.particle)
        s = math.sqrt(1.0 - mu_in * mu_in)
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
        emitted = list(simulation["bank_active"]["particle_data"][:size])
        if particles[0]["alive"]:
            emitted.append(particles[0])
        score = np.zeros((G, J))
        for P in emitted:
            g = np.searchsorted(E_edges, P["E"], side="right") - 1
            j = min(np.searchsorted(MU_EDGES, P["uz"], side="right") - 1, J - 1)
            if 0 <= g < G:
                score[g, j] += weight
        tally += score
        tally_square += score**2
    mean = tally / N
    std = np.sqrt(np.maximum(tally_square / N - mean**2, 0.0) / N)
    return mean, std


def check_row_against_monte_carlo(
    sampler, reaction, nuclide, simulation, data, E_edges, gp
):
    row = transfer_row(reaction, nuclide, simulation, data, E_edges, gp)
    for jp in range(2):
        mean, std = monte_carlo_row(
            sampler, reaction, nuclide, simulation, data, E_edges, gp, jp, 20000
        )
        expected = row[:, :, jp]
        assert expected.sum() > 0.0
        assert expected.sum() == pytest.approx(mean.sum(), rel=0.03)
        resolved = std > 0.0
        z = np.abs(mean - expected)[resolved] / std[resolved]
        assert np.all(z < 5.0), (z.max(), mean, expected)


def find_inelastic(simulation, data, nuclide, predicate):
    for base, inelastic in inelastic_reactions(simulation, data, nuclide):
        if predicate(base, inelastic):
            return base
    raise AssertionError("no matching reaction")


@requires_nuclide("O16")
def test_elastic_row_monte_carlo(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    ID = mcdc_get.nuclide.neutron_elastic_scattering_reaction_IDs(0, nuclide, data)
    reaction = simulation["neutron_reactions"][ID]
    E_edges = np.logspace(6.0, 7.0, 11)
    check_row_against_monte_carlo(
        sample_elastic_scattering, reaction, nuclide, simulation, data, E_edges, 6
    )


@requires_nuclide("O16")
def test_level_row_monte_carlo(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    reaction = find_inelastic(
        simulation,
        data,
        nuclide,
        lambda base, inelastic: kernel_type(base, simulation, data) == KERNEL_LEVEL
        and inelastic["angle_type"] != 0,
    )
    E_edges = np.logspace(5.0, 7.2, 12)
    check_row_against_monte_carlo(
        sample_inelastic_scattering, reaction, nuclide, simulation, data, E_edges, 10
    )


@pytest.mark.parametrize(
    "name, sub_type, E_low, E_high, gp",
    [
        pytest.param(
            "O16", DISTRIBUTION_KALBACH_MANN, 1e5, 2e7, 8, marks=requires_nuclide("O16")
        ),
        pytest.param(
            "U235",
            DISTRIBUTION_TABULATED_ENERGY_ANGLE,
            1e5,
            2e7,
            8,
            marks=requires_nuclide("U235"),
        ),
    ],
)
def test_correlated_row_monte_carlo(
    nuclide_simulation, name, sub_type, E_low, E_high, gp
):
    simulation, data, nuclide = nuclide_simulation(name)
    reaction = find_inelastic(
        simulation,
        data,
        nuclide,
        lambda base, inelastic: inelastic["multiplicity"] == 1
        and spectrum_distribution(simulation, data, inelastic)["sub_type"] == sub_type,
    )
    E_edges = np.logspace(np.log10(E_low), np.log10(E_high), 10)
    check_row_against_monte_carlo(
        sample_inelastic_scattering, reaction, nuclide, simulation, data, E_edges, gp
    )


@requires_nuclide("U235")
def test_fission_row_monte_carlo(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("U235")
    ID = mcdc_get.nuclide.neutron_fission_reaction_IDs(0, nuclide, data)
    reaction = simulation["neutron_reactions"][ID]
    E_edges = np.logspace(4.0, 7.3, 12)
    check_row_against_monte_carlo(
        sample_fission, reaction, nuclide, simulation, data, E_edges, 8
    )
