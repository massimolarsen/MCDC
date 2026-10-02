"""
Evaluator-vs-sampler consistency: the density returned by each RMC evaluator must be
the density MC/DC's sampler actually produces (chi-square on histograms of samples).
"""

import numpy as np
import pytest

import mcdc.mcdc_get as mcdc_get
from mcdc.constant import (
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
)
from mcdc.rmc.evaluate import (
    correlated_distribution_support,
    distribution_support,
    evaluate_correlated_distribution,
    evaluate_distribution,
)
from mcdc.transport.distribution import (
    sample_correlated_distribution_with_scale,
    sample_distribution,
    sample_distribution_with_scale,
)

from .conftest import (
    numba_only,
    bin_integral_1d,
    bin_integral_2d,
    chi_square_pvalue,
    inelastic_reactions,
    interior_energy,
    requires_nuclide,
    rng_container,
    spectrum_distribution,
)

N_SAMPLE = 20000
P_MIN = 1.0e-4

pytestmark = numba_only


def elastic_mu_distribution(simulation, data, nuclide):
    ID = mcdc_get.nuclide.neutron_elastic_scattering_reaction_IDs(0, nuclide, data)
    base = simulation["neutron_reactions"][ID]
    elastic = simulation["neutron_elastic_scattering_reactions"][base["sub_ID"]]
    return simulation["distributions"][elastic["mu_table_ID"]]


def fission_spectrum(simulation, data, nuclide):
    ID = mcdc_get.nuclide.neutron_fission_reaction_IDs(0, nuclide, data)
    base = simulation["neutron_reactions"][ID]
    fission = simulation["neutron_fission_reactions"][base["sub_ID"]]
    return simulation["distributions"][fission["spectrum_ID"]]


def find_correlated(simulation, data, nuclide, sub_type):
    for base, inelastic in inelastic_reactions(simulation, data, nuclide):
        for n in range(inelastic["N_spectrum"]):
            spectrum = spectrum_distribution(simulation, data, inelastic, n)
            if spectrum["sub_type"] == sub_type:
                return spectrum
    raise AssertionError("no such spectrum")


def correlated_grid(simulation, data, spectrum):
    if spectrum["sub_type"] == DISTRIBUTION_KALBACH_MANN:
        km = simulation["kalbach_mann_distributions"][spectrum["sub_ID"]]
        return mcdc_get.kalbach_mann_distribution.energy_all(km, data)
    table = simulation["tabulated_energy_angle_distributions"][spectrum["sub_ID"]]
    return mcdc_get.tabulated_energy_angle_distribution.energy_all(table, data)


# ======================================================================================
# Multi-table
# ======================================================================================


@requires_nuclide("O16")
def test_multi_table_unscaled_elastic_angle(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("O16")
    distribution = elastic_mu_distribution(simulation, data, nuclide)
    E = 3.7e6  # anisotropic COM elastic, between tabulated energies

    rng = rng_container(11)
    samples = [
        sample_distribution(E, distribution, rng, simulation, data)
        for _ in range(N_SAMPLE)
    ]
    edges = np.linspace(-1.0, 1.0, 21)
    observed, _ = np.histogram(samples, edges)
    probability = bin_integral_1d(
        lambda x: evaluate_distribution(E, x, distribution, simulation, data, False),
        edges,
    )
    assert probability.sum() == pytest.approx(1.0, abs=1e-6)
    assert chi_square_pvalue(observed, probability) > P_MIN


@requires_nuclide("U235")
def test_multi_table_scaled_fission_spectrum(nuclide_simulation):
    simulation, data, nuclide = nuclide_simulation("U235")
    distribution = fission_spectrum(simulation, data, nuclide)
    E = 2.3e6

    rng = rng_container(12)
    samples = [
        sample_distribution_with_scale(E, distribution, rng, simulation, data)
        for _ in range(N_SAMPLE)
    ]
    low, high = distribution_support(E, distribution, simulation, data, True)
    assert low <= min(samples) and max(samples) <= high

    # Bins in sqrt(E) to resolve the low-energy peak
    edges = np.linspace(np.sqrt(low), np.sqrt(min(high, 2.0e7)), 31) ** 2
    observed, _ = np.histogram(samples, edges)
    probability = bin_integral_1d(
        lambda x: evaluate_distribution(E, x, distribution, simulation, data, True),
        edges,
    )
    assert chi_square_pvalue(observed, probability) > P_MIN


# ======================================================================================
# Correlated energy-angle: Kalbach-Mann (O-16) and tabulated energy-angle (U-235)
# ======================================================================================


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
@pytest.mark.parametrize("interval", ["middle", "last"])
def test_correlated_energy_angle(nuclide_simulation, name, sub_type, interval):
    simulation, data, nuclide = nuclide_simulation(name)
    spectrum = find_correlated(simulation, data, nuclide, sub_type)
    grid = correlated_grid(simulation, data, spectrum)
    idx = len(grid) // 2 if interval == "middle" else len(grid) - 2
    E = interior_energy(grid, idx)

    rng = rng_container(13)
    samples = np.array(
        [
            sample_correlated_distribution_with_scale(
                E, spectrum, rng, simulation, data
            )
            for _ in range(N_SAMPLE)
        ]
    )
    E_low, E_high = correlated_distribution_support(E, spectrum, simulation, data)
    assert E_low <= samples[:, 0].min() and samples[:, 0].max() <= E_high

    edges_E = np.linspace(E_low, E_high, 9)
    edges_mu = np.linspace(-1.0, 1.0, 6)
    observed, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], [edges_E, edges_mu])
    probability = bin_integral_2d(
        lambda e, mu: evaluate_correlated_distribution(
            E, e, mu, spectrum, simulation, data
        ),
        edges_E,
        edges_mu,
    )
    assert probability.sum() == pytest.approx(1.0, abs=2e-3)
    assert chi_square_pvalue(observed, probability) > P_MIN
