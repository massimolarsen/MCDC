import numpy as np
import pytest

from mcdc.coupling.geant4_distribution import build_source_distribution_payload

from mcdc.constant import PARTICLE_PROTON

from ._helpers import (
    distribution_config,
    distribution_simulation_and_data,
    two_species_simulation_and_data,
)


def test_build_source_distribution_payload_uses_current_in_tally():
    simulation, data, weights = distribution_simulation_and_data()
    payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(
        payload["box_bounds_mm"], [[-10, 10], [-20, 20], [-30, 30]]
    )
    (block,) = payload["sources"]
    np.testing.assert_allclose(block["mu_edges"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(block["azi_edges"], [-np.pi, 0.0, np.pi])
    np.testing.assert_allclose(block["energy_edges_mev"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(block["weights"], weights.ravel())
    assert block["Nu"] == 2
    assert block["Nv"] == 3
    assert block["n_events"] == 12
    assert block["pdg"] == 2112
    assert payload["source_size"] == 12
    assert payload["source_total_weight"] == np.sum(weights)
    assert payload["source_tally_name"] == "source_tally"


def test_build_source_distribution_payload_recenters_source_box_for_geant4():
    simulation, data, _ = distribution_simulation_and_data()
    surface_tally = simulation["surface_crossing_tallies"][0]
    surface_tally["surface_mesh_x_min"] = 1.0
    surface_tally["surface_mesh_x_max"] = 2.0
    surface_tally["surface_mesh_y_min"] = -2.0
    surface_tally["surface_mesh_y_max"] = 2.0
    surface_tally["surface_mesh_z_min"] = 10.0
    surface_tally["surface_mesh_z_max"] = 12.0

    payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(
        payload["box_bounds_mm"], [[-5, 5], [-20, 20], [-10, 10]]
    )


def test_build_source_distribution_payload_unnormalizes_by_n_particle():
    # The tally mean is normalized by the global source count, not the local MPI
    # work size; the payload must scale it back by global N_particle.
    simulation, data, weights = distribution_simulation_and_data(
        n_particle=2000, mpi_size=4
    )
    payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(payload["sources"][0]["weights"], 2000.0 * weights.ravel())
    assert payload["source_total_weight"] == 2000.0 * np.sum(weights)


def test_build_source_distribution_payload_rejects_zero_global_n_particle():
    simulation, data, _ = distribution_simulation_and_data(n_particle=0, mpi_size=4)

    with pytest.raises(RuntimeError, match="global N_particle"):
        build_source_distribution_payload(simulation, data, distribution_config())


def test_build_source_distribution_payload_uses_structured_mcdc_tally():
    simulation, data, weights = distribution_simulation_and_data()

    payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(payload["sources"][0]["weights"], weights.ravel())
    assert payload["source_tally_name"] == "source_tally"


def test_build_source_distribution_payload_returns_none_for_zero_weight():
    simulation, data, _ = distribution_simulation_and_data(
        weights=np.zeros((2, 2, 2, 6, 2, 3))
    )

    assert (
        build_source_distribution_payload(simulation, data, distribution_config())
        is None
    )


def test_build_source_distribution_payload_warns_when_collapsing_time_bins():
    simulation, data, weights = distribution_simulation_and_data(time_bins=2)

    with pytest.warns(RuntimeWarning, match="collapses tally time bins"):
        payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(payload["sources"][0]["weights"], (2.0 * weights).ravel())


@pytest.mark.parametrize(
    "config_update, match",
    [
        ({"n_geant4_particles": 0}, "n_geant4_particles > 0"),
        ({"source_tally_name": ""}, "source_tally_name"),
    ],
)
def test_build_source_distribution_payload_rejects_invalid_config(config_update, match):
    simulation, data, _ = distribution_simulation_and_data()

    with pytest.raises(RuntimeError, match=match):
        build_source_distribution_payload(
            simulation, data, distribution_config(**config_update)
        )


@pytest.mark.parametrize(
    "simulation_kwargs, match",
    [
        ({"tally_name": "different"}, "Could not find source tally"),
        ({"scores": [99]}, "current-in"),
        ({"filter_direction": False}, "mu/azi"),
        ({"filter_energy": False}, "energy"),
        ({"sub_type": 1}, "surface-mesh"),
        ({"polar_reference": (1.0, 0.0, 0.0)}, "polar_reference"),
        ({"surface_mesh": False}, "surface_mesh"),
    ],
)
def test_build_source_distribution_payload_rejects_invalid_tally(
    simulation_kwargs, match
):
    simulation, data, _ = distribution_simulation_and_data(**simulation_kwargs)

    with pytest.raises(RuntimeError, match=match):
        build_source_distribution_payload(simulation, data, distribution_config())


def _species_config(**kwargs):
    return distribution_config(
        n_geant4_particles=0,
        source_tally_name="",
        species_sources=[
            {"particle": "proton", "tally": "p_src", "n_events": 5},
            {"particle": "neutron", "tally": "n_src", "n_events": 7},
        ],
        **kwargs,
    )


def test_build_source_distribution_payload_one_block_per_species():
    simulation, data, proton, neutron = two_species_simulation_and_data(
        neutron_weights=np.full((2, 2, 2, 6, 2, 3), 0.5)
    )

    payload = build_source_distribution_payload(simulation, data, _species_config())

    proton_block, neutron_block = payload["sources"]
    assert (proton_block["pdg"], proton_block["n_events"]) == (2212, 5)
    assert (neutron_block["pdg"], neutron_block["n_events"]) == (2112, 7)
    np.testing.assert_allclose(proton_block["weights"], proton.ravel())
    np.testing.assert_allclose(neutron_block["weights"], neutron.ravel())
    assert payload["source_size"] == 12
    assert payload["source_total_weight"] == np.sum(proton) + np.sum(neutron)
    assert payload["source_tally_name"] == "p_src,n_src"


def test_build_source_distribution_payload_drops_species_without_current():
    simulation, data, proton, _ = two_species_simulation_and_data(
        neutron_weights=np.zeros((2, 2, 2, 6, 2, 3))
    )

    payload = build_source_distribution_payload(simulation, data, _species_config())

    (block,) = payload["sources"]
    assert block["pdg"] == 2212
    assert payload["source_size"] == 5


def test_build_source_distribution_payload_requires_matching_tally_species():
    simulation, data, _, _ = two_species_simulation_and_data()
    config = _species_config()
    config.species_sources[0]["tally"] = "n_src"
    config.species_sources[1]["tally"] = "p_src"

    with pytest.raises(RuntimeError, match="particle_type='proton'"):
        build_source_distribution_payload(simulation, data, config)


def test_build_source_distribution_payload_requires_shared_box():
    simulation, data, _, _ = two_species_simulation_and_data()
    simulation["surface_crossing_tallies"][1]["surface_mesh_x_max"] = 5.0

    with pytest.raises(RuntimeError, match="same surface_mesh box"):
        build_source_distribution_payload(simulation, data, _species_config())


def test_shorthand_takes_species_from_tally_filter():
    simulation, data, _ = distribution_simulation_and_data(
        particle_type=PARTICLE_PROTON
    )

    payload = build_source_distribution_payload(simulation, data, distribution_config())

    assert payload["sources"][0]["pdg"] == 2212
