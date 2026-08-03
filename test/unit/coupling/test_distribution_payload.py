import numpy as np
import pytest

from mcdc.coupling.geant4_distribution import build_source_distribution_payload

from ._helpers import distribution_config, distribution_simulation_and_data


def test_build_source_distribution_payload_uses_current_in_tally():
    simulation, data, weights = distribution_simulation_and_data()
    payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(
        payload["box_bounds_mm"], [[-10, 10], [-20, 20], [-30, 30]]
    )
    np.testing.assert_allclose(payload["mu_edges"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(payload["azi_edges"], [-np.pi, 0.0, np.pi])
    np.testing.assert_allclose(payload["energy_edges_mev"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(payload["weights"], weights.ravel())
    assert payload["Nu"] == 2
    assert payload["Nv"] == 3
    assert payload["n_events"] == 12
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

    np.testing.assert_allclose(payload["weights"], 2000.0 * weights.ravel())
    assert payload["source_total_weight"] == 2000.0 * np.sum(weights)


def test_build_source_distribution_payload_rejects_zero_global_n_particle():
    simulation, data, _ = distribution_simulation_and_data(n_particle=0, mpi_size=4)

    with pytest.raises(RuntimeError, match="global N_particle"):
        build_source_distribution_payload(simulation, data, distribution_config())


def test_build_source_distribution_payload_uses_structured_mcdc_tally():
    simulation, data, weights = distribution_simulation_and_data()

    payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(payload["weights"], weights.ravel())
    assert payload["source_tally_name"] == "source_tally"


def test_build_source_distribution_payload_warns_when_collapsing_time_bins():
    simulation, data, weights = distribution_simulation_and_data(time_bins=2)

    with pytest.warns(RuntimeWarning, match="collapses tally time bins"):
        payload = build_source_distribution_payload(simulation, data, distribution_config())

    np.testing.assert_allclose(payload["weights"], (2.0 * weights).ravel())


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
        ({"child_type": 1}, "surface-mesh"),
        ({"surface_mesh": False}, "surface_mesh"),
        ({"weights": np.zeros((2, 2, 2, 6, 2, 3))}, "zero total"),
    ],
)
def test_build_source_distribution_payload_rejects_invalid_tally(
    simulation_kwargs, match
):
    simulation, data, _ = distribution_simulation_and_data(**simulation_kwargs)

    with pytest.raises(RuntimeError, match=match):
        build_source_distribution_payload(simulation, data, distribution_config())
