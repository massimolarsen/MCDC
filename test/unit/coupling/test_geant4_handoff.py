import numpy as np
import pytest

from mcdc.constant import SCORE_CURRENT_IN
from mcdc.coupling import geant4_config, geant4_handoff
from mcdc.coupling.geant4_bank import convert_handoff_bank_to_geant4
from mcdc.coupling.geant4_distribution import build_source_distribution_payload
from mcdc.coupling.geant4_config import Geant4HandoffConfig

convert_handoff_bank = convert_handoff_bank_to_geant4
build_distribution_payload = build_source_distribution_payload


PARTICLE_DTYPE = np.dtype(
    [
        ("particle_type", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("ux", np.float64),
        ("uy", np.float64),
        ("uz", np.float64),
        ("E", np.float64),
        ("w", np.float64),
        ("t", np.float64),
    ]
)


def _distribution_simulation_and_data(
    *,
    tally_name="source_tally",
    scores=None,
    filter_direction=True,
    filter_energy=True,
    weights=None,
):
    if scores is None:
        scores = [99, SCORE_CURRENT_IN]
    if weights is None:
        weights = np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 1.0

    scores = np.asarray(scores, dtype=np.float64)
    mu = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    azi = np.asarray([-np.pi, 0.0, np.pi], dtype=np.float64)
    energy = np.asarray([0.0, 1.0e6, 2.0e6], dtype=np.float64)
    shape = np.asarray([2, 2, 2, 1, len(scores)], dtype=np.float64)
    mean = np.zeros(tuple(shape.astype(int)), dtype=np.float64)
    if SCORE_CURRENT_IN in scores:
        current_idx = int(np.where(scores == SCORE_CURRENT_IN)[0][0])
        mean[..., 0, current_idx] = weights

    offsets = {}
    chunks = []
    for name, values in (
        ("scores", scores),
        ("mu", mu),
        ("azi", azi),
        ("energy", energy),
        ("bin_shape", shape),
        ("bin_sum", mean.ravel()),
    ):
        offsets[name] = sum(len(chunk) for chunk in chunks)
        chunks.append(np.asarray(values, dtype=np.float64).ravel())
    data = np.concatenate(chunks)

    tally = {
        "name": tally_name,
        "filter_direction": filter_direction,
        "filter_energy": filter_energy,
        "scores_offset": offsets["scores"],
        "scores_length": len(scores),
        "mu_offset": offsets["mu"],
        "mu_length": len(mu),
        "azi_offset": offsets["azi"],
        "azi_length": len(azi),
        "energy_offset": offsets["energy"],
        "energy_length": len(energy),
        "bin_shape_offset": offsets["bin_shape"],
        "bin_shape_length": len(shape),
        "bin_sum_offset": offsets["bin_sum"],
        "bin_length": mean.size,
    }
    simulation = {
        "mpi_size": 1,
        "tallies": [tally],
        "bank_handoff": {
            "size": np.asarray([0], dtype=np.int64),
            "particle_data": np.array([], dtype=PARTICLE_DTYPE),
        },
    }
    return simulation, data, weights


def _distribution_config(**kwargs):
    values = {
        "source_mode": "distribution",
        "n_geant4_particles": 12,
        "source_tally_name": "source_tally",
        "distribution_box_cm": ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)),
    }
    values.update(kwargs)
    return Geant4HandoffConfig(**values)


def test_convert_handoff_bank_to_geant4_converts_units_and_layout():
    particles = np.array(
        [
            (0, 1.25, -2.0, 0.5, 0.0, 0.6, 0.8, 14.0e6, 0.75, 2.5e-9),
            (0, -0.1, 0.2, -0.3, -1.0, 0.0, 0.0, 2.0e6, 1.25, 1.0e-8),
        ],
        dtype=PARTICLE_DTYPE,
    )

    bank = convert_handoff_bank(particles)

    assert bank.shape == (2, 10)
    np.testing.assert_allclose(bank[:, 0], [2112.0, 2112.0])
    np.testing.assert_allclose(bank[:, 1], [12.5, -1.0])
    np.testing.assert_allclose(bank[:, 2], [-20.0, 2.0])
    np.testing.assert_allclose(bank[:, 3], [5.0, -3.0])
    np.testing.assert_allclose(bank[:, 4], [0.0, -1.0])
    np.testing.assert_allclose(bank[:, 5], [0.6, 0.0])
    np.testing.assert_allclose(bank[:, 6], [0.8, 0.0])
    np.testing.assert_allclose(bank[:, 7], [14.0, 2.0])
    np.testing.assert_allclose(bank[:, 8], [0.75, 1.25])
    np.testing.assert_allclose(bank[:, 9], [2.5, 10.0])


def test_convert_handoff_bank_to_geant4_handles_empty_arrays():
    particles = np.array([], dtype=PARTICLE_DTYPE)

    bank = convert_handoff_bank(particles)

    assert bank.shape == (0, 10)


def test_convert_handoff_bank_to_geant4_rejects_non_neutrons():
    particles = np.array(
        [(1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0e6, 1.0, 0.0)],
        dtype=PARTICLE_DTYPE,
    )

    with pytest.raises(RuntimeError, match="supports neutrons only"):
        convert_handoff_bank(particles)


def test_build_source_distribution_payload_uses_current_in_tally():
    simulation, data, weights = _distribution_simulation_and_data()
    payload = build_distribution_payload(simulation, data, _distribution_config())

    np.testing.assert_allclose(
        payload["box_bounds_mm"], [[-10, 10], [-20, 20], [-30, 30]]
    )
    np.testing.assert_allclose(payload["mu_edges"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(payload["azi_edges"], [-np.pi, 0.0, np.pi])
    np.testing.assert_allclose(payload["energy_edges_mev"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(payload["weights"], weights.ravel())
    assert payload["n_events"] == 12
    assert payload["total_weight"] == np.sum(weights)


def test_build_source_distribution_payload_warns_when_collapsing_time_bins():
    simulation, data, weights = _distribution_simulation_and_data()
    tally = simulation["tallies"][0]

    shape_offset = int(tally["bin_shape_offset"])
    mean_offset = int(tally["bin_sum_offset"])
    scores_offset = int(tally["scores_offset"])
    scores_length = int(tally["scores_length"])
    scores = data[scores_offset : scores_offset + scores_length].astype(int)
    current_idx = int(np.where(scores == SCORE_CURRENT_IN)[0][0])

    shape = np.asarray([2, 2, 2, 2, len(scores)], dtype=np.float64)
    mean = np.zeros(tuple(shape.astype(int)), dtype=np.float64)
    mean[..., 0, current_idx] = weights
    mean[..., 1, current_idx] = weights

    data[shape_offset : shape_offset + len(shape)] = shape
    data = np.concatenate([data[:mean_offset], mean.ravel()])
    tally["bin_length"] = mean.size

    with pytest.warns(RuntimeWarning, match="collapses tally axes"):
        payload = build_distribution_payload(simulation, data, _distribution_config())

    np.testing.assert_allclose(payload["weights"], (2.0 * weights).ravel())


def test_build_source_distribution_payload_rejects_missing_tally_field():
    simulation, data, _ = _distribution_simulation_and_data()
    del simulation["tallies"][0]["bin_sum_offset"]

    with pytest.raises(RuntimeError, match="missing required fields: bin_sum_offset"):
        build_distribution_payload(simulation, data, _distribution_config())


@pytest.mark.parametrize(
    "config_update, match",
    [
        ({"n_geant4_particles": 0}, "n_geant4_particles > 0"),
        ({"source_tally_name": ""}, "source_tally_name"),
        ({"distribution_box_cm": None}, "distribution_box_cm"),
        ({"distribution_box_cm": ((0.0, 0.0), (-1.0, 1.0), (-1.0, 1.0))}, "min < max"),
    ],
)
def test_build_source_distribution_payload_rejects_invalid_config(config_update, match):
    simulation, data, _ = _distribution_simulation_and_data()

    with pytest.raises(RuntimeError, match=match):
        build_distribution_payload(
            simulation, data, _distribution_config(**config_update)
        )


@pytest.mark.parametrize(
    "simulation_kwargs, match",
    [
        ({"tally_name": "different"}, "Could not find source tally"),
        ({"scores": [99]}, "current-in"),
        ({"filter_direction": False}, "mu/azi"),
        ({"filter_energy": False}, "energy"),
        ({"weights": np.zeros((2, 2, 2))}, "zero total"),
    ],
)
def test_build_source_distribution_payload_rejects_invalid_tally(
    simulation_kwargs, match
):
    simulation, data, _ = _distribution_simulation_and_data(**simulation_kwargs)

    with pytest.raises(RuntimeError, match=match):
        build_distribution_payload(simulation, data, _distribution_config())


def test_run_handoff_distribution_loads_source_distribution(monkeypatch):
    simulation, data, weights = _distribution_simulation_and_data()

    class FakeResults:
        loaded_primaries = 12
        last_events_run = 12
        status = "ok"
        first_primary = []
        last_primary = []
        min_position_mm = [0.0, 0.0, 0.0]
        max_position_mm = [0.0, 0.0, 0.0]
        min_direction = [0.0, 0.0, 0.0]
        max_direction = [0.0, 0.0, 0.0]
        min_energy_mev = 0.0
        max_energy_mev = 0.0

    class FakeSession:
        def __init__(self):
            self.loaded_distribution = None

        def load_source_distribution(self, *args):
            self.loaded_distribution = args

        def beam_on(self):
            pass

        def get_results(self):
            return FakeResults()

        def close(self):
            pass

    fake_session = FakeSession()
    monkeypatch.setattr(geant4_handoff, "CONFIG", _distribution_config())
    monkeypatch.setattr(geant4_config, "create_session", lambda: fake_session)

    summary = geant4_handoff.run_handoff_from_simulation(simulation, data)

    assert summary["source_mode"] == "distribution"
    assert summary["events_run"] == 12
    assert fake_session.loaded_distribution is not None
    np.testing.assert_allclose(fake_session.loaded_distribution[4], weights.ravel())
