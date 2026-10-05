import numpy as np
import pytest

from mcdc.coupling.geant4_bank import (
    build_bank_payload,
    convert_handoff_bank_to_geant4,
)

from ._helpers import PARTICLE_DTYPE


def test_convert_handoff_bank_to_geant4_converts_units_and_layout():
    particles = np.array(
        [
            (0, 1.25, -2.0, 0.5, 0.0, 0.6, 0.8, 14.0e6, 0.75, 2.5e-9),
            (0, -0.1, 0.2, -0.3, -1.0, 0.0, 0.0, 2.0e6, 1.25, 1.0e-8),
        ],
        dtype=PARTICLE_DTYPE,
    )

    bank = convert_handoff_bank_to_geant4(particles)

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

    bank = convert_handoff_bank_to_geant4(particles)

    assert bank.shape == (0, 10)


def test_convert_handoff_bank_to_geant4_maps_mixed_species():
    # neutron, electron, proton
    particles = np.array(
        [
            (0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0e6, 1.0, 0.0),
            (1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0e6, 1.0, 0.0),
            (2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0e8, 1.0, 0.0),
        ],
        dtype=PARTICLE_DTYPE,
    )

    bank = convert_handoff_bank_to_geant4(particles)

    np.testing.assert_allclose(bank[:, 0], [2112.0, 11.0, 2212.0])
    np.testing.assert_allclose(bank[:, 7], [1.0, 2.0, 100.0])


def test_convert_handoff_bank_to_geant4_rejects_unsupported_types():
    particles = np.array(
        [(7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0e6, 1.0, 0.0)],
        dtype=PARTICLE_DTYPE,
    )

    with pytest.raises(RuntimeError, match="does not support: \\[7\\]"):
        convert_handoff_bank_to_geant4(particles)


def test_build_bank_payload_summarizes_species():
    particles = np.array(
        [
            (2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0e8, 0.5, 0.0),
            (0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0e6, 2.0, 0.0),
            (2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9.0e7, 0.25, 0.0),
        ],
        dtype=PARTICLE_DTYPE,
    )
    simulation = {
        "bank_handoff": {
            "size": np.asarray([3], dtype=np.int64),
            "particle_data": particles,
        }
    }

    payload = build_bank_payload(simulation)

    np.testing.assert_array_equal(payload["handoff_species_pdg"], [2112, 2212])
    np.testing.assert_array_equal(payload["handoff_species_count"], [1, 2])
    np.testing.assert_allclose(payload["handoff_species_weight"], [2.0, 0.75])
