import numpy as np
import pytest

from mcdc.coupling.geant4_bank import convert_handoff_bank_to_geant4

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


def test_convert_handoff_bank_to_geant4_rejects_non_neutrons():
    particles = np.array(
        [(1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0e6, 1.0, 0.0)],
        dtype=PARTICLE_DTYPE,
    )

    with pytest.raises(RuntimeError, match="supports neutrons only"):
        convert_handoff_bank_to_geant4(particles)
