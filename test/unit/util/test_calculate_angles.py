import numpy as np
import pytest

from mcdc.transport.util import calculate_angles
from mcdc.numba_types import particle_data


@pytest.mark.parametrize(
    "ux, uy, uz, expected_mu, expected_phi",
    [
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, np.pi),
        (0.0, 1.0, 0.0, 0.0, np.pi / 2.0),
        (0.0, -1.0, 0.0, 0.0, -np.pi / 2.0),
        (0.0, 0.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, -1.0, -1.0, 0.0),
    ],
)
def test_calculate_angles(ux, uy, uz, expected_mu, expected_phi):
    """Check the six coordinate directions against the positive-Z reference.

    Verify polar cosines and signed azimuthal angles, including the zero
    azimuth returned for directions along the reference axis.
    """
    particle_container = np.zeros(1, particle_data)
    particle = particle_container[0]

    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz
    polar_reference = np.array([0.0, 0.0, 1.0])

    mu, phi = calculate_angles(particle_container, polar_reference)

    assert np.isclose(mu, expected_mu)
    assert np.isclose(phi, expected_phi)


@pytest.mark.parametrize(
    "polar_reference, direction, expected_azimuthal",
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0),
        ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), np.pi / 2.0),
        ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), np.pi / 2.0),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), np.pi),
        ((-1.0, 0.0, 1.0e-12), (0.0, 1.0, 0.0), np.pi),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), 0.0),
    ],
)
def test_calculate_angles_reference_axes(
    polar_reference, direction, expected_azimuthal
):
    """Check the shared azimuthal-basis convention for signed reference axes.

    Exact and near-negative-X references must produce valid angles without
    a degenerate transverse basis. All test directions are perpendicular
    to their reference axis, so the expected polar cosine is zero.
    """
    particle_container = np.zeros(1, particle_data)
    particle = particle_container[0]
    particle["ux"], particle["uy"], particle["uz"] = direction
    polar_reference = np.array(polar_reference)
    polar_reference /= np.linalg.norm(polar_reference)

    mu, azimuthal = calculate_angles(particle_container, polar_reference)

    assert np.isclose(mu, 0.0)
    assert np.isclose(azimuthal, expected_azimuthal)
