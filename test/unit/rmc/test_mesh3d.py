"""Face jumps and residual masses of the 3D trial space (mcdc.rmc.mesh3d)."""

import numpy as np

from mcdc.rmc.mesh3d import CartesianMesh, Residual3D, face_jumps_3d
from mcdc.rmc.patches import AXIS_X, AXIS_Y, AXIS_Z, AngularPatches
from mcdc.rmc.residual import BOUNDARY_REFLECTIVE, BOUNDARY_VACUUM

PATCHES = AngularPatches([-1.0, 0.0, 1.0], np.linspace(-np.pi, np.pi, 5))


def test_uniform_flux_reflective_box_has_no_face_residual():
    """A uniform, isotropic psi~ in a reflective box is continuous everywhere."""
    psi = np.full((2, 3, 1, 2, PATCHES.J), 0.7)
    for axis in (AXIS_X, AXIS_Y, AXIS_Z):
        D = face_jumps_3d(
            psi,
            axis,
            (BOUNDARY_REFLECTIVE, BOUNDARY_REFLECTIVE),
            PATCHES,
            PATCHES.mirror(axis),
        )
        assert np.all(D == 0.0)


def test_vacuum_jumps_are_incoming_only():
    rng = np.random.default_rng(0)
    psi = rng.random((3, 2, 2, 1, PATCHES.J))
    D = face_jumps_3d(psi, AXIS_X, (BOUNDARY_VACUUM, BOUNDARY_VACUUM), PATCHES)
    sign = PATCHES.direction_sign[AXIS_X]
    assert D.shape == (4, 2, 2, 1, PATCHES.J)
    assert np.allclose(D[1:3], psi[1:] - psi[:-1])
    # low face: D = psi~(cell 0) - 0 for Omega_x > 0; high face: 0 - psi~ for < 0
    assert np.allclose(D[0][..., sign > 0], psi[0][..., sign > 0])
    assert np.all(D[0][..., sign < 0] == 0.0)
    assert np.allclose(D[3][..., sign < 0], -psi[2][..., sign < 0])
    assert np.all(D[3][..., sign > 0] == 0.0)


def test_reflective_jump_uses_mirrored_patch():
    rng = np.random.default_rng(1)
    psi = rng.random((1, 1, 2, 1, PATCHES.J))
    mirror = PATCHES.mirror(AXIS_Z)
    D = face_jumps_3d(
        psi, AXIS_Z, (BOUNDARY_REFLECTIVE, BOUNDARY_VACUUM), PATCHES, mirror
    )
    up = PATCHES.direction_sign[AXIS_Z] > 0
    assert np.allclose(
        D[:, :, 0][..., up], (psi[:, :, 0] - psi[:, :, 0][..., mirror])[..., up]
    )


def test_face_mass_of_uniform_flux_in_vacuum_box():
    """
    Uniform psi~ in a vacuum box: the face residual is the incoming boundary jump,
    with total mass psi dE sum_a (boundary area on a) int_{Omega_a > 0} |Omega_a|
    (= pi per face pair... per boundary face: pi psi dE area for each axis side).
    """
    mesh = CartesianMesh([0.0, 1.0, 3.0], [0.0, 2.0], [0.0, 0.5, 1.0, 1.5])
    psi = np.full(mesh.shape + (1, PATCHES.J), 2.0)
    dE = 0.25
    residual = Residual3D(
        psi,
        np.zeros_like(psi),
        np.zeros_like(psi),
        mesh,
        np.array([1.0, 1.0 + dE]),
        PATCHES,
        ((BOUNDARY_VACUUM, BOUNDARY_VACUUM),) * 3,
        (None, None, None),
        np.zeros(mesh.shape + (1,)),
        None,
    )
    Lx, Ly, Lz = 3.0, 2.0, 1.5
    # int over the incoming hemisphere of |Omega . n| = pi, per boundary face
    expected = 2.0 * dE * np.pi * 2.0 * (Ly * Lz + Lx * Lz + Lx * Ly)
    assert np.isclose(sum(m.sum() for m in residual.face_mass), expected)
    assert np.all(residual.collision_mass == 0.0)
