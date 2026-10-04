"""Continuous piecewise-linear spatial trial space (mcdc.rmc.space)."""

import numpy as np
import pytest

from mcdc.rmc.residual import BOUNDARY_REFLECTIVE, BOUNDARY_VACUUM
from mcdc.rmc.space import (
    LinearZProjection,
    cell_average,
    constraint_map,
    mass_matrix,
    node_moments,
)

Z_EDGES = np.array([0.0, 0.5, 1.5, 2.0, 3.5])
E_EDGES = np.array([1.0, 2.0, 5.0])
MU_EDGES = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])


def moments_of(Phi, z_edges, E_edges, mu_edges):
    """Exact t[i, g, j, a] = int psi h_i P_a of the nodal function Phi."""
    t = np.einsum("il,lgja->igja", mass_matrix(z_edges), Phi)
    P = Phi.shape[-1]
    weight = np.diff(E_edges)[:, None] / (2.0 * np.arange(P) + 1.0)[None, :]
    return t * weight[None, :, None, :] * np.diff(mu_edges)[None, None, :, None]


def test_mass_matrix():
    M = mass_matrix(Z_EDGES)
    # Rows sum to int h_i dz; the total is the slab length
    assert M.sum() == pytest.approx(Z_EDGES[-1] - Z_EDGES[0])
    np.testing.assert_allclose(M.sum(axis=1)[1:-1], 0.5 * (Z_EDGES[2:] - Z_EDGES[:-2]))


@pytest.mark.parametrize(
    "boundary",
    [
        (BOUNDARY_VACUUM, BOUNDARY_VACUUM),
        (BOUNDARY_REFLECTIVE, BOUNDARY_REFLECTIVE),
        (BOUNDARY_REFLECTIVE, BOUNDARY_VACUUM),
    ],
)
def test_projection_reproduces_trial_functions(boundary):
    """Pi v = v for every nodal function satisfying the boundary constraints."""
    K, G, J, P = len(Z_EDGES) - 1, len(E_EDGES) - 1, len(MU_EDGES) - 1, 2
    T = constraint_map(K, MU_EDGES, *boundary)
    rng = np.random.default_rng(5)
    u = rng.normal(size=(T.shape[1], G * P))
    Phi = np.transpose((T @ u).reshape(J, K + 1, G, P), (1, 2, 0, 3))
    projection = LinearZProjection(Z_EDGES, E_EDGES, MU_EDGES, *boundary)
    np.testing.assert_allclose(
        projection.project(moments_of(Phi, Z_EDGES, E_EDGES, MU_EDGES)),
        Phi,
        atol=1e-12,
    )


def test_constraints():
    K = len(Z_EDGES) - 1
    mu_center = 0.5 * (MU_EDGES[:-1] + MU_EDGES[1:])
    # Vacuum: incoming nodal values are fixed to zero
    T = constraint_map(K, MU_EDGES, BOUNDARY_VACUUM, BOUNDARY_VACUUM)
    free = T.sum(axis=1).reshape(len(mu_center), K + 1)
    assert np.all(free[mu_center > 0.0, 0] == 0.0)
    assert np.all(free[mu_center < 0.0, K] == 0.0)
    assert np.all(free[:, 1:K] == 1.0)
    # Reflective: a bin and its mirror share the boundary unknown
    T = constraint_map(K, MU_EDGES, BOUNDARY_REFLECTIVE, BOUNDARY_REFLECTIVE)
    rows = T.reshape(len(mu_center), K + 1, -1)
    J = len(mu_center)
    for j in range(J):
        np.testing.assert_array_equal(rows[j, 0], rows[J - 1 - j, 0])
        np.testing.assert_array_equal(rows[j, K], rows[J - 1 - j, K])


def test_node_moments_and_cell_average():
    """Hat moments from cell Legendre moments of a linear function, exactly."""
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(len(Z_EDGES), 3))
    h = np.diff(Z_EDGES)
    left, right = Phi[:-1], Phi[1:]
    m0 = h[:, None] * 0.5 * (left + right)  # int_k psi
    m1 = h[:, None] * (right - left) / 6.0  # int_k psi x, x = 2s - 1
    np.testing.assert_allclose(node_moments(m0, m1), mass_matrix(Z_EDGES) @ Phi)
    np.testing.assert_allclose(cell_average(Phi), 0.5 * (left + right))
