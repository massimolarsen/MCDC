import numpy as np
import pytest

from mcdc.rmc.residual import (
    BOUNDARY_REFLECTIVE,
    BOUNDARY_VACUUM,
    _abs_linear_integral,
    _abs_quadratic_integral,
    abs_mu_integral,
    binned_source,
    collision_residual_mass,
    face_jumps,
    projected_removal,
)


@pytest.mark.parametrize("y0, y1", [(1.0, 3.0), (-2.0, 1.0), (2.0, -1.0), (-1.0, -4.0)])
def test_abs_linear_integral(y0, y1):
    x = np.linspace(0.5, 2.0, 200001)
    y = y0 + (y1 - y0) * (x - 0.5) / 1.5
    assert _abs_linear_integral(0.5, 2.0, y0, y1) == pytest.approx(
        np.trapezoid(np.abs(y), x), rel=1e-8
    )


@pytest.mark.parametrize(
    "y", [(1.0, 2.0, 0.5), (1.0, -1.0, 1.0), (-1.0, 0.5, 2.0), (0.0, 1.0, 2.0)]
)
def test_abs_quadratic_integral(y):
    x = np.linspace(0.5, 2.0, 400001)
    t = (x - 0.5) / 1.5
    y0, ym, y1 = y
    values = y0 * (1 - t) * (1 - 2 * t) + 4 * ym * t * (1 - t) + y1 * t * (2 * t - 1)
    assert _abs_quadratic_integral(0.5, 2.0, y0, ym, y1) == pytest.approx(
        np.trapezoid(np.abs(values), x), rel=1e-8
    )


@pytest.mark.parametrize(
    "c, psi",
    [([2.0], [0.9]), ([2.0, 1.5], [0.9, -0.6])],
)
def test_collision_residual_mass(c, psi):
    """Constant and linear energy coefficients against a brute-force integral."""
    E_grid = np.array([1.0, 2.0, 2.5, 4.0, 7.0])
    Sigma = np.array([1.0, 5.0, 0.5, 3.0, 2.0])
    c, psi = np.array(c), np.array(psi)
    for E_a, E_b in [(1.0, 7.0), (1.3, 2.2), (2.6, 3.9)]:
        E = np.linspace(E_a, E_b, 400001)
        x = (2 * E - E_a - E_b) / (E_b - E_a)
        c_E = c[0] + (c[1] * x if len(c) > 1 else 0.0)
        psi_E = psi[0] + (psi[1] * x if len(psi) > 1 else 0.0)
        brute = np.trapezoid(np.abs(c_E - psi_E * np.interp(E, E_grid, Sigma)), E)
        assert collision_residual_mass(
            c, psi, E_a, E_b, E_grid, Sigma
        ) == pytest.approx(brute, rel=1e-8)


def test_abs_mu_integral():
    np.testing.assert_allclose(
        abs_mu_integral(np.array([-1.0, -0.5, 0.0, 0.5, 1.0])),
        [0.375, 0.125, 0.125, 0.375],
    )
    with pytest.raises(ValueError):
        abs_mu_integral(np.array([-1.0, 0.5, 1.0]))


def test_face_jumps():
    mu_edges = np.array([-1.0, 0.0, 1.0])
    psi = np.array([[[1.0, 2.0]], [[3.0, 5.0]]])  # (K=2, G=1, J=2)

    D = face_jumps(psi, mu_edges, BOUNDARY_VACUUM, BOUNDARY_VACUUM)
    np.testing.assert_allclose(D[1], psi[1] - psi[0])
    # Left boundary, incoming mu > 0 only: psi - 0
    np.testing.assert_allclose(D[0], [[0.0, 2.0]])
    # Right boundary, incoming mu < 0 only: 0 - psi
    np.testing.assert_allclose(D[2], [[-3.0, 0.0]])

    D = face_jumps(psi, mu_edges, BOUNDARY_REFLECTIVE, BOUNDARY_REFLECTIVE)
    # Reflected value: psi(-mu) of the boundary cell
    np.testing.assert_allclose(D[0], [[0.0, 2.0 - 1.0]])
    np.testing.assert_allclose(D[2], [[5.0 - 3.0, 0.0]])

    # Symmetric psi in a single reflective cell: no edge residual
    symmetric = np.array([[[4.0, 4.0]]])
    D = face_jumps(symmetric, mu_edges, BOUNDARY_REFLECTIVE, BOUNDARY_REFLECTIVE)
    assert np.all(D == 0.0)


@pytest.mark.parametrize("P", [1, 2])
def test_binned_source(P):
    rng = np.random.default_rng(3)
    G, J = 3, 2
    M = rng.random((P, G, J, P, G, J))
    psi = rng.random((G, J, P))
    E_edges = np.array([1.0, 2.0, 4.0, 8.0])
    mu_edges = np.array([-1.0, 0.0, 1.0])
    S = binned_source(M, psi, E_edges, mu_edges)
    for a in range(P):
        for g in range(G):
            for j in range(J):
                expected = (
                    (2 * a + 1)
                    * sum(
                        M[b, gp, jp, a, g, j] * psi[gp, jp, b]
                        for b in range(P)
                        for gp in range(G)
                        for jp in range(J)
                    )
                    / ((E_edges[g + 1] - E_edges[g]) * (mu_edges[j + 1] - mu_edges[j]))
                )
                assert S[g, j, a] == pytest.approx(expected)


def test_projected_removal():
    """R[g, a, b] = (2a + 1)/dE int_g Sigma P_a P_b against a fine trapezoid."""
    E_grid = np.array([1.0, 2.0, 2.5, 4.0, 7.0])
    Sigma = np.array([1.0, 5.0, 0.5, 3.0, 2.0])
    E_edges = np.array([1.0, 2.2, 7.0])
    R = projected_removal((E_grid, Sigma), E_edges, 2)
    for g in range(2):
        E = np.linspace(E_edges[g], E_edges[g + 1], 400001)
        x = (2 * E - E_edges[g] - E_edges[g + 1]) / (E_edges[g + 1] - E_edges[g])
        P = [np.ones_like(x), x]
        for a in range(2):
            for b in range(2):
                expected = (
                    (2 * a + 1)
                    * np.trapezoid(np.interp(E, E_grid, Sigma) * P[a] * P[b], E)
                    / (E[-1] - E[0])
                )
                assert R[g, a, b] == pytest.approx(expected, rel=1e-8)
