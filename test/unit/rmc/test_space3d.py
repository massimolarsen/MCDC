"""
Polynomial spatial bases of the 3D solver (mcdc.rmc.space3d) and the flux-xyz tally
moments they are projected from (mcdc.transport.tally.score).
"""

import numpy as np
import pytest

from mcdc.rmc.mesh3d import CartesianMesh
from mcdc.rmc.patches import AngularPatches
from mcdc.rmc.residual import BOUNDARY_REFLECTIVE as R, BOUNDARY_VACUUM as V
from mcdc.rmc.space3d import SpatialBasis, legendre
from mcdc.transport.tally.score import _segment_legendre_mean

MESH = CartesianMesh([0.0, 0.3, 1.0, 1.4], [0.0, 0.5, 1.2], [0.0, 1.0, 2.5, 3.0])
PATCHES = AngularPatches([-1.0, 0.0, 1.0], np.linspace(-np.pi, np.pi, 5))
G, J = 2, 8


def _cell_moments(f, basis):
    """int_V f P_m dV per cell, exact for polynomials (6-point Gauss per axis)."""
    x, w = np.polynomial.legendre.leggauss(6)
    T = np.meshgrid(x, x, x, indexing="ij")
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    out = np.zeros(MESH.shape + (G, J, basis.M))
    for k in np.ndindex(MESH.shape):
        h = [MESH.width[d][k[d]] for d in range(3)]
        X = [MESH.edges[d][k[d]] + 0.5 * (T[d] + 1.0) * h[d] for d in range(3)]
        values = f(*X)
        for i, (a, b, c) in enumerate(basis.modes):
            P = legendre(a, T[0]) * legendre(b, T[1]) * legendre(c, T[2])
            out[k][..., i] = np.einsum("xyz,xyzgj->gj", W * P, values) * np.prod(h) / 8
    return out


@pytest.mark.parametrize("name", ["dg1", "dg2", "cg1", "cg2"])
def test_projection_reproduces_its_space(name):
    """
    The projection of a polynomial of degree p (global, so continuous) returns it
    exactly, and continuous bases have equal traces on both sides of every face.
    """
    basis = SpatialBasis(name, MESH, PATCHES, ((R, R),) * 3)
    p = basis.p
    coefficient = np.random.default_rng(0).normal(size=(G, J))

    def f(x, y, z):
        value = 1.0 + x - 0.5 * y + 0.3 * z
        if p == 2:
            value = value + x * x - 0.7 * y * z
        return value[..., None, None] * coefficient

    C = basis.project_modal(_cell_moments(f, basis))
    point = np.array([0.31, -0.62, 0.15])
    for k in np.ndindex(MESH.shape):
        position = [
            MESH.edges[d][k[d]] + 0.5 * (point[d] + 1.0) * MESH.width[d][k[d]]
            for d in range(3)
        ]
        value = sum(
            C[k][..., i] * legendre(a, point[0]) * legendre(b, point[1]) * legendre(c, point[2])
            for i, (a, b, c) in enumerate(basis.modes)
        )
        assert np.allclose(value, f(*position), atol=1e-12)
    if basis.continuous:
        N = np.random.default_rng(1).normal(size=basis.node_shape + (G, J))
        C = basis.nodal_to_modal(N)
        for axis in range(3):
            low = np.einsum("fm,xyzgjm->xyzgjf", basis.trace[axis][0], C)
            high = np.einsum("fm,xyzgjm->xyzgjf", basis.trace[axis][1], C)
            low, high = np.moveaxis(low, axis, 0), np.moveaxis(high, axis, 0)
            assert np.allclose(low[1:], high[:-1], atol=1e-13)


def test_continuous_vacuum_is_strong():
    """Incoming patches have zero nodal values on vacuum faces after projection."""
    basis = SpatialBasis("cg2", MESH, PATCHES, ((V, V), (R, R), (V, R)))
    moments = np.random.default_rng(2).normal(size=MESH.shape + (G, J, basis.M))
    N = basis.project(moments)
    sign = PATCHES.direction_sign
    assert np.all(N[0][..., sign[0] > 0] == 0.0)
    assert np.all(N[-1][..., sign[0] < 0] == 0.0)
    assert np.all(N[:, :, 0][..., sign[2] > 0] == 0.0)
    assert np.any(N[:, :, -1][..., sign[2] < 0] != 0.0)  # reflective: free


def test_segment_moments():
    """The tally's segment mean of P_a P_b P_c is exact (degree <= 6, 4 Gauss points)."""
    rng = np.random.default_rng(3)
    low, high = np.array([0.0, 0.5, 2.0]), np.array([0.3, 1.2, 2.5])
    start, distance = np.array([0.1, 0.7, 2.2]), 0.3
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    s, w = np.polynomial.legendre.leggauss(10)
    s, w = 0.5 * (s + 1.0) * distance, 0.5 * w
    for order in range(27):
        orders = (order % 3, (order // 3) % 3, order // 9)
        t = [
            (2.0 * (start[d] + s * u[d]) - low[d] - high[d]) / (high[d] - low[d])
            for d in range(3)
        ]
        exact = np.sum(w * np.prod([legendre(n, t[d]) for d, n in enumerate(orders)], axis=0))
        mean = _segment_legendre_mean(
            order, *start, *u, distance, low[0], high[0], low[1], high[1], low[2], high[2]
        )
        assert abs(mean - exact) < 1e-13
