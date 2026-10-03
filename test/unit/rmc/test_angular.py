import math

import numpy as np
import pytest
from scipy.integrate import quad

from mcdc.rmc.angular import (
    angular_transfer,
    angular_transfer_exact,
    build_angular_transfer_table,
)
from mcdc.rmc.kinematics import azimuthal_bin_probability


@pytest.mark.parametrize("J", [1, 2, 4])
def test_angular_transfer_conserves_incident_bin(J):
    edges = np.linspace(-1.0, 1.0, J + 1)
    for mu0 in [-1.0, -0.6, 0.0, 0.37, 0.99, 1.0]:
        for jp in range(J):
            total = sum(angular_transfer_exact(mu0, edges, j, jp) for j in range(J))
            assert total == pytest.approx(edges[jp + 1] - edges[jp], abs=1e-13)


def test_angular_transfer_against_adaptive_quadrature():
    edges = np.array([-1.0, -0.3, 0.0, 0.5, 1.0])
    for mu0 in [-0.8, 0.1, 0.7]:
        for j, jp in [(0, 0), (3, 0), (2, 1), (1, 3)]:
            kinks = []
            for b in (edges[j], edges[j + 1]):
                s = math.sqrt(1 - b * b) * math.sqrt(1 - mu0 * mu0)
                kinks += [
                    p
                    for p in (b * mu0 - s, b * mu0 + s)
                    if edges[jp] < p < edges[jp + 1]
                ]
            reference, _ = quad(
                lambda m: azimuthal_bin_probability(m, mu0, edges[j], edges[j + 1]),
                edges[jp],
                edges[jp + 1],
                points=kinks or None,
                limit=500,
                epsabs=1e-14,
            )
            assert angular_transfer_exact(mu0, edges, j, jp) == pytest.approx(
                reference, abs=1e-11
            )


def test_angular_transfer_table_interpolation():
    edges = np.linspace(-1.0, 1.0, 3)
    table = build_angular_transfer_table(edges, 2049)
    out = np.zeros((2, 2))
    rng = np.random.default_rng(1)
    for mu0 in rng.uniform(-1.0, 1.0, 50):
        angular_transfer(mu0, table, out)
        for j in range(2):
            for jp in range(2):
                assert out[j, jp] == pytest.approx(
                    angular_transfer_exact(mu0, edges, j, jp), abs=5e-8
                )
