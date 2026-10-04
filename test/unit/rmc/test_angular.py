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


@pytest.mark.parametrize("order", [0, 1])
def test_angular_transfer_table_interpolation(order):
    edges = np.linspace(-1.0, 1.0, 3)
    B = order + 1
    table = build_angular_transfer_table(edges, 2049, order)
    out = np.zeros((B, B, 2, 2))
    rng = np.random.default_rng(1)
    for mu0 in rng.uniform(-1.0, 1.0, 50):
        angular_transfer(mu0, table, out)
        for bp in range(B):
            for b in range(B):
                for j in range(2):
                    for jp in range(2):
                        # Linear interpolation in theta0 (2049 nodes): the slope
                        #   blocks curve more in theta0
                        tol = 5e-8 if bp == b == 0 else 3e-7
                        assert out[bp, b, j, jp] == pytest.approx(
                            angular_transfer_exact(mu0, edges, j, jp, bp, b), abs=tol
                        )


def test_angular_moments_against_quadrature():
    """
    A^{b' b}_{j j'}(mu0) = int_{j'} P_b'(y_j'(mu_in)) Pi^b_j dmu_in against a brute-force
    double integral over mu_in and the azimuth, and the sum rule
    sum_j A^{b' 0}_{j j'} = int_{j'} P_b' (= dmu_j' for b' = 0, 0 for b' = 1).
    """
    edges = np.array([-1.0, -0.3, 0.0, 0.5, 1.0])
    J = len(edges) - 1
    mu_in = (np.arange(4000) + 0.5) / 4000
    theta = (np.arange(4000) + 0.5) / 4000 * math.pi
    for mu0 in [-0.8, 0.1, 0.7]:
        for jp in range(J):
            low, high = edges[jp], edges[jp + 1]
            m = low + (high - low) * mu_in
            y_in = (2 * m - low - high) / (high - low)
            mu_out = m[:, None] * mu0 + np.sqrt(1 - m[:, None] ** 2) * math.sqrt(
                1 - mu0 * mu0
            ) * np.cos(theta[None, :])
            for bp in range(2):
                total = sum(
                    angular_transfer_exact(mu0, edges, j, jp, bp, 0) for j in range(J)
                )
                assert total == pytest.approx(
                    (high - low) if bp == 0 else 0.0, abs=1e-12
                )
                for j in range(J):
                    inside = (mu_out >= edges[j]) & (mu_out < edges[j + 1])
                    y_out = (2 * mu_out - edges[j] - edges[j + 1]) / (
                        edges[j + 1] - edges[j]
                    )
                    for b in range(2):
                        brute = (high - low) * np.mean(
                            (y_in**bp)[:, None] * (y_out**b) * inside
                        )
                        assert angular_transfer_exact(
                            mu0, edges, j, jp, bp, b
                        ) == pytest.approx(brute, abs=2e-4)
