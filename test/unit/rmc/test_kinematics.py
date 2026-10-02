import numpy as np
import pytest

from mcdc.rmc.kinematics import (
    azimuthal_bin_probability,
    azimuthal_kernel,
    com_to_lab,
    com_to_lab_jacobian,
    elastic_dmu_cm_dE_out,
    elastic_E_out,
    elastic_E_out_range,
    elastic_mu_cm,
    elastic_mu_lab,
    lab_to_com,
    level_dmu_cm_dE_out,
    level_E_out_range,
    level_mu_cm,
    mu0_from_theta,
)

A_VALUES = [0.999167, 15.857510, 233.024800]


@pytest.mark.parametrize("A", A_VALUES)
def test_com_lab_round_trip(A):
    E_in = 2.0e6
    for E_cm in [1.0e3, 3.0e5, 1.5e6]:
        for mu_cm in [-0.9, -0.2, 0.4, 0.95]:
            E_out, mu_lab = com_to_lab(E_in, E_cm, mu_cm, A)
            E_cm_back, mu_cm_back = lab_to_com(E_in, E_out, mu_lab, A)
            assert E_cm_back == pytest.approx(E_cm, rel=1e-10)
            assert mu_cm_back == pytest.approx(mu_cm, abs=1e-10)


@pytest.mark.parametrize("A", A_VALUES)
def test_com_lab_joint_jacobian(A):
    """|d(E_out, mu_lab)/d(E_cm, mu_cm)| = sqrt(E_cm / E_out), so f_L = f_C sqrt(E_out/E_cm)."""
    E_in = 2.0e6
    for E_cm, mu_cm in [(1.0e3, 0.3), (3.0e5, -0.7), (1.5e6, 0.9)]:
        hE = 1e-6 * E_cm
        hm = 1e-6
        Ep, mp = com_to_lab(E_in, E_cm + hE, mu_cm, A)
        Em, mm = com_to_lab(E_in, E_cm - hE, mu_cm, A)
        dE_dEcm, dmu_dEcm = (Ep - Em) / (2 * hE), (mp - mm) / (2 * hE)
        Ep, mp = com_to_lab(E_in, E_cm, mu_cm + hm, A)
        Em, mm = com_to_lab(E_in, E_cm, mu_cm - hm, A)
        dE_dmu, dmu_dmu = (Ep - Em) / (2 * hm), (mp - mm) / (2 * hm)
        determinant = abs(dE_dEcm * dmu_dmu - dE_dmu * dmu_dEcm)

        E_out, _ = com_to_lab(E_in, E_cm, mu_cm, A)
        assert determinant == pytest.approx(
            1.0 / com_to_lab_jacobian(E_out, E_cm), rel=1e-6
        )


@pytest.mark.parametrize("A", A_VALUES)
def test_level_kinematics(A):
    E_in, E_cm = 9.0e6, 2.0e6
    mu_cm = np.linspace(-1.0, 1.0, 2001)
    E_out = np.array([com_to_lab(E_in, E_cm, m, A)[0] for m in mu_cm])

    # Inverse and derivative of the (affine) E_out(mu_cm) relation
    for m, e in zip(mu_cm[::200], E_out[::200]):
        assert level_mu_cm(E_in, e, E_cm, A) == pytest.approx(m, abs=1e-10)
    slope = (E_out[-1] - E_out[0]) / 2.0
    assert level_dmu_cm_dE_out(E_in, E_cm, A) == pytest.approx(1.0 / slope, rel=1e-10)

    # Support over a range of E_cm, against brute force
    for low, high in [(1.0e5, 2.0e6), (1.0e3, 1.0e4), (2.0e6, 2.0e6)]:
        brute = [
            com_to_lab(E_in, ec, m, A)[0]
            for ec in np.linspace(low, high, 201)
            for m in mu_cm[::20]
        ]
        E_low, E_high = level_E_out_range(E_in, low, high, A)
        assert E_low <= min(brute) * (1 + 1e-12) + 1e-9
        assert E_high == pytest.approx(max(brute), rel=1e-12)
        assert min(brute) - E_low < 1e-3 * (E_high - E_low) + 1e-6


@pytest.mark.parametrize("A", A_VALUES)
def test_elastic_kinematics(A):
    E_in = 2.0e6
    for mu_cm in [-1.0, -0.3, 0.5, 1.0]:
        E_out = elastic_E_out(E_in, mu_cm, A)
        assert elastic_mu_cm(E_in, E_out, A) == pytest.approx(mu_cm, abs=1e-9)
    assert elastic_E_out_range(E_in, A)[1] == pytest.approx(E_in, rel=1e-12)

    # Derivative of mu_cm(E_out)
    E_out = elastic_E_out(E_in, 0.2, A)
    h = 1e-6 * E_out
    numeric = (
        elastic_mu_cm(E_in, E_out + h, A) - elastic_mu_cm(E_in, E_out - h, A)
    ) / (2 * h)
    assert elastic_dmu_cm_dE_out(E_in, E_out, A) == pytest.approx(numeric, rel=1e-6)

    # Classical limit: isotropic COM gives g = 1 / ((1 - alpha) E_in)
    alpha = ((A - 1) / (A + 1)) ** 2
    g = 0.5 * elastic_dmu_cm_dE_out(E_in, E_out, A)
    assert g == pytest.approx(1.0 / ((1.0 - alpha) * E_in), rel=1e-2)

    # Lab cosine from velocity addition
    assert elastic_mu_lab(1.0, A) == pytest.approx(1.0)


def test_azimuthal_bin_probability():
    edges = np.linspace(-1.0, 1.0, 5)
    for mu_in, mu0 in [(0.3, 0.8), (-0.95, 0.1), (0.0, -0.5), (1.0, 0.4)]:
        P = [
            azimuthal_bin_probability(mu_in, mu0, edges[j], edges[j + 1])
            for j in range(4)
        ]
        assert sum(P) == pytest.approx(1.0, abs=1e-12)

        # Against direct azimuth averaging
        phi = (np.arange(200000) + 0.5) / 200000 * 2 * np.pi
        mu_out = mu_in * mu0 + np.sqrt(1 - mu_in**2) * np.sqrt(1 - mu0**2) * np.cos(phi)
        counts = np.histogram(mu_out, edges)[0] / len(phi)
        np.testing.assert_allclose(P, counts, atol=1e-4)


def test_theta_substitution_removes_singularity():
    """int K(mu_in, mu_out | mu0) dmu0 over its support = 1 = (1/pi) int_0^pi dtheta."""
    mu_in, mu_out = 0.4, -0.3
    theta, w = np.polynomial.legendre.leggauss(8)
    theta = 0.5 * np.pi * (theta + 1)
    w = 0.5 * np.pi * w
    # Smooth test function of mu0 integrated against K
    f = lambda mu0: 1.0 + mu0 + mu0**2
    via_theta = (
        sum(wi * f(mu0_from_theta(mu_in, mu_out, t)) for t, wi in zip(theta, w)) / np.pi
    )
    s = np.sqrt(1 - mu_in**2) * np.sqrt(1 - mu_out**2)
    mu0 = np.linspace(mu_in * mu_out - s, mu_in * mu_out + s, 101)[1:-1]
    assert all(azimuthal_kernel(mu_in, mu_out, m) > 0.0 for m in mu0)
    exact = 1.0 + mu_in * mu_out + (mu_in * mu_out) ** 2 + 0.5 * s**2
    assert via_theta == pytest.approx(exact, rel=1e-9)
