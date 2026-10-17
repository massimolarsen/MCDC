"""Angular patches of the 3D trial space (mcdc.rmc.patches)."""

import math

import numpy as np
import pytest

from mcdc.rmc.angular import angular_transfer_exact
from mcdc.rmc.patches import (
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    AngularPatches,
    arc_fractions,
    build_patch_transfer_table,
    patch_transfer_row,
    sample_direction_abs,
)

PHI_8 = np.linspace(-np.pi, np.pi, 9)
MU_4 = np.array([-1.0, -0.4, 0.0, 0.5, 1.0])


def transfer(patches, mu0):
    A = np.zeros((patches.J, patches.J))
    for jp in range(patches.J):
        patch_transfer_row(mu0, jp, patches.mu_edges, patches.phi_edges, A[:, jp])
    return A


def test_layout_validation():
    with pytest.raises(ValueError, match="mu = 0"):
        AngularPatches([-1.0, 0.3, 1.0], PHI_8)
    with pytest.raises(ValueError, match="phi = pi/2"):
        AngularPatches([-1.0, 0.0, 1.0], [-np.pi, -np.pi / 2, 0.0, 2.0, np.pi])
    patches = AngularPatches(MU_4, PHI_8)
    assert patches.J == 32
    assert np.isclose(patches.area.sum(), 4.0 * np.pi)
    with pytest.raises(ValueError, match="axis z"):
        patches.mirror(AXIS_Z)  # the mu edges are not symmetric


@pytest.mark.parametrize("mu0", [-0.97, -0.5, 0.0, 0.23, 0.7, 0.999])
def test_transfer_identities(mu0):
    """
    sum_j A_jj' = |A_j'| (every circle lies on the sphere), A_jj' = A_j'j (both are
    the measure of {(Omega', Omega) : Omega' . Omega = mu0} in the patch pair), and
    summing out the azimuth gives 2 pi times the slab table.
    """
    patches = AngularPatches(MU_4, PHI_8)
    A = transfer(patches, mu0)
    assert np.allclose(A.sum(axis=0), patches.area, rtol=1e-13)
    assert np.max(np.abs(A - A.T)) < 1e-5 * patches.area.max()
    J_mu, J_phi = patches.J_mu, patches.J_phi
    slab = np.array(
        [
            [
                2.0 * math.pi * angular_transfer_exact(mu0, MU_4, j, jp)
                for jp in range(J_mu)
            ]
            for j in range(J_mu)
        ]
    )
    summed = A.reshape(J_mu, J_phi, J_mu, J_phi).sum(axis=(1, 3))
    assert np.max(np.abs(summed - slab)) < 1e-7 * slab.max()


@pytest.mark.parametrize("mu0", [-1.0, 1.0])
def test_transfer_limits(mu0):
    """mu0 = 1 keeps the direction (A = diag |A_j|); mu0 = -1 reverses it."""
    patches = AngularPatches([-1.0, 0.0, 1.0], np.linspace(-np.pi, np.pi, 5))
    A = transfer(patches, mu0)
    if mu0 > 0:
        expected = np.diag(patches.area)
    else:
        antipode = patches.index(
            -patches.center[:, 2],
            np.arctan2(-patches.center[:, 1], -patches.center[:, 0]),
        )
        expected = np.zeros_like(A)
        expected[antipode, np.arange(patches.J)] = patches.area
    assert np.allclose(A, expected, atol=1e-12)


def test_transfer_brute_force():
    """A_jj'(mu0) / |A_j'| against Omega' uniform on patch j' and a random azimuth."""
    patches = AngularPatches(MU_4, PHI_8)
    mu0, jp, N = 0.23, 5, 1_000_000
    rng = np.random.default_rng(3)
    mu_p = rng.uniform(patches.mu_low[jp], patches.mu_high[jp], N)
    phi_p = rng.uniform(patches.phi_low[jp], patches.phi_high[jp], N)
    psi = rng.uniform(0.0, 2.0 * np.pi, N)
    s_p, s0 = np.sqrt(1 - mu_p**2), math.sqrt(1 - mu0**2)
    o = np.stack((s_p * np.cos(phi_p), s_p * np.sin(phi_p), mu_p))
    e1 = np.stack((mu_p * np.cos(phi_p), mu_p * np.sin(phi_p), -s_p))
    e2 = np.stack((-np.sin(phi_p), np.cos(phi_p), 0.0 * mu_p))
    w = mu0 * o + s0 * (np.cos(psi) * e1 + np.sin(psi) * e2)
    p = (
        np.bincount(patches.index(w[2], np.arctan2(w[1], w[0])), minlength=patches.J)
        / N
    )
    expected = transfer(patches, mu0)[:, jp] / patches.area[jp]
    sigma = np.sqrt(np.maximum(expected * (1 - expected), 1e-12) / N)
    z = (p - expected) / sigma
    assert np.max(np.abs(z)) < 4.5, z


def test_symmetry_filled_table():
    """
    Columns filled from the rotation and mirror symmetries equal direct ones, to the
    quadrature accuracy of the direct computation (the fill itself is exact).
    """
    patches = AngularPatches([-1.0, -0.4, 0.0, 0.4, 1.0], PHI_8)
    table = build_patch_transfer_table(patches, N_theta=9)
    for k in range(9):
        mu0 = math.cos(math.pi * k / 8)
        error = np.abs(table[0, 0, :, :, k] - transfer(patches, mu0))
        assert np.max(error) < 1e-5 * patches.area.max()


def test_arc_fraction_single_patch():
    """A circle entirely inside one patch puts all of its length there."""
    mu_edges, phi_edges = np.array([-1.0, 0.0, 1.0]), np.linspace(-np.pi, np.pi, 5)
    out = np.zeros(8)
    arc_fractions(0.7, 0.8, 0.999, mu_edges, phi_edges, 1.0, out)
    patches = AngularPatches(mu_edges, phi_edges)
    assert np.isclose(out[patches.index(0.7, 0.8)], 1.0) and np.isclose(out.sum(), 1.0)


def test_abs_direction_integral():
    """int_patch |Omega_a| dOmega against a fine midpoint rule."""
    patches = AngularPatches(MU_4, PHI_8)
    n = 400
    for j in range(patches.J):
        mu = patches.mu_low[j] + (np.arange(n) + 0.5) / n * (
            patches.mu_high[j] - patches.mu_low[j]
        )
        phi = patches.phi_low[j] + (np.arange(n) + 0.5) / n * (
            patches.phi_high[j] - patches.phi_low[j]
        )
        M, F = np.meshgrid(mu, phi, indexing="ij")
        S = np.sqrt(1 - M**2)
        for axis, omega in ((0, S * np.cos(F)), (1, S * np.sin(F)), (2, M)):
            numeric = np.abs(omega).mean() * patches.area[j]
            assert np.isclose(
                patches.abs_direction_integral[axis, j], numeric, rtol=2e-4
            )


@pytest.mark.parametrize("axis", [AXIS_X, AXIS_Y, AXIS_Z])
def test_direction_sampler(axis):
    """Directions drawn with density |Omega_a| / int |Omega_a|: sub-bin frequencies."""
    patches = AngularPatches(MU_4, PHI_8)
    rng = np.random.default_rng(axis)
    N, n = 200_000, 4
    for j in (0, 13, 26):
        lo_m, hi_m = patches.mu_low[j], patches.mu_high[j]
        lo_f, hi_f = patches.phi_low[j], patches.phi_high[j]
        xi = rng.random((N, 2))
        samples = np.array(
            [sample_direction_abs(axis, lo_m, hi_m, lo_f, hi_f, a, b) for a, b in xi]
        )
        mu_e = np.linspace(lo_m, hi_m, n + 1)
        phi_e = np.linspace(lo_f, hi_f, n + 1)
        counts, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], [mu_e, phi_e])
        # exact sub-bin probabilities from the separable antiderivatives
        G = lambda m: 0.5 * (m * np.sqrt(1 - m**2) + np.arcsin(m))
        if axis == AXIS_Z:
            pm = np.abs(np.diff(mu_e * np.abs(mu_e)))
            pf = np.diff(phi_e)
        elif axis == AXIS_X:
            pm = np.diff(G(mu_e))
            pf = np.abs(np.diff(np.sin(phi_e)))
        else:
            pm = np.diff(G(mu_e))
            pf = np.abs(np.diff(np.cos(phi_e)))
        p = np.outer(pm / pm.sum(), pf / pf.sum())
        z = (counts / N - p) / np.sqrt(p * (1 - p) / N)
        assert np.max(np.abs(z)) < 4.5, (j, z)


def test_mirror_maps():
    """Reflection maps each patch onto the patch of the reflected center direction."""
    patches = AngularPatches([-1.0, -0.5, 0.0, 0.5, 1.0], PHI_8)
    c = patches.center
    for axis in (AXIS_X, AXIS_Y, AXIS_Z):
        r = c.copy()
        r[:, axis] *= -1.0
        expected = patches.index(r[:, 2], np.arctan2(r[:, 1], r[:, 0]))
        assert np.array_equal(patches.mirror(axis), expected)
        assert np.allclose(patches.area[patches.mirror(axis)], patches.area)
