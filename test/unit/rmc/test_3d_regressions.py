"""Proposal support and envelopes for the spatial-polynomial residual."""

from pathlib import Path

import mcdc.rmc.driver3d as driver

import numpy as np
import pytest

from mcdc.rmc.correction3d import CorrectionNodes
from mcdc.rmc.mesh3d import CartesianMesh
from mcdc.rmc.patches import AngularPatches
from mcdc.rmc.polynomial3d import volume_masses
from mcdc.rmc.residual import BOUNDARY_REFLECTIVE as R
from mcdc.rmc.space3d import SpatialBasis


def test_streaming_proposal_covers_interior_cancellation():
    n_angle = 2
    mesh = CartesianMesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    patches = AngularPatches([-1.0, 0.0, 1.0], np.linspace(-np.pi, np.pi, 5))
    basis = SpatialBasis("dg1", mesh, patches, ((R, R),) * 3)
    nodes = CorrectionNodes(np.array([1.0, 2.0]), patches, [], 1, n_angle, n_angle)
    C = np.zeros(mesh.shape + (1, patches.J, basis.M))
    Q = np.zeros_like(C)
    j = patches.index(0.5, np.pi / 4)
    for mode in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        C[0, 0, 0, 0, j, basis.index[mode]] = 0.5
    Q[0, 0, 0, 0, j, 0] = 1.0
    Tn = np.zeros((1, basis.M, len(nodes.E_nodes), patches.J, n_angle, n_angle))
    mass = volume_masses(
        basis,
        C,
        Q,
        Tn,
        basis.gradient(C),
        np.zeros((1, len(nodes.E_nodes))),
        np.array([0]),
        nodes,
        patches,
        np.zeros((1, len(nodes.interval_left))),
    ).reshape(1, 1, patches.J, n_angle - 1, n_angle - 1)
    # In a void r = 1 - (Omega_x + Omega_y + Omega_z). For two nodes,
    # every corner is zero but every interior direction has nonzero r.
    rng = np.random.default_rng(2)
    for a in range(n_angle - 1):
        for b in range(n_angle - 1):
            mu = (a + rng.random(1000)) / (n_angle - 1)
            phi = (b + rng.random(1000)) * (np.pi / 2) / (n_angle - 1)
            r = 1.0 - mu - np.sqrt(1 - mu**2) * (np.cos(phi) + np.sin(phi))
            measure = (np.pi / 2) / (n_angle - 1) ** 2
            bound = mass[0, 0, j, a, b] / measure
            assert bound > 0.0
            assert np.all(np.abs(r) <= bound + 1e-14)


def test_harmonic_projection_reproduces_p7(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "examples/rmc/basis_study")
    )
    from project import AngularBasis
    from sn3d import product_quadrature

    with pytest.raises(ValueError, match="underresolved"):
        AngularBasis("P7", *product_quadrature(4, 8, 4))
    basis = AngularBasis("P7", *product_quadrature(4, 8, 8))
    assert np.allclose(basis.P.T @ basis.B, np.eye(basis.F), atol=1e-8)


def test_table_setup_is_outside_pass_timer(monkeypatch):
    for name in ("constant", "dg1"):
        _check_table_timing(monkeypatch, name)


def _check_table_timing(monkeypatch, name):
    clock = [0.0]
    monkeypatch.setattr(driver.MPI, "Wtime", lambda: clock[0])

    class Solver:
        def __init__(self, *args):
            self.mesh = CartesianMesh([0, 1], [0, 1], [0, 1])
            self.E_edges = np.array([1.0, 2.0])
            self.patches = AngularPatches([-1, 0, 1], np.linspace(-np.pi, np.pi, 5))
            self.basis = (
                None
                if name == "constant"
                else SpatialBasis(name, self.mesh, self.patches, ((R, R),) * 3)
            )
            self.shape = (1, 1, 1, 1, 8)
            self.N_total = 80
            self.measure = np.ones(self.shape)
            self.last_residual_mass = (1.0, 0.0)
            self.last_correction_mass = (1.0, 0.0)
            self.last_scalar = (np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1, 1)))
            self.prepared = False

        def prepare_correction(self):
            if not self.prepared:
                clock[0] += 10
                self.time_table = 10
                self.prepared = True

        def iterate(self, psi, n):
            clock[0] += 3
            return np.zeros_like(psi)

        iterate_polynomial = iterate

        def correct(self, psi, n):
            assert self.prepared
            clock[0] += 2
            return np.zeros_like(psi), np.zeros_like(psi)

        correct_polynomial = correct

    monkeypatch.setattr(driver, "RMC3DSolver", Solver)
    result = driver.run3d(
        None,
        *([0, 1],) * 3,
        [1, 2],
        [-1, 0, 1],
        np.linspace(-np.pi, np.pi, 5),
        None,
        None,
        1,
        10,
        N_correction=2,
        spatial_basis=name,
    )
    assert result.time_iteration == [3]
    assert result.time_correction == [2, 2]
    assert result.time_table == 10
    assert (
        result.time_precompute
        + sum(result.time_iteration)
        + sum(result.time_correction)
        + result.time_table
        == clock[0]
    )
