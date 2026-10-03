"""
End-to-end RMC on analytic 0D slowing-down problems (synthetic nuclides).
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import mcdc
from mcdc.rmc.driver import run

from .conftest import numba_only, write_synthetic_nuclide

pytestmark = numba_only


def reflective_box(material):
    """Single cell, reflective on all six faces: an infinite medium."""
    surfaces = []
    for plane, low, high in (
        (mcdc.Surface.PlaneX, -0.5, 0.5),
        (mcdc.Surface.PlaneY, -0.5, 0.5),
        (mcdc.Surface.PlaneZ, 0.0, 1.0),
    ):
        surfaces.append(
            (
                plane(
                    **{plane.__name__[-1].lower(): low}, boundary_condition="reflective"
                ),
                plane(
                    **{plane.__name__[-1].lower(): high},
                    boundary_condition="reflective",
                ),
            )
        )
    (x0, x1), (y0, y1), (z0, z1) = surfaces
    return mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=material)


def test_analytic_slowing_down_A1(tmp_path, monkeypatch):
    """
    A = 1, constant sigma_s = sigma_c (c = 1/2), isotropic COM elastic, monoenergetic
    source S0 = 1 at E0 (smeared over a narrow top bin). Below E0:
        phi(E) = c S0 / (Sigma_t E0) (E0 / E)^c.
    """
    E0, E_min, delta = 101.0, 1.0, 0.1
    sigma_s, sigma_c = 1.0, 1.0
    write_synthetic_nuclide(
        tmp_path, "HX", 1.0, [1.0e-5, 1.0e3], [sigma_s, sigma_s], [sigma_c, sigma_c]
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))

    material = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
    simulation = mcdc.Simulation("rmc-analytic")
    simulation.set_model([reflective_box(material)])
    simulation.set_sources(
        [mcdc.Source(position=[0.0, 0.0, 0.5], isotropic=True, energy=E0)]
    )

    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 21), [E0]))
    mu_edges = np.array([-1.0, 0.0, 1.0])
    G = len(E_edges) - 1
    Q = np.zeros((1, G, 2))
    Q[0, -1, :] = 1.0 / (delta * 2.0)  # S0 = 1 per unit area, isotropic
    # (the source bin stays much wider than MC/DC's 1e-5 eV tally energy tolerance)

    result = run(
        simulation,
        [0.0, 1.0],
        E_edges,
        mu_edges,
        Q,
        [material],
        N_iteration=8,
        N_per_bin=500,
    )

    Sigma_t = sigma_s + sigma_c
    c = sigma_s / Sigma_t
    # Bin averages of the collided flux (all bins below the source bin)
    a, b = E_edges[:-2], E_edges[1:-1]
    exact = (
        c / (Sigma_t * E0) * E0**c * (b ** (1 - c) - a ** (1 - c)) / (1 - c) / (b - a)
    )
    dmu = np.diff(mu_edges)
    phi = np.array(
        [np.einsum("kgj,j->kg", psi, dmu)[0, :-1] for psi in result.psi_history]
    )

    # Iteration 0 solves the source problem; afterwards eps~ only corrects it
    norms = np.array(result.epsilon_norm)
    assert norms[1] < 2e-2 * norms[0]

    # The iteration stagnates at the trial-space floor of the scattering correction
    # (see mcdc.rmc.driver); the stagnated iterates are unbiased
    relative = phi[2:].mean(axis=0) / exact - 1.0
    assert np.sqrt(np.mean(relative**2)) < 0.06, relative
    assert abs(np.mean(relative)) < 0.04, relative


def small_problem(tmp_path, monkeypatch, E_low=1.0):
    write_synthetic_nuclide(
        tmp_path, "HX", 1.0, [1.0e-6, 1.0e3], [1.0, 1.0], [1.0, 1.0]
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    material = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
    simulation = mcdc.Simulation("rmc-small")
    simulation.set_model([reflective_box(material)])
    simulation.set_sources(
        [mcdc.Source(position=[0.0, 0.0, 0.5], isotropic=True, energy=10.0)]
    )
    E_edges = np.array([E_low, 3.0, 9.9, 10.0])
    mu_edges = np.array([-1.0, 0.0, 1.0])
    Q = np.zeros((1, 3, 2))
    Q[0, -1, :] = 1.0 / (0.1 * 2.0)
    return simulation, material, E_edges, mu_edges, Q


@pytest.mark.parametrize("case", ["mu", "Q", "eigenvalue", "free-gas"])
def test_guards(tmp_path, monkeypatch, capsys, case):
    E_low = 1.0e-3 if case == "free-gas" else 1.0
    simulation, material, E_edges, mu_edges, Q = small_problem(
        tmp_path, monkeypatch, E_low
    )
    if case == "mu":
        mu_edges = np.array([-1.0, 0.5, 1.0])
    if case == "Q":
        Q = np.zeros((1, 2, 2))
    if case == "eigenvalue":
        simulation.settings.neutron_eigenvalue_mode = True
    with pytest.raises(SystemExit):
        run(simulation, [0.0, 1.0], E_edges, mu_edges, Q, [material], 1, 10)
    message = {
        "mu": "mu = 0 must be a bin edge",
        "Q": "Q must have shape",
        "eigenvalue": "only fixed-source",
        "free-gas": "free-gas threshold",
    }[case]
    assert message in capsys.readouterr().out


def test_reproducible(tmp_path, monkeypatch):
    results = []
    for _ in range(2):
        simulation, material, E_edges, mu_edges, Q = small_problem(
            tmp_path, monkeypatch
        )
        result = run(
            simulation, [0.0, 1.0], E_edges, mu_edges, Q, [material], 3, 50, seed=7
        )
        results.append(result.psi)
    np.testing.assert_array_equal(results[0], results[1])


def test_moment_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    results = []
    for _ in range(2):
        simulation, material, E_edges, mu_edges, Q = small_problem(
            tmp_path, monkeypatch
        )
        result = run(
            simulation,
            [0.0, 1.0],
            E_edges,
            mu_edges,
            Q,
            [material],
            2,
            50,
            seed=3,
            cache_dir=str(cache),
        )
        results.append(result.psi)
    assert len(list(cache.iterdir())) == 1
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.skipif(
    shutil.which("mpiexec", path=os.path.dirname(sys.executable)) is None,
    reason="mpiexec not available",
)
def test_mpi_rank_independence(tmp_path):
    script = os.path.join(os.path.dirname(__file__), "scripts", "mpi_small_problem.py")
    mpiexec = shutil.which("mpiexec", path=os.path.dirname(sys.executable))
    psi = []
    for N_rank in (1, 2):
        out = tmp_path / f"psi_{N_rank}.npy"
        subprocess.run(
            [
                mpiexec,
                "-n",
                str(N_rank),
                sys.executable,
                script,
                str(tmp_path),
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        psi.append(np.load(out))
    np.testing.assert_allclose(psi[0], psi[1], rtol=1e-12)
