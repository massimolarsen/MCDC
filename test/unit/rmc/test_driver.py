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
        N_correction=12,
    )

    Sigma_t = sigma_s + sigma_c
    c = sigma_s / Sigma_t
    # Bin averages of the collided flux (all bins below the source bin)
    a, b = E_edges[:-2], E_edges[1:-1]
    exact = (
        c / (Sigma_t * E0) * E0**c * (b ** (1 - c) - a ** (1 - c)) / (1 - c) / (b - a)
    )
    dmu = np.diff(mu_edges)

    def flux(psi):
        return np.einsum("kgj,j->kg", psi, dmu)[0, :-1]

    # Phase 1: exponential convergence of the collision-only iteration
    norms = np.array(result.epsilon_norm)
    assert norms[5] < 1e-6 * norms[0], norms
    # ... to the binned fixed point, whose bias is the bin-shape floor (G-dependent)
    phi_fixed = flux(result.psi_fixed_point)
    assert np.max(np.abs(phi_fixed / exact - 1.0)) < 3e-3

    # Phase 2: the averaged corrections estimate exact - fixed point without bias
    #   (12 passes: the correction weights are heavy-tailed, so fewer passes give an
    #   unreliable standard error)
    corrections = np.array([flux(eps) for eps in result.corrections])
    mean = corrections.mean(axis=0)
    error = corrections.std(axis=0, ddof=1) / np.sqrt(len(corrections))
    z = (mean - (exact - phi_fixed)) / error
    assert np.sqrt(np.mean(z**2)) < 2.5, z


def small_problem(tmp_path, monkeypatch, E_low=1.0, anisotropic=False):
    path = write_synthetic_nuclide(
        tmp_path, "HX", 1.0, [1.0e-6, 1.0e3], [1.0, 1.0], [1.0, 1.0]
    )
    if anisotropic:
        # Linearly anisotropic COM elastic scattering, p(mu) = (1 + mu / 2) / 2
        import h5py

        with h5py.File(path, "r+") as f:
            pdf = f["neutron_reactions/elastic_scattering/MT-002"][
                "angular_cosine_distribution/pdf"
            ]
            pdf[...] = [0.25, 0.75, 0.25, 0.75]
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
        tmp_path, monkeypatch, E_low, anisotropic=case == "free-gas"
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
        "free-gas": "anisotropic free-gas kernel is not supported",
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


def test_analytic_slowing_down_inverse_sqrt(tmp_path, monkeypatch):
    """
    A = 1 pure scatterer with Sigma_s = Sigma_t = E^(-1/2) (energy-dependent within
    every trial bin): below the source, phi(E) = S0 / (Sigma_s(E) E) = S0 E^(-1/2).
    """
    E0, delta = 101.0, 0.1
    energy = np.logspace(-6.0, 3.0, 2000)
    write_synthetic_nuclide(
        tmp_path, "HY", 1.0, energy, energy**-0.5, np.zeros_like(energy)
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))

    material = mcdc.Material(nuclide_composition={"HY": 1.0}, temperature=0.1)
    simulation = mcdc.Simulation("rmc-analytic-sqrt")
    simulation.set_model([reflective_box(material)])
    simulation.set_sources(
        [mcdc.Source(position=[0.0, 0.0, 0.5], isotropic=True, energy=E0)]
    )

    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 21), [E0]))
    mu_edges = np.array([-1.0, 0.0, 1.0])
    G = len(E_edges) - 1
    Q = np.zeros((1, G, 2))
    Q[0, -1, :] = 1.0 / (delta * 2.0)

    result = run(simulation, [0.0, 1.0], E_edges, mu_edges, Q, [material], 8, 500)

    a, b = E_edges[:-2], E_edges[1:-1]
    exact = 2.0 * (np.sqrt(b) - np.sqrt(a)) / (b - a)
    phi = np.einsum("kgj,j->kg", result.psi, np.diff(mu_edges))[0, :-1]
    norms = np.array(result.epsilon_norm)
    print("sqrt: eps", norms, "max rel", np.max(np.abs(phi / exact - 1.0)))
    assert norms[5] < 1e-4 * norms[0], norms
    assert np.max(np.abs(phi / exact - 1.0)) < 1e-2


def slab_model(material_a, material_b, E0, delta, smc):
    """z in [0, 4]: material a in [0, 2], b in [2, 4]; reflective x/y, vacuum z."""
    x0 = mcdc.Surface.PlaneX(x=-0.5, boundary_condition="reflective")
    x1 = mcdc.Surface.PlaneX(x=0.5, boundary_condition="reflective")
    y0 = mcdc.Surface.PlaneY(y=-0.5, boundary_condition="reflective")
    y1 = mcdc.Surface.PlaneY(y=0.5, boundary_condition="reflective")
    z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
    z2 = mcdc.Surface.PlaneZ(z=2.0)
    z4 = mcdc.Surface.PlaneZ(z=4.0, boundary_condition="vacuum")
    box = +x0 & -x1 & +y0 & -y1
    simulation = mcdc.Simulation("rmc-slab")
    simulation.set_model(
        [
            mcdc.Cell(box & +z0 & -z2, fill=material_a),
            mcdc.Cell(box & +z2 & -z4, fill=material_b),
        ]
    )
    # Isotropic, uniform in z in [0, 1], uniform in energy over [E0 - delta, E0]
    energy = np.array([[E0 - delta, E0], [1.0 / delta, 1.0 / delta]])
    simulation.set_sources(
        [
            mcdc.Source(
                x=[-0.5, 0.5],
                y=[-0.5, 0.5],
                z=[0.0, 1.0],
                isotropic=True,
                energy=energy,
            )
        ]
    )
    return simulation


def test_slab_against_smc(tmp_path, monkeypatch):
    """1D two-material slab with vacuum boundaries: RMC (stagnated iterates) vs SMC."""
    import h5py

    write_synthetic_nuclide(
        tmp_path, "HX", 1.0, [1.0e-6, 1.0e3], [1.0, 1.0], [1.0, 1.0]
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    E0, delta = 10.0, 0.1
    z_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 9), [E0]))
    mu_edges = np.array([-1.0, 0.0, 1.0])
    K, G = 4, len(E_edges) - 1

    # Standard Monte Carlo reference
    material_a = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
    material_b = mcdc.Material(nuclide_composition={"HX": 0.4}, temperature=0.1)
    simulation = slab_model(material_a, material_b, E0, delta, True)
    simulation.settings.N_particle = 200_000
    simulation.settings.use_neutron_energy_window = True
    simulation.settings.neutron_energy_min = E_edges[0]
    simulation.settings.neutron_energy_max = E_edges[-1]
    simulation.settings.output_name = "smc"
    mesh = mcdc.MeshStructured("slab", z=z_edges)
    simulation.set_tallies([mcdc.Tally(mesh=mesh, scores=["flux"], energy=E_edges)])
    simulation.run()
    with h5py.File("smc.h5", "r") as f:
        group = f["tallies"][list(f["tallies"])[0]]
        mean = np.squeeze(group["flux/mean"][()])
        sdev = np.squeeze(group["flux/sdev"][()])
    # (energy, z) -> per unit z and energy
    width = np.diff(z_edges)[None, :] * np.diff(E_edges)[:, None]
    phi_smc = (mean / width).T
    sd_smc = (sdev / width).T

    # RMC
    material_a = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
    material_b = mcdc.Material(nuclide_composition={"HX": 0.4}, temperature=0.1)
    simulation = slab_model(material_a, material_b, E0, delta, False)
    Q = np.zeros((K, G, 2))
    Q[0, -1, :] = 1.0 / (1.0 * delta * 2.0)
    result = run(
        simulation,
        z_edges,
        E_edges,
        mu_edges,
        Q,
        [material_a, material_a, material_b, material_b],
        8,
        200,
        N_correction=8,
        boundary=("vacuum", "vacuum"),
    )
    norms = np.array(result.epsilon_norm)
    dmu = np.diff(mu_edges)
    phi_rmc = np.einsum("kgj,j->kg", result.psi, dmu)
    # The collision-only fixed point is biased (binned in-scatter, flat in-cell
    #   shape); the averaged full-residual corrections remove that without bias, so
    #   compare with the corrections' own standard error and SMC's combined
    corrections = np.array([np.einsum("kgj,j->kg", e, dmu) for e in result.corrections])
    se_rmc = corrections.std(axis=0, ddof=1) / np.sqrt(len(corrections))
    z = (phi_rmc - phi_smc) / np.sqrt(sd_smc**2 + se_rmc**2)
    print(
        "slab: eps",
        norms,
        "z rms",
        np.sqrt(np.mean(z**2)),
        "max rel",
        np.max(np.abs(phi_rmc / phi_smc - 1)),
    )
    # In 1D the face residual keeps an in-cell shape, so ||eps~|| drops in the first
    #   iteration and then plateaus at the noise of one iteration (NOTES.md, 2b)
    assert np.all(norms[1:] < 0.1 * norms[0]), norms
    assert np.sqrt(np.mean(z**2)) < 2.0, z
    assert np.all(np.abs(z) < 5.0), z
