"""
End-to-end tests of the 3D Cartesian RMC solver (mcdc.rmc.driver3d) on small
problems: against standard Monte Carlo, an analytic solution, and the slab solver.
"""

import numpy as np
import pytest

import mcdc
from mcdc.rmc.driver import run
from mcdc.rmc.driver3d import run3d

from .conftest import numba_only, write_synthetic_nuclide
from .test_driver import reflective_box, slab_model

pytestmark = numba_only

MU = np.array([-1.0, 0.0, 1.0])
PHI = np.linspace(-np.pi, np.pi, 5)  # octant patches
J = 8


def box_model(material, boundary):
    """[0, 2]^3 with the given (low, high) boundary conditions per axis."""
    planes = []
    for plane, name, (low, high) in zip(
        (mcdc.Surface.PlaneX, mcdc.Surface.PlaneY, mcdc.Surface.PlaneZ), "xyz", boundary
    ):
        planes.append(
            (
                plane(**{name: 0.0}, boundary_condition=low),
                plane(**{name: 2.0}, boundary_condition=high),
            )
        )
    (x0, x1), (y0, y1), (z0, z1) = planes
    simulation = mcdc.Simulation("rmc-box3d")
    simulation.set_model([mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=material)])
    simulation.set_sources(
        [
            mcdc.Source(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                z=[0.0, 1.0],
                isotropic=True,
                energy=np.array([[9.9, 10.0], [10.0, 10.0]]),
            )
        ]
    )
    simulation.settings.use_progress_bar = False
    return simulation


def test_absorber_box_against_smc(tmp_path, monkeypatch):
    """
    Pure absorber (Sigma_t = 1 /cm), source in the corner cell of a 2 x 2 x 2 mesh,
    reflective on x-low and vacuum elsewhere. Without scattering the residual is
    exact, so the iterates scatter without bias about the bin averages of the true
    flux: their mean must agree with SMC per cell and patch, and bins no particle can
    reach (e.g. the far corner, directions toward the source) must stay exactly 0.
    """
    import h5py

    write_synthetic_nuclide(tmp_path, "AX", 1.0, [1e-5, 1e3], [0.0, 0.0], [1.0, 1.0])
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    boundary = (("reflective", "vacuum"), ("vacuum", "vacuum"), ("vacuum", "vacuum"))
    E_edges, edges = np.array([9.9, 10.0]), np.array([0.0, 1.0, 2.0])
    area = 2.0 * np.pi / 4.0

    material = mcdc.Material(nuclide_composition={"AX": 1.0}, temperature=0.1)
    simulation = box_model(material, boundary)
    simulation.settings.N_particle = 200_000
    simulation.settings.output_name = "smc"
    simulation.settings.use_neutron_energy_window = True
    simulation.settings.neutron_energy_min = E_edges[0]
    simulation.settings.neutron_energy_max = E_edges[-1]
    mesh = mcdc.MeshStructured("box", x=edges, y=edges, z=edges)
    simulation.set_tallies(
        [mcdc.Tally(mesh=mesh, scores=["flux"], energy=E_edges, mu=MU, azi=PHI)]
    )
    simulation.run()
    with h5py.File("smc.h5", "r") as f:
        group = f["tallies"][list(f["tallies"])[0]]
        mean, sdev = group["flux/mean"][()], group["flux/sdev"][()]
    # (mu, azi, x, y, z) totals -> psi~ per unit volume, energy and solid angle
    to_psi = lambda a: np.transpose(a, (2, 3, 4, 0, 1)).reshape(2, 2, 2, 1, J) / (
        0.1 * area
    )
    psi_smc, sd_smc = to_psi(mean), to_psi(sdev)

    material = mcdc.Material(nuclide_composition={"AX": 1.0}, temperature=0.1)
    simulation = box_model(material, boundary)
    Q = np.zeros((2, 2, 2, 1, J))
    Q[0, 0, 0, 0, :] = 1.0 / (1.0 * 0.1 * 4.0 * np.pi)
    result = run3d(
        simulation,
        edges,
        edges,
        edges,
        E_edges,
        MU,
        PHI,
        Q,
        material,
        N_iteration=8,
        N_per_bin=1000,
        boundary=boundary,
        cache_dir=str(tmp_path / "cache"),
        N_theta=129,
    )
    norms = np.array(result.epsilon_norm)
    assert np.all(norms[1:] < 0.1 * norms[0]), norms
    psi, error = result.psi_average(2)
    unreached = psi_smc == 0.0
    assert np.all(psi[unreached] == 0.0)
    z = (psi - psi_smc)[~unreached] / np.sqrt(sd_smc**2 + error**2)[~unreached]
    print("absorber box: z rms", np.sqrt(np.mean(z**2)), "max", np.max(np.abs(z)))
    assert np.sqrt(np.mean(z**2)) < 2.0, z
    assert np.all(np.abs(z) < 5.0), z


def test_infinite_medium_A1(tmp_path, monkeypatch):
    """
    A = 1, sigma_s = sigma_c = 1 b, monoenergetic source in a reflective box: an
    infinite medium. psi~ is isotropic and the face residual vanishes, so the
    iteration converges exponentially to the binned fixed point, within the bin-shape
    floor of the analytic flux phi(E) = c S0 / (Sigma_t E0) (E0 / E)^c (as in the
    slab test, test_analytic_slowing_down_A1).
    """
    E0, delta = 101.0, 0.1
    write_synthetic_nuclide(
        tmp_path, "HX", 1.0, [1.0e-5, 1.0e3], [1.0, 1.0], [1.0, 1.0]
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    material = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
    simulation = mcdc.Simulation("rmc-infinite-3d")
    simulation.set_model([reflective_box(material)])
    simulation.set_sources(
        [mcdc.Source(position=[0.0, 0.0, 0.5], isotropic=True, energy=E0)]
    )
    simulation.settings.use_progress_bar = False
    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 21), [E0]))
    G = len(E_edges) - 1
    Q = np.zeros((1, 1, 1, G, J))
    Q[0, 0, 0, -1, :] = 1.0 / (delta * 4.0 * np.pi)  # S0 = 1 per unit volume
    result = run3d(
        simulation,
        [-0.5, 0.5],
        [-0.5, 0.5],
        [0.0, 1.0],
        E_edges,
        MU,
        PHI,
        Q,
        material,
        N_iteration=8,
        N_per_bin=200,
        cache_dir=str(tmp_path / "cache"),
        N_theta=513,
    )
    norms = np.array(result.epsilon_norm)
    # Geometric, about 10x per iteration at 200 histories per (bin, patch)
    assert np.all(np.diff(norms[:6]) < 0.0), norms
    assert norms[7] < 1e-6 * norms[0], norms
    Sigma_t, c = 2.0, 0.5
    a, b = E_edges[:-2], E_edges[1:-1]
    exact = (
        c / (Sigma_t * E0) * E0**c * (b ** (1 - c) - a ** (1 - c)) / (1 - c) / (b - a)
    )
    phi = result.scalar_flux()[0, 0, 0, :-1]
    print("infinite medium: max rel error", np.max(np.abs(phi / exact - 1)))
    assert np.max(np.abs(phi / exact - 1.0)) < 3e-3


def test_slab_matches_slab_solver(tmp_path, monkeypatch):
    """
    The two-material slab of test_slab_against_smc, built in 3D (one cell in x and y,
    reflective there). For a slab-symmetric problem the 3D constant-basis system has
    the slab solver's fixed point (same mu bins, isotropic in phi, no x/y jumps), so
    the averaged plateau iterates of the two solvers must agree within statistics.
    """
    write_synthetic_nuclide(
        tmp_path, "HX", 1.0, [1.0e-6, 1.0e3], [1.0, 1.0], [1.0, 1.0]
    )
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    E0, delta = 10.0, 0.1
    z_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 9), [E0]))
    K, G = 4, len(E_edges) - 1

    def materials():
        a = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
        b = mcdc.Material(nuclide_composition={"HX": 0.4}, temperature=0.1)
        return a, b

    # Slab solver (collision-only iterations)
    a, b = materials()
    simulation = slab_model(a, b, E0, delta, False)
    simulation.settings.use_progress_bar = False
    Q1 = np.zeros((K, G, 2))
    Q1[0, -1, :] = 1.0 / (1.0 * delta * 2.0)
    slab = run(
        simulation,
        z_edges,
        E_edges,
        MU,
        Q1,
        [a, a, b, b],
        10,
        400,
        boundary=("vacuum", "vacuum"),
    )
    history = np.array(
        [np.einsum("kgj,j->kg", p, np.diff(MU)) for p in slab.psi_history[2:]]
    )
    phi_1d = history.mean(axis=0)
    se_1d = history.std(axis=0, ddof=1) / np.sqrt(len(history))

    # 3D solver on the same slab
    a, b = materials()
    simulation = slab_model(a, b, E0, delta, False)
    simulation.settings.use_progress_bar = False
    Q3 = np.zeros((1, 1, K, G, J))
    Q3[0, 0, 0, -1, :] = 1.0 / (1.0 * delta * 4.0 * np.pi)
    cells = np.array([a, a, b, b], dtype=object).reshape(1, 1, K)
    result = run3d(
        simulation,
        [-0.5, 0.5],
        [-0.5, 0.5],
        z_edges,
        E_edges,
        MU,
        PHI,
        Q3,
        cells,
        N_iteration=10,
        N_per_bin=100,
        boundary=(("reflective",) * 2, ("reflective",) * 2, ("vacuum", "vacuum")),
        cache_dir=str(tmp_path / "cache"),
        N_theta=513,
    )
    history = np.array([result.scalar_flux(p)[0, 0] for p in result.psi_history[2:]])
    phi_3d = history.mean(axis=0)
    se_3d = history.std(axis=0, ddof=1) / np.sqrt(len(history))
    z = (phi_3d - phi_1d) / np.sqrt(se_1d**2 + se_3d**2)
    print(
        "slab: z rms",
        np.sqrt(np.mean(z**2)),
        "max",
        np.max(np.abs(z)),
        "max rel",
        np.max(np.abs(phi_3d / phi_1d - 1)),
    )
    assert np.sqrt(np.mean(z**2)) < 2.0, z
    assert np.all(np.abs(z) < 5.0), z
