"""
Verification of the 3D Cartesian RMC solver (mcdc.rmc.run3d, piecewise-constant trial
space, collision-only iterations) against standard Monte Carlo (SMC) with MC/DC on the
same model, and against analytic references. See README.md and
mcdc/rmc/writeups/cartesian_3d/cartesian_3d.tex.

  1. infinite: A = 1 slowing down in a reflective box (an infinite medium), sigma_s =
     sigma_c = 1 b, monoenergetic isotropic source; analytic flux
     phi(E) = c S0 / (Sigma_t E0) (E0 / E)^c.
  2. absorber: pure absorber (Sigma_t = 1 /cm) in [0, 2]^3 on a 2 x 2 x 2 mesh with
     octant patches, source in the corner cell [0, 1]^3, reflective on x = 0 and
     vacuum elsewhere; reference: the exact ray integral of the source (`reference_absorber`).
  3. slab: the two-material slab of the 1D tests built in 3D (reflective in x and y);
     compared with the slab solver (mcdc.rmc.run) and SMC.

Writes output.h5; then python plot.py.
    python input.py --mode=numba --no-progress_bar
    mpiexec -n 4 python input.py --mode=numba --no-progress_bar
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

HERE = os.path.dirname(os.path.abspath(__file__))
MU = np.array([-1.0, 0.0, 1.0])
PHI = np.linspace(-np.pi, np.pi, 5)  # octant patches
J = 8
N_THETA = 2049
N_ITERATION = 10


def box(material, bounds, boundary):
    """Rectangular box cell; bounds ((x0, x1), (y0, y1), (z0, z1)), boundary per side."""
    import mcdc

    planes = []
    for plane, name, (low, high), (bc_low, bc_high) in zip(
        (mcdc.Surface.PlaneX, mcdc.Surface.PlaneY, mcdc.Surface.PlaneZ),
        "xyz",
        bounds,
        boundary,
    ):
        planes.append(
            (
                plane(**{name: low}, boundary_condition=bc_low),
                plane(**{name: high}, boundary_condition=bc_high),
            )
        )
    (x0, x1), (y0, y1), (z0, z1) = planes
    return mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=material)


def smc_tally(make_model, edges, E_edges, N_particle, name, angular=True):
    """SMC flux per (cell, energy, patch) bin, per unit volume, energy and solid angle."""
    import mcdc

    simulation = make_model()
    settings = simulation.settings
    settings.N_particle = int(N_particle)
    settings.use_progress_bar = False
    settings.use_neutron_energy_window = True
    settings.neutron_energy_min = E_edges[0]
    settings.neutron_energy_max = E_edges[-1]
    settings.output_name = f"{name}-tmp"
    mesh = mcdc.MeshStructured(f"{name}-mesh", x=edges[0], y=edges[1], z=edges[2])
    kwargs = dict(mu=MU, azi=PHI) if angular else {}
    simulation.set_tallies(
        [mcdc.Tally(mesh=mesh, scores=["flux"], energy=E_edges, **kwargs)]
    )
    simulation.run()
    if not common.is_master():
        return None, None
    with h5py.File(f"{name}-tmp.h5", "r") as f:
        group = f["tallies"][list(f["tallies"])[0]]
        mean, sdev = group["flux/mean"][()], group["flux/sdev"][()]
    os.remove(f"{name}-tmp.h5")
    shape = [len(e) - 1 for e in edges]
    G = len(E_edges) - 1
    if angular:
        full = (len(MU) - 1, len(PHI) - 1, G, *shape)
        order = (3, 4, 5, 2, 0, 1)
    else:
        full = (G, *shape)
        order = (1, 2, 3, 0)
    mean, sdev = (np.transpose(a.reshape(full), order) for a in (mean, sdev))
    volume = np.einsum("i,j,k->ijk", *(np.diff(e) for e in edges))
    measure = volume[..., None] * np.diff(E_edges)
    if angular:
        mean = mean.reshape(*shape, G, J)
        sdev = sdev.reshape(*shape, G, J)
        area = np.full(J, 2.0 * np.pi / 4.0)
        measure = measure[..., None] * area
    return mean / measure, sdev / measure


# ======================================================================================
# 1. Infinite medium, A = 1
# ======================================================================================


def infinite(output):
    import mcdc
    from mcdc.rmc.driver3d import run3d

    E0, delta = 101.0, 0.1
    if common.is_master():
        common.write_nuclide("data", "HX", 1.0, [1e-5, 1e3], [1.0, 1.0], [1.0, 1.0])
    common.barrier()
    common.use_library("data")
    bounds = ((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0))
    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 41), [E0]))
    G = len(E_edges) - 1

    def make_model():
        material = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
        simulation = mcdc.Simulation("verification-infinite")
        simulation.set_model([box(material, bounds, (("reflective",) * 2,) * 3)])
        simulation.set_sources(
            [mcdc.Source(position=[0.0, 0.0, 0.5], isotropic=True, energy=E0)]
        )
        simulation.settings.use_progress_bar = False
        return simulation, material

    phi_smc, sd_smc = smc_tally(
        lambda: make_model()[0], bounds, E_edges, 1e6, "infinite", angular=False
    )
    simulation, material = make_model()
    Q = np.zeros((1, 1, 1, G, J))
    Q[0, 0, 0, -1, :] = 1.0 / (delta * 4.0 * np.pi)  # S0 = 1 per unit volume
    result = run3d(
        simulation,
        *bounds,
        E_edges,
        MU,
        PHI,
        Q,
        material,
        N_iteration=N_ITERATION,
        N_per_bin=100,
        cache_dir="cache",
        N_theta=N_THETA,
    )
    if common.is_master():
        Sigma_t, c = 2.0, 0.5
        a, b = E_edges[:-1], E_edges[1:]
        analytic = (
            c
            / (Sigma_t * E0)
            * E0**c
            * (b ** (1 - c) - a ** (1 - c))
            / (1 - c)
            / (b - a)
        )
        analytic[-1] = np.nan  # the source bin (uncollided source) is not compared
        with output.write() as f:
            group = f.create_group("infinite")
            result.write(group.create_group("rmc"))
            group.create_dataset("smc", data=phi_smc[0, 0, 0])
            group.create_dataset("smc_sdev", data=sd_smc[0, 0, 0])
            group.create_dataset("analytic", data=analytic)


# ======================================================================================
# 2. Pure absorber box
# ======================================================================================


def reference_absorber(edges, mu_edges, phi_edges, Sigma, q, N_points=2**20, seed=7):
    """
    Bin averages of the exact angular flux of a pure absorber with a uniform isotropic
    source q (per unit volume and solid angle) in [0, 1]^3, inside [0, 2]^3 with a
    reflective plane x = 0 and vacuum elsewhere:

        psi(r, Omega) = q (exp(-Sigma s1) - exp(-Sigma s2)) / Sigma,

    with [s1, s2] the part of the backward ray r - s Omega inside the source. The
    reflection is unfolded: the ray continues to x < 0, where the image source is
    [-1, 0] x [0, 1]^2 and the image domain extends to x = -2. The average over each
    cell and patch (dOmega = dmu dphi) is a scrambled Sobol quasi-Monte Carlo mean,
    returned with the spread of two independent scrambles as its error.
    """
    from scipy.stats import qmc

    J_mu, J_phi = len(mu_edges) - 1, len(phi_edges) - 1
    shape = tuple(len(e) - 1 for e in edges)
    estimates = []
    for scramble in range(2):
        sobol = qmc.Sobol(d=5, scramble=True, seed=seed + scramble)
        u = sobol.random(N_points)
        values = np.zeros(shape + (J_mu * J_phi,))
        for kx, ky, kz in np.ndindex(shape):
            point = np.stack(
                [
                    edges[d][k] + u[:, d] * (edges[d][k + 1] - edges[d][k])
                    for d, k in enumerate((kx, ky, kz))
                ]
            )
            for jm in range(J_mu):
                for jf in range(J_phi):
                    mu = mu_edges[jm] + u[:, 3] * (mu_edges[jm + 1] - mu_edges[jm])
                    phi = phi_edges[jf] + u[:, 4] * (phi_edges[jf + 1] - phi_edges[jf])
                    s = np.sqrt(1.0 - mu**2)
                    omega = np.stack((s * np.cos(phi), s * np.sin(phi), mu))
                    values[kx, ky, kz, jm * J_phi + jf] = np.mean(
                        _absorber_flux(point, omega, Sigma, q)
                    )
        estimates.append(values)
    return 0.5 * (estimates[0] + estimates[1]), 0.5 * np.abs(
        estimates[0] - estimates[1]
    )


def _absorber_flux(point, omega, Sigma, q):
    """psi at points r along directions Omega (unfolded backward ray, see above)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # Backward ray r - s Omega; the domain (unfolded) is [-2, 2] x [0, 2]^2
        exit = np.full(point.shape[1], np.inf)
        start = np.zeros(point.shape[1])
        end = np.full(point.shape[1], np.inf)
        for d, (low, high, src_low, src_high) in enumerate(
            ((-2.0, 2.0, -1.0, 1.0), (0.0, 2.0, 0.0, 1.0), (0.0, 2.0, 0.0, 1.0))
        ):
            v = -omega[d]
            t_low = (low - point[d]) / v
            t_high = (high - point[d]) / v
            exit = np.minimum(
                exit, np.where(v > 0, t_high, np.where(v < 0, t_low, np.inf))
            )
            a = (src_low - point[d]) / v
            b = (src_high - point[d]) / v
            inside = (point[d] >= src_low) & (point[d] <= src_high)
            lo = np.where(v != 0, np.minimum(a, b), np.where(inside, -np.inf, np.inf))
            hi = np.where(v != 0, np.maximum(a, b), np.where(inside, np.inf, -np.inf))
            start = np.maximum(start, lo)
            end = np.minimum(end, hi)
        end = np.minimum(end, exit)
        start = np.maximum(start, 0.0)
        hit = end > start
        end = np.where(hit, end, start)  # masked bins: no exp of -inf
        return np.where(
            hit, q * (np.exp(-Sigma * start) - np.exp(-Sigma * end)) / Sigma, 0.0
        )


def absorber(output):
    import mcdc
    from mcdc.rmc.driver3d import run3d

    if common.is_master():
        common.write_nuclide("data", "AX", 1.0, [1e-5, 1e3], [0.0, 0.0], [1.0, 1.0])
    common.barrier()
    common.use_library("data")
    edges = (np.array([0.0, 1.0, 2.0]),) * 3
    bounds = ((0.0, 2.0),) * 3
    boundary = (("reflective", "vacuum"), ("vacuum", "vacuum"), ("vacuum", "vacuum"))
    E_edges = np.array([9.9, 10.0])
    dE = 0.1

    def make_model():
        material = mcdc.Material(nuclide_composition={"AX": 1.0}, temperature=0.1)
        simulation = mcdc.Simulation("verification-absorber")
        simulation.set_model([box(material, bounds, boundary)])
        simulation.set_sources(
            [
                mcdc.Source(
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    z=[0.0, 1.0],
                    isotropic=True,
                    energy=np.array([[9.9, 10.0], [1.0 / dE, 1.0 / dE]]),
                )
            ]
        )
        simulation.settings.use_progress_bar = False
        return simulation, material

    psi_smc, sd_smc = smc_tally(
        lambda: make_model()[0], edges, E_edges, 1e6, "absorber"
    )
    simulation, material = make_model()
    q = 1.0 / (1.0 * dE * 4.0 * np.pi)  # S0 = 1, per unit volume, energy, solid angle
    Q = np.zeros((2, 2, 2, 1, J))
    Q[0, 0, 0, 0, :] = q
    result = run3d(
        simulation,
        *edges,
        E_edges,
        MU,
        PHI,
        Q,
        material,
        N_iteration=N_ITERATION,
        N_per_bin=2000,
        boundary=boundary,
        cache_dir="cache",
        N_theta=N_THETA,
    )
    if common.is_master():
        analytic, analytic_error = reference_absorber(edges, MU, PHI, 1.0, q)
        with output.write() as f:
            group = f.create_group("absorber")
            result.write(group.create_group("rmc"))
            group.create_dataset("smc", data=psi_smc[..., 0, :])
            group.create_dataset("smc_sdev", data=sd_smc[..., 0, :])
            group.create_dataset("analytic", data=analytic)
            group.create_dataset("analytic_error", data=analytic_error)


# ======================================================================================
# 3. Two-material slab, in 3D and in the slab solver
# ======================================================================================


def slab(output):
    import mcdc
    from mcdc.rmc.driver import run
    from mcdc.rmc.driver3d import run3d

    if common.is_master():
        common.write_nuclide("data", "HX", 1.0, [1e-6, 1e3], [1.0, 1.0], [1.0, 1.0])
    common.barrier()
    common.use_library("data")
    E0, delta = 10.0, 0.1
    z_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 9), [E0]))
    K, G = 4, len(E_edges) - 1
    xy = np.array([-0.5, 0.5])

    def make_model():
        a = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
        b = mcdc.Material(nuclide_composition={"HX": 0.4}, temperature=0.1)
        x0 = mcdc.Surface.PlaneX(x=-0.5, boundary_condition="reflective")
        x1 = mcdc.Surface.PlaneX(x=0.5, boundary_condition="reflective")
        y0 = mcdc.Surface.PlaneY(y=-0.5, boundary_condition="reflective")
        y1 = mcdc.Surface.PlaneY(y=0.5, boundary_condition="reflective")
        z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
        z2 = mcdc.Surface.PlaneZ(z=2.0)
        z4 = mcdc.Surface.PlaneZ(z=4.0, boundary_condition="vacuum")
        column = +x0 & -x1 & +y0 & -y1
        simulation = mcdc.Simulation("verification-slab")
        simulation.set_model(
            [
                mcdc.Cell(column & +z0 & -z2, fill=a),
                mcdc.Cell(column & +z2 & -z4, fill=b),
            ]
        )
        simulation.set_sources(
            [
                mcdc.Source(
                    x=[-0.5, 0.5],
                    y=[-0.5, 0.5],
                    z=[0.0, 1.0],
                    isotropic=True,
                    energy=np.array([[E0 - delta, E0], [1.0 / delta, 1.0 / delta]]),
                )
            ]
        )
        simulation.settings.use_progress_bar = False
        return simulation, a, b

    phi_smc, sd_smc = smc_tally(
        lambda: make_model()[0], (xy, xy, z_edges), E_edges, 1e6, "slab", angular=False
    )

    simulation, a, b = make_model()
    Q1 = np.zeros((K, G, 2))
    Q1[0, -1, :] = 1.0 / (1.0 * delta * 2.0)
    slab_1d = run(
        simulation,
        z_edges,
        E_edges,
        MU,
        Q1,
        [a, a, b, b],
        N_ITERATION,
        400,
        boundary=("vacuum", "vacuum"),
        cache_dir="cache",
    )

    simulation, a, b = make_model()
    Q3 = np.zeros((1, 1, K, G, J))
    Q3[0, 0, 0, -1, :] = 1.0 / (1.0 * delta * 4.0 * np.pi)
    slab_3d = run3d(
        simulation,
        xy,
        xy,
        z_edges,
        E_edges,
        MU,
        PHI,
        Q3,
        np.array([a, a, b, b], dtype=object).reshape(1, 1, K),
        N_iteration=N_ITERATION,
        N_per_bin=100,
        boundary=(("reflective",) * 2, ("reflective",) * 2, ("vacuum", "vacuum")),
        cache_dir="cache",
        N_theta=N_THETA,
    )
    if common.is_master():
        with output.write() as f:
            group = f.create_group("slab")
            slab_3d.write(group.create_group("rmc3d"))
            history = np.array(
                [np.einsum("kgj,j->kg", p, np.diff(MU)) for p in slab_1d.psi_history]
            )
            group.create_dataset("rmc1d_phi_history", data=history)
            group.create_dataset("rmc1d_epsilon_norm", data=slab_1d.epsilon_norm)
            group.create_dataset("smc", data=phi_smc[0, 0])
            group.create_dataset("smc_sdev", data=sd_smc[0, 0])


if __name__ == "__main__":
    os.chdir(HERE)
    output = common.open_output("output.h5")
    for case in os.environ.get("VERIFICATION_CASES", "infinite,absorber,slab").split(
        ","
    ):
        {"infinite": infinite, "absorber": absorber, "slab": slab}[case](output)
    output.close()
