"""
Shared helpers for the RMC examples that reproduce the figures of

  [D] M. Larsen, Continuous Energy Residual Monte Carlo, PhD dissertation,
      Oregon State University (2025);
  [N] the NSE article "Continuous Energy Residual Monte Carlo".

Every problem directory has an input.py (runs RMC and the references, writes
output.h5) and a plot.py (reads output.h5, writes the figures).

All RMC runs use the scheme of the papers: collision-only iterations with the binned
in-scatter and bin-averaged cross sections, and no correction passes (N_correction =
0). See mcdc/rmc/NOTES.md.

Units: energies in eV. psi~ and SMC tallies are per unit z (cm), energy (eV) and polar
cosine about z, per unit x-y area, per source neutron.
"""

import os
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MU_EDGES = np.array([-1.0, 0.0, 1.0])


# ======================================================================================
# Nuclear data
# ======================================================================================


def real_library():
    """Path of the MC/DC library with real (ENDF) data; regenerated with the MT-5 fix."""
    if "MCDC_LIB" not in os.environ:
        raise RuntimeError("Set MCDC_LIB to the MC/DC HDF5 data library")
    return os.environ["MCDC_LIB"]


def use_library(path):
    """Point MC/DC at a library directory (read when materials are created)."""
    os.environ["MCDC_LIB"] = os.path.abspath(path)


def read_xs(name, temperature, reaction, library=None):
    """Microscopic cross section (energy [eV], xs [b]) summed over a reaction group."""
    path = os.path.join(library or real_library(), f"{name}-{temperature}K.h5")
    with h5py.File(path, "r") as f:
        energy = f["neutron_reactions/xs_energy_grid"][()] * 1e6
        xs = np.zeros_like(energy)
        if reaction in f["neutron_reactions"]:
            for group in f[f"neutron_reactions/{reaction}"].values():
                if not isinstance(group, h5py.Group) or "xs" not in group:
                    continue
                offset = int(group["xs"].attrs["offset"])
                values = group["xs"][()]
                xs[offset : offset + len(values)] += values
    return energy, xs


def read_total_xs(name, temperature, library=None):
    """Microscopic total cross section (energy [eV], xs [b]): the sum of all reactions."""
    total = None
    for reaction in (
        "elastic_scattering",
        "capture",
        "inelastic_scattering",
        "fission",
    ):
        energy, xs = read_xs(name, temperature, reaction, library)
        total = xs if total is None else total + xs
    return energy, total


def read_atomic_weight_ratio(name, temperature, library=None):
    path = os.path.join(library or real_library(), f"{name}-{temperature}K.h5")
    with h5py.File(path, "r") as f:
        return float(f["atomic_weight_ratio"][()])


def _tabulated_distribution(group, values, pdf, energy_MeV):
    """Energy-independent tabulated distribution, repeated at two incident energies."""
    group.attrs["type"] = "tabulated"
    N = len(values)
    group.create_dataset("energy", data=np.asarray(energy_MeV)[[0, -1]])
    group.create_dataset("offset", data=np.array([0, N]))
    group.create_dataset("value", data=np.tile(values, 2))
    group.create_dataset("pdf", data=np.tile(pdf, 2))


def write_nuclide(
    directory,
    name,
    A,
    energy,
    elastic,
    capture,
    fission=None,
    nu=2.0,
    spectrum=None,
):
    """
    Synthetic nuclide in the MC/DC library format, at 0.1 K: elastic scattering
    isotropic in the COM frame (target at rest above 400 kT = 3.4 meV), capture, and
    optionally fission with a constant multiplicity and an energy-independent
    tabulated spectrum (energy [eV], pdf [/eV]).

    Cross sections [b] are tabulated on `energy` [eV] and interpolated linearly, as
    MC/DC does. Written as <name>-0.1K.h5.
    """
    energy_MeV = np.asarray(energy, dtype=float) * 1e-6
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}-0.1K.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("nuclide_name", data=name)
        f.create_dataset("excitation_level", data=0)
        f.create_dataset("temperature", data=0.1).attrs["unit"] = "K"
        f.create_dataset("atomic_number", data=0)
        f.create_dataset("mass_number", data=int(round(A)))
        f.create_dataset("atomic_weight_ratio", data=float(A))
        f.create_dataset("fissionable", data=fission is not None)
        reactions = f.create_group("neutron_reactions")
        reactions.create_dataset("xs_energy_grid", data=energy_MeV).attrs["unit"] = (
            "MeV"
        )

        def basic(group_name, MT, xs, frame):
            group = reactions.create_group(f"{group_name}/MT-{MT:03}")
            group.attrs["MT"] = MT
            dataset = group.create_dataset("xs", data=np.asarray(xs, dtype=float))
            dataset.attrs["offset"] = 0
            group.create_dataset("reference_frame", data=frame)
            group.create_dataset("Q-value", data=0.0)
            return group

        group = basic("elastic_scattering", 2, elastic, "COM")
        _tabulated_distribution(
            group.create_group("angular_cosine_distribution"),
            np.array([-1.0, 1.0]),
            np.array([0.5, 0.5]),
            energy_MeV,
        )
        basic("capture", 102, capture, "LAB")

        if fission is not None:
            group = basic("fission", 18, fission, "LAB")
            group.create_group("angular_cosine_distribution").attrs[
                "type"
            ] = "isotropic"
            values, pdf = spectrum
            spectrum_MeV = (np.asarray(values) * 1e-6, np.asarray(pdf) * 1e6)
            _tabulated_distribution(
                group.create_group("energy_spectrum-1"), *spectrum_MeV, energy_MeV
            )
            group.create_dataset("spectrum_probability", data=np.ones((1, 1)))
            group.create_dataset("spectrum_probability_grid", data=[0.0, 30.0])

            parent = reactions["fission"]
            for kind, value in (("prompt", nu), ("delayed", 0.0)):
                multiplicity = parent.create_group(f"{kind}_multiplicity")
                multiplicity.attrs["type"] = "tabulated"
                multiplicity.create_dataset("energy", data=energy_MeV[[0, -1]])
                multiplicity.create_dataset("value", data=[value, value])
            precursors = parent.create_group("delayed_neutron_precursors")
            precursors.create_dataset("fractions", data=[1.0])
            precursors.create_dataset("decay_rates", data=[1.0])
            _tabulated_distribution(
                precursors.create_group("energy_spectrum-1"), *spectrum_MeV, energy_MeV
            )
    return path


def watt_spectrum(a, b, E_min, E_max, N=4000):
    """Watt spectrum chi(E) ~ exp(-E/a) sinh(sqrt(bE)), normalized on [E_min, E_max]."""
    E = np.logspace(np.log10(E_min), np.log10(E_max), N)
    pdf = np.exp(-E / a) * np.sinh(np.sqrt(b * E))
    pdf /= np.trapezoid(pdf, E)
    return E, pdf


# ======================================================================================
# Models
# ======================================================================================


def slab(cells, z_boundary=("reflective", "reflective")):
    """
    Slab along z, reflective in x and y (unit width): cells = [(z_low, z_high,
    material), ...]. A single reflective cell is an infinite medium (0D).
    """
    import mcdc

    x0 = mcdc.Surface.PlaneX(x=-0.5, boundary_condition="reflective")
    x1 = mcdc.Surface.PlaneX(x=0.5, boundary_condition="reflective")
    y0 = mcdc.Surface.PlaneY(y=-0.5, boundary_condition="reflective")
    y1 = mcdc.Surface.PlaneY(y=0.5, boundary_condition="reflective")
    edges = [cells[0][0]] + [cell[1] for cell in cells]
    planes = []
    for i, z in enumerate(edges):
        kwargs = {}
        if i == 0:
            kwargs["boundary_condition"] = z_boundary[0]
        elif i == len(edges) - 1:
            kwargs["boundary_condition"] = z_boundary[1]
        planes.append(mcdc.Surface.PlaneZ(z=z, **kwargs))
    box = +x0 & -x1 & +y0 & -y1
    simulation = mcdc.Simulation("rmc-example")
    simulation.set_model(
        [
            mcdc.Cell(box & +planes[i] & -planes[i + 1], fill=cell[2])
            for i, cell in enumerate(cells)
        ]
    )
    return simulation


def volume_source(z_low, z_high, energy, polar_cosine=None, probability=1.0):
    """Source uniform in the slab volume, uniform in `energy` = (E_low, E_high)."""
    import mcdc

    E_low, E_high = energy
    pdf = np.array([[E_low, E_high], [1.0, 1.0]]) / [[1.0], [E_high - E_low]]
    direction = {"isotropic": True}
    if polar_cosine is not None:
        direction = {"direction": [0.0, 0.0, 1.0], "polar_cosine": polar_cosine}
    return mcdc.Source(
        x=[-0.5, 0.5],
        y=[-0.5, 0.5],
        z=[z_low, z_high],
        energy=pdf,
        probability=probability,
        **direction,
    )


def uniform_Q(z_edges, E_edges, source_cells, energy, mu_weights=(0.5, 0.5)):
    """
    Binned source density Q[k, g, j] for a unit source spread uniformly over the z
    cells `source_cells`, uniform in energy over `energy` (aligned with bin edges),
    and with probability mu_weights[j] in each polar bin (uniform inside it).
    """
    z_edges, E_edges = np.asarray(z_edges), np.asarray(E_edges)
    K, G, J = len(z_edges) - 1, len(E_edges) - 1, len(MU_EDGES) - 1
    Q = np.zeros((K, G, J))
    height = sum(z_edges[k + 1] - z_edges[k] for k in source_cells)
    E_low, E_high = energy
    in_source = (E_edges[:-1] >= E_low * (1 - 1e-12)) & (
        E_edges[1:] <= E_high * (1 + 1e-12)
    )
    for k in source_cells:
        for j in range(J):
            Q[k, in_source, j] = mu_weights[j] / (
                height * (E_high - E_low) * (MU_EDGES[j + 1] - MU_EDGES[j])
            )
    return Q


def smc_source_energy(E_low, E_high):
    """
    SMC source energy range for a narrow source bin [E_low, E_high]: starts 3x the
    tally energy tolerance above E_low. MC/DC scores particles within 1e-5 eV above an
    edge in the bin below, which would move part of the large uncollided source-bin
    flux into the top collided bin. The shift changes the collided flux by ~1e-7.
    (RMC is unaffected: its 0D fixed point is set by the binned equations, not the
    tally.)
    """
    from mcdc.constant import COINCIDENCE_TOLERANCE_ENERGY

    return (E_low + 3.0 * COINCIDENCE_TOLERANCE_ENERGY, E_high)


def narrow_source_grid(E_min, E0, delta, G):
    """G log-spaced collided bins on [E_min, E0 - delta] plus the source bin [E0 - delta, E0]."""
    return np.concatenate(
        (np.logspace(np.log10(E_min), np.log10(E0 - delta), G + 1), [E0])
    )


# ======================================================================================
# Runs
# ======================================================================================


def run_rmc(make_model, z_edges, E_edges, Q, N_iteration, N_per_bin, **kwargs):
    """make_model() -> (simulation, cell_materials); returns RMCResult."""
    from mcdc.rmc.driver import run

    simulation, cell_materials = make_model()
    return run(
        simulation,
        z_edges,
        E_edges,
        MU_EDGES,
        Q,
        cell_materials,
        N_iteration,
        N_per_bin,
        **kwargs,
    )


def run_smc(make_model, z_edges, E_edges, N_particle, seed=1, name="smc"):
    """
    Standard Monte Carlo with MC/DC on the same model, energy window and trial-space
    binning; returns psi mean and standard error [K, G, J] and the wall time.
    """
    import mcdc

    simulation, _ = make_model()
    settings = simulation.settings
    settings.N_particle = int(N_particle)
    settings.rng_seed = seed
    settings.use_neutron_energy_window = True
    settings.neutron_energy_min = E_edges[0]
    settings.neutron_energy_max = E_edges[-1]
    settings.neutron_fission_all_prompt = True
    settings.use_progress_bar = False
    settings.output_name = f"{name}-tmp"
    mesh = mcdc.MeshStructured("smc-trial-space", z=np.asarray(z_edges))
    tally = mcdc.Tally(
        name="smc", mesh=mesh, scores=["flux"], energy=np.asarray(E_edges), mu=MU_EDGES
    )
    simulation.set_tallies([tally])
    time_start = time.perf_counter()
    simulation.run()
    wall = time.perf_counter() - time_start

    if not is_master():
        return None, None, wall
    K, G, J = len(z_edges) - 1, len(E_edges) - 1, len(MU_EDGES) - 1
    volume = (
        np.diff(z_edges)[:, None, None]
        * np.diff(E_edges)[None, :, None]
        * np.diff(MU_EDGES)[None, None, :]
    )
    path = f"{name}-tmp.h5"
    with h5py.File(path, "r") as f:
        mean = f["tallies/smc/flux/mean"][()].reshape(J, G, K).transpose(2, 1, 0)
        sdev = f["tallies/smc/flux/sdev"][()].reshape(J, G, K).transpose(2, 1, 0)
    os.remove(path)
    return mean / volume, sdev / volume, wall


def barrier():
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()


def is_master():
    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank() == 0


class Output:
    """
    output.h5 on the master rank, reopened (append) and closed for every write, so
    results saved during a run of several hours are on disk as soon as they exist.
    """

    def __init__(self, path="output.h5"):
        self.path = path
        if is_master():
            h5py.File(path, "w").close()

    def write(self):
        return h5py.File(self.path, "a")

    def create_dataset(self, name, data):
        if is_master():
            with self.write() as f:
                f.create_dataset(name, data=data)

    def __contains__(self, name):
        if not is_master():
            return True
        with h5py.File(self.path, "r") as f:
            return name in f

    def close(self):
        pass


def open_output(path="output.h5"):
    return Output(path)


def save_rmc(output, name, result, **attrs):
    if not is_master():
        return
    with output.write() as f:
        group = f.require_group("rmc").create_group(name)
        result.write(group)
        group.attrs.update(attrs)


def save_smc(output, name, mean, sdev, wall, N_particle, **attrs):
    if not is_master():
        return
    with output.write() as f:
        group = f.require_group("smc").create_group(name)
        group.create_dataset("psi", data=mean)
        group.create_dataset("psi_sdev", data=sdev)
        group.attrs.update(dict(N_particle=int(N_particle), time=wall, **attrs))


def scalar_flux(psi):
    """phi[..., k, g] = sum_j psi[..., k, g, j] dmu_j."""
    return np.einsum("...kgj,j->...kg", psi, np.diff(MU_EDGES))


# ======================================================================================
# Deterministic reference: infinite-medium slowing down, isotropic-COM elastic
# ======================================================================================


def slowing_down_reference(
    E_edges, nuclides, source, h=1e-4, extra_nodes=(), N_source=400
):
    """
    Bin-averaged scalar flux [per eV] of the infinite-medium slowing-down problem with
    target-at-rest elastic scattering isotropic in the COM frame (what MC/DC samples
    for the synthetic nuclides above), solved deterministically to ~1e-8.

    nuclides : [(A, density, energy, sigma_s, sigma_t), ...]
        Tabulated microscopic data (interpolated linearly in energy, as in MC/DC).
    source : (E_low, E_high)
        Unit source, uniform in energy over [E_low, E_high] (inside the window).

    Collision density F = q + F_s, with F_s(E) = sum_n int_E^{min(E/alpha_n, E_max)}
    c_n(E') F(E') / ((1 - alpha_n) E') dE', c_n = N_n sigma_s,n / Sigma_t. In lethargy
    u = ln(E_max / E), with F~ = E F, the kernel factorizes, exp(-(u - u')), so the
    Volterra equation is marched with running integrals H_n(u) = int c_n F~ e^u' du'
    (trapezoid, implicit in the current node). The source part of H is integrated
    with Gauss-Legendre on the source interval.
    """
    from numba import njit

    E_edges = np.asarray(E_edges, dtype=float)
    E_min, E_max = E_edges[0], E_edges[-1]
    E_low, E_high = source

    # Lethargy nodes: uniform, plus bin edges, source edges, and data points
    u_end = np.log(E_max / E_min)
    nodes = [np.linspace(0.0, u_end, int(np.ceil(u_end / h)) + 1)]
    nodes.append(np.log(E_max / E_edges))
    nodes.append(np.log(E_max / np.linspace(E_low, E_high, N_source)))
    for _, _, energy, _, _ in nuclides:
        inside = (energy > E_min) & (energy < E_max)
        nodes.append(np.log(E_max / energy[inside]))
    for E in extra_nodes:
        nodes.append(np.log(E_max / np.asarray(E)))
    u = np.unique(np.clip(np.concatenate(nodes), 0.0, u_end))
    E = E_max * np.exp(-u)

    Sigma_t = np.zeros_like(E)
    for A, N, energy, _, sigma_t in nuclides:
        Sigma_t += N * np.interp(E, energy, sigma_t)
    c = np.array(
        [
            N * np.interp(E, energy, sigma_s) / Sigma_t
            for _, N, energy, sigma_s, _ in nuclides
        ]
    )
    alpha = np.array([((A - 1.0) / (A + 1.0)) ** 2 for A, _, _, _, _ in nuclides])
    width = np.where(alpha > 0.0, np.log(1.0 / np.maximum(alpha, 1e-300)), np.inf)

    # Source part of the running integrals: H_q,n(u) = int_0^u c_n E q e^u' du'
    #   with E e^u' = E_max and q = 1 / (E_high - E_low) on the source interval
    #   (Gauss-Legendre on each node interval; data points are nodes)
    q = 1.0 / (E_high - E_low)
    xg, wg = np.polynomial.legendre.leggauss(8)
    cells = np.nonzero(
        (E[:-1] <= E_high * (1 + 1e-14)) & (E[1:] >= E_low * (1 - 1e-14))
    )[0]
    a, b = u[cells], u[cells + 1]
    points = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * xg[None, :]
    E_points = E_max * np.exp(-points)
    Sigma_t_points = sum(N * np.interp(E_points, e, st) for _, N, e, _, st in nuclides)
    Hq = np.zeros_like(c)
    for n, (A, N, energy, sigma_s, _) in enumerate(nuclides):
        c_points = N * np.interp(E_points, energy, sigma_s) / Sigma_t_points
        Hq[n, cells + 1] = 0.5 * (b - a) * ((c_points * E_max * q) @ wg)
    Hq = np.cumsum(Hq, axis=1)

    F_s = _march(u, c, alpha, width, Hq)

    # Bin averages of phi = (E q + F~_s) / (E Sigma_t), with dE = E du: trapezoid in u
    #   on the nodes of each bin (the uncollided part only inside the source interval)
    in_source = (E >= E_low * (1 - 1e-14)) & (E <= E_high * (1 + 1e-14))
    flux = np.zeros(len(E_edges) - 1)
    u_edges = np.log(E_max / E_edges)
    for g in range(len(flux)):
        mask = (u >= u_edges[g + 1] - 1e-15) & (u <= u_edges[g] + 1e-15)
        flux[g] = np.trapezoid(F_s[mask] / Sigma_t[mask], u[mask])
        source_mask = mask & in_source
        if np.count_nonzero(source_mask) > 1:
            flux[g] += np.trapezoid((E * q / Sigma_t)[source_mask], u[source_mask])
    return flux / np.diff(E_edges)


def _march(u, c, alpha, width, Hq):
    from numba import njit

    @njit(cache=False)
    def march(u, c, alpha, width, Hq):
        N_node = len(u)
        N_nuclide = len(alpha)
        F = np.zeros(N_node)
        Hs = np.zeros((N_nuclide, N_node))  # int_0^u c_n F~_s e^u' du'
        for i in range(1, N_node):
            step = u[i] - u[i - 1]
            rhs = 0.0
            diagonal = 0.0
            for n in range(N_nuclide):
                g_previous = c[n, i - 1] * F[i - 1] * np.exp(u[i - 1])
                H_partial = Hs[n, i - 1] + 0.5 * step * g_previous + Hq[n, i]
                # H_n at the lower limit u_i - width_n (zero below u = 0)
                lower = u[i] - width[n]
                H_low = 0.0
                if lower > 0.0:
                    j = np.searchsorted(u, lower) - 1
                    s = lower - u[j]
                    h = u[j + 1] - u[j]
                    g0 = c[n, j] * F[j] * np.exp(u[j])
                    g1 = c[n, j + 1] * F[j + 1] * np.exp(u[j + 1])
                    slope = (g1 - g0) / h
                    H_low = Hs[n, j] + s * (g0 + 0.5 * slope * s)
                    # Source part: linear interpolation of the cumulative integral
                    H_low += Hq[n, j] + (Hq[n, j + 1] - Hq[n, j]) * s / h
                factor = np.exp(-u[i]) / (1.0 - alpha[n])
                rhs += factor * (H_partial - H_low)
                diagonal += 0.5 * step * c[n, i] / (1.0 - alpha[n])
            F[i] = rhs / (1.0 - diagonal)
            for n in range(N_nuclide):
                Hs[n, i] = Hs[n, i - 1] + 0.5 * step * (
                    c[n, i - 1] * F[i - 1] * np.exp(u[i - 1])
                    + c[n, i] * F[i] * np.exp(u[i])
                )
        return F

    return march(u, c, alpha, width, Hq)


def linf(phi, reference):
    """L-infinity norm of the scalar flux error, as in [N] Eq. 30 and [D] Eq. 2.28."""
    return np.max(np.abs(np.asarray(phi) - reference), axis=-1)


# ======================================================================================
# Plot style
# ======================================================================================

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e7e6e2"
SURFACE = "#fcfcfb"
REGION = {"H2O": "#cde2fb", "UO2": "#f8d3c3"}


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt


def steps(edges, values):
    """Piecewise-constant line through bin edges."""
    return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)


def sweep_runs(f):
    """RMC sweep runs in an output file, as (G, P, group), sorted by P then G."""
    runs = [
        (int(group.attrs["G"]), int(group.attrs["P"]), group)
        for group in f["rmc"].values()
        if "G" in group.attrs and "P" in group.attrs
    ]
    return sorted(runs, key=lambda run: (run[1], run[0]))


def plot_sweep(f, error, x_axis, smc_error=None, grids=None):
    """
    L-infinity convergence of an RMC sweep (color: G, line style: P/B/I) and,
    optionally, of SMC: error(group) -> array per iteration; smc_error(group) -> float.
    x_axis: "histories" or "iterations".
    """
    plt = style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    runs = sweep_runs(f)
    grids = grids or sorted({G for G, _, _ in runs})
    styles = {P: s for P, s in zip(sorted({P for _, P, _ in runs}), [(0, (2, 2)), "-"])}
    if smc_error is not None and x_axis == "histories":
        smc = sorted(f["smc"].values(), key=lambda g: g.attrs["N_particle"])
        smc = [g for g in smc if "G" in g.attrs]
        ax.plot(
            [g.attrs["N_particle"] for g in smc],
            [smc_error(g) for g in smc],
            color=MUTED,
            marker="o",
            ms=4,
            label=f"SMC: G = {smc[0].attrs['G']}",
        )
    for G, P, group in runs:
        values = error(group)
        n = np.arange(1, len(values) + 1)
        x = n * group.attrs["N_history"] if x_axis == "histories" else n
        ax.plot(
            x,
            values,
            color=SERIES[grids.index(G)],
            ls=styles[P],
            label=f"RMC: {P} P/B/I, G = {G}",
        )
    ax.set_yscale("log")
    if x_axis == "histories":
        ax.set_xscale("log")
        ax.set_xlabel("Particle histories")
    else:
        ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$L_\infty$ norm")
    ax.legend(loc="upper right")
    return fig, ax


def bin_xs_variation(energy, xs, E_edges):
    """max/min of a lin-lin cross section over each bin (1 where it is flat)."""
    out = np.ones(len(E_edges) - 1)
    for g in range(len(out)):
        inside = (energy > E_edges[g]) & (energy < E_edges[g + 1])
        x = np.concatenate(([E_edges[g]], energy[inside], [E_edges[g + 1]]))
        values = np.interp(x, energy, xs)
        if values.min() > 0.0:
            out[g] = values.max() / values.min()
    return out


def plot_rmc_vs_smc(f, name, title, path, smooth=None):
    """
    One window of an RMC run against SMC (output.h5 groups rmc/<name>, smc/<name>):
    flux (top), ratio to SMC with the SMC 2-sigma band (middle), and the per-bin
    z of RMC against SMC (bottom). With correction passes the RMC result is the
    corrected one and z includes their standard error; without, it is the phase-1
    fixed point and z uses the SMC error only. `smooth` (bool per bin) marks bins
    where the trial-space bias of the fixed point is negligible (Sigma_t nearly flat
    in the bin); the rms of z is reported over all bins and over those bins.
    Returns (rms z over all bins, rms z over the smooth bins).
    """
    plt = style()
    rmc, smc = f[f"rmc/{name}"], f[f"smc/{name}"]
    E_edges = rmc["E_edges"][()]
    unit, scale = ("MeV", 1e-6) if E_edges[-1] > 1e5 else ("eV", 1.0)
    E_MeV = E_edges * scale
    phi_smc = scalar_flux(smc["psi"][()])[0]
    # Polar bins of one history are correlated (in 0D every track scores both): the
    #   fully correlated sum is an upper bound of the scalar-flux standard error
    sdev_smc = scalar_flux(smc["psi_sdev"][()])[0]
    phi_fixed = scalar_flux(rmc["psi_fixed_point"][()])[0]
    phi = scalar_flux(rmc["psi"][()])[0]
    sdev_corr = np.zeros_like(phi)
    N_corr = 0
    if "corrections" in rmc:
        corrections = scalar_flux(rmc["corrections"][()])[:, 0]
        N_corr = len(corrections)
        sdev_corr = corrections.std(axis=0, ddof=1) / np.sqrt(N_corr)
    z = (phi - phi_smc) / np.sqrt(sdev_smc**2 + sdev_corr**2)
    rms_z = np.sqrt(np.mean(z**2))
    smooth = np.ones(len(z), dtype=bool) if smooth is None else np.asarray(smooth)
    rms_z_smooth = np.sqrt(np.mean(z[smooth] ** 2)) if smooth.any() else np.nan

    fig, (ax, ax_ratio, ax_z) = plt.subplots(
        3,
        1,
        figsize=(7.5, 8.0),
        sharex=True,
        gridspec_kw=dict(height_ratios=(3, 1.5, 1.2)),
    )
    N_smc = int(np.log10(smc.attrs["N_particle"]))
    ax.plot(*steps(E_MeV, phi_smc), color=MUTED, label=f"SMC: $10^{{{N_smc}}}$")
    ax.plot(
        *steps(E_MeV, phi_fixed),
        color=SERIES[0],
        ls=(0, (4, 2)) if N_corr else "-",
        label="RMC fixed point (phase 1)",
    )
    if N_corr:
        ax.plot(
            *steps(E_MeV, phi),
            color=SERIES[1],
            label=f"RMC + {N_corr} correction passes",
        )
    ax.set_xscale("log")
    ax.set_ylabel(f"$\\phi(E)$ [per eV]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    band = 2.0 * sdev_smc / phi_smc
    ax_ratio.fill_between(
        np.repeat(E_MeV, 2)[1:-1],
        np.repeat(1.0 - band, 2),
        np.repeat(1.0 + band, 2),
        color=GRID,
        label="SMC $\\pm2\\sigma$",
    )
    ax_ratio.plot(*steps(E_MeV, phi / phi_smc), color=SERIES[1 if N_corr else 0])
    ax_ratio.set_ylabel("RMC / SMC")
    ax_ratio.legend(fontsize=8)
    ax_z.axhline(0.0, color=MUTED, lw=0.8)
    for level in (-2.0, 2.0):
        ax_z.axhline(level, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    color = SERIES[1 if N_corr else 0]
    ax_z.plot(*steps(E_MeV, z), color=color, lw=0.8, alpha=0.5)
    z_smooth = np.where(smooth, z, np.nan)
    ax_z.plot(
        *steps(E_MeV, z_smooth),
        color=color,
        label=f"rms z: all bins {rms_z:.2f}, flat-$\\Sigma_t$ bins {rms_z_smooth:.2f}",
    )
    ax_z.set_ylabel("z")
    ax_z.set_xlabel(f"E ({unit})")
    ax_z.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    return rms_z, rms_z_smooth
