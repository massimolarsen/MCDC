"""
Residual Monte Carlo iteration driver.

A normal MC/DC fixed-source CE model (a slab along z, reflective in x and y; a single
reflective box for 0D) is solved for the piecewise-constant angular flux psi~[k, g, j]
on a trial space (z_edges, E_edges, mu_edges):

  1. compile the model once, with the neutron energy window, all-prompt fission,
     one batch, and an internal track-length mesh tally on the trial space;
  2. precompute (or load) the transfer moments of every nuclide;
  3. iterate: residual -> signed residual particles -> one MC/DC transport pass ->
     eps~ = tally / (h dE dmu) -> psi~ += eps~.

Units: psi~ is per unit z (cm), energy (eV), and polar cosine about z, per unit x-y
area; Q uses the same units. MC/DC tallies are per source history, so residual
weights carry the residual's absolute normalization.
"""

import hashlib
import os

import h5py
import numpy as np

####

import mcdc
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.simulation as simulation_module
import mcdc.transport.tally as tally_module

from mcdc.constant import (
    BOLTZMANN_K,
    COINCIDENCE_TOLERANCE_ENERGY,
    THERMAL_THRESHOLD_FACTOR,
)
from mcdc.main import prepare
from mcdc.print_ import print_error, print_msg, print_warning
from mcdc.rmc.angular import N_THETA_DEFAULT, build_angular_transfer_table
from mcdc.rmc.kernel import nuclide_reactions, nuclide_transfer_moments
from mcdc.rmc.reaction import kernel_type
from mcdc.rmc.residual import (
    BOUNDARY_REFLECTIVE,
    BOUNDARY_VACUUM,
    Residual,
    material_total_xs,
)
from mcdc.rmc.source import (
    _emission_rate,
    PROPOSAL_DEFENSIVE,
    PROPOSAL_UNIFORM_LETHARGY,
    PROPOSAL_UNIFORM_LINEAR,
    sample_collision_edge,
    sample_correction,
)

import mcdc.mcdc_get as mcdc_get

PROPOSALS = {
    "defensive": PROPOSAL_DEFENSIVE,
    "uniform-linear": PROPOSAL_UNIFORM_LINEAR,
    "uniform-lethargy": PROPOSAL_UNIFORM_LETHARGY,
}
BOUNDARIES = {"vacuum": BOUNDARY_VACUUM, "reflective": BOUNDARY_REFLECTIVE}


class RMCResult:
    """psi~ and the convergence history of an RMC run."""

    def __init__(self, z_edges, E_edges, mu_edges):
        self.z_edges = z_edges
        self.E_edges = E_edges
        self.mu_edges = mu_edges
        self.psi = None
        self.epsilon_norm = []
        self.psi_history = []

    @property
    def scalar_flux(self):
        """phi~[k, g] = sum_j psi~[k, g, j] dmu_j (per unit z and energy)."""
        return np.einsum("kgj,j->kg", self.psi, np.diff(self.mu_edges))


# ======================================================================================
# Transfer moments with an on-disk cache per nuclide
# ======================================================================================


def _cache_key(nuclide, E_edges, mu_edges, tol, tol_low):
    text = f"{nuclide['name']}|{nuclide['temperature']}|{tol}|{tol_low}|"
    text += np.asarray(E_edges).tobytes().hex() + np.asarray(mu_edges).tobytes().hex()
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def nuclide_moments(
    simulation, data, nuclide, E_edges, mu_edges, table, tol, tol_low, cache_dir
):
    if cache_dir is not None:
        key = _cache_key(nuclide, E_edges, mu_edges, tol, tol_low)
        path = os.path.join(cache_dir, f"{nuclide['name']}-{key}.h5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                return f["M_scatter"][()], f["M_fission"][()]
    M_scatter, M_fission = nuclide_transfer_moments(
        simulation, data, nuclide, E_edges, mu_edges, table, tol, tol_low
    )
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("M_scatter", data=M_scatter)
            f.create_dataset("M_fission", data=M_fission)
            f.attrs["nuclide"] = str(nuclide["name"])
            f.create_dataset("E_edges", data=E_edges)
            f.create_dataset("mu_edges", data=mu_edges)
    return M_scatter, M_fission


def _material_nuclides(simulation, data, material):
    for i in range(material["N_nuclide"]):
        ID = mcdc_get.material.nuclide_IDs(i, material, data)
        density = mcdc_get.material.nuclide_densities(i, material, data)
        yield ID, density


def _reaction_tables(simulation, data, material_IDs, fission):
    """Concatenated (by material) reaction arrays of the scattering or fission term."""
    offsets = [0]
    rx_ID, rx_nuclide, rx_density, rx_type = [], [], [], []
    for m in material_IDs:
        material = simulation["materials"][m]
        for nuclide_ID, density in _material_nuclides(simulation, data, material):
            nuclide = simulation["nuclides"][nuclide_ID]
            for reaction, is_fission in nuclide_reactions(simulation, data, nuclide):
                if is_fission != fission:
                    continue
                rx_ID.append(reaction["ID"])
                rx_nuclide.append(nuclide_ID)
                rx_density.append(density)
                rx_type.append(kernel_type(reaction, simulation, data))
        offsets.append(len(rx_ID))
    return (
        np.array(offsets, dtype=np.int64),
        np.array(rx_ID, dtype=np.int64),
        np.array(rx_nuclide, dtype=np.int64),
        np.array(rx_density, dtype=np.float64),
        np.array(rx_type, dtype=np.int64),
    )


def _emission_integrals(program, data, tables, xs, E_edges):
    """
    int_g Psi(E) dE per material and energy bin, Psi = sum_r N_n sigma_r y_r (all
    emissions, inside or outside the energy grid), by trapezoid on the material's
    union xs grid. Only used to shape the q_T proposal, so it need not be exact.
    """
    offsets, rx_ID, rx_nuclide, rx_density, rx_type = tables
    G = len(E_edges) - 1
    result = np.zeros((len(xs), G))
    for m, (E_grid, _) in enumerate(xs):
        start, end = offsets[m], offsets[m + 1]
        if start == end:
            continue
        for g in range(G):
            E = E_grid[(E_grid > E_edges[g]) & (E_grid < E_edges[g + 1])]
            E = np.concatenate(([E_edges[g]], E, [E_edges[g + 1]]))
            rate = [
                _emission_rate(
                    e, start, end, rx_ID, rx_nuclide, rx_density, rx_type, program, data
                )
                for e in E
            ]
            result[m, g] = np.trapezoid(rate, E)
    return result


# ======================================================================================
# Driver
# ======================================================================================


class RMCSolver:
    """
    Compiled RMC problem: set up once, then `iterate(psi, n)` performs one residual
    solve and returns eps~. See `run` for the parameters.
    """

    def __init__(
        self,
        simulation,
        z_edges,
        E_edges,
        mu_edges,
        Q,
        cell_materials,
        N_per_bin,
        collision_fraction=0.9,
        proposal="defensive",
        defensive_fraction=0.5,
        quadrature_tol=1e-8,
        quadrature_tol_low=1e-10,
        boundary=("reflective", "reflective"),
        cache_dir=None,
        seed=1,
    ):
        z_edges = np.asarray(z_edges, dtype=float)
        E_edges = np.asarray(E_edges, dtype=float)
        mu_edges = np.asarray(mu_edges, dtype=float)
        K, G, J = len(z_edges) - 1, len(E_edges) - 1, len(mu_edges) - 1
        Q = np.asarray(Q, dtype=float)

        # ==============================================================================
        # Validation
        # ==============================================================================

        if Q.shape != (K, G, J):
            print_error(f"RMC: Q must have shape {(K, G, J)}")
        if len(cell_materials) != K:
            print_error("RMC: one material per trial-space z cell is required")
        if not np.any(np.isclose(mu_edges, 0.0)):
            print_error("RMC: mu = 0 must be a bin edge")
        if proposal not in PROPOSALS:
            print_error(f"RMC: unknown proposal {proposal}")
        settings = simulation.settings
        if settings.neutron_eigenvalue_mode:
            print_error("RMC: only fixed-source problems are supported")
        if settings.N_census > 1:
            print_error("RMC: time census is not supported")
        if np.min(np.diff(E_edges)) < 1e3 * COINCIDENCE_TOLERANCE_ENERGY:
            print_warning(
                "RMC: energy bins narrower than 1e3 x the tally energy tolerance "
                f"({COINCIDENCE_TOLERANCE_ENERGY} eV) are scored inaccurately"
            )

        # ==============================================================================
        # Configure and compile the MC/DC model
        # ==============================================================================

        N_total = N_per_bin * K * G * J
        settings.N_particle = N_total
        settings.N_batch = 1
        settings.use_source_bank = True
        settings.use_neutron_energy_window = True
        settings.neutron_energy_min = E_edges[0]
        settings.neutron_energy_max = E_edges[-1]
        settings.neutron_fission_all_prompt = True
        settings.use_progress_bar = False

        mesh = mcdc.MeshStructured("rmc-trial-space", z=z_edges)
        epsilon_tally = mcdc.Tally(
            mesh=mesh, scores=["flux"], energy=E_edges, mu=mu_edges
        )
        simulation.set_tallies(list(simulation.tallies) + [epsilon_tally])

        simulation.compile()
        simulation_container, data = prepare(simulation)
        program = simulation_container[0]

        material_IDs = [material.ID for material in cell_materials]
        unique_materials = sorted(set(material_IDs))
        material_index = {m: i for i, m in enumerate(unique_materials)}
        cell_material = np.array(
            [material_index[m] for m in material_IDs], dtype=np.int64
        )

        # Free-gas threshold: target-at-rest elastic kernels require E_min above it
        for m in unique_materials:
            for nuclide_ID, _ in _material_nuclides(
                program, data, program["materials"][m]
            ):
                nuclide = program["nuclides"][nuclide_ID]
                threshold = (
                    THERMAL_THRESHOLD_FACTOR * BOLTZMANN_K * nuclide["temperature"]
                )
                if E_edges[0] <= threshold:
                    print_error(
                        f"RMC: E_min = {E_edges[0]} eV is below the free-gas "
                        f"threshold ({threshold} eV) of {nuclide['name']}"
                    )

        # ==============================================================================
        # Transfer moments, cross sections, and reaction tables per material
        # ==============================================================================

        table = build_angular_transfer_table(mu_edges, N_THETA_DEFAULT)
        nuclide_cache = {}
        M_scatter = np.zeros((len(unique_materials), G, J, G, J))
        M_fission = np.zeros((len(unique_materials), G, J, G, J))
        xs = []
        for i, m in enumerate(unique_materials):
            material = program["materials"][m]
            for nuclide_ID, density in _material_nuclides(program, data, material):
                if nuclide_ID not in nuclide_cache:
                    name = program["nuclides"][nuclide_ID]["name"]
                    print_msg(f"RMC: transfer moments of {name}")
                    nuclide_cache[nuclide_ID] = nuclide_moments(
                        program,
                        data,
                        program["nuclides"][nuclide_ID],
                        E_edges,
                        mu_edges,
                        table,
                        quadrature_tol,
                        quadrature_tol_low,
                        cache_dir,
                    )
                M_scatter[i] += density * nuclide_cache[nuclide_ID][0]
                M_fission[i] += density * nuclide_cache[nuclide_ID][1]
            xs.append(
                material_total_xs(program, data, material, E_edges[0], E_edges[-1])
            )

        # Store the state
        self.z_edges, self.E_edges, self.mu_edges = z_edges, E_edges, mu_edges
        self.shape = (K, G, J)
        self.Q = Q
        self.N_total = N_total
        self.collision_fraction = collision_fraction
        self.proposal = PROPOSALS[proposal]
        self.defensive_fraction = defensive_fraction
        self.boundary = (BOUNDARIES[boundary[0]], BOUNDARIES[boundary[1]])
        self.seed = seed
        self.simulation_container, self.data, self.program = (
            simulation_container,
            data,
            program,
        )
        self.epsilon_tally = epsilon_tally
        self.cell_material = cell_material
        self.M_scatter, self.M_fission = M_scatter, M_fission
        self.M_total = M_scatter + M_fission
        self.xs = xs
        self.xs_offsets = np.concatenate(
            ([0], np.cumsum([len(x[0]) for x in xs]))
        ).astype(np.int64)
        self.xs_energy = np.concatenate([x[0] for x in xs])
        self.xs_total = np.concatenate([x[1] for x in xs])
        self.tables_scatter = _reaction_tables(program, data, unique_materials, False)
        self.tables_fission = _reaction_tables(program, data, unique_materials, True)
        self.emission_scatter = _emission_integrals(
            program, data, self.tables_scatter, xs, E_edges
        )
        self.emission_fission = _emission_integrals(
            program, data, self.tables_fission, xs, E_edges
        )
        self.volume = (
            np.diff(z_edges)[:, None, None]
            * np.diff(E_edges)[None, :, None]
            * np.diff(mu_edges)[None, None, :]
        )

    def residual(self, psi):
        K = self.shape[0]
        return Residual(
            psi,
            self.Q,
            [self.M_total[self.cell_material[k]] for k in range(K)],
            [self.xs[self.cell_material[k]] for k in range(K)],
            self.z_edges,
            self.E_edges,
            self.mu_edges,
            *self.boundary,
        )

    def iterate(self, psi, n, terms=("collision", "scatter", "fission")):
        """One residual solve from psi~; returns eps~ (None if the residual is empty)."""
        K = self.shape[0]
        h = np.diff(self.z_edges)
        program, data = self.program, self.data
        N_total = self.N_total
        residual = self.residual(psi)

        # Particle budget: collision/edge, then corrections by binned source size
        cm = self.cell_material
        size_scatter = sum(
            h[k] * np.abs(np.einsum("pqgj,pq->", self.M_scatter[cm[k]], psi[k]))
            for k in range(K)
        )
        size_fission = sum(
            h[k] * np.abs(np.einsum("pqgj,pq->", self.M_fission[cm[k]], psi[k]))
            for k in range(K)
        )
        N_correction = 0
        if size_scatter + size_fission > 0.0:
            N_correction = N_total - int(self.collision_fraction * N_total)
        N_collision = N_total - N_correction
        N_fission = int(
            round(
                N_correction * size_fission / max(size_scatter + size_fission, 1e-300)
            )
        )
        N_scatter = N_correction - N_fission

        # Fill the source bank
        bank = program["bank_source"]
        particle_bank_module.set_bank_size(bank, 0)
        seed_iteration = np.uint64(self.seed * 1000003 + n)
        if "collision" in terms:
            mass = np.concatenate(
                (residual.collision_mass.ravel(), residual.edge_mass.ravel())
            )
            sample_collision_edge(
                N_collision,
                N_total,
                seed_iteration,
                self.z_edges,
                self.E_edges,
                self.mu_edges,
                psi,
                residual.c,
                mass,
                residual.jump,
                self.xs_offsets,
                self.xs_energy,
                self.xs_total,
                cm,
                program,
            )
        for name, N_term, M_term, emission, tables, salt in (
            (
                "scatter",
                N_scatter,
                self.M_scatter,
                self.emission_scatter,
                self.tables_scatter,
                1,
            ),
            (
                "fission",
                N_fission,
                self.M_fission,
                self.emission_fission,
                self.tables_fission,
                2,
            ),
        ):
            if name not in terms:
                continue
            sample_correction(
                N_term,
                N_total,
                np.uint64(seed_iteration * 7 + salt),
                self.proposal,
                self.defensive_fraction,
                self.z_edges,
                self.E_edges,
                self.mu_edges,
                psi,
                M_term,
                emission,
                cm,
                *tables,
                program,
                data,
            )

        # Rescale to the number of banked histories (skipped samples carry no weight)
        N_bank = particle_bank_module.get_bank_size(bank)
        if N_bank == 0:
            return None
        bank["particle_data"][:N_bank]["w"] *= N_bank / N_total
        program["settings"]["N_particle"] = N_bank

        # Transport the residual source and read eps~
        tally_module.closeout.reset_statistics(program, data)
        simulation_module.fixed_source_simulation(self.simulation_container, data)
        record = program["tallies"][self.epsilon_tally.ID]
        start = record["bin_mean_offset"]
        mean = data[start : start + record["bin_length"]].reshape(
            self.epsilon_tally.bin_shape
        )
        # bin shape: (mu, azi, energy, time, x, y, z, score)
        return np.transpose(mean[:, 0, :, 0, 0, 0, :, 0], (2, 1, 0)) / self.volume


def run(
    simulation,
    z_edges,
    E_edges,
    mu_edges,
    Q,
    cell_materials,
    N_iteration,
    N_per_bin,
    collision_fraction=0.9,
    proposal="defensive",
    defensive_fraction=0.5,
    quadrature_tol=1e-8,
    quadrature_tol_low=1e-10,
    stop="fixed",
    tol=1e-6,
    boundary=("reflective", "reflective"),
    cache_dir=None,
    seed=1,
):
    """
    Solve a fixed-source CE problem with Residual Monte Carlo.

    Parameters
    ----------
    simulation : mcdc.Simulation
        The model (cells, materials, settings). Its geometry must be a slab along z
        (reflective in x and y), consistent with `z_edges` and `cell_materials`.
    z_edges, E_edges, mu_edges : array_like
        Trial-space grids. E_edges [eV] also define the neutron energy window;
        mu_edges (polar cosine about z) must contain 0.
    Q : ndarray (K, G, J)
        Binned external source density.
    cell_materials : sequence of mcdc.Material
        Material of each trial-space z cell.
    N_iteration : int
        Maximum number of RMC iterations.
    N_per_bin : int
        Histories per trial-space bin per iteration.
    collision_fraction : float
        Fraction of histories assigned to r_c + r_e; the rest go to the scattering
        and fission corrections in proportion to their binned source magnitudes.
    proposal : {"defensive", "uniform-linear", "uniform-lethargy"}
        Proposal for the scattering/fission corrections.
    stop : {"fixed", "tol"}
        Run N_iteration iterations, or stop when ||eps~||_2 / ||psi~||_2 < tol.
    boundary : (str, str)
        Boundary conditions at z_low and z_high: "reflective" or "vacuum".

    Returns
    -------
    RMCResult
    """
    if stop not in ("fixed", "tol"):
        print_error(f"RMC: unknown stop {stop}")
    solver = RMCSolver(
        simulation,
        z_edges,
        E_edges,
        mu_edges,
        Q,
        cell_materials,
        N_per_bin,
        collision_fraction,
        proposal,
        defensive_fraction,
        quadrature_tol,
        quadrature_tol_low,
        boundary,
        cache_dir,
        seed,
    )
    result = RMCResult(solver.z_edges, solver.E_edges, solver.mu_edges)
    psi = np.zeros(solver.shape)
    for n in range(N_iteration):
        epsilon = solver.iterate(psi, n)
        if epsilon is None:
            break
        psi = psi + epsilon
        result.psi_history.append(psi.copy())
        result.epsilon_norm.append(np.linalg.norm(epsilon))
        if stop == "tol" and np.linalg.norm(epsilon) < tol * np.linalg.norm(psi):
            break
    result.psi = psi
    return result
