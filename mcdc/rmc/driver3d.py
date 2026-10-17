"""
3D Cartesian Residual Monte Carlo on the piecewise-constant trial space
(writeups/cartesian_3d.tex).

The trial space is constant on each structured x-y-z cell, energy bin and angular
patch (`mcdc.rmc.patches`). Each iteration samples the collision residual (binned
in-scatter) and the face residual of psi~ (`mcdc.rmc.mesh3d`), transports the
signed-weight particles with MC/DC unchanged, and adds the tallied error eps~ to psi~.
The energy physics (inverted kernels, transfer moments, cache) is shared with the
slab solver: the patch transfer table takes the place of the polar-bin table.

Supported here: collision-only iterations (the binned fixed point, which carries the
in-bin shape bias of the constant trial space when there is scattering), vacuum and
reflective boundaries on each of the six faces, one material per trial cell. Not yet:
the unbiased correction passes, linear bases, cells with mixed materials.
"""

import hashlib
import os
import uuid

import h5py
import numpy as np

from mpi4py import MPI

####

import mcdc
import mcdc.transport.mpi as mpi
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.simulation as simulation_module
import mcdc.transport.tally as tally_module

from mcdc.constant import (
    COINCIDENCE_TOLERANCE_ENERGY,
    NEUTRON_REACTION_ELASTIC_SCATTERING,
)
from mcdc.main import prepare
from mcdc.print_ import print_error, print_msg, print_warning
from mcdc.rmc.driver import (
    BOUNDARIES,
    _material_nuclides,
    _xs_arrays,
    nuclide_moments,
)
from mcdc.rmc.kernel import nuclide_reactions
from mcdc.rmc.mesh3d import CartesianMesh, Residual3D, sample_residual_3d
from mcdc.rmc.patches import (
    _GL_X,
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    N_THETA_PATCHES,
    AngularPatches,
    build_patch_transfer_table,
)
from mcdc.rmc.reaction import elastic_isotropic_below, free_gas_threshold
from mcdc.rmc.residual import (
    BOUNDARY_REFLECTIVE,
    material_total_xs,
    projected_removal,
)

PATCH_TABLE_FORMAT_VERSION = 1


class RMC3DResult:
    """psi~ and the convergence history of a 3D RMC run."""

    def __init__(self, mesh, E_edges, patches):
        self.x_edges, self.y_edges, self.z_edges = mesh.edges
        self.E_edges = E_edges
        self.mu_edges, self.phi_edges = patches.mu_edges, patches.phi_edges
        self.patch_area = patches.area
        self.psi = None  # last iterate psi~[kx, ky, kz, g, j], j = j_mu J_phi + j_phi
        self.psi_history = []  # psi~ after each iteration
        self.epsilon_norm = []  # ||eps~|| per iteration
        self.N_history = 0  # histories per iteration
        self.time_precompute = 0.0  # [s] model compile, patch table, moments
        self.time_iteration = []  # [s] per iteration

    def psi_average(self, start):
        """
        Mean of the iterates from `start` on and its standard error. When the
        iteration has reached its noise plateau, the iterates scatter about the
        fixed point and their mean reduces that noise (NOTES.md, section 2b).
        """
        history = np.array(self.psi_history[start:])
        error = (
            history.std(axis=0, ddof=1) / np.sqrt(len(history))
            if len(history) > 1
            else None
        )
        return history.mean(axis=0), error

    def scalar_flux(self, psi=None):
        """phi~[kx, ky, kz, g] = sum_j psi~ |A_j| (per unit volume and energy)."""
        psi = self.psi if psi is None else psi
        return psi @ self.patch_area

    def write(self, group):
        for name in ("x_edges", "y_edges", "z_edges", "E_edges", "mu_edges"):
            group.create_dataset(name, data=getattr(self, name))
        group.create_dataset("phi_edges", data=self.phi_edges)
        group.create_dataset("psi", data=self.psi)
        group.create_dataset("scalar_flux", data=self.scalar_flux())
        group.create_dataset("epsilon_norm", data=np.array(self.epsilon_norm))
        group.create_dataset("psi_history", data=np.array(self.psi_history))
        group.create_dataset("time_iteration", data=np.array(self.time_iteration))
        group.attrs["N_history"] = self.N_history
        group.attrs["time_precompute"] = self.time_precompute


# ======================================================================================
# Patch transfer table (cached on disk)
# ======================================================================================


def patch_table(patches, N_theta, cache_dir):
    """The patch transfer table, from `cache_dir` if a matching file exists."""
    comm = MPI.COMM_WORLD
    path = None
    if cache_dir is not None:
        text = f"format={PATCH_TABLE_FORMAT_VERSION}|N_theta={N_theta}|GL={len(_GL_X)}|"
        text += patches.key.tobytes().hex()
        key = hashlib.sha256(text.encode()).hexdigest()[:20]
        path = os.path.join(cache_dir, f"patch-table-{key}.h5")
        found = None
        if comm.Get_rank() == 0:
            found = os.path.exists(path)
        if comm.bcast(found, root=0):
            with h5py.File(path, "r") as f:
                return f["table"][()]
    print_msg(f"RMC: patch transfer table ({patches.J} patches, {N_theta} angles)")
    table = build_patch_transfer_table(patches, N_theta, comm)
    if path is not None and comm.Get_rank() == 0:
        os.makedirs(cache_dir, exist_ok=True)
        temporary = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with h5py.File(temporary, "w") as f:
            f.create_dataset("table", data=table)
            f.create_dataset("mu_edges", data=patches.mu_edges)
            f.create_dataset("phi_edges", data=patches.phi_edges)
            f.attrs["N_theta"] = N_theta
            f.attrs["format_version"] = PATCH_TABLE_FORMAT_VERSION
        os.replace(temporary, path)
    return table


# ======================================================================================
# Solver
# ======================================================================================


def _boundaries(boundary):
    """((x_low, x_high), (y_low, y_high), (z_low, z_high)) as boundary codes."""
    if isinstance(boundary, str):
        boundary = ((boundary, boundary),) * 3
    if len(boundary) != 3 or any(len(pair) != 2 for pair in boundary):
        print_error("RMC: boundary must be three (low, high) pairs, for x, y and z")
    for pair in boundary:
        for name in pair:
            if name not in BOUNDARIES:
                print_error(f"RMC: unknown boundary {name}")
    return tuple((BOUNDARIES[low], BOUNDARIES[high]) for low, high in boundary)


class RMC3DSolver:
    """Compiled 3D RMC problem; `iterate(psi, n)` returns eps~. See `run3d`."""

    def __init__(
        self,
        simulation,
        x_edges,
        y_edges,
        z_edges,
        E_edges,
        mu_edges,
        phi_edges,
        Q,
        cell_materials,
        N_per_bin,
        boundary="reflective",
        quadrature_tol=1e-8,
        quadrature_tol_low=1e-8,
        cache_dir=None,
        seed=1,
        collision_xs="binned",
        N_theta=N_THETA_PATCHES,
    ):
        mesh = CartesianMesh(x_edges, y_edges, z_edges)
        try:
            patches = AngularPatches(mu_edges, phi_edges)
        except ValueError as error:
            print_error(str(error))
        E_edges = np.asarray(E_edges, dtype=float)
        G, J = len(E_edges) - 1, patches.J
        shape = mesh.shape + (G, J)
        Q = np.asarray(Q, dtype=float)
        if Q.shape != shape:
            print_error(f"RMC: Q must have shape {shape} (cells, energy bins, patches)")
        cells = np.empty(mesh.shape, dtype=object)
        if isinstance(cell_materials, mcdc.Material):
            cells[...] = cell_materials
        else:
            cells[...] = np.array(cell_materials, dtype=object).reshape(mesh.shape)
        boundary = _boundaries(boundary)
        mirrors = []
        for axis in (AXIS_X, AXIS_Y, AXIS_Z):
            mirror = None
            if BOUNDARY_REFLECTIVE in boundary[axis]:
                try:
                    mirror = patches.mirror(axis)
                except ValueError as error:
                    print_error(str(error))
            mirrors.append(mirror)
        if collision_xs not in ("binned", "pointwise"):
            print_error(f"RMC: unknown collision_xs {collision_xs}")
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

        N_total = int(round(N_per_bin * mesh.N_cell * G * J))
        settings.N_particle = N_total
        settings.N_batch = 1
        settings.use_source_bank = True
        settings.use_neutron_energy_window = True
        settings.neutron_energy_min = E_edges[0]
        settings.neutron_energy_max = E_edges[-1]
        settings.neutron_fission_all_prompt = True
        settings.use_progress_bar = False

        x, y, z = mesh.edges
        tally_mesh = mcdc.MeshStructured("rmc-trial-space-3d", x=x, y=y, z=z)
        epsilon_tally = mcdc.Tally(
            mesh=tally_mesh,
            scores=["flux"],
            energy=E_edges,
            mu=patches.mu_edges,
            azi=patches.phi_edges,
        )
        simulation.set_tallies(list(simulation.tallies) + [epsilon_tally])
        simulation.compile()
        simulation_container, data = prepare(simulation)
        program = simulation_container[0]

        material_IDs = [cell.ID for cell in cells.ravel()]
        unique_materials = sorted(set(material_IDs))
        material_index = {m: i for i, m in enumerate(unique_materials)}
        cell_material = np.array(
            [material_index[m] for m in material_IDs], dtype=np.int64
        )
        # Free-gas range: the inverted kernel assumes isotropic COM elastic scattering
        for m in unique_materials:
            for nuclide_ID, _ in _material_nuclides(
                program, data, program["materials"][m]
            ):
                nuclide = program["nuclides"][nuclide_ID]
                threshold = free_gas_threshold(nuclide)
                if E_edges[0] > threshold:
                    continue
                for reaction, _ in nuclide_reactions(program, data, nuclide):
                    if reaction[
                        "sub_type"
                    ] == NEUTRON_REACTION_ELASTIC_SCATTERING and not (
                        elastic_isotropic_below(reaction, threshold, program, data)
                    ):
                        print_error(
                            f"RMC: {nuclide['name']} has anisotropic COM elastic data "
                            "below the free-gas threshold; not supported"
                        )

        # ==============================================================================
        # Patch table, transfer moments and cross sections per material
        # ==============================================================================

        table = patch_table(patches, N_theta, cache_dir)
        nuclide_cache = {}
        M = np.zeros((len(unique_materials), G * J, G * J))
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
                        patches.measure_edges,
                        table,
                        quadrature_tol,
                        quadrature_tol_low,
                        cache_dir,
                        0,
                        angular_key=patches.key,
                        N_theta=N_theta,
                    )
                M_scatter, M_fission = nuclide_cache[nuclide_ID]
                # [c', g', j', c, g, j] with one coefficient -> [(g', j'), (g, j)]
                M[i] += density * (M_scatter + M_fission)[0, :, :, 0].reshape(
                    G * J, G * J
                )
            xs.append(
                material_total_xs(program, data, material, E_edges[0], E_edges[-1])
            )

        self.mesh, self.patches = mesh, patches
        self.E_edges = E_edges
        self.shape = shape
        self.Q = Q
        self.boundary, self.mirrors = boundary, mirrors
        self.N_total = N_total
        self.seed = seed
        self.simulation_container, self.data, self.program = (
            simulation_container,
            data,
            program,
        )
        self.epsilon_tally = epsilon_tally
        self.cell_material = cell_material
        self.M = M
        self.xs = xs
        self.cell_xs = [xs[m] for m in cell_material]
        self.removal = None
        self.xs_arrays = _xs_arrays(xs)
        if collision_xs == "binned":
            # Bin-averaged Sigma_t per cell; r_c then sees Sigma_t = 0
            sigma_bar = np.array([projected_removal(x, E_edges, 1)[:, 0, 0] for x in xs])
            self.removal = sigma_bar[cell_material].reshape(mesh.shape + (G,))
            no_xs = (np.array([E_edges[0], E_edges[-1]]), np.zeros(2))
            self.xs_arrays = _xs_arrays([no_xs] * len(xs))
        self.measure = (
            mesh.volume[..., None, None]
            * np.diff(E_edges)[:, None]
            * patches.area[None, None, None, None, :]
        )

    def binned_in_scatter(self, psi):
        """S_bar = M psi~ / (dE_g |A_j|), per cell with its material's moments."""
        G, J = self.shape[3:]
        flat = psi.reshape(-1, G * J)
        S = np.zeros_like(flat)
        for m in range(len(self.M)):
            cells = self.cell_material == m
            S[cells] = flat[cells] @ self.M[m]
        norm = np.outer(np.diff(self.E_edges), self.patches.area).ravel()
        return (S / norm).reshape(self.shape)

    def residual(self, psi):
        return Residual3D(
            psi,
            self.Q,
            self.binned_in_scatter(psi),
            self.mesh,
            self.E_edges,
            self.patches,
            self.boundary,
            self.mirrors,
            self.removal,
            self.cell_xs,
        )

    def iterate(self, psi, n):
        """One residual solve from psi~ (collision-only iteration); returns eps~."""
        residual = self.residual(psi)
        mass = np.concatenate(
            [residual.collision_mass.ravel()] + [m.ravel() for m in residual.face_mass]
        )
        program, data = self.program, self.data
        bank = program["bank_source"]
        particle_bank_module.set_bank_size(bank, 0)
        program["settings"]["N_particle"] = self.N_total
        mpi.distribute_work(self.N_total, program)
        start = program["mpi_work_start"]
        x, y, z = self.mesh.edges
        p = self.patches
        sample_residual_3d(
            start,
            start + program["mpi_work_size"],
            np.uint64(self.seed * 1000003 + n),
            x,
            y,
            z,
            self.E_edges,
            p.mu_low,
            p.mu_high,
            p.phi_low,
            p.phi_high,
            p.area,
            p.abs_direction_integral,
            psi,
            residual.c,
            *residual.jump,
            mass,
            *self.xs_arrays,
            self.cell_material,
            program,
        )
        return self._read_epsilon()

    def _read_epsilon(self):
        """Transport the banked residual source; eps~ from the tally (all ranks)."""
        program, data = self.program, self.data
        tally_module.closeout.reset_statistics(program, data)
        simulation_module.fixed_source_simulation(self.simulation_container, data)
        record = program["tallies"][self.epsilon_tally.ID]
        start = record["bin_mean_offset"]
        mean = data[start : start + record["bin_length"]].reshape(
            self.epsilon_tally.bin_shape
        )
        # bin shape: (mu, azi, energy, time, x, y, z, score)
        flux = np.transpose(mean[:, :, :, 0, :, :, :, 0], (3, 4, 5, 2, 0, 1))
        epsilon = np.ascontiguousarray(flux.reshape(self.shape) / self.measure)
        MPI.COMM_WORLD.Bcast(epsilon, root=0)
        return epsilon


def run3d(
    simulation,
    x_edges,
    y_edges,
    z_edges,
    E_edges,
    mu_edges,
    phi_edges,
    Q,
    cell_materials,
    N_iteration,
    N_per_bin,
    boundary="reflective",
    stop="fixed",
    tol=1e-3,
    quadrature_tol=1e-8,
    quadrature_tol_low=1e-8,
    cache_dir=None,
    seed=1,
    collision_xs="binned",
    N_theta=N_THETA_PATCHES,
):
    """
    Solve a fixed-source CE problem on a 3D Cartesian, piecewise-constant trial space
    with collision-only RMC iterations.

    Parameters
    ----------
    simulation : mcdc.Simulation
        The MC/DC model (geometry, materials, boundary conditions). Its outer
        boundary must coincide with the trial mesh and its boundary conditions with
        `boundary`; each trial cell must hold a single material.
    x_edges, y_edges, z_edges : array
        Trial mesh edges [cm].
    E_edges : array
        Energy bin edges [eV]; the energy window.
    mu_edges : array
        Polar-cosine patch edges, from -1 to 1, including 0.
    phi_edges : array
        Azimuthal patch edges, from -pi to pi, including -pi/2, 0 and pi/2.
    Q : ndarray (Kx, Ky, Kz, G, J)
        Bin averages of the external source per unit volume, energy and solid angle,
        with J = (len(mu_edges) - 1)(len(phi_edges) - 1) and j = j_mu J_phi + j_phi.
    cell_materials : mcdc.Material or array (Kx, Ky, Kz) of mcdc.Material
    N_iteration : int
        Iterations (the cap with stop="tol").
    N_per_bin : float
        Histories per (cell, energy bin, patch) per iteration.
    boundary : str or ((x_low, x_high), (y_low, y_high), (z_low, z_high))
        "vacuum" or "reflective". A reflective axis needs patch edges symmetric under
        that reflection.
    stop : {"fixed", "tol"}
        Run N_iteration iterations, or stop when ||eps~|| < tol ||psi~||.
    quadrature_tol, quadrature_tol_low : float
        Transfer-moment quadrature tolerances (as in `mcdc.rmc.run`).
    cache_dir : str, optional
        Directory for the patch table and transfer-moment caches.
    collision_xs : {"binned", "pointwise"}
        Removal in r_c: bin-averaged or pointwise Sigma_t.
    N_theta : int
        Points of the patch transfer table in theta0.

    Returns
    -------
    RMC3DResult
    """
    if stop not in ("fixed", "tol"):
        print_error(f"RMC: unknown stop {stop}")
    time_start = MPI.Wtime()
    solver = RMC3DSolver(
        simulation,
        x_edges,
        y_edges,
        z_edges,
        E_edges,
        mu_edges,
        phi_edges,
        Q,
        cell_materials,
        N_per_bin,
        boundary,
        quadrature_tol,
        quadrature_tol_low,
        cache_dir,
        seed,
        collision_xs,
        N_theta,
    )
    result = RMC3DResult(solver.mesh, solver.E_edges, solver.patches)
    result.N_history = solver.N_total
    result.time_precompute = MPI.Wtime() - time_start

    # L2 norm of a trial-space function: sqrt(sum psi^2 x bin measure)
    def norm(values):
        return np.sqrt(np.sum(values**2 * solver.measure))

    psi = np.zeros(solver.shape)
    for n in range(N_iteration):
        time_start = MPI.Wtime()
        epsilon = solver.iterate(psi, n)
        result.time_iteration.append(MPI.Wtime() - time_start)
        psi = psi + epsilon
        result.psi_history.append(psi.copy())
        result.epsilon_norm.append(norm(epsilon))
        if stop == "tol" and norm(epsilon) < tol * norm(psi):
            break
    result.psi = psi
    return result
