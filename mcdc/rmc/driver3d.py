"""
3D Cartesian Residual Monte Carlo on the piecewise-constant trial space
(writeups/cartesian_3d/cartesian_3d.tex).

The trial space is constant on each structured x-y-z cell, energy bin and angular
patch (`mcdc.rmc.patches`). Each iteration samples the collision residual (binned
in-scatter) and the face residual of psi~ (`mcdc.rmc.mesh3d`), transports the
signed-weight particles with MC/DC unchanged, and adds the tallied error eps~ to psi~.
The energy physics (inverted kernels, transfer moments, cache) is shared with the
slab solver: the patch transfer table takes the place of the polar-bin table.

Supported here: collision-only iterations (the binned fixed point, which carries the
in-bin shape bias of the constant trial space when there is scattering), then
full-residual correction passes with a tabulated in-scatter kernel
(`mcdc.rmc.correction3d`, writeups/cartesian_3d/tabulated_correction.tex), which reduce
that bin-shape error while retaining interpolation bias; vacuum and reflective boundaries on each of the six faces; one material per
trial cell. Polynomial spatial bases use the shared modal residual implementation.
Cells with mixed materials are not supported. Tabulated correction passes retain
the interpolation bias described in tabulated_correction.tex.
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
from mcdc.rmc.correction3d import (
    CorrectionNodes,
    linear_departure,
    material_table,
    node_in_scatter,
    sample_correction_3d,
    volume_masses,
)
from mcdc.rmc.driver import (
    BOUNDARIES,
    _material_nuclides,
    _reaction_tables,
    _xs_arrays,
    nuclide_data_path,
    nuclide_moments,
)
from mcdc.rmc.kernel import nuclide_reactions
from mcdc.rmc.mesh3d import CartesianMesh, Residual3D, sample_residual_3d
from mcdc.rmc.polynomial3d import (
    face_masses as polynomial_face_masses,
    sample_polynomial_3d,
    volume_masses as polynomial_volume_masses,
)
from mcdc.rmc.space3d import SPATIAL_BASES, SpatialBasis
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
        # Final psi~[kx, ky, kz, g, j], j = j_mu J_phi + j_phi: the last iterate plus
        #   the averaged correction passes (the last iterate if there are none)
        self.psi = None
        self.psi_fixed_point = None  # last collision-only iterate
        self.spatial_basis = "constant"
        # Polynomial bases: Legendre coefficients C[kx, ky, kz, g, j, m] of the last
        #   iterate and the modes m = (a, b, c); psi and psi_history hold cell averages
        self.coefficients = None
        self.modes = None
        self.psi_history = []  # psi~ after each collision-only iteration
        self.epsilon_norm = []  # ||eps~|| per collision-only iteration
        # L1 masses (collision, faces) of the residual sampled in each iteration
        self.residual_mass = []
        self.corrections = []  # eps~ of each correction pass
        # Standard error of psi (history statistics of the correction tallies)
        # This excludes deterministic in-scatter interpolation bias.
        self.psi_sdev = None
        self.N_history = 0  # histories per iteration (and per correction pass)
        self.time_precompute = 0.0  # [s] model compile, patch table, moments
        self.time_iteration = []  # [s] per collision-only iteration
        self.time_table = 0.0  # [s] in-scatter tables of the correction passes
        self.time_correction = []  # [s] per correction pass
        # L1 masses (volume bound, faces) sampled in each correction pass
        self.correction_mass = []
        # Scalar flux per (cell, energy bin), per unit volume and energy: the final
        #   estimate and its history standard error (with correction passes)
        self.scalar = None
        self.scalar_sdev = None

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
        group.create_dataset("residual_mass", data=np.array(self.residual_mass))
        group.attrs["spatial_basis"] = self.spatial_basis
        if self.coefficients is not None:
            group.create_dataset("coefficients", data=self.coefficients)
            group.create_dataset("modes", data=self.modes)
        group.attrs["N_history"] = self.N_history
        group.attrs["time_precompute"] = self.time_precompute
        if self.corrections:
            group.create_dataset("psi_fixed_point", data=self.psi_fixed_point)
            group.create_dataset("corrections", data=np.array(self.corrections))
            group.create_dataset("psi_sdev", data=self.psi_sdev)
            group.create_dataset(
                "time_correction", data=np.array(self.time_correction)
            )
            group.create_dataset(
                "correction_mass", data=np.array(self.correction_mass)
            )
            group.attrs["time_table"] = self.time_table
            group.create_dataset("scalar", data=self.scalar)
            group.create_dataset("scalar_sdev", data=self.scalar_sdev)


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
        correction_nodes=(4, 3, 3),
        spatial_basis="constant",
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
        if spatial_basis not in SPATIAL_BASES:
            print_error(f"RMC: unknown spatial_basis {spatial_basis}")
        basis = None
        unknowns = mesh.N_cell
        if spatial_basis != "constant":
            if collision_xs != "binned":
                print_error("RMC: polynomial spatial bases use collision_xs='binned'")
            basis = SpatialBasis(spatial_basis, mesh, patches, boundary)
            unknowns = (
                int(np.prod(basis.node_shape))
                if basis.continuous
                else mesh.N_cell * basis.M
            )
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

        # Histories per iteration: N_per_bin per spatial unknown, energy bin and patch
        N_total = int(round(N_per_bin * unknowns * G * J))
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
            scores=["flux"] if basis is None else basis.tally_scores,
            energy=E_edges,
            mu=patches.mu_edges,
            azi=patches.phi_edges,
        )
        # Scalar flux per (cell, energy bin): its history standard error compares runs
        #   with different patch layouts and bases against one SMC reference
        scalar_tally = mcdc.Tally(mesh=tally_mesh, scores=["flux"], energy=E_edges)
        simulation.set_tallies(
            list(simulation.tallies) + [epsilon_tally, scalar_tally]
        )
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
        self.basis = basis
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
        self.scalar_tally = scalar_tally
        self.last_scalar = None  # (mean, sdev) of the last transport, per unit V and E
        self.cell_material = cell_material
        self.M = M
        self.xs = xs
        self.cell_xs = [xs[m] for m in cell_material]
        self.removal = None
        self.xs_arrays = _xs_arrays(xs)
        self.xs_arrays_pointwise = self.xs_arrays
        self.unique_materials = unique_materials
        self.cache_dir = cache_dir
        self.correction_nodes = correction_nodes
        self.correction = None  # (nodes, U per material, Sigma_t departure), built once
        self.time_table = 0.0
        self.iteration_nodes = None  # one interval per bin and patch (polynomial bases)
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
        self.last_residual_mass = (
            float(residual.collision_mass.sum()),
            float(sum(m.sum() for m in residual.face_mass)),
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

    def _correction_tables(self):
        """Interpolation nodes and the in-scatter table U of every material."""
        program, data = self.program, self.data
        alphas, U, labels = set(), [], []
        rx_tables, nuclides = [], []
        for m in self.unique_materials:
            material = program["materials"][m]
            names = []
            for nuclide_ID, _ in _material_nuclides(program, data, material):
                nuclide = program["nuclides"][nuclide_ID]
                A = nuclide["atomic_weight_ratio"]
                alphas.add(((A - 1.0) / (A + 1.0)) ** 2)
                names.append((str(nuclide["name"]), nuclide_data_path(nuclide)))
            scatter = _reaction_tables(program, data, [m], False)[1:]
            fission = _reaction_tables(program, data, [m], True)[1:]
            rx_tables.append(tuple(np.concatenate(p) for p in zip(scatter, fission)))
            nuclides.append(names)
            labels.append("+".join(name for name, _ in names))
        try:
            nodes = CorrectionNodes(
                self.E_edges, self.patches, sorted(alphas), *self.correction_nodes
            )
        except ValueError as error:
            print_error(str(error))
        for i in range(len(self.unique_materials)):
            U.append(
                material_table(
                    program,
                    data,
                    rx_tables[i],
                    nuclides[i],
                    nodes,
                    self.E_edges,
                    self.patches,
                    self.cache_dir,
                    labels[i],
                )
            )
        return nodes, U, linear_departure(nodes, self.xs)

    def prepare_correction(self):
        """Build correction tables once, outside the timed transport passes."""
        if self.correction is None:
            time_start = MPI.Wtime()
            self.correction = self._correction_tables()
            self.time_table = MPI.Wtime() - time_start

    def correct(self, psi, n):
        """
        One full-residual correction pass from psi~; returns eps~ and its standard
        error (writeups/cartesian_3d/tabulated_correction.tex).
        """
        self.prepare_correction()
        nodes, U, departure = self.correction
        p = self.patches
        Tn = node_in_scatter(
            psi, self.binned_in_scatter(psi), U, self.cell_material, nodes, p
        )
        volume = volume_masses(
            psi,
            self.Q,
            Tn,
            self.cell_material,
            nodes,
            self.mesh,
            p,
            self.xs,
            departure,
        )
        jumps = Residual3D(
            psi,
            self.Q,
            np.zeros_like(psi),
            self.mesh,
            self.E_edges,
            p,
            self.boundary,
            self.mirrors,
            np.zeros(psi.shape[:-1]),
            self.cell_xs,
        )
        mass = np.concatenate(
            [volume.ravel()] + [m.ravel() for m in jumps.face_mass]
        )
        self.last_correction_mass = (
            float(volume.sum()),
            float(sum(m.sum() for m in jumps.face_mass)),
        )
        program = self.program
        particle_bank_module.set_bank_size(program["bank_source"], 0)
        program["settings"]["N_particle"] = self.N_total
        mpi.distribute_work(self.N_total, program)
        start = program["mpi_work_start"]
        x, y, z = self.mesh.edges
        sample_correction_3d(
            start,
            start + program["mpi_work_size"],
            np.uint64(self.seed * 1000003 + n),
            x,
            y,
            z,
            self.E_edges,
            nodes.E_nodes,
            nodes.interval_left,
            nodes.interval_bin,
            nodes.n_mu,
            nodes.n_phi,
            p.mu_low,
            p.mu_high,
            p.phi_low,
            p.phi_high,
            p.abs_direction_integral,
            psi,
            self.Q,
            Tn,
            *jumps.jump,
            mass,
            *self.xs_arrays_pointwise,
            self.cell_material,
            program,
        )
        return self._read_epsilon(with_sdev=True)

    def _read_epsilon(self, with_sdev=False):
        """
        Transport the banked residual source; eps~ from the tally (all ranks), and
        with with_sdev its standard error (history statistics).
        """
        epsilon, sdev = self._transport()
        epsilon = epsilon[..., 0] / self.mesh.volume[..., None, None]
        if not with_sdev:
            return epsilon
        return epsilon, sdev[..., 0] / self.mesh.volume[..., None, None]

    def _transport(self):
        """
        Transport the banked residual source; every score of the eps~ tally per unit
        energy and solid angle, (Kx, Ky, Kz, G, J, score), and its standard error
        (history statistics; after the closeout the deviation buffer holds it). For
        the "flux-xyz" scores these are the cell moments int_V eps~ P_m.
        """
        program, data = self.program, self.data
        tally_module.closeout.reset_statistics(program, data)
        simulation_module.fixed_source_simulation(self.simulation_container, data)
        record = program["tallies"][self.epsilon_tally.ID]
        G, J = self.shape[3:]
        per_unit = (
            np.diff(self.E_edges)[:, None] * self.patches.area[None, :]
        )[..., None]

        def read(offset):
            values = data[offset : offset + record["bin_length"]].reshape(
                self.epsilon_tally.bin_shape
            )
            # bin shape: (mu, azi, energy, time, x, y, z, score)
            flux = np.transpose(values[:, :, :, 0], (3, 4, 5, 2, 0, 1, 6))
            out = flux.reshape(self.mesh.shape + (G, J, -1)) / per_unit
            out = np.ascontiguousarray(out)
            MPI.COMM_WORLD.Bcast(out, root=0)
            return out

        scalar = program["tallies"][self.scalar_tally.ID]
        per_volume_energy = self.mesh.volume[..., None] * np.diff(self.E_edges)

        def read_scalar(offset):
            values = data[offset : offset + scalar["bin_length"]].reshape(
                self.scalar_tally.bin_shape
            )
            # bin shape: (mu, azi, energy, time, x, y, z, score)
            flux = np.transpose(values[0, 0, :, 0, :, :, :, 0], (1, 2, 3, 0))
            out = np.ascontiguousarray(flux / per_volume_energy)
            MPI.COMM_WORLD.Bcast(out, root=0)
            return out

        self.last_scalar = (
            read_scalar(scalar["bin_mean_offset"]),
            read_scalar(scalar["bin_sum_squared_deviations_offset"]),
        )
        return read(record["bin_mean_offset"]), read(
            record["bin_sum_squared_deviations_offset"]
        )

    # ==================================================================================
    # Polynomial spatial bases (mcdc.rmc.space3d, mcdc.rmc.polynomial3d)
    # ==================================================================================

    def binned_in_scatter_modes(self, C):
        """S_bar of every spatial mode: the transfer moments act mode by mode."""
        return np.stack(
            [self.binned_in_scatter(C[..., m]) for m in range(C.shape[-1])], axis=-1
        )

    def _source_modes(self, M):
        Q = np.zeros(self.shape + (M,))
        Q[..., 0] = self.Q  # Q is uniform in each cell
        return Q

    def _sample_polynomial(self, C, Tn, nodes, Sigma_nodes, departure, xs_arrays, n):
        """Masses, then bank the particles of the residual (polynomial3d)."""
        basis, p = self.basis, self.patches
        gradient = basis.gradient(C)
        Q = self._source_modes(basis.M)
        jumps = [
            basis.face_jumps(C, axis, self.boundary[axis], p, self.mirrors[axis])
            for axis in (AXIS_X, AXIS_Y, AXIS_Z)
        ]
        volume = polynomial_volume_masses(
            basis, C, Q, Tn, gradient, Sigma_nodes, self.cell_material, nodes, p,
            departure,
        )
        faces = polynomial_face_masses(basis, jumps, self.E_edges, p)
        mass = np.concatenate([volume] + faces)
        masses = (float(volume.sum()), float(sum(f.sum() for f in faces)))
        program = self.program
        particle_bank_module.set_bank_size(program["bank_source"], 0)
        program["settings"]["N_particle"] = self.N_total
        mpi.distribute_work(self.N_total, program)
        start = program["mpi_work_start"]
        x, y, z = self.mesh.edges
        sample_polynomial_3d(
            start,
            start + program["mpi_work_size"],
            np.uint64(self.seed * 1000003 + n),
            x,
            y,
            z,
            self.E_edges,
            nodes.E_nodes,
            nodes.interval_left,
            nodes.interval_bin,
            nodes.n_mu,
            nodes.n_phi,
            p.mu_low,
            p.mu_high,
            p.phi_low,
            p.phi_high,
            p.abs_direction_integral,
            basis.modes,
            np.stack(basis.face_modes),
            C,
            Q,
            gradient,
            Tn,
            *jumps,
            mass,
            *xs_arrays,
            self.cell_material,
            program,
        )
        return masses

    def iterate_polynomial(self, C, n):
        """
        One collision-only iteration from the coefficients C; returns the projected
        update of the coefficients.
        """
        if self.iteration_nodes is None:
            self.iteration_nodes = CorrectionNodes(self.E_edges, self.patches, [], 1, 2, 2)
        nodes = self.iteration_nodes
        K = self.mesh.N_cell
        M = C.shape[-1]
        G, J = self.shape[3:]
        # Binned in-scatter minus binned removal, constant on each bin and patch
        binned = self.binned_in_scatter_modes(C) - self.removal[..., None, None] * C
        Tn = binned.reshape(K, G, J, M)[:, nodes.node_bin]  # (K, N_nodes, J, M)
        Tn = np.ascontiguousarray(
            np.broadcast_to(
                np.transpose(Tn, (0, 3, 1, 2))[..., None, None],
                (K, M, len(nodes.E_nodes), J, nodes.n_mu, nodes.n_phi),
            )
        )
        zeros_nodes = np.zeros((len(self.xs), len(nodes.E_nodes)))
        zeros_departure = np.zeros((len(self.xs), len(nodes.interval_left)))
        self.last_residual_mass = self._sample_polynomial(
            C, Tn, nodes, zeros_nodes, zeros_departure, self.xs_arrays, n
        )
        moments, _ = self._transport()
        return self.basis.project_modal(moments)

    def correct_polynomial(self, C, n):
        """
        One full-residual correction pass from C; returns the tallied cell moments of
        eps~ (per unit energy and solid angle) and their standard errors.
        """
        self.prepare_correction()
        nodes, U, departure = self.correction
        S = self.binned_in_scatter_modes(C)
        Tn = np.stack(
            [
                node_in_scatter(
                    C[..., m], S[..., m], U, self.cell_material, nodes, self.patches
                )
                for m in range(C.shape[-1])
            ],
            axis=1,
        )
        Sigma_nodes = np.array(
            [np.interp(nodes.E_nodes, grid, Sigma) for grid, Sigma in self.xs]
        )
        self.last_correction_mass = self._sample_polynomial(
            C, Tn, nodes, Sigma_nodes, departure, self.xs_arrays_pointwise, n
        )
        return self._transport()


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
    N_correction=0,
    correction_nodes=(4, 3, 3),
    spatial_basis="constant",
):
    """
    Solve a fixed-source CE problem on a 3D Cartesian, piecewise-constant trial space
    with collision-only RMC iterations, then N_correction full-residual correction
    passes from the last iterate, averaged and added to it (as in `mcdc.rmc.run`).

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
        Q defines the piecewise-constant external source actually solved here.
        A within-bin source remainder q-Q is not sampled from simulation.sources.
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
    N_correction : int
        Full-residual correction passes (0: the result is the collision-only fixed
        point). Each pass uses the histories of one iteration.
    correction_nodes : (int, int, int)
        Interpolation nodes of the in-scatter T in the correction passes: uniform
        intervals per energy bin (the elastic kinks are added), and direction nodes
        per patch in mu and in phi (edges included).
    spatial_basis : {"constant", "dg1", "dg2", "cg1", "cg2"}
        Spatial trial space in each cell (`mcdc.rmc.space3d`): constant, discontinuous
        complete polynomials of degree 1 or 2, or continuous trilinear or
        triquadratic. N_per_bin then counts histories per spatial unknown (cell
        coefficient or node), energy bin and patch. The result's psi holds cell
        averages; with corrections, the last iterate's plus the tallied (unprojected)
        cell averages of the passes. An exact residual gives unbiased cell averages;
        the tabulated in-scatter retains interpolation bias. psi_sdev measures
        sampling uncertainty only; refine correction_nodes to assess that bias.

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
        correction_nodes,
        spatial_basis,
    )
    result = RMC3DResult(solver.mesh, solver.E_edges, solver.patches)
    result.spatial_basis = spatial_basis
    result.N_history = solver.N_total
    result.time_precompute = MPI.Wtime() - time_start

    # L2 norm of a trial-space function: sqrt(sum psi^2 x bin measure)
    def norm(values):
        return np.sqrt(np.sum(values**2 * solver.measure))

    if solver.basis is not None:
        return _run_polynomial(solver, result, N_iteration, stop, tol, N_correction)

    psi = np.zeros(solver.shape)
    for n in range(N_iteration):
        time_start = MPI.Wtime()
        epsilon = solver.iterate(psi, n)
        result.time_iteration.append(MPI.Wtime() - time_start)
        psi = psi + epsilon
        result.psi_history.append(psi.copy())
        result.epsilon_norm.append(norm(epsilon))
        result.residual_mass.append(solver.last_residual_mass)
        if stop == "tol" and norm(epsilon) < tol * norm(psi):
            break
    result.psi_fixed_point = psi.copy()

    # Full-residual corrections from the last iterate, averaged
    variance = np.zeros(solver.shape)
    scalar = []
    if N_correction > 0:
        solver.prepare_correction()
    for m in range(N_correction):
        time_start = MPI.Wtime()
        epsilon, sdev = solver.correct(psi, N_iteration + m)
        result.time_correction.append(MPI.Wtime() - time_start)
        result.corrections.append(epsilon)
        result.correction_mass.append(solver.last_correction_mass)
        variance += sdev**2
        scalar.append(solver.last_scalar)
    if N_correction > 0:
        result.time_table = solver.time_table
        psi = psi + np.mean(result.corrections, axis=0)
        result.psi_sdev = np.sqrt(variance) / N_correction
        _scalar_result(result, solver, result.psi_fixed_point, scalar)
    result.psi = psi
    return result


def _scalar_result(result, solver, cell_averages, passes):
    """Scalar flux: the last iterate's plus the averaged tallied corrections."""
    N = len(passes)
    result.scalar = cell_averages @ solver.patches.area + sum(p[0] for p in passes) / N
    result.scalar_sdev = np.sqrt(sum(p[1] ** 2 for p in passes)) / N


def _run_polynomial(solver, result, N_iteration, stop, tol, N_correction):
    """run3d with a polynomial spatial basis (coefficients C[..., m])."""
    basis = solver.basis
    # L2 norm over cells: |V| dE |A| sum_m C_m^2 / norm_m (Legendre orthogonality)
    weights = solver.measure[..., None] / basis.norm

    def norm(C):
        return np.sqrt(np.sum(C**2 * weights))

    C = np.zeros(solver.shape + (basis.M,))
    for n in range(N_iteration):
        time_start = MPI.Wtime()
        update = solver.iterate_polynomial(C, n)
        result.time_iteration.append(MPI.Wtime() - time_start)
        C = C + update
        result.psi_history.append(C[..., 0].copy())
        result.epsilon_norm.append(norm(update))
        result.residual_mass.append(solver.last_residual_mass)
        if stop == "tol" and norm(update) < tol * norm(C):
            break
    result.coefficients = C
    result.modes = basis.modes
    result.psi_fixed_point = C[..., 0].copy()

    # Corrections: the tallied cell averages of eps~ (mode 0), averaged over passes
    volume = solver.mesh.volume[..., None, None]
    variance = np.zeros(solver.shape)
    scalar = []
    if N_correction > 0:
        solver.prepare_correction()
    for m in range(N_correction):
        time_start = MPI.Wtime()
        moments, sdev = solver.correct_polynomial(C, N_iteration + m)
        result.time_correction.append(MPI.Wtime() - time_start)
        result.corrections.append(moments[..., 0] / volume)
        result.correction_mass.append(solver.last_correction_mass)
        variance += (sdev[..., 0] / volume) ** 2
        scalar.append(solver.last_scalar)
    psi = C[..., 0]
    if N_correction > 0:
        result.time_table = solver.time_table
        psi = psi + np.mean(result.corrections, axis=0)
        result.psi_sdev = np.sqrt(variance) / N_correction
        _scalar_result(result, solver, C[..., 0], scalar)
    result.psi = psi
    return result
