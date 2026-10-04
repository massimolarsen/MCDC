"""
Residual Monte Carlo iteration driver.

A normal MC/DC fixed-source CE model (a slab along z, reflective in x and y; a single
reflective box for 0D) is solved for the angular flux psi~[k, g, j, a] on a trial space
(z_edges, E_edges, mu_edges), piecewise constant in z and mu and, in energy, constant or
linear discontinuous on each bin (coefficients a of P_0 = 1, P_1 = x_g(E)):

  1. compile the model once, with the neutron energy window, all-prompt fission,
     one batch, and an internal track-length mesh tally on the trial space;
  2. precompute (or load) the transfer moments of every nuclide;
  3. iterate: residual -> signed residual particles -> one MC/DC transport pass ->
     eps~ = (2a + 1) tally_a / (h dE dmu) -> psi~ += eps~, where tally_a scores the
     flux times P_a (the "flux" and "flux-energy-slope" scores).

Units: psi~ is per unit z (cm), energy (eV), and polar cosine about z, per unit x-y
area; Q uses the same units. MC/DC tallies are per source history, so residual
weights carry the residual's absolute normalization.

Schedule: collision-only iterations (exponential convergence to the fixed point of the
binned in-scatter operator), then optional averaged full-residual correction passes
that remove the remaining bin-shape bias without bias of their own. Sampling the
scattering correction in every iteration stalls the iteration; see mcdc/rmc/NOTES.md
for the investigation and the deferred low-variance fix.
"""

import datetime
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
from mcdc.rmc.angular import N_THETA_DEFAULT, build_angular_transfer_table
from mcdc.rmc.kernel import (
    MAX_DEPTH,
    TOLERANCE_FLOOR,
    nuclide_reactions,
    nuclide_transfer_moments,
)
from mcdc.rmc.reaction import (
    elastic_isotropic_below,
    free_gas_threshold,
    kernel_type,
)
from mcdc.rmc.residual import (
    BOUNDARY_REFLECTIVE,
    BOUNDARY_VACUUM,
    Residual,
    ResidualLinearZ,
    material_total_xs,
    projected_removal,
)
from mcdc.rmc.space import LinearZProjection, cell_average, node_moments
from mcdc.rmc.source import (
    _emission_rate,
    PROPOSAL_DEFENSIVE,
    PROPOSAL_UNIFORM_LETHARGY,
    PROPOSAL_UNIFORM_LINEAR,
    binned_in_scatter,
    correction_masses,
    sample_collision_edge,
    sample_collision_linear_z,
    sample_correction,
    sample_correction_integrated,
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

    def __init__(
        self,
        z_edges,
        E_edges,
        mu_edges,
        energy_basis="constant",
        spatial_basis="constant",
    ):
        self.z_edges = z_edges
        self.E_edges = E_edges
        self.mu_edges = mu_edges
        self.energy_basis = energy_basis
        # Spatial basis "linear": psi~ arrays hold nodal values on z_edges (K + 1),
        #   and psi_cell the cell averages; "constant": cell values (K)
        self.spatial_basis = spatial_basis
        # Bin averages psi~[k, g, j]; with the linear energy basis also the P_1
        #   coefficients (psi~(E) = psi + psi_slope x_g(E)), else None
        self.psi = None  # final answer (fixed point + averaged correction)
        self.psi_slope = None
        self.psi_fixed_point = None  # converged collision-only iterate
        self.psi_slope_fixed_point = None
        self.epsilon_norm = []  # ||eps~|| per collision-only iteration
        self.psi_history = []  # psi~ (bin averages) after each collision-only iteration
        self.corrections = []  # full-residual corrections (bin averages)
        self.corrections_slope = []  # and their P_1 coefficients (linear basis)
        self.N_history = 0  # histories per iteration (and per correction pass)
        self.time_precompute = 0.0  # [s] model compile and transfer moments
        self.time_iteration = []  # [s] per collision-only iteration
        self.time_correction = []  # [s] per correction pass

    @property
    def psi_cell(self):
        """Cell averages of psi~ [k, g, j] (nodal averages for the linear basis)."""
        if self.spatial_basis == "linear":
            return cell_average(self.psi)
        return self.psi

    @property
    def scalar_flux(self):
        """phi~[k, g] = sum_j psi~[k, g, j] dmu_j of the cell averages (per unit z, E)."""
        return np.einsum("kgj,j->kg", self.psi_cell, np.diff(self.mu_edges))

    @property
    def correction_standard_error(self):
        """Standard error of the averaged correction (None with < 2 passes)."""
        if len(self.corrections) < 2:
            return None
        return np.std(self.corrections, axis=0, ddof=1) / np.sqrt(len(self.corrections))

    def write(self, group):
        """Write the result into an open h5py group."""
        for name in ("z_edges", "E_edges", "mu_edges", "psi", "psi_fixed_point"):
            group.create_dataset(name, data=getattr(self, name))
        if self.psi_slope is not None:
            group.create_dataset("psi_slope", data=self.psi_slope)
            group.create_dataset(
                "psi_slope_fixed_point", data=self.psi_slope_fixed_point
            )
        group.create_dataset("scalar_flux", data=self.scalar_flux)
        group.create_dataset("epsilon_norm", data=np.array(self.epsilon_norm))
        group.create_dataset("psi_history", data=np.array(self.psi_history))
        group.create_dataset("time_iteration", data=np.array(self.time_iteration))
        if self.corrections:
            group.create_dataset("corrections", data=np.array(self.corrections))
            if self.corrections_slope:
                group.create_dataset(
                    "corrections_slope", data=np.array(self.corrections_slope)
                )
            group.create_dataset("time_correction", data=np.array(self.time_correction))
        group.attrs["N_history"] = self.N_history
        group.attrs["energy_basis"] = self.energy_basis
        group.attrs["spatial_basis"] = self.spatial_basis
        if self.spatial_basis == "linear":
            group.create_dataset("psi_cell", data=self.psi_cell)
        group.attrs["time_precompute"] = self.time_precompute


# ======================================================================================
# Transfer moments with an on-disk cache per nuclide
# ======================================================================================


# Version of the cached transfer-moment content: bump whenever the moment math, the
#   kernels, or the file layout change, so older cache files are never reused
MOMENT_FORMAT_VERSION = 3
ENERGY_BASES = {"constant": 0, "linear": 1}
SPATIAL_BASES = ("constant", "linear")

_fingerprints = {}


def data_fingerprint(path):
    """SHA-256 of a nuclear data file, memoized by (path, size, modification time)."""
    status = os.stat(path)
    memo = (os.path.abspath(path), status.st_size, status.st_mtime_ns)
    if memo not in _fingerprints:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        _fingerprints[memo] = digest.hexdigest()
    return _fingerprints[memo]


def nuclide_data_path(nuclide):
    """The library file MC/DC loaded the nuclide from (resolved as `Nuclide` does)."""
    name = f"{nuclide['name']}-{nuclide['temperature']}K.h5"
    return os.path.join(os.environ["MCDC_LIB"], name)


def _cache_provenance(
    nuclide, fingerprint, E_edges, mu_edges, tol, tol_low, energy_order
):
    """
    Everything the moments depend on: the data contents, the moment-format version,
    the grids, and the effective quadrature settings (tolerances after the floor,
    angular-table resolution, refinement depth).
    """
    return {
        "format_version": MOMENT_FORMAT_VERSION,
        "nuclide": str(nuclide["name"]),
        "temperature": float(nuclide["temperature"]),
        "data_sha256": fingerprint,
        "tol": max(float(tol), TOLERANCE_FLOOR),
        "tol_low": max(float(tol_low), TOLERANCE_FLOOR),
        "N_theta": N_THETA_DEFAULT,
        "max_depth": MAX_DEPTH,
        "energy_order": int(energy_order),
    }


def _cache_key(provenance, E_edges, mu_edges):
    text = "|".join(f"{k}={provenance[k]}" for k in sorted(provenance)) + "|"
    text += np.asarray(E_edges, dtype=float).tobytes().hex()
    text += np.asarray(mu_edges, dtype=float).tobytes().hex()
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _read_cache(path, provenance):
    """Moments from a cache file whose stored provenance matches, else None."""
    try:
        with h5py.File(path, "r") as f:
            for key, value in provenance.items():
                stored = f.attrs.get(key)
                if isinstance(stored, bytes):
                    stored = stored.decode()
                if stored is None or stored != value:
                    return None
            return f["M_scatter"][()], f["M_fission"][()]
    except (OSError, KeyError):
        return None


def _write_cache(path, provenance, M_scatter, M_fission, E_edges, mu_edges):
    """Write to a unique temporary file, then rename into place (atomic on POSIX)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with h5py.File(temporary, "w") as f:
            f.create_dataset("M_scatter", data=M_scatter)
            f.create_dataset("M_fission", data=M_fission)
            f.create_dataset("E_edges", data=E_edges)
            f.create_dataset("mu_edges", data=mu_edges)
            for key, value in provenance.items():
                f.attrs[key] = value
            f.attrs["mcdc_version"] = str(getattr(mcdc, "__version__", "unknown"))
            f.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def nuclide_moments(
    simulation,
    data,
    nuclide,
    E_edges,
    mu_edges,
    table,
    tol,
    tol_low,
    cache_dir,
    energy_order=0,
):
    """
    Transfer moments of one nuclide, from the on-disk cache when a file with the
    same provenance (data contents, format version, grids, effective quadrature
    settings) exists. Rank 0 checks the cache and writes it; all ranks agree on
    whether to enter the collective computation.
    """
    comm = MPI.COMM_WORLD
    if cache_dir is not None:
        provenance = None
        if comm.Get_rank() == 0:
            fingerprint = data_fingerprint(nuclide_data_path(nuclide))
            provenance = _cache_provenance(
                nuclide, fingerprint, E_edges, mu_edges, tol, tol_low, energy_order
            )
        provenance = comm.bcast(provenance, root=0)
        key = _cache_key(provenance, E_edges, mu_edges)
        path = os.path.join(cache_dir, f"{nuclide['name']}-{key}.h5")
        valid = (
            _read_cache(path, provenance) is not None if comm.Get_rank() == 0 else None
        )
        if comm.bcast(valid, root=0):
            # Validated on rank 0; every rank reads the arrays itself
            with h5py.File(path, "r") as f:
                return f["M_scatter"][()], f["M_fission"][()]
    M_scatter, M_fission = nuclide_transfer_moments(
        simulation, data, nuclide, E_edges, mu_edges, table, tol, tol_low, energy_order
    )
    if cache_dir is not None and comm.Get_rank() == 0:
        _write_cache(path, provenance, M_scatter, M_fission, E_edges, mu_edges)
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


def _xs_arrays(xs):
    offsets = np.concatenate(([0], np.cumsum([len(x[0]) for x in xs]))).astype(np.int64)
    return (
        offsets,
        np.concatenate([x[0] for x in xs]),
        np.concatenate([x[1] for x in xs]),
    )


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
        quadrature_tol_low=1e-8,
        boundary=("reflective", "reflective"),
        cache_dir=None,
        seed=1,
        collision_xs="binned",
        correction_sampler="integrated",
        energy_basis="constant",
        spatial_basis="constant",
    ):
        z_edges = np.asarray(z_edges, dtype=float)
        E_edges = np.asarray(E_edges, dtype=float)
        mu_edges = np.asarray(mu_edges, dtype=float)
        K, G, J = len(z_edges) - 1, len(E_edges) - 1, len(mu_edges) - 1
        Q = np.asarray(Q, dtype=float)

        # ==============================================================================
        # Validation
        # ==============================================================================

        if energy_basis not in ENERGY_BASES:
            print_error(f"RMC: unknown energy_basis {energy_basis}")
        if spatial_basis not in SPATIAL_BASES:
            print_error(f"RMC: unknown spatial_basis {spatial_basis}")
        linear_z = spatial_basis == "linear"
        if linear_z and correction_sampler == "pointwise":
            print_error(
                "RMC: the linear spatial basis supports the integrated correction "
                "sampler only"
            )
        P = ENERGY_BASES[energy_basis] + 1
        if Q.shape == (K, G, J):
            Q = np.concatenate((Q[..., None], np.zeros((K, G, J, P - 1))), axis=-1)
        if Q.shape != (K, G, J, P):
            print_error(f"RMC: Q must have shape {(K, G, J)} or {(K, G, J, P)}")
        if len(cell_materials) != K:
            print_error("RMC: one material per trial-space z cell is required")
        if not np.any(np.isclose(mu_edges, 0.0)):
            print_error("RMC: mu = 0 must be a bin edge")
        if proposal not in PROPOSALS:
            print_error(f"RMC: unknown proposal {proposal}")
        if collision_xs not in ("binned", "pointwise"):
            print_error(f"RMC: unknown collision_xs {collision_xs}")
        if correction_sampler not in ("integrated", "pointwise"):
            print_error(f"RMC: unknown correction_sampler {correction_sampler}")
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

        N_total = int(round(N_per_bin * K * G * J))
        settings.N_particle = N_total
        settings.N_batch = 1
        settings.use_source_bank = True
        settings.use_neutron_energy_window = True
        settings.neutron_energy_min = E_edges[0]
        settings.neutron_energy_max = E_edges[-1]
        settings.neutron_fission_all_prompt = True
        settings.use_progress_bar = False

        mesh = mcdc.MeshStructured("rmc-trial-space", z=z_edges)
        # Scores: flux x P_a(energy) x P_b(z within the cell), a < P, b < 1 or 2
        scores = ["flux", "flux-energy-slope"][:P]
        if linear_z:
            scores += ["flux-z-slope", "flux-z-energy-slope"][:P]
        epsilon_tally = mcdc.Tally(
            mesh=mesh, scores=scores, energy=E_edges, mu=mu_edges
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

        # Free-gas range (E <= 400 kT): the inverted kernel assumes isotropic COM
        #   elastic scattering there (anisotropic free gas is not supported)
        for m in unique_materials:
            for nuclide_ID, _ in _material_nuclides(
                program, data, program["materials"][m]
            ):
                nuclide = program["nuclides"][nuclide_ID]
                threshold = free_gas_threshold(nuclide)
                if E_edges[0] > threshold:
                    continue
                for reaction, _ in nuclide_reactions(program, data, nuclide):
                    if reaction["sub_type"] != NEUTRON_REACTION_ELASTIC_SCATTERING:
                        continue
                    if not elastic_isotropic_below(reaction, threshold, program, data):
                        print_error(
                            f"RMC: {nuclide['name']} has anisotropic COM elastic data "
                            f"below the free-gas threshold ({threshold} eV); the "
                            "anisotropic free-gas kernel is not supported"
                        )

        # ==============================================================================
        # Transfer moments, cross sections, and reaction tables per material
        # ==============================================================================

        table = build_angular_transfer_table(mu_edges, N_THETA_DEFAULT)
        nuclide_cache = {}
        M_scatter = np.zeros((len(unique_materials), P, G, J, P, G, J))
        M_fission = np.zeros((len(unique_materials), P, G, J, P, G, J))
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
                        P - 1,
                    )
                M_scatter[i] += density * nuclide_cache[nuclide_ID][0]
                M_fission[i] += density * nuclide_cache[nuclide_ID][1]
            xs.append(
                material_total_xs(program, data, material, E_edges[0], E_edges[-1])
            )

        # Store the state
        self.z_edges, self.E_edges, self.mu_edges = z_edges, E_edges, mu_edges
        # psi~ holds cell values (K) or, for the linear spatial basis, nodal values
        self.shape = (K + 1 if linear_z else K, G, J, P)
        self.K = K
        self.energy_basis = energy_basis
        self.spatial_basis = spatial_basis
        self.linear_z = linear_z
        self.projection = None
        if linear_z:
            self.projection = LinearZProjection(
                z_edges, E_edges, mu_edges, *(BOUNDARIES[b] for b in boundary)
            )
        self.Q = Q
        self.N_total = N_total
        self.collision_fraction = collision_fraction
        self.proposal = PROPOSALS[proposal]
        self.correction_sampler = correction_sampler
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
        # Collision-only iterations use the projected removal P[Sigma_t psi~] when
        #   binned (the bin-averaged Sigma_t for the constant basis); it is folded
        #   into the residual's c, so their r_c sampling sees Sigma_t = 0
        self.xs = xs
        self.xs_arrays = _xs_arrays(self.xs)
        self.removal = None
        self.xs_arrays_iteration = self.xs_arrays
        if collision_xs == "binned":
            self.removal = [projected_removal(x, E_edges, P) for x in xs]
            no_xs = (np.array([E_edges[0], E_edges[-1]]), np.zeros(2))
            self.xs_arrays_iteration = _xs_arrays([no_xs] * len(xs))
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

    def residual(self, psi, corrections=True):
        K = self.K
        cm = self.cell_material
        if self.linear_z:
            return ResidualLinearZ(
                psi,
                self.Q,
                self.M_total,
                self.xs,
                self.z_edges,
                self.E_edges,
                self.mu_edges,
                cm,
                removal=None if corrections else self.removal,
            )
        removal = None
        if not corrections and self.removal is not None:
            removal = [self.removal[cm[k]] for k in range(K)]
        return Residual(
            psi,
            self.Q,
            [self.M_total[cm[k]] for k in range(K)],
            [self.xs[cm[k]] for k in range(K)],
            self.z_edges,
            self.E_edges,
            self.mu_edges,
            *self.boundary,
            removal=removal,
        )

    def iterate(self, psi, n, corrections=True):
        """
        One residual solve from psi~; returns eps~. With corrections=False only the
        collision and edge residual (binned in-scatter) is sampled.
        """
        K = self.K
        h = np.diff(self.z_edges)
        program, data = self.program, self.data
        # Per-cell left and right values of psi~ (equal for the constant basis)
        psi_left, psi_right = psi, psi
        if self.linear_z:
            psi_left = np.ascontiguousarray(psi[:-1])
            psi_right = np.ascontiguousarray(psi[1:])
        N_total = self.N_total
        residual = self.residual(psi, corrections)
        xs_arrays = self.xs_arrays if corrections else self.xs_arrays_iteration

        # Particle budget: collision/edge, then corrections by binned source size
        cm = self.cell_material
        integrated = corrections and self.correction_sampler == "integrated"
        if integrated:
            # Sampling masses of r = T - S_bar per bin (deterministic pilot)
            terms = []
            for M_term, tables in (
                (self.M_scatter, self.tables_scatter),
                (self.M_fission, self.tables_fission),
            ):
                S_bar = binned_in_scatter(
                    psi_left, M_term, cm, self.E_edges, self.mu_edges
                )
                S_right = S_bar
                if self.linear_z:
                    S_right = binned_in_scatter(
                        psi_right, M_term, cm, self.E_edges, self.mu_edges
                    )
                mass = correction_masses(
                    psi_left,
                    S_bar,
                    psi_right,
                    S_right,
                    self.linear_z,
                    self.z_edges,
                    self.E_edges,
                    self.mu_edges,
                    cm,
                    *tables,
                    program,
                    data,
                )
                terms.append((S_bar, S_right, mass))
            size_scatter = np.sum(terms[0][2])
            size_fission = np.sum(terms[1][2])
        else:
            size_scatter = sum(
                h[k]
                * np.abs(
                    np.einsum(
                        "bpqgj,pqb->", self.M_scatter[cm[k]][:, :, :, 0], psi_left[k]
                    )
                )
                for k in range(K)
            )
            size_fission = sum(
                h[k]
                * np.abs(
                    np.einsum(
                        "bpqgj,pqb->", self.M_fission[cm[k]][:, :, :, 0], psi_left[k]
                    )
                )
                for k in range(K)
            )
        N_correction = 0
        if corrections and size_scatter + size_fission > 0.0:
            N_correction = N_total - int(self.collision_fraction * N_total)
        N_collision = N_total - N_correction
        N_fission = int(
            round(
                N_correction * size_fission / max(size_scatter + size_fission, 1e-300)
            )
        )
        N_scatter = N_correction - N_fission

        # Fill the source bank: this rank's slice of the global sample indices
        #   (collision/edge, then scattering, then fission); seeds depend only on the
        #   global index, so the result does not depend on the number of ranks
        bank = program["bank_source"]
        particle_bank_module.set_bank_size(bank, 0)
        program["settings"]["N_particle"] = N_total
        mpi.distribute_work(N_total, program)
        work_start = program["mpi_work_start"]
        work_end = work_start + program["mpi_work_size"]

        def local_range(offset, N_term):
            start = min(max(work_start - offset, 0), N_term)
            end = min(max(work_end - offset, 0), N_term)
            return start, end

        seed_iteration = np.uint64(self.seed * 1000003 + n)
        if self.linear_z:
            sample_collision_linear_z(
                *local_range(0, N_collision),
                N_collision,
                N_total,
                seed_iteration,
                self.z_edges,
                self.E_edges,
                self.mu_edges,
                residual.c_left,
                residual.c_right,
                residual.psi_left,
                residual.psi_right,
                residual.delta,
                residual.collision_mass,
                *xs_arrays,
                cm,
                program,
                self.shape[3] > 1,  # antithetic energy pairs for the linear basis
            )
        else:
            mass = np.concatenate(
                (residual.collision_mass.ravel(), residual.edge_mass.ravel())
            )
            self._sample_collision_edge(
                local_range,
                N_collision,
                N_total,
                seed_iteration,
                psi,
                residual,
                mass,
                xs_arrays,
                cm,
                program,
            )
        offset = N_collision
        self._sample_corrections(
            local_range,
            offset,
            N_scatter,
            N_fission,
            N_total,
            seed_iteration,
            integrated,
            terms if integrated else None,
            psi,
            psi_left,
            psi_right,
            cm,
            program,
            data,
        )
        return self._read_epsilon(program, data)

    def _sample_collision_edge(
        self,
        local_range,
        N_collision,
        N_total,
        seed_iteration,
        psi,
        residual,
        mass,
        xs_arrays,
        cm,
        program,
    ):
        sample_collision_edge(
            *local_range(0, N_collision),
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
            *xs_arrays,
            cm,
            program,
            self.shape[3] > 1,  # antithetic energy pairs for the linear basis
        )

    def _sample_corrections(
        self,
        local_range,
        offset,
        N_scatter,
        N_fission,
        N_total,
        seed_iteration,
        integrated,
        terms,
        psi,
        psi_left,
        psi_right,
        cm,
        program,
        data,
    ):
        for N_term, M_term, emission, tables, salt in (
            (N_scatter, self.M_scatter, self.emission_scatter, self.tables_scatter, 1),
            (N_fission, self.M_fission, self.emission_fission, self.tables_fission, 2),
        ):
            if integrated:
                S_bar, S_right, mass = terms[salt - 1]
                sample_correction_integrated(
                    *local_range(offset, N_term),
                    N_term,
                    N_total,
                    np.uint64(seed_iteration * 7 + salt),
                    self.z_edges,
                    self.E_edges,
                    self.mu_edges,
                    psi_left,
                    S_bar,
                    psi_right,
                    S_right,
                    self.linear_z,
                    mass,
                    cm,
                    *tables,
                    program,
                    data,
                )
                offset += N_term
                continue
            sample_correction(
                *local_range(offset, N_term),
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
            offset += N_term

    def _read_epsilon(self, program, data):
        """Transport the banked residual source; eps~ from the tally (all ranks)."""
        tally_module.closeout.reset_statistics(program, data)
        simulation_module.fixed_source_simulation(self.simulation_container, data)
        record = program["tallies"][self.epsilon_tally.ID]
        start = record["bin_mean_offset"]
        mean = data[start : start + record["bin_length"]].reshape(
            self.epsilon_tally.bin_shape
        )
        # bin shape: (mu, azi, energy, time, x, y, z, score); score index b P + a is
        #   the flux times P_a(energy) times P_b(z in the cell)
        P = self.shape[3]

        def moment(a, b):
            return np.transpose(mean[:, 0, :, 0, 0, 0, :, b * P + a], (2, 1, 0))

        if self.linear_z:
            # Hat-function moments t[i] = int eps h_i P_a, then the L2 projection
            m0 = np.stack([moment(a, 0) for a in range(P)], axis=-1)
            m1 = np.stack([moment(a, 1) for a in range(P)], axis=-1)
            epsilon = self.projection.project(node_moments(m0, m1))
        else:
            # Coefficient (2a + 1) score / volume
            epsilon = np.empty(self.shape)
            for a in range(P):
                epsilon[..., a] = (2 * a + 1) * moment(a, 0) / self.volume
        epsilon = np.ascontiguousarray(epsilon)
        MPI.COMM_WORLD.Bcast(epsilon, root=0)
        return epsilon


def run(
    simulation,
    z_edges,
    E_edges,
    mu_edges,
    Q,
    cell_materials,
    N_iteration,
    N_per_bin,
    N_correction=0,
    collision_fraction=0.9,
    proposal="defensive",
    defensive_fraction=0.5,
    quadrature_tol=1e-8,
    quadrature_tol_low=1e-8,
    stop="fixed",
    tol=1e-10,
    boundary=("reflective", "reflective"),
    cache_dir=None,
    seed=1,
    collision_xs="binned",
    correction_sampler="integrated",
    energy_basis="constant",
    spatial_basis="constant",
):
    """
    Solve a fixed-source CE problem with Residual Monte Carlo, in two phases:

    1. Collision-only iterations: the residual uses the binned in-scatter (no
       scattering/fission correction) and, by default, the bin-averaged Sigma_t
       (piecewise-constant cross sections, as in the dissertation). Its fixed point is
       the flat-flux-weighted multigroup solution, reached exponentially.
    2. N_correction passes with the full residual (collision, edge, scattering and
       fission corrections) at that fixed point. Each pass is an unbiased estimate of
       the remaining difference to the true bin averages; the passes are averaged
       (not accumulated) and added to the fixed point. See mcdc/rmc/NOTES.md.

    Parameters
    ----------
    simulation : mcdc.Simulation
        The model (cells, materials, settings). Its geometry must be a slab along z
        (reflective in x and y), consistent with `z_edges` and `cell_materials`.
    z_edges, E_edges, mu_edges : array_like
        Trial-space grids. E_edges [eV] also define the neutron energy window;
        mu_edges (polar cosine about z) must contain 0.
    Q : ndarray (K, G, J) or (K, G, J, P)
        Projected external source density: bin averages, and optionally the P_1
        coefficients for the linear energy basis (zero if omitted).
    cell_materials : sequence of mcdc.Material
        Material of each trial-space z cell.
    N_iteration : int
        Maximum number of collision-only iterations.
    N_per_bin : float
        Histories per trial-space bin per iteration (and per correction pass); the
        total, N_per_bin * K * G * J, is rounded to an integer.
    N_correction : int
        Number of final full-residual correction passes (0: none; the result is the
        collision-only fixed point, as in the NSE article).
    collision_fraction : float
        In correction passes, the fraction of histories assigned to r_c + r_e; the
        rest go to the scattering and fission corrections in proportion to their
        binned source magnitudes.
    proposal : {"defensive", "uniform-linear", "uniform-lethargy"}
        Proposal for the scattering/fission corrections.
    stop : {"fixed", "tol"}
        Run N_iteration collision-only iterations, or stop once
        ||eps~||_2 / ||psi~||_2 < tol.
    boundary : (str, str)
        Boundary conditions at z_low and z_high: "reflective" or "vacuum".
    collision_xs : {"binned", "pointwise"}
        Removal in the collision-only iterations: projected onto the trial space
        (P[Sigma_t psi~]; the bin-averaged Sigma_t for the constant basis), or
        pointwise Sigma_t(E) psi~(E).
        Pointwise keeps the in-bin shape of Sigma_t(E) psi~ in every iteration; that
        part of the residual never vanishes, so the iteration stalls at the noise of
        one iteration (see mcdc/rmc/NOTES.md). Correction passes always use the
        pointwise Sigma_t(E).
    correction_sampler : {"integrated", "pointwise"}
        Scattering/fission correction in the correction passes: "integrated" samples
        (z, E_out, mu_out) per bin and integrates the incident energy
        deterministically (weights ~ |T - S_bar|); "pointwise" samples (E_in, E_out,
        mu_out) with `proposal` (weights ~ the in-scatter density, much noisier).
    energy_basis : {"constant", "linear"}
        Trial space in energy on each bin: constant, or linear discontinuous
        (P_0, P_1 Legendre coefficients; transfer moments 4x larger).
    spatial_basis : {"constant", "linear"}
        Trial space in z: piecewise constant on the cells, or continuous piecewise
        linear on the nodes z_edges with the boundary conditions imposed on it (no
        face residual; result.psi then holds nodal values and result.psi_cell the
        cell averages). See writeups/continuous_linear_space.tex.

    Returns
    -------
    RMCResult
    """
    if stop not in ("fixed", "tol"):
        print_error(f"RMC: unknown stop {stop}")
    time_start = MPI.Wtime()
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
        collision_xs,
        correction_sampler,
        energy_basis,
        spatial_basis,
    )
    result = RMCResult(
        solver.z_edges, solver.E_edges, solver.mu_edges, energy_basis, spatial_basis
    )
    result.N_history = solver.N_total
    result.time_precompute = MPI.Wtime() - time_start

    # L2 norm of a trial-space function (per unit bin volume): P_a has mean square
    #   1 / (2a + 1) on a bin
    P = solver.shape[3]

    def norm(coefficients):
        return np.sqrt(np.sum(coefficients**2 / (2.0 * np.arange(P) + 1.0)))

    def slope(coefficients):
        return coefficients[..., 1].copy() if P > 1 else None

    # Phase 1: collision-only iterations
    psi = np.zeros(solver.shape)
    for n in range(N_iteration):
        time_start = MPI.Wtime()
        epsilon = solver.iterate(psi, n, corrections=False)
        result.time_iteration.append(MPI.Wtime() - time_start)
        psi = psi + epsilon
        result.psi_history.append(psi[..., 0].copy())
        result.epsilon_norm.append(norm(epsilon))
        if stop == "tol" and norm(epsilon) < tol * norm(psi):
            break
    result.psi_fixed_point = psi[..., 0].copy()
    result.psi_slope_fixed_point = slope(psi)

    # Phase 2: averaged full-residual corrections at the fixed point
    corrections = []
    for m in range(N_correction):
        time_start = MPI.Wtime()
        corrections.append(solver.iterate(psi, N_iteration + m))
        result.time_correction.append(MPI.Wtime() - time_start)
        result.corrections.append(corrections[-1][..., 0])
        if P > 1:
            result.corrections_slope.append(corrections[-1][..., 1])
    if corrections:
        psi = psi + np.mean(corrections, axis=0)
    result.psi = psi[..., 0].copy()
    result.psi_slope = slope(psi)
    return result
