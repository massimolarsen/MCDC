from pathlib import Path
import os

import numpy as np
import mcdc

from cubesat_G4_devices import build_detector_sizes_mm, build_device_components

EXAMPLE_DIR = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get("MCDC_RUN_DIR", EXAMPLE_DIR)).resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(RUN_DIR)

# =============================================================================
# CARRE project — 1U CubeSat simplified geometry
#
# Geometry is modeled with rectangular boxes. The CubeSat footprint is
# x/y = 0->10 cm, the large rails span z = 0->11 cm, and the board stack is
# vertically centered in that rail height. A 15 cm boundary cube surrounds the
# model and is centered at (5, 5, 5).
#
# Continuous-energy materials use the local HDF5 nuclear data library.
# Nuclide compositions are atom densities in atoms/barn-cm.
#
# Neutron + proton run: ECSS neutrons and AP9 trapped protons, sampled in equal
# numbers, both up to 200 MeV. MCDC_LIB must hold combined neutron + proton
# nuclide files at 300 K (hdf5libProton/with_neutron_200MeV): JENDL-5 neutrons
# (FENDL-3.0 for Li7) and TENDL-2021 protons with Geant4 stopping powers, see
# tools/data_library_generator/proton.
# =============================================================================

# Library temperature of the 200 MeV neutron + proton data
XS_TEMPERATURE_K = 300.0

# =============================================================================
# MATERIALS
# =============================================================================

# Al7075, rho=2.81 g/cm3, wt%: 90 Al-27, 5 Zn-64, 3 Mg-24, 2 Cu-63.
m_al7075 = mcdc.Material(
    name="Al7075",
    temperature=XS_TEMPERATURE_K,
    nuclide_composition={
        "Al27": 0.056445980580537,
        "Zn64": 0.0013235134207385,
        "Mg24": 0.0021165961369498,
        "Cu63": 0.0005378141989737,
    },
)
# Al6061, rho=2.70 g/cm3, wt%: 98 Al-27, 1.2 Mg-24, 0.8 Si-28.
m_al6061 = mcdc.Material(
    name="Al6061",
    temperature=XS_TEMPERATURE_K,
    nuclide_composition={
        "Al27": 0.059057360465045,
        "Mg24": 0.00081349602416576,
        "Si28": 0.00046494828605733,
    },
)
# FR4 proxy, rho=1.85 g/cm3, wt%: 28.0 Si-28, 51.9 O-16, 18.3 C-12, 1.8 H-1.
m_epoxy = mcdc.Material(
    name="FR4_proxy",
    temperature=XS_TEMPERATURE_K,
    nuclide_composition={
        "Si28": 0.01115014871193,
        "O16": 0.036149980092044,
        "C12": 0.01698996461915,
        "H1": 0.019898026035757,
    },
)
# Silicon, rho=2.329 g/cm3, wt%: 100 Si-28.
m_silicon = mcdc.Material(
    name="Silicon",
    temperature=XS_TEMPERATURE_K,
    nuclide_composition={"Si28": 0.050132618436459},
)
# LiCoO2 proxy, rho=5.05 g/cm3, wt%: 6.661 Li-7, 60.551 Co-59, 32.788 O-16.
m_licoo2 = mcdc.Material(
    name="LiCoO2_proxy",
    temperature=XS_TEMPERATURE_K,
    nuclide_composition={
        "Li7": 0.028874848294873,
        "Co59": 0.031246477802075,
        "O16": 0.062341083116754,
    },
)
# Copper, rho=8.96 g/cm3, wt%: 68.4792 Cu-63, 31.5208 Cu-65.
m_copper = mcdc.Material(
    name="Copper",
    temperature=XS_TEMPERATURE_K,
    nuclide_composition={
        "Cu63": 0.058716830763647,
        "Cu65": 0.026195433536638,
    },
)
# Void, rho=0.0 g/cm3, wt%: 100 void.
m_void = mcdc.Material(
    name="Void", temperature=XS_TEMPERATURE_K, nuclide_composition={"Si28": 0.0}
)

# =============================================================================
# BOX HELPER
# =============================================================================

def box(x0, x1, y0, y1, z0, z1, bcx0='none', bcx1='none',
        bcy0='none', bcy1='none', bcz0='none', bcz1='none'):
    sx0 = mcdc.Surface.PlaneX(x=x0, boundary_condition=bcx0)
    sx1 = mcdc.Surface.PlaneX(x=x1, boundary_condition=bcx1)
    sy0 = mcdc.Surface.PlaneY(y=y0, boundary_condition=bcy0)
    sy1 = mcdc.Surface.PlaneY(y=y1, boundary_condition=bcy1)
    sz0 = mcdc.Surface.PlaneZ(z=z0, boundary_condition=bcz0)
    sz1 = mcdc.Surface.PlaneZ(z=z1, boundary_condition=bcz1)
    return +sx0 & -sx1 & +sy0 & -sy1 & +sz0 & -sz1


def stacked_board_sv(board_bounds, sv_bounds, board_fill, sv_fill):
    bx0, bx1, by0, by1, bz0, bz1 = board_bounds
    sx0, sx1, sy0, sy1, _, sz1 = sv_bounds

    board_x0 = mcdc.Surface.PlaneX(x=bx0)
    board_x1 = mcdc.Surface.PlaneX(x=bx1)
    board_y0 = mcdc.Surface.PlaneY(y=by0)
    board_y1 = mcdc.Surface.PlaneY(y=by1)
    board_z0 = mcdc.Surface.PlaneZ(z=bz0)
    # board top and SV bottom use the same surface
    shared_z = mcdc.Surface.PlaneZ(z=bz1)

    sv_x0 = mcdc.Surface.PlaneX(x=sx0)
    sv_x1 = mcdc.Surface.PlaneX(x=sx1)
    sv_y0 = mcdc.Surface.PlaneY(y=sy0)
    sv_y1 = mcdc.Surface.PlaneY(y=sy1)
    sv_z1 = mcdc.Surface.PlaneZ(z=sz1)

    board_region = (
        +board_x0 & -board_x1 & +board_y0 & -board_y1 & +board_z0 & -shared_z
    )
    sv_region = +sv_x0 & -sv_x1 & +sv_y0 & -sv_y1 & +shared_z & -sv_z1
    return (
        mcdc.Cell(region=board_region, fill=board_fill),
        mcdc.Cell(region=sv_region, fill=sv_fill),
    )


def g4_size_mm(bounds, padding_scale):
    x0, x1, y0, y1, z0, z1 = bounds
    return (
        padding_scale * 10.0 * (x1 - x0),
        padding_scale * 10.0 * (y1 - y0),
        padding_scale * 10.0 * (z1 - z0),
    )


def geant4_output_path(filename):
    return str(RUN_DIR / "geant4_h5" / filename)

# =============================================================================
# OUTER BOUNDARY
# 15 cm x 15 cm x 15 cm vacuum cube centered on the CubeSat.
# =============================================================================

boundary_center = np.array([5.0, 5.0, 5.0])
boundary_half_width = 7.5

boundary_x0, boundary_y0, boundary_z0 = boundary_center - boundary_half_width
boundary_x1, boundary_y1, boundary_z1 = boundary_center + boundary_half_width

outer = box(
    boundary_x0, boundary_x1,
    boundary_y0, boundary_y1,
    boundary_z0, boundary_z1,
    bcx0='vacuum', bcx1='vacuum',
    bcy0='vacuum', bcy1='vacuum',
    bcz0='vacuum', bcz1='vacuum',
)

# =============================================================================
# MAIN RAILS
# Al 7075, 0.5 x 0.5 x 11 cm, one at each corner
# =============================================================================

rail_regions = [
    box(0.0, 0.5,  0.0, 0.5,  0.0, 11.0),   # -X -Y corner
    box(9.5, 10.0, 0.0, 0.5,  0.0, 11.0),   # +X -Y corner
    box(0.0, 0.5,  9.5, 10.0, 0.0, 11.0),   # -X +Y corner
    box(9.5, 10.0, 9.5, 10.0, 0.0, 11.0),   # +X +Y corner
]
rail_cells = [mcdc.Cell(region=r, fill=m_al7075) for r in rail_regions]

# =============================================================================
# SMALL RAILS
# Al 7075, 9.0 x 0.5 x 0.5 cm edge-to-edge spans (shrunk from the original
# 0.5361 cm cross-section to match the corner rails' 0.5 cm width exactly,
# so they no longer clip into the Z-face shear panels or each other at the
# corners).
# Four bottom rails and four top rails, centered vertically on the large rails.
# =============================================================================

small_rail_dims = [
    (0.5, 9.5, 0.0, 0.5, 0.5, 1.0),
    (0.5, 9.5, 9.5, 10.0, 0.5, 1.0),
    (0.0, 0.5, 0.5, 9.5, 0.5, 1.0),
    (9.5, 10.0, 0.5, 9.5, 0.5, 1.0),
    (0.5, 9.5, 0.0, 0.5, 10.0, 10.5),
    (0.5, 9.5, 9.5, 10.0, 10.0, 10.5),
    (0.0, 0.5, 0.5, 9.5, 10.0, 10.5),
    (9.5, 10.0, 0.5, 9.5, 10.0, 10.5),
]
small_rail_cells = [mcdc.Cell(region=box(*d), fill=m_al7075) for d in small_rail_dims]

# =============================================================================
# SHEAR PANELS
# Al 6061, 9.0 x 9.0 x 0.3 cm, fit between rail frames
# =============================================================================

shear_panel_dims = [
    (0.0,  0.3,  0.5, 9.5, 1.0, 10.0),     # -X face
    (9.7, 10.0,  0.5, 9.5, 1.0, 10.0),     # +X face
    (0.5, 9.5,  0.0, 0.3, 1.0, 10.0),      # -Y face
    (0.5, 9.5,  9.7, 10.0, 1.0, 10.0),     # +Y face
    (0.5, 9.5,  0.5, 9.5, 0.5, 0.8),       # -Z face
    (0.5, 9.5,  0.5, 9.5, 10.2, 10.5),     # +Z face
]
shear_cells = [mcdc.Cell(region=box(*d), fill=m_al6061) for d in shear_panel_dims]

# =============================================================================
# SOLAR PANELS
# Ten total: two per face except the -Z face.
# Silicon, 3.5 x 8.0 x 0.2 cm, mounted on the outer face of each shear panel
# (shifted outward by the panel's own 0.2 cm thickness so it sits on top of
# the shear panel's exterior surface, not nested inside it).
# =============================================================================

solar_panel_dims = [
    (-0.2, 0.0,  1.0, 9.0, 1.75, 5.25),   # -X face, lower
    (-0.2, 0.0,  1.0, 9.0, 5.75, 9.25),   # -X face, upper
    (10.0, 10.2, 1.0, 9.0, 1.75, 5.25),   # +X face, lower
    (10.0, 10.2, 1.0, 9.0, 5.75, 9.25),   # +X face, upper
    (1.0,  9.0, -0.2, 0.0, 1.75, 5.25),   # -Y face, lower
    (1.0,  9.0, -0.2, 0.0, 5.75, 9.25),   # -Y face, upper
    (1.0,  9.0, 10.0, 10.2, 1.75, 5.25),  # +Y face, lower
    (1.0,  9.0, 10.0, 10.2, 5.75, 9.25),  # +Y face, upper
    (1.0,  9.0,  1.25, 4.75, 10.5, 10.7), # +Z face, lower
    (1.0,  9.0,  5.25, 8.75, 10.5, 10.7), # +Z face, upper
]
solar_cells = [mcdc.Cell(region=box(*d), fill=m_silicon) for d in solar_panel_dims]

# =============================================================================
# ANTENNA
# Four thin deployable-antenna strips (Copper), framing the edges of the -Z
# face rather than one solid slab covering it. Each strip is 0.3 x 0.3 cm in
# cross-section, ~9 cm long, tucked between the rail bottom (z=0) and the -Z
# shear panel (z=0.5) so nothing protrudes past the satellite's outer
# envelope, matching how real stowed tape-spring monopole/dipole antennas
# sit close to the body rather than as a face-covering block.
# =============================================================================

antenna_strip_dims = [
    (0.5, 9.5, 0.5, 0.8, 0.2, 0.5),   # -Y edge
    (0.5, 9.5, 9.2, 9.5, 0.2, 0.5),   # +Y edge
    (0.5, 0.8, 0.8, 9.2, 0.2, 0.5),   # -X edge (trimmed to fit between the Y-edge strips)
    (9.2, 9.5, 0.8, 9.2, 0.2, 0.5),   # +X edge (trimmed to fit between the Y-edge strips)
]
antenna_cells = [mcdc.Cell(region=box(*d), fill=m_copper) for d in antenna_strip_dims]

# =============================================================================
# BOARD STACK
# Epoxy boards are 9.0 x 9.0 x 0.16 cm and centered at x/y = 0.5->9.5.
# Each sensitive volume is mounted directly above its board.
# =============================================================================

# OBC board, z = 2.285 -> 2.445 cm
obc_board_bounds = (0.5, 9.5, 0.5, 9.5, 2.285, 2.445)
# OBC sensitive volume: silicon, 3.6 x 2.7 x 0.6 cm
obc_sv_bounds = (3.2, 6.8, 3.65, 6.35, 2.445, 3.045)
obc_board, obc_sv = stacked_board_sv(obc_board_bounds, obc_sv_bounds, m_epoxy, m_silicon)

# EPS board, z = 3.785 -> 3.945 cm
eps_board_bounds = (0.5, 9.5, 0.5, 9.5, 3.785, 3.945)
# EPS sensitive volume: LiCoO2, 2.5 x 6.5 x 0.23 cm
eps_sv_bounds = (3.75, 6.25, 1.75, 8.25, 3.945, 4.175)
eps_board, eps_sv = stacked_board_sv(eps_board_bounds, eps_sv_bounds, m_epoxy, m_licoo2)

# ADCS board, z = 5.785 -> 5.945 cm
adcs_board_bounds = (0.5, 9.5, 0.5, 9.5, 5.785, 5.945)
# ADCS sensitive volume: silicon, 4.5 x 4.5 x 1.8 cm
adcs_sv_bounds = (2.75, 7.25, 2.75, 7.25, 5.945, 7.745)
adcs_board, adcs_sv = stacked_board_sv(adcs_board_bounds, adcs_sv_bounds, m_epoxy, m_silicon)

# Comms board, z = 8.285 -> 8.445 cm
comms_board_bounds = (0.5, 9.5, 0.5, 9.5, 8.285, 8.445)
# Comms sensitive volume: silicon, 1.1 x 0.97 x 0.27 cm
comms_sv_bounds = (4.45, 5.55, 4.515, 5.485, 8.445, 8.715)
comms_board, comms_sv = stacked_board_sv(comms_board_bounds, comms_sv_bounds, m_epoxy, m_silicon)

sensitive_volumes = [
    ("obc", obc_sv, obc_sv_bounds),
    ("eps", eps_sv, eps_sv_bounds),
    ("adcs", adcs_sv, adcs_sv_bounds),
    ("comms", comms_sv, comms_sv_bounds),
]

device_components = build_device_components()
detector_sizes_mm = build_detector_sizes_mm()

# =============================================================================
# VOID FILL
# All remaining space inside the boundary cube.
# =============================================================================

all_component_cells = (
    rail_cells + small_rail_cells + shear_cells + solar_cells + antenna_cells +
    [obc_board, obc_sv,
     eps_board, eps_sv,
     adcs_board, adcs_sv] +
    [comms_board, comms_sv]
)

void_region = outer
for c in all_component_cells:
    void_region = void_region & ~c.region

void_cell = mcdc.Cell(region=void_region, fill=m_void)

# =============================================================================
# SOURCE
# Tabulated CE source, uniformly white and directed inward across all six
# boundary-cube faces.
# =============================================================================

mu_bins = np.linspace(-1.0, 1.0, 9)
azi_bins = np.linspace(-np.pi, np.pi, 9)
surface_mesh = (5, 5)
source_inset = 1.0e-6  # Keep source points just inside the vacuum boundary.

# Source spectra per species, both truncated at the 200 MeV data limit.
# Neutrons: ECSS, default 100 km altitude (altitude scales the flux, not the
# shape). Protons: AP9 mean trapped protons at 550 km / 97 deg.
SOURCE_SPECTRA = {
    "neutron": EXAMPLE_DIR / "ecss_200MeV.csv",
    "proton": EXAMPLE_DIR / "ap9_200MeV.csv",
}
source_spectra = {}
for particle, path in SOURCE_SPECTRA.items():
    energy_ev, pdf = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    if energy_ev.size < 2: raise ValueError(f"Invalid source spectrum: {path}")
    source_spectra[particle] = (energy_ev, pdf)

# Tally energy grids per species, about 5.6 bins per decade as before.
# Proton-induced neutrons reach 200 MeV and protons slow down to the 250 keV
# transport cutoff, so the grids span the energies that reach each SV.
energy_bins_ev = {
    "neutron": np.geomspace(source_spectra["neutron"][0][0], 2.0e8, 47),
    "proton": np.geomspace(2.5e5, 2.0e8, 17),
}

boundary_sources = [
    dict(
        x=[boundary_x0 + source_inset, boundary_x0 + source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
        white_direction=[1.0, 0.0, 0.0],
    ),
    dict(
        x=[boundary_x1 - source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
        white_direction=[-1.0, 0.0, 0.0],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y0 + source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
        white_direction=[0.0, 1.0, 0.0],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y1 - source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
        white_direction=[0.0, -1.0, 0.0],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z0 + source_inset],
        white_direction=[0.0, 0.0, 1.0],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z1 - source_inset, boundary_z1 - source_inset],
        white_direction=[0.0, 0.0, -1.0],
    ),
]

# Equal numbers of neutron and proton histories: each species gets half the
# source probability, spread evenly over the six faces.
sources = []
for particle, (energy_ev, pdf) in source_spectra.items():
    for source_bounds in boundary_sources:
        sources.append(
            mcdc.Source(
                **source_bounds,
                energy=[energy_ev, pdf],
                particle_type=particle,
                probability=1.0 / (6.0 * len(source_spectra)),
            )
        )

# =============================================================================
# TALLIES
# =============================================================================

# One distribution-source tally per sensitive volume and species
tallies = []
for name, cell, _ in sensitive_volumes:
    for particle in source_spectra:
        tallies.append(
            mcdc.Tally(
                name=f"{name}_{particle}_g4_source",
                cell=cell,
                scores=["current-in", "current-out"],
                particle_type=particle,
                mu=mu_bins,
                azi=azi_bins,
                energy=energy_bins_ev[particle],
                surface_mesh=surface_mesh,
            )
        )


# =============================================================================
# SETTINGS AND RUN
# =============================================================================

simulation = mcdc.Simulation("cubesat_CE_G4_NP")
simulation.set_model(all_component_cells + [void_cell])
simulation.set_sources(sources)
simulation.set_tallies(tallies)

# Proton slowing down (Geant4 QGSP_BIC stopping powers in the proton data);
# steps lose at most 1% of the proton energy on average
simulation.settings.condensed_interactions(max_fractional_energy_loss=0.01)

# set parllel g4 workers
simulation.settings.geant4_max_workers = 4
simulation.settings.geant4_n_threads = 2
simulation.settings.geant4_payload_dir = "geant4_payloads"

BRIDGE_BUILD_DIR = Path(__file__).resolve().parents[3] / "couple-mcdc-g4" / "build"
# Geant4 events per species and region (same total per region as the
# neutron-only run)
n_geant4_particles_per_species = 25000000
geant4_random_seeds = {
    "obc": 10011,
    "eps": 10021,
    "adcs": 10031,
    "comms": 10041,
}

for i, (name, _, bounds) in enumerate(sensitive_volumes):
    handoff = dict(
        name=name,
        bridge_build_dir=str(BRIDGE_BUILD_DIR),
        world_size_mm=g4_size_mm(bounds, padding_scale=1.2),
        detector_size_mm=detector_sizes_mm[name],
        detector_material="G4_Galactic",
        envelope_material="G4_Galactic",
        device_components=device_components[name],
        physics_list="QGSP_BIC_HP",
        em_production_cut_mm=0.001,  # provisional; compare effective thresholds and cut convergence
        record_seu_events=False,  # opt in to selected-event diagnostic tables
        diagnostic_min_Eion_mev=0.001,
        source_mode="distribution",
        species_sources=[
            {
                "particle": particle,
                "tally": f"{name}_{particle}_g4_source",
                "n_events": n_geant4_particles_per_species,
            }
            for particle in source_spectra
        ],
        geant4_output_path=geant4_output_path(f"cubesat_{name}_geant4.h5"),
        random_seed=geant4_random_seeds[name],
    )
    if i == 0:
        mcdc.enable_geant4_handoff(**handoff)
    else:
        mcdc.add_geant4_handoff(**handoff)

simulation.settings.N_particle = 1000000
simulation.settings.output_name = "mcdc_h5/cubesat_CE_G4_NP"
simulation.settings.active_bank_buffer = 2000
simulation.settings.use_progress_bar = False
simulation.run()
