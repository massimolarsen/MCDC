import numpy as np
import mcdc

# =============================================================================
# CARRE project — 1U CubeSat simplified geometry
#
# Geometry is modeled with rectangular boxes. The CubeSat footprint is
# x/y = 0->10 cm, the large rails span z = 0->11 cm, and the board stack is
# vertically centered in that rail height. A 1 m boundary cube surrounds the
# model and is centered at (5, 5, 5).
#
# Continuous-energy materials use the local HDF5 nuclear data library.
# Nuclide compositions are atom densities in atoms/barn-cm.
# =============================================================================

# =============================================================================
# MATERIALS
# =============================================================================

# Al7075, rho=2.81 g/cm3, wt%: 90 Al-27, 5 Zn-64, 3 Mg-24, 2 Cu-63.
m_al7075 = mcdc.Material(
    name="Al7075",
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
    nuclide_composition={
        "Al27": 0.059057360465045,
        "Mg24": 0.00081349602416576,
        "Si28": 0.00046494828605733,
    },
)
# FR4 proxy, rho=1.85 g/cm3, wt%: 28.0 Si-28, 51.9 O-16, 18.3 C-12, 1.8 H-1.
m_epoxy = mcdc.Material(
    name="FR4_proxy",
    nuclide_composition={
        "Si28": 0.01115014871193,
        "O16": 0.036149980092044,
        "C12": 0.01698996461915,
        "H1": 0.019898026035757,
    },
)
# Silicon, rho=2.329 g/cm3, wt%: 100 Si-28.
m_silicon = mcdc.Material(name="Silicon", nuclide_composition={"Si28": 0.050132618436459})
# LiCoO2 proxy, rho=5.05 g/cm3, wt%: 6.661 Li-7, 60.551 Co-59, 32.788 O-16.
m_licoo2 = mcdc.Material(
    name="LiCoO2_proxy",
    nuclide_composition={
        "Li7": 0.028874848294873,
        "Co59": 0.031246477802075,
        "O16": 0.062341083116754,
    },
)
# Copper, rho=8.96 g/cm3, wt%: 68.4792 Cu-63, 31.5208 Cu-65.
m_copper = mcdc.Material(
    name="Copper",
    nuclide_composition={
        "Cu63": 0.058716830763647,
        "Cu65": 0.026195433536638,
    },
)
# Void, rho=0.0 g/cm3, wt%: 100 void.
m_void = mcdc.Material(name="Void", nuclide_composition={"Si28": 0.0})

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

# =============================================================================
# OUTER BOUNDARY
# 100 cm x 100 cm x 100 cm vacuum cube centered on the CubeSat.
# =============================================================================

boundary_center = np.array([5.0, 5.0, 5.0])
boundary_half_width = 50.0

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
obc_board = mcdc.Cell(region=box(0.5, 9.5, 0.5, 9.5, 2.285, 2.445), fill=m_epoxy)
# OBC sensitive volume: silicon, 3.6 x 2.7 x 0.6 cm
obc_sv    = mcdc.Cell(region=box(3.2, 6.8, 3.65, 6.35, 2.445, 3.045), fill=m_silicon)

# EPS board, z = 3.785 -> 3.945 cm
eps_board = mcdc.Cell(region=box(0.5, 9.5, 0.5, 9.5, 3.785, 3.945), fill=m_epoxy)
# EPS sensitive volume: LiCoO2, 2.5 x 6.5 x 0.23 cm
eps_sv    = mcdc.Cell(region=box(3.75, 6.25, 1.75, 8.25, 3.945, 4.175), fill=m_licoo2)

# ADCS board, z = 5.785 -> 5.945 cm
adcs_board = mcdc.Cell(region=box(0.5, 9.5, 0.5, 9.5, 5.785, 5.945), fill=m_epoxy)
# ADCS sensitive volume: silicon, 4.5 x 4.5 x 1.8 cm
adcs_sv = mcdc.Cell(region=box(2.75, 7.25, 2.75, 7.25, 5.945, 7.745), fill=m_silicon)

# Comms board, z = 8.285 -> 8.445 cm
comms_board = mcdc.Cell(region=box(0.5, 9.5, 0.5, 9.5, 8.285, 8.445), fill=m_epoxy)
# Comms sensitive volume: silicon, 1.1 x 0.97 x 0.27 cm
comms_sv    = mcdc.Cell(region=box(4.45, 5.55, 4.515, 5.485, 8.445, 8.715), fill=m_silicon)

# =============================================================================
# VOID FILL
# All remaining space inside the boundary cube.
# =============================================================================

all_component_cells = (
    rail_cells + small_rail_cells + shear_cells + solar_cells + antenna_cells +
    [obc_board, obc_sv,
     eps_board, eps_sv,
     adcs_board, adcs_sv] +
    [
     comms_board, comms_sv]
)

void_region = outer
for c in all_component_cells:
    void_region = void_region & ~c.region

void_cell = mcdc.Cell(region=void_region, fill=m_void)

# =============================================================================
# SOURCE
# Monoenergetic CE source, isotropic across all six boundary-cube faces.
# =============================================================================

source_inset = 1.0e-6  # Keep source points just inside the vacuum boundary.
source_energy_ev = 1.0e6

boundary_sources = [
    dict(
        x=[boundary_x0 + source_inset, boundary_x0 + source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[boundary_x1 - source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y0 + source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y1 - source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z0 + source_inset],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z1 - source_inset, boundary_z1 - source_inset],
    ),
]

for source_bounds in boundary_sources:
    mcdc.Source(
        **source_bounds,
        isotropic=True,
        energy=source_energy_ev,
        probability=1.0 / 6.0,
    )

# =============================================================================
# TALLIES
# =============================================================================

mcdc.Tally(name="OBC SV current-in", cell=obc_sv, scores=["current-in"])
mcdc.Tally(name="EPS SV current-in", cell=eps_sv, scores=["current-in"])
mcdc.Tally(name="ADCS SV current-in", cell=adcs_sv, scores=["current-in"])
mcdc.Tally(name="Comms SV current-in", cell=comms_sv, scores=["current-in"])


# =============================================================================
# SETTINGS AND RUN
# =============================================================================

mcdc.settings.N_particle = 100
mcdc.settings.output_name = "cubesat_CE"
mcdc.run()
