import argparse

import numpy as np
import mcdc
from mcdc.viewer import geo_viewer_3d


def bool_arg(value):
    if value.lower() in {"true", "1", "yes", "on"}:
        return True
    if value.lower() in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


parser = argparse.ArgumentParser()
parser.add_argument("--labels", type=bool_arg, default=True)
parser.add_argument("--color-by", choices=["material", "cell"], default="material")
parser.add_argument("--opacity-slider", type=bool_arg, default=True)
args = parser.parse_args()

# =============================================================================
# CARRE project — 1U CubeSat simplified geometry
#
# Geometry is modeled with rectangular boxes. The CubeSat footprint is
# x/y = 0->10 cm, the large rails span z = 0->11 cm, and the board stack is
# vertically centered in that rail height. A 1 m boundary cube surrounds the
# model and is centered at (5, 5, 5).
#
# Cross sections are placeholders; replace them with real nuclear data before
# production analysis.
# =============================================================================

# =============================================================================
# MATERIALS
# =============================================================================

m_al7075 = mcdc.MaterialMG(
    name="Al7075", capture=np.array([0.02815]), scatter=np.array([[0.01408]])
)
m_al6061 = mcdc.MaterialMG(
    name="Al6061", capture=np.array([0.02700]), scatter=np.array([[0.01350]])
)
m_epoxy = mcdc.MaterialMG(
    name="Epoxy", capture=np.array([0.01200]), scatter=np.array([[0.00600]])
)
m_silicon = mcdc.MaterialMG(
    name="Silicon", capture=np.array([0.02330]), scatter=np.array([[0.01165]])
)
m_licoo2 = mcdc.MaterialMG(
    name="LiCoO2", capture=np.array([0.05030]), scatter=np.array([[0.02515]])
)
m_copper = mcdc.MaterialMG(
    name="Copper", capture=np.array([0.08960]), scatter=np.array([[0.04480]])
)
m_void = mcdc.MaterialMG(
    name="Vacuum", capture=np.array([0.00000]), scatter=np.array([[0.00000]])
)

# =============================================================================
# BOX HELPER
# =============================================================================


def box(
    x0,
    x1,
    y0,
    y1,
    z0,
    z1,
    bcx0="none",
    bcx1="none",
    bcy0="none",
    bcy1="none",
    bcz0="none",
    bcz1="none",
):
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
    boundary_x0,
    boundary_x1,
    boundary_y0,
    boundary_y1,
    boundary_z0,
    boundary_z1,
    bcx0="vacuum",
    bcx1="vacuum",
    bcy0="vacuum",
    bcy1="vacuum",
    bcz0="vacuum",
    bcz1="vacuum",
)

# =============================================================================
# MAIN RAILS
# Al 7075, 0.5 x 0.5 x 11 cm, one at each corner
# =============================================================================

rail_regions = [
    box(0.0, 0.5, 0.0, 0.5, 0.0, 11.0),  # -X -Y corner
    box(9.5, 10.0, 0.0, 0.5, 0.0, 11.0),  # +X -Y corner
    box(0.0, 0.5, 9.5, 10.0, 0.0, 11.0),  # -X +Y corner
    box(9.5, 10.0, 9.5, 10.0, 0.0, 11.0),  # +X +Y corner
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
    (0.0, 0.3, 0.5, 9.5, 1.0, 10.0),  # -X face
    (9.7, 10.0, 0.5, 9.5, 1.0, 10.0),  # +X face
    (0.5, 9.5, 0.0, 0.3, 1.0, 10.0),  # -Y face
    (0.5, 9.5, 9.7, 10.0, 1.0, 10.0),  # +Y face
    (0.5, 9.5, 0.5, 9.5, 0.5, 0.8),  # -Z face
    (0.5, 9.5, 0.5, 9.5, 10.2, 10.5),  # +Z face
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
    (-0.2, 0.0, 1.0, 9.0, 1.75, 5.25),  # -X face, lower
    (-0.2, 0.0, 1.0, 9.0, 5.75, 9.25),  # -X face, upper
    (10.0, 10.2, 1.0, 9.0, 1.75, 5.25),  # +X face, lower
    (10.0, 10.2, 1.0, 9.0, 5.75, 9.25),  # +X face, upper
    (1.0, 9.0, -0.2, 0.0, 1.75, 5.25),  # -Y face, lower
    (1.0, 9.0, -0.2, 0.0, 5.75, 9.25),  # -Y face, upper
    (1.0, 9.0, 10.0, 10.2, 1.75, 5.25),  # +Y face, lower
    (1.0, 9.0, 10.0, 10.2, 5.75, 9.25),  # +Y face, upper
    (1.0, 9.0, 1.25, 4.75, 10.5, 10.7),  # +Z face, lower
    (1.0, 9.0, 5.25, 8.75, 10.5, 10.7),  # +Z face, upper
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
    (0.5, 9.5, 0.5, 0.8, 0.2, 0.5),  # -Y edge
    (0.5, 9.5, 9.2, 9.5, 0.2, 0.5),  # +Y edge
    (
        0.5,
        0.8,
        0.8,
        9.2,
        0.2,
        0.5,
    ),  # -X edge (trimmed to fit between the Y-edge strips)
    (
        9.2,
        9.5,
        0.8,
        9.2,
        0.2,
        0.5,
    ),  # +X edge (trimmed to fit between the Y-edge strips)
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
obc_sv = mcdc.Cell(region=box(3.2, 6.8, 3.65, 6.35, 2.445, 3.045), fill=m_silicon)

# EPS board, z = 3.785 -> 3.945 cm
eps_board = mcdc.Cell(region=box(0.5, 9.5, 0.5, 9.5, 3.785, 3.945), fill=m_epoxy)
# EPS sensitive volume: LiCoO2, 2.5 x 6.5 x 0.23 cm
eps_sv = mcdc.Cell(region=box(3.75, 6.25, 1.75, 8.25, 3.945, 4.175), fill=m_licoo2)

# ADCS board, z = 5.785 -> 5.945 cm
adcs_board = mcdc.Cell(region=box(0.5, 9.5, 0.5, 9.5, 5.785, 5.945), fill=m_epoxy)
# ADCS sensitive volume: silicon, 4.5 x 4.5 x 1.8 cm
adcs_sv = mcdc.Cell(region=box(2.75, 7.25, 2.75, 7.25, 5.945, 7.745), fill=m_silicon)

# Comms board, z = 8.285 -> 8.445 cm
comms_board = mcdc.Cell(region=box(0.5, 9.5, 0.5, 9.5, 8.285, 8.445), fill=m_epoxy)
# Comms sensitive volume: silicon, 1.1 x 0.97 x 0.27 cm
comms_sv = mcdc.Cell(region=box(4.45, 5.55, 4.515, 5.485, 8.445, 8.715), fill=m_silicon)

# =============================================================================
# VOID FILL
# All remaining space inside the boundary cube.
# =============================================================================

all_component_cells = (
    rail_cells
    + small_rail_cells
    + shear_cells
    + solar_cells
    + antenna_cells
    + [obc_board, obc_sv, eps_board, eps_sv, adcs_board, adcs_sv]
    + [comms_board, comms_sv]
)

void_region = outer
for c in all_component_cells:
    void_region = void_region & ~c.region

void_cell = mcdc.Cell(name="vacuum", region=void_region, fill=m_void)

# =============================================================================
# SOURCE
# Isotropic source distributed across thin slabs on all six boundary-cube faces.
# =============================================================================

source_inset = 1.0e-6  # Keep source points just inside the vacuum boundary.
source_marker_thickness = 0.1  # Give the 3D viewer finite source-marker volume.

boundary_sources = [
    dict(
        x=[
            boundary_x0 + source_inset,
            boundary_x0 + source_inset + source_marker_thickness,
        ],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[
            boundary_x1 - source_inset - source_marker_thickness,
            boundary_x1 - source_inset,
        ],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[
            boundary_y0 + source_inset,
            boundary_y0 + source_inset + source_marker_thickness,
        ],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[
            boundary_y1 - source_inset - source_marker_thickness,
            boundary_y1 - source_inset,
        ],
        z=[boundary_z0 + source_inset, boundary_z1 - source_inset],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[
            boundary_z0 + source_inset,
            boundary_z0 + source_inset + source_marker_thickness,
        ],
    ),
    dict(
        x=[boundary_x0 + source_inset, boundary_x1 - source_inset],
        y=[boundary_y0 + source_inset, boundary_y1 - source_inset],
        z=[
            boundary_z1 - source_inset - source_marker_thickness,
            boundary_z1 - source_inset,
        ],
    ),
]

for source_bounds in boundary_sources:
    mcdc.Source(**source_bounds, isotropic=True, probability=1.0 / 6.0)

# =============================================================================
# TALLIES
# =============================================================================

mcdc.Tally(name="OBC SV current-in", cell=obc_sv, scores=["current-in"])
mcdc.Tally(name="EPS SV current-in", cell=eps_sv, scores=["current-in"])
mcdc.Tally(name="ADCS SV current-in", cell=adcs_sv, scores=["current-in"])
mcdc.Tally(name="Comms SV current-in", cell=comms_sv, scores=["current-in"])


# =============================================================================
# SETTINGS AND VIEWER
# =============================================================================

mcdc.settings.N_particle = 100
geo_viewer_3d(
    mcdc.object_.simulation.simulation,
    labels=args.labels,
    color_by=args.color_by,
    opacity_slider=args.opacity_slider,
)
