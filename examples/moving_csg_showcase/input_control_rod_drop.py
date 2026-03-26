from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc
from mcdc.visualization.geometry import geo_viewer_3d

# ======================================================================================
# Time-dependent nuclear engineering showcase:
# Control rod dropping into a small fuel assembly.
# ======================================================================================

# Materials (visualization-focused placeholders)
fuel = mcdc.Material(
    name="Fuel (UO2)",
    nuclide_composition={"H1": 0.04, "H1": 0.96, "H1": 2.0},
)
guide = mcdc.Material(name="Guide Tube", nuclide_composition={"H1": 1.0})
coolant = mcdc.Material(name="Coolant", nuclide_composition={"H1": 2.0, "H1": 1.0})
control_rod = mcdc.Material(name="Control Rod", nuclide_composition={"H1": 1.0})
void = mcdc.Material(name="Void", nuclide_composition={"H1": 0.0})

# --------------------------------------------------------------------------------------
# Global container
# --------------------------------------------------------------------------------------
outer = 5.0
xm = mcdc.Surface.PlaneX(x=-outer, boundary_condition="vacuum")
xp = mcdc.Surface.PlaneX(x=+outer, boundary_condition="vacuum")
ym = mcdc.Surface.PlaneY(y=-outer, boundary_condition="vacuum")
yp = mcdc.Surface.PlaneY(y=+outer, boundary_condition="vacuum")
zm = mcdc.Surface.PlaneZ(z=-6.0, boundary_condition="vacuum")
zp = mcdc.Surface.PlaneZ(z=+6.0, boundary_condition="vacuum")
world = +xm & -xp & +ym & -yp & +zm & -zp

# --------------------------------------------------------------------------------------
# Active assembly box
# --------------------------------------------------------------------------------------
core_half_xy = 2.0
core_half_z = 4.0
cxm = mcdc.Surface.PlaneX(x=-core_half_xy)
cxp = mcdc.Surface.PlaneX(x=+core_half_xy)
cym = mcdc.Surface.PlaneY(y=-core_half_xy)
cyp = mcdc.Surface.PlaneY(y=+core_half_xy)
czm = mcdc.Surface.PlaneZ(z=-core_half_z)
czp = mcdc.Surface.PlaneZ(z=+core_half_z)
core = +cxm & -cxp & +cym & -cyp & +czm & -czp

# --------------------------------------------------------------------------------------
# Four static fuel pins
# --------------------------------------------------------------------------------------
pin_r = 0.34
pin_centers = [(-0.9, -0.9), (+0.9, -0.9), (-0.9, +0.9), (+0.9, +0.9)]
pin_regions = []
for cx, cy in pin_centers:
    cyl = mcdc.Surface.CylinderZ(center=[cx, cy], radius=pin_r)
    pin_regions.append((-cyl) & +czm & -czp)

fuel_region = pin_regions[0] | pin_regions[1] | pin_regions[2] | pin_regions[3]

# --------------------------------------------------------------------------------------
# Static central guide tube (annulus)
# --------------------------------------------------------------------------------------
guide_outer = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.56)
guide_inner = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.38)
guide_shell_region = (-guide_outer) & (+guide_inner) & +czm & -czp

# --------------------------------------------------------------------------------------
# Moving control rod: capped cylinder dropping downward through guide tube
# --------------------------------------------------------------------------------------
rod_cyl = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.30)
rod_bot = mcdc.Surface.PlaneZ(z=+3.4)
rod_top = mcdc.Surface.PlaneZ(z=+5.8)

# Drop for 4 seconds, then remain static.
rod_velocity = [[0.0, 0.0, -1.6]]
rod_duration = [4.0]
rod_cyl.move(rod_velocity, rod_duration)
rod_bot.move(rod_velocity, rod_duration)
rod_top.move(rod_velocity, rod_duration)

rod_region = (-rod_cyl) & +rod_bot & -rod_top

# --------------------------------------------------------------------------------------
# Remaining regions
# --------------------------------------------------------------------------------------
core_coolant_region = core & ~fuel_region & ~guide_shell_region & ~rod_region
outer_void_region = world & ~core & ~rod_region

# --------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------
mcdc.Cell(name="fuel_pins", region=fuel_region, fill=fuel)
mcdc.Cell(name="guide_tube", region=guide_shell_region, fill=guide)
mcdc.Cell(name="control_rod_drop", region=rod_region, fill=control_rod)
mcdc.Cell(name="core_coolant", region=core_coolant_region, fill=coolant)
mcdc.Cell(name="outer_void", region=outer_void_region, fill=void)

# --------------------------------------------------------------------------------------
# Visualize only
# --------------------------------------------------------------------------------------
sim = mcdc.object_.simulation.simulation

time_steps = np.linspace(0.0, 6.0, 31)
geo_viewer_3d(
    sim,
    primitive_resolution=56,
    time_steps=time_steps,
    dynamic_bounds=False,
    save_animation_path="control_rod_drop.gif",
    animation_fps=15,
)

# Keyboard controls:
#   Left/Right or p/n to step through precomputed frames
