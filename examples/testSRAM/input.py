import numpy as np
import mcdc
#from mcdc.tools.visualize_geometry import visualize_simulation
from mcdc.object_.tools.visualize_geometry import visualizer_3d

# =============================================================================
# Materials (Continuous-Energy)
# =============================================================================
# Defined by nuclide composition for continuous-energy lookup
#silicon  = mcdc.Material(name="Silicon", nuclide_composition={"Si28": 1.0})
#aluminum = mcdc.Material(name="Aluminum", nuclide_composition={"Al27": 1.0})
silicon  = mcdc.Material(name="Silicon", nuclide_composition={"Si28": 1.0})
aluminum = mcdc.Material(name="Aluminum", nuclide_composition={"H1": 1.0})
# Vacuum modeled as low-density Hydrogen as per original input logic
vacuum   = mcdc.Material(name="Vacuum", nuclide_composition={"H1": 0.0})

# =============================================================================
# Geometry / Surfaces (cm conversion: 1 um = 1e-4 cm)
# =============================================================================

# X-Y Extents (Half-widths)
w_bulk = 250.0e-4   # Total 50 um -> +/- 25 um
w_sv   = 5.0e-4    # Total 10 um -> +/- 5 um

x_bulk_pos = mcdc.Surface.PlaneX(x=w_bulk)
x_bulk_neg = mcdc.Surface.PlaneX(x=-w_bulk)
y_bulk_pos = mcdc.Surface.PlaneY(y=w_bulk)
y_bulk_neg = mcdc.Surface.PlaneY(y=-w_bulk)

x_sv_pos = mcdc.Surface.PlaneX(x=w_sv)
x_sv_neg = mcdc.Surface.PlaneX(x=-w_sv)
y_sv_pos = mcdc.Surface.PlaneY(y=w_sv)
y_sv_neg = mcdc.Surface.PlaneY(y=-w_sv)

# Z-Planes (Depths)
z_beol_top = mcdc.Surface.PlaneZ(z=0.05e-4)   # Metal1 top surface
z_surface  = mcdc.Surface.PlaneZ(z=0.0)       # Al/Si interface
z_sv_top   = mcdc.Surface.PlaneZ(z=-2.4e-4)   # SV 200nm thick centered at -2.5um
z_sv_bot   = mcdc.Surface.PlaneZ(z=-2.6e-4)   
z_bulk_bot = mcdc.Surface.PlaneZ(z=-5.0e-4)   # 5um Bulk depth

# Outer bounding box (Exactly as original input)
box_extent = 5.0
x_min = mcdc.Surface.PlaneX(x=-box_extent, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=box_extent,  boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=-box_extent, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=box_extent,  boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=-box_extent, boundary_condition="vacuum")
z_max = mcdc.Surface.PlaneZ(z=box_extent,  boundary_condition="vacuum")

box_extent2 = 1.0
x_min2 = mcdc.Surface.PlaneX(x=-box_extent2,)
x_max2 = mcdc.Surface.PlaneX(x=box_extent2, )
y_min2 = mcdc.Surface.PlaneY(y=-box_extent2,)
y_max2 = mcdc.Surface.PlaneY(y=box_extent2, )
z_min2 = mcdc.Surface.PlaneZ(z=-box_extent2,)
z_max2 = mcdc.Surface.PlaneZ(z=box_extent2, )

# =============================================================================
# Cells
# =============================================================================

# BEOL Layer (Metal 1)
mcdc.Cell(region=+z_surface & -z_beol_top & +x_bulk_neg & -x_bulk_pos & +y_bulk_neg & -y_bulk_pos, fill=aluminum)

# Sensitive Volume (SRAM cell region)
sv_region = +z_sv_bot & -z_sv_top & +x_sv_neg & -x_sv_pos & +y_sv_neg & -y_sv_pos
mcdc.Cell(region=sv_region, fill=silicon)

# Bulk Substrate (Excluding the SV volume)
bulk_region = +z_bulk_bot & -z_surface & +x_bulk_neg & -x_bulk_pos & +y_bulk_neg & -y_bulk_pos
mcdc.Cell(region=bulk_region & ~sv_region, fill=silicon)

# Surrounding Vacuum (Fills the rest of the 5cm box)
mcdc.Cell(region=+x_min2 & -x_max2 & +y_min2 & -y_max2 & +z_min2 & -z_max2, fill=silicon)
mcdc.Cell(region=+x_min & -x_max & +y_min & -y_max & +z_min & -z_max, fill=vacuum)

# =============================================================================
# Source, Tallies, and Settings (Same as original)
# =============================================================================
mcdc.Source(x=[-4.5, 4.5], y=[-4.5, 4.5], z=[-4.5, 4.5], isotropic=True, energy=1e6)

energy_grid = np.loadtxt("energy_grid.txt")

mcdc.TallyGlobal(
    scores=["flux"],
    multipliers=["energy"],
    energy=energy_grid,
)

mcdc.TallySurface(
    surface=z_beol_top, # Net current through the very top surface
    scores=["net-current"],
    energy=energy_grid,
)

mcdc.settings.N_particle = 1000

if __name__ == "__main__":
    sim = mcdc.object_.simulation.simulation
    print("Cells and their surfaces:")
    for cell in sim.cells:
        s_info = [(s.ID, getattr(s, 'type', None)) for s in cell.surfaces]
        print(f"  Cell {cell.ID}: fill={getattr(cell.fill, 'name', cell.fill)} surfaces={s_info}")

    # Visualize the geometry (samples points inside inferred bounding box)
    visualizer_3d(sim, alpha=0.6, interactive=True)

    # Run
    # mcdc.run()   