import numpy as np
import mcdc
import numpy as np
#from mcdc.tools.visualize_geometry import visualize_simulation
from mcdc.object_.tools.visualize_geometry import visualize_simulation

SV = mcdc.Material(
    name="SV",
    nuclide_composition={
        "Si28": 1.0,
    }
)

silicon = mcdc.Material(
    name="Silicon",
    nuclide_composition={
        "Si28": 1.0,
    }
)

void = mcdc.Material(
    name="Void",
    nuclide_composition={
        "H1": 0.0,
    }
)

# Geometry: diode centered at origin
radius = 0.4  # cm (8 mm diameter)
half_length = 0.3  # cm (6 mm length)

# cyl = mcdc.Surface.CylinderX(center=[0.0, 0.0], radius=radius)
# z_front = mcdc.Surface.PlaneX(x=half_length)
# z_back = mcdc.Surface.PlaneX(x=-half_length)

cyl = mcdc.Surface.CylinderY(center=[0.0, 0.0], radius=radius)
z_front = mcdc.Surface.PlaneY(y=half_length)
z_back = mcdc.Surface.PlaneY(y=-half_length)

#cyl = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius, boundary_condition="vacuum")
#z_front = mcdc.Surface.PlaneZ(z=half_length, boundary_condition="vacuum")
#z_back = mcdc.Surface.PlaneZ(z=-half_length, boundary_condition="vacuum")

mcdc.Cell(region=-cyl & +z_back & -z_front, fill=SV)


# Outer bounding box (vacuum boundaries) large enough to surround the source
box_extent = 1.0  # cm
x_min = mcdc.Surface.PlaneX(x=-box_extent,)
x_max = mcdc.Surface.PlaneX(x=box_extent, )
y_min = mcdc.Surface.PlaneY(y=-box_extent,)
y_max = mcdc.Surface.PlaneY(y=box_extent, )
z_min = mcdc.Surface.PlaneZ(z=-box_extent,)
z_max = mcdc.Surface.PlaneZ(z=box_extent, )
mcdc.Cell(region=+z_min & -z_max & +x_min & -x_max & +y_min & -y_max, fill=silicon)


box_extent2 = 5.0  # cm
x_min2 = mcdc.Surface.PlaneX(x=-box_extent2, boundary_condition="vacuum")
x_max2 = mcdc.Surface.PlaneX(x=box_extent2, boundary_condition="vacuum")
y_min2 = mcdc.Surface.PlaneY(y=-box_extent2, boundary_condition="vacuum")
y_max2 = mcdc.Surface.PlaneY(y=box_extent2, boundary_condition="vacuum")
z_min2 = mcdc.Surface.PlaneZ(z=-box_extent2, boundary_condition="vacuum")
z_max2 = mcdc.Surface.PlaneZ(z=box_extent2, boundary_condition="vacuum")
mcdc.Cell(region=+z_min2 & -z_max2 & +x_min2 & -x_max2 & +y_min2 & -y_max2, fill=void)

# Vacuum outside the cylinder within the central slab
#mcdc.Cell(region=+cyl & +z_back & -z_front & +x_min & -x_max & +y_min & -y_max & +z_min & -z_max, fill=vacuum)

# Vacuum above and below the central slab (to enclose domain)
#mcdc.Cell(region=+z_min & -z_back & +x_min & -x_max & +y_min & -y_max, fill=vacuum)
#mcdc.Cell(region=+z_front & -z_max & +x_min & -x_max & +y_min & -y_max, fill=vacuum)


# Source: isotropic source surrounding the material (fills the box)
#mcdc.Source(x=[-1.5, 1.5], y=[-1.5, 1.5], z=[-1.5, 1.5], isotropic=True, energy=1e6)
mcdc.Source(x=[-2, 2], y=[-2, 2], z=[-2, 2], isotropic=True, energy=1e6)

# Energy grid for spectrum tally (must exist in this folder)
energy_grid = np.loadtxt("energy_grid.txt")  # 50 bins from 0 to 1 MeV

# Surface tally on the front surface (z = +half_length)
# Tally only particles crossing from the outside into the diode (toward -z):
# polar_reference = +z, and mu range [-1, 0] selects directions with negative z-component.

# Outer bounding box (vacuum boundaries) large enough to surround the source
box_extent = 5.0  # cm
x_min = mcdc.Surface.PlaneX(x=-box_extent, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=box_extent, boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=-box_extent, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=box_extent, boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=-box_extent, boundary_condition="vacuum")
z_max = mcdc.Surface.PlaneZ(z=box_extent, boundary_condition="vacuum")

box_extent2 = 1.0  # cm
x_min2 = mcdc.Surface.PlaneX(x=-box_extent2, )
x_max2 = mcdc.Surface.PlaneX(x=box_extent2, )
y_min2 = mcdc.Surface.PlaneY(y=-box_extent2, )
y_max2 = mcdc.Surface.PlaneY(y=box_extent2, )
z_min2 = mcdc.Surface.PlaneZ(z=-box_extent2, )
z_max2 = mcdc.Surface.PlaneZ(z=box_extent2, )


#mcdc.Cell(region=+z_min3 & -z_max3 & +x_min3 & -x_max3 & +y_min3 & -y_max3, fill=silicon)
mcdc.Cell(region=+z_min2 & -z_max2 & +x_min2 & -x_max2 & +y_min2 & -y_max2, fill=silicon)
mcdc.Cell(region=+z_min & -z_max & +x_min & -x_max & +y_min & -y_max, fill=void)



mcdc.TallyGlobal(
	scores=["flux"],
    #multipliers=["energy"],
	energy=energy_grid,
)

# Diagnostic: also record net current across the same surface
mcdc.TallySurface(
    surface=z_front,
    scores=["net-current"],
    energy=energy_grid,
)

# Settings
mcdc.settings.N_particle = 10000

sim = mcdc.object_.simulation.simulation
print("Cells and their surfaces:")
for cell in sim.cells:
    s_info = [(s.ID, getattr(s, 'type', None)) for s in cell.surfaces]
    print(f"  Cell {cell.ID}: fill={getattr(cell.fill, 'name', cell.fill)} surfaces={s_info}")

# Visualize the geometry (samples points inside inferred bounding box)
visualize_simulation(sim, alpha=0.6, interactive=True) 

# Run
#mcdc.run()   