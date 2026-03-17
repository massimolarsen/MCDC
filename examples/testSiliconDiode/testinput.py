import numpy as np
import mcdc
import numpy as np
#from mcdc.tools.visualize_geometry import visualize_simulation
from mcdc.object_.tools.visualize_geometry_trimesh import visualize_simulation

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

box_extent3 = 500.0 * 10e-6  # um
box_extent4 = 50.0 * 10e-6  # um
x_min3 = mcdc.Surface.PlaneX(x=-box_extent3, )
x_max3 = mcdc.Surface.PlaneX(x=box_extent3, )
y_min3 = mcdc.Surface.PlaneY(y=-box_extent4, )
y_max3 = mcdc.Surface.PlaneY(y=box_extent4, )
z_min3 = mcdc.Surface.PlaneZ(z=-box_extent3, )
z_max3 = mcdc.Surface.PlaneZ(z=box_extent3, )



mcdc.Cell(region=+z_min3 & -z_max3 & +x_min3 & -x_max3 & +y_min3 & -y_max3, fill=SV)
device = mcdc.Cell(region=+z_min2 & -z_max2 & +x_min2 & -x_max2 & +y_min2 & -y_max2, fill=silicon)
#mcdc.Cell(region=+z_min & -z_max & +x_min & -x_max & +y_min & -y_max, fill=void)


# Source: isotropic source surrounding the material (fills the box)
#mcdc.Source(x=[-1.5, 1.5], y=[-1.5, 1.5], z=[-1.5, 1.5], isotropic=True, energy=1e6)
mcdc.Source(x=[-2, 2], y=[-2, 2], z=[-2, 2], isotropic=True, energy=1e6)

# Energy grid for spectrum tally (must exist in this folder)
energy_grid = np.loadtxt("energy_grid.txt")  # 50 bins from 0 to 1 MeV

# Surface tally on the front surface (z = +half_length)
# Tally only particles crossing from the outside into the diode (toward -z):
# polar_reference = +z, and mu range [-1, 0] selects directions with negative z-component.
mcdc.TallyGlobal(
	scores=["flux"],
	#multipliers=["energy"],
	energy=energy_grid,
)

mcdc.TallyCell(
    cell=device,
    scores=["flux"],
    #multipliers=["energy"],
    energy=energy_grid
)

# Settings
mcdc.settings.N_particle = 1000

sim = mcdc.object_.simulation.simulation
print("Cells and their surfaces:")
for cell in sim.cells:
    s_info = [(s.ID, getattr(s, 'type', None)) for s in cell.surfaces]
    print(f"  Cell {cell.ID}: fill={getattr(cell.fill, 'name', cell.fill)} surfaces={s_info}")

# Visualize the geometry (samples points inside inferred bounding box)
visualize_simulation(sim, alpha=0.6, interactive=True) 

# Run
#mcdc.run()   
