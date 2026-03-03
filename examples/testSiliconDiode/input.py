import numpy as np
import mcdc
import numpy as np
#from mcdc.tools.visualize_geometry import visualize_simulation
from mcdc.object_.tools.visualize_geometry import visualizer_3d

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

# Outer bounding box (vacuum boundaries) large enough to surround the source
box_extent = 2.0  # cm
x_min = mcdc.Surface.PlaneX(x=-box_extent)
x_max = mcdc.Surface.PlaneX(x=box_extent)
y_min = mcdc.Surface.PlaneY(y=-box_extent)
y_max = mcdc.Surface.PlaneY(y=box_extent)
z_min = mcdc.Surface.PlaneZ(z=-box_extent)
z_max = mcdc.Surface.PlaneZ(z=box_extent)

box_extent2 = 5.0  # cm
x_min2 = mcdc.Surface.PlaneX(x=-box_extent2, boundary_condition="vacuum")
x_max2 = mcdc.Surface.PlaneX(x=box_extent2, boundary_condition="vacuum")
y_min2 = mcdc.Surface.PlaneY(y=-box_extent2, boundary_condition="vacuum")
y_max2 = mcdc.Surface.PlaneY(y=box_extent2, boundary_condition="vacuum")
z_min2 = mcdc.Surface.PlaneZ(z=-box_extent2, boundary_condition="vacuum")
z_max2 = mcdc.Surface.PlaneZ(z=box_extent2, boundary_condition="vacuum")

# Cells: diode (silicon) and surrounding vacuum
# Silicon cylinder limited in z
silicon_cell = mcdc.Cell(region=-cyl & +z_back & -z_front, fill=silicon)

#box_extent2 = 2.0  # cm
#x_min2 = mcdc.Surface.PlaneX(x=-box_extent2, boundary_condition="reflective")
#x_max2 = mcdc.Surface.PlaneX(x=box_extent2, boundary_condition="reflective")
#y_min2 = mcdc.Surface.PlaneY(y=-box_extent2, boundary_condition="reflective")
#y_max2 = mcdc.Surface.PlaneY(y=box_extent2, boundary_condition="reflective")
#z_min2 = mcdc.Surface.PlaneZ(z=-box_extent2, boundary_condition="reflective")
#z_max2 = mcdc.Surface.PlaneZ(z=box_extent2, boundary_condition="reflective")
#mcdc.Cell(region=+z_min2 & -z_max2 & +x_min2 & -x_max2 & +y_min2 & -y_max2, fill=vacuum)

# Vacuum outside the cylinder within the central slab
#mcdc.Cell(region=+cyl & +z_back & -z_front & +x_min & -x_max & +y_min & -y_max & +z_min & -z_max, fill=vacuum)

# Vacuum above and below the central slab (to enclose domain)
#mcdc.Cell(region=+z_min & -z_back & +x_min & -x_max & +y_min & -y_max, fill=vacuum)
#mcdc.Cell(region=+z_front & -z_max & +x_min & -x_max & +y_min & -y_max, fill=vacuum)

# also add outer box cells
box_silicon_cell = mcdc.Cell(region=+z_min & -z_max & +x_min & -x_max & +y_min & -y_max, fill=silicon)
void_cell = mcdc.Cell(region=+z_min2 & -z_max2 & +x_min2 & -x_max2 & +y_min2 & -y_max2, fill=void)


# Source: isotropic source surrounding the material (fills the box)
#mcdc.Source(x=[-1.5, 1.5], y=[-1.5, 1.5], z=[-1.5, 1.5], isotropic=True, energy=1e6)
mcdc.Source(x=[-2.0, 2.0], y=[-2.0, 2.0], z=[-2.0, 2.0], isotropic=True, energy=1e6)

# Energy grid for spectrum tally (must exist in this folder)
energy_grid = np.loadtxt("energy_grid.txt")  # 50 bins from 0 to 1 MeV

# global flux tally with energy binning
mcdc.TallyGlobal(
	scores=["flux"],
	multipliers=["energy"],
	energy=energy_grid,
)

# Angular (mu = cos(theta)) bins and tally inside the silicon cell
mu_bins = np.linspace(-1.0, 1.0, 41)  # 40 bins from -1 to 1

# Tally angular distribution (flux vs mu) restricted to the silicon-filled cell
mcdc.TallyCell(
    cell=silicon_cell,
    scores=["flux"],
    mu=mu_bins,
)

# also record net current across the same surface
mcdc.TallySurface(
    surface=z_front,
    scores=["net-current"],
    multipliers=["energy"],
    energy=energy_grid
)

mcdc.TallySurface(
    surface=z_front,
    scores=["flux"],
    multipliers=["energy"],
    mu=mu_bins,
)

# Settings
mcdc.settings.N_particle = 1000

# Visualize the geometry
#visualizer_3d(mcdc.object_.simulation.simulation, alpha=0.6, interactive=True) 

# Run
mcdc.run()   