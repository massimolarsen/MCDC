import numpy as np
import mcdc
import numpy as np
from mcdc.object_.tools.visualize_geometry_trimesh import visualizer_3d

silicon = mcdc.Material(
    name="SV",
    nuclide_composition={
        "Si28": 1.0,
    }
)

void = mcdc.Material(
    name="Silicon",
    nuclide_composition={
        "H1": 0.0,
    }
)


# Outer bounding box (vacuum boundaries) large enough to surround the source
box_extent3 = 500.0e-5 # um
box_extent4 = 50.0e-5 # um
x_min3 = mcdc.Surface.PlaneX(x=-box_extent3)
x_max3 = mcdc.Surface.PlaneX(x=box_extent3)
y_min3 = mcdc.Surface.PlaneY(y=-box_extent4)
y_max3 = mcdc.Surface.PlaneY(y=box_extent4)
z_min3 = mcdc.Surface.PlaneZ(z=-box_extent3)
z_max3 = mcdc.Surface.PlaneZ(z=box_extent3)

box_extent = 1.0  # cm
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


mcdc.Cell(region=+z_min3 & -z_max3 & +x_min3 & -x_max3 & +y_min3 & -y_max3, fill=silicon)
device = mcdc.Cell(region=+z_min & -z_max & +x_min & -x_max & +y_min & -y_max, fill=void)
#void_cell = mcdc.Cell(region=+z_min2 & -z_max2 & +x_min2 & -x_max2 & +y_min2 & -y_max2, fill=void)


# Source: isotropic source surrounding the material (fills the box)
mcdc.Source(x=[-2.0, 2.0], y=[-2.0, 2.0], z=[-2.0, 2.0], isotropic=True, energy=1e6)

# Energy grid for spectrum tally (must exist in this folder)
energy_grid = np.loadtxt("energy_grid.txt")  # 50 bins from 0 to 1 MeV

# Tally only inside the silicon-filled cell using a cell tally
mcdc.TallyCell(
    cell=device,
    scores=["flux"],
    multipliers=["energy"],
    energy=energy_grid,
)

# Settings
mcdc.settings.N_particle = 10000

# Visualize the geometry
visualizer_3d(mcdc.object_.simulation.simulation, alpha=0.6, interactive=True) 

# Run
#mcdc.run()   
