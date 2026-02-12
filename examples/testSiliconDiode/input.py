import numpy as np
import mcdc

# Silicon diode test input
# Cylinder: 8 mm width (diameter) -> radius = 0.4 cm
# Length: ~6 mm -> +/- 0.3 cm in z

# Materials (one-group MG approximations)
#silicon = mcdc.MaterialMG(capture=np.array([0.01]), scatter=np.array([[0.99]]))
#vacuum = mcdc.MaterialMG(capture=np.array([5e-5]), scatter=np.array([[0.0]]))

silicon = mcdc.Material(
    nuclide_composition={
        "Si28": 1.0,
    }
)

vacuum = mcdc.Material(
    nuclide_composition={
        "H1": 1.0,
    }
)

# Geometry: diode centered at origin
radius = 0.4  # cm (8 mm diameter)
half_length = 0.3  # cm (6 mm length)

cyl = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius)
z_front = mcdc.Surface.PlaneZ(z=half_length)
z_back = mcdc.Surface.PlaneZ(z=-half_length)

# Outer bounding box (vacuum boundaries) large enough to surround the source
box_extent = 5.0  # cm
x_min = mcdc.Surface.PlaneX(x=-box_extent, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=box_extent, boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=-box_extent, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=box_extent, boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=-box_extent, boundary_condition="vacuum")
z_max = mcdc.Surface.PlaneZ(z=box_extent, boundary_condition="vacuum")

# Cells: diode (silicon) and surrounding vacuum
# Silicon cylinder limited in z
mcdc.Cell(region=-cyl & +z_back & -z_front, fill=silicon)

# Vacuum outside the cylinder within the central slab
mcdc.Cell(region=+cyl & +z_back & -z_front & +x_min & -x_max & +y_min & -y_max & +z_min & -z_max, fill=vacuum)

# Vacuum above and below the central slab (to enclose domain)
mcdc.Cell(region=+z_min & -z_back & +x_min & -x_max & +y_min & -y_max, fill=vacuum)
mcdc.Cell(region=+z_front & -z_max & +x_min & -x_max & +y_min & -y_max, fill=vacuum)

# Source: isotropic source surrounding the material (fills the box)
# For a surrounding source, use a broad spatial box; the diode is small at center.
mcdc.Source(x=[-4.5, 4.5], y=[-4.5, 4.5], z=[-4.5, 4.5], isotropic=True, energy=1e6)

# Energy grid for spectrum tally (must exist in this folder)
energy_grid = np.loadtxt("energy_grid.txt")  # 50 bins from 0 to 1 MeV

# Surface tally on the front surface (z = +half_length)
# Tally only particles crossing from the outside into the diode (toward -z):
# polar_reference = +z, and mu range [-1, 0] selects directions with negative z-component.
mcdc.TallyGlobal(
	scores=["flux"],
	multipliers=["energy"],
	energy=energy_grid,
)

# Diagnostic: also record net current across the same surface
mcdc.TallySurface(
    surface=z_front,
    scores=["net-current"],
    energy=energy_grid,
)

# Settings
mcdc.settings.N_particle = 1000

# Run
mcdc.run()
