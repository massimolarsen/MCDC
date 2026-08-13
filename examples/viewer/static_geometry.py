import numpy as np

import mcdc
from mcdc.viewer import geo_viewer_3d

simulation = mcdc.Simulation("Static viewer geometry")

mat_inner = mcdc.Material.multigroup(name="inner", capture=np.array([1.0]))
mat_outer = mcdc.Material.multigroup(name="outer", capture=np.array([1.0]))

x0 = mcdc.Surface.PlaneX(x=-3.0, boundary_condition="vacuum")
x1 = mcdc.Surface.PlaneX(x=3.0, boundary_condition="vacuum")
y0 = mcdc.Surface.PlaneY(y=-3.0, boundary_condition="vacuum")
y1 = mcdc.Surface.PlaneY(y=3.0, boundary_condition="vacuum")
z0 = mcdc.Surface.PlaneZ(z=-3.0, boundary_condition="vacuum")
z1 = mcdc.Surface.PlaneZ(z=3.0, boundary_condition="vacuum")
sphere = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=1.0)

world = +x0 & -x1 & +y0 & -y1 & +z0 & -z1
inside = -sphere

inner_cell = mcdc.Cell(name="inner_sphere", region=inside, fill=mat_inner)
outer_cell = mcdc.Cell(name="outer_box", region=world & ~inside, fill=mat_outer)
simulation.set_model([inner_cell, outer_cell])
simulation.compile()

geo_viewer_3d(simulation)
