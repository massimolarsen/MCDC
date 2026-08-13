import numpy as np

import mcdc
from mcdc.viewer import geo_viewer_3d

simulation = mcdc.Simulation("Moving viewer geometry")

mat_block = mcdc.Material.multigroup(name="moving_block", capture=np.array([1.0]))
mat_void = mcdc.Material.multigroup(name="void", capture=np.array([1.0]))

x0 = mcdc.Surface.PlaneX(x=-4.0, boundary_condition="vacuum")
x1 = mcdc.Surface.PlaneX(x=4.0, boundary_condition="vacuum")
y0 = mcdc.Surface.PlaneY(y=-3.0, boundary_condition="vacuum")
y1 = mcdc.Surface.PlaneY(y=3.0, boundary_condition="vacuum")
z0 = mcdc.Surface.PlaneZ(z=-3.0, boundary_condition="vacuum")
z1 = mcdc.Surface.PlaneZ(z=3.0, boundary_condition="vacuum")
world = +x0 & -x1 & +y0 & -y1 & +z0 & -z1

bx0 = mcdc.Surface.PlaneX(x=-0.75)
bx1 = mcdc.Surface.PlaneX(x=0.75)
by0 = mcdc.Surface.PlaneY(y=-0.75)
by1 = mcdc.Surface.PlaneY(y=0.75)
bz0 = mcdc.Surface.PlaneZ(z=-0.75)
bz1 = mcdc.Surface.PlaneZ(z=0.75)

velocity = [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]
duration = [2.0, 2.0]
for surface in (bx0, bx1, by0, by1, bz0, bz1):
    surface.move(velocity, duration)

block = +bx0 & -bx1 & +by0 & -by1 & +bz0 & -bz1

block_cell = mcdc.Cell(name="moving_block", region=block, fill=mat_block)
background_cell = mcdc.Cell(name="background", region=world & ~block, fill=mat_void)
simulation.set_model([block_cell, background_cell])
simulation.compile()

geo_viewer_3d(
    simulation,
    time_steps=np.linspace(0.0, 4.0, 17),
    dynamic_bounds=True,
)
