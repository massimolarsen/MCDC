import numpy as np

import mcdc
from mcdc.object_.tools.visualize_geometry_trimesh import geo_viewer_3d

# ======================================================================================
# Stress test 2: robust but heavy transient CSG
# ======================================================================================
# Design goal:
# - still stress the viewer (many moving CSG features + dense timeline)
# - avoid manifold "Not all meshes are volumes!" failures by isolating each moving
#   feature inside a dedicated static sub-box.

# --------------------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------------------
mat_a = mcdc.Material(name="MoverA", nuclide_composition={"H1": 1.0})
mat_b = mcdc.Material(name="MoverB", nuclide_composition={"H1": 1.0})
mat_c = mcdc.Material(name="MoverC", nuclide_composition={"H1": 1.0})
mat_d = mcdc.Material(name="MoverD", nuclide_composition={"H1": 1.0})
mat_e = mcdc.Material(name="MoverE", nuclide_composition={"H1": 1.0})
mat_bg = mcdc.Material(name="Coolant", nuclide_composition={"H1": 2.0, "O16": 1.0})
mat_void = mcdc.Material(name="Void", nuclide_composition={"H1": 0.0})

# --------------------------------------------------------------------------------------
# World and core
# --------------------------------------------------------------------------------------
W = 10.0
xm = mcdc.Surface.PlaneX(x=-W, boundary_condition="vacuum")
xp = mcdc.Surface.PlaneX(x=+W, boundary_condition="vacuum")
ym = mcdc.Surface.PlaneY(y=-W, boundary_condition="vacuum")
yp = mcdc.Surface.PlaneY(y=+W, boundary_condition="vacuum")
zm = mcdc.Surface.PlaneZ(z=-W, boundary_condition="vacuum")
zp = mcdc.Surface.PlaneZ(z=+W, boundary_condition="vacuum")
world = +xm & -xp & +ym & -yp & +zm & -zp

core_half = 7.5
cxm = mcdc.Surface.PlaneX(x=-core_half)
cxp = mcdc.Surface.PlaneX(x=+core_half)
cym = mcdc.Surface.PlaneY(y=-core_half)
cyp = mcdc.Surface.PlaneY(y=+core_half)
czm = mcdc.Surface.PlaneZ(z=-core_half)
czp = mcdc.Surface.PlaneZ(z=+core_half)
core = +cxm & -cxp & +cym & -cyp & +czm & -czp


def make_box(center, half):
    cx, cy, cz = center
    hx, hy, hz = half
    sx0 = mcdc.Surface.PlaneX(x=cx - hx)
    sx1 = mcdc.Surface.PlaneX(x=cx + hx)
    sy0 = mcdc.Surface.PlaneY(y=cy - hy)
    sy1 = mcdc.Surface.PlaneY(y=cy + hy)
    sz0 = mcdc.Surface.PlaneZ(z=cz - hz)
    sz1 = mcdc.Surface.PlaneZ(z=cz + hz)
    return +sx0 & -sx1 & +sy0 & -sy1 & +sz0 & -sz1


# Disjoint static sub-boxes used as safe "containers" for each moving feature
box_a = make_box(center=(-4.2, -4.2, 0.0), half=(1.8, 1.8, 2.0))
box_b = make_box(center=(+4.2, -4.2, 0.0), half=(1.8, 1.8, 2.0))
box_c = make_box(center=(-4.2, +4.2, 0.0), half=(1.8, 1.8, 2.0))
box_d = make_box(center=(+4.2, +4.2, 0.0), half=(1.8, 1.8, 2.0))
box_e = make_box(center=(0.0, 0.0, 0.0), half=(1.8, 1.8, 2.0))

# --------------------------------------------------------------------------------------
# Feature A: moving capped X-cylinder in box A
# --------------------------------------------------------------------------------------
a_cyl = mcdc.Surface.CylinderX(center=[-4.2, -4.2], radius=0.70)
a_x0 = mcdc.Surface.PlaneX(x=-5.3)
a_x1 = mcdc.Surface.PlaneX(x=-3.1)
a_vel = [[0.28, 0.20, 0.10], [-0.24, 0.12, -0.08], [0.10, -0.22, 0.05], [0.0, 0.0, 0.0]]
a_dt = [2.0, 2.0, 2.0, 2.0]
for s in (a_cyl, a_x0, a_x1):
    s.move(a_vel, a_dt)
region_a = (-a_cyl & +a_x0 & -a_x1) & box_a

# --------------------------------------------------------------------------------------
# Feature B: moving triple-sphere intersection in box B
# --------------------------------------------------------------------------------------
b_s1 = mcdc.Surface.Sphere(center=[3.8, -4.6, -0.3], radius=1.05)
b_s2 = mcdc.Surface.Sphere(center=[4.8, -4.1, -0.2], radius=1.05)
b_s3 = mcdc.Surface.Sphere(center=[4.2, -3.4, 0.0], radius=1.05)
b_vel = [[-0.18, 0.24, 0.08], [0.20, -0.12, -0.09], [0.08, 0.12, 0.08], [0.0, 0.0, 0.0]]
b_dt = [2.0, 2.0, 2.0, 2.0]
for s in (b_s1, b_s2, b_s3):
    s.move(b_vel, b_dt)
region_b = ((-b_s1) & (-b_s2) & (-b_s3)) & box_b

# --------------------------------------------------------------------------------------
# Feature C: moving clipped twin-sphere union in box C (robust manifold CSG)
# --------------------------------------------------------------------------------------
c_s1 = mcdc.Surface.Sphere(center=[-4.6, 4.0, 0.1], radius=0.95)
c_s2 = mcdc.Surface.Sphere(center=[-3.8, 4.4, -0.1], radius=0.95)
c_clip0 = mcdc.Surface.PlaneZ(z=-0.7)
c_clip1 = mcdc.Surface.PlaneZ(z=+0.7)
c_vel = [[0.20, -0.20, 0.08], [-0.12, 0.06, -0.12], [0.08, 0.12, 0.06], [0.0, 0.0, 0.0]]
c_dt = [2.0, 2.0, 2.0, 2.0]
for s in (c_s1, c_s2, c_clip0, c_clip1):
    s.move(c_vel, c_dt)
region_c = (((-c_s1) | (-c_s2)) & +c_clip0 & -c_clip1) & box_c

# --------------------------------------------------------------------------------------
# Feature D: moving cross-block in box D
# --------------------------------------------------------------------------------------
d_x0 = mcdc.Surface.PlaneX(x=+3.1)
d_x1 = mcdc.Surface.PlaneX(x=+5.3)
d_y0 = mcdc.Surface.PlaneY(y=+3.1)
d_y1 = mcdc.Surface.PlaneY(y=+5.3)
d_z0 = mcdc.Surface.PlaneZ(z=-1.1)
d_z1 = mcdc.Surface.PlaneZ(z=+1.1)
d_block = +d_x0 & -d_x1 & +d_y0 & -d_y1 & +d_z0 & -d_z1
d_tun_x = mcdc.Surface.CylinderX(center=[4.2, 0.0], radius=0.28)
d_tun_y = mcdc.Surface.CylinderY(center=[4.2, 0.0], radius=0.28)
d_tun_z = mcdc.Surface.CylinderZ(center=[4.2, 4.2], radius=0.28)
d_vel = [[0.0, 0.0, 0.24], [0.0, 0.18, -0.20], [0.0, -0.18, 0.06], [0.0, 0.0, 0.0]]
d_dt = [2.0, 2.0, 2.0, 2.0]
for s in (d_x0, d_x1, d_y0, d_y1, d_z0, d_z1, d_tun_x, d_tun_y, d_tun_z):
    s.move(d_vel, d_dt)
region_d = (d_block & ~((-d_tun_x) | (-d_tun_y) | (-d_tun_z))) & box_d

# --------------------------------------------------------------------------------------
# Feature E: moving clipped capsule in center box E
# --------------------------------------------------------------------------------------
e_s = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=0.95)
e_c = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.65)
e_z0 = mcdc.Surface.PlaneZ(z=-0.8)
e_z1 = mcdc.Surface.PlaneZ(z=+0.8)
e_vel = [[-0.20, -0.12, 0.14], [0.16, -0.08, -0.14], [0.10, 0.14, 0.06], [0.0, 0.0, 0.0]]
e_dt = [2.0, 2.0, 2.0, 2.0]
for s in (e_s, e_c, e_z0, e_z1):
    s.move(e_vel, e_dt)
region_e = (((-e_s) | (-e_c)) & +e_z0 & -e_z1) & box_e

# --------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------
mcdc.Cell(name="feature_a", region=region_a, fill=mat_a)
mcdc.Cell(name="feature_b", region=region_b, fill=mat_b)
mcdc.Cell(name="feature_c", region=region_c, fill=mat_c)
mcdc.Cell(name="feature_d", region=region_d, fill=mat_d)
mcdc.Cell(name="feature_e", region=region_e, fill=mat_e)

# Core coolant excludes only static sub-boxes (not moving feature complements)
core_coolant = core & ~box_a & ~box_b & ~box_c & ~box_d & ~box_e
mcdc.Cell(name="core_coolant", region=core_coolant, fill=mat_bg)
mcdc.Cell(name="outer_void", region=world & ~core, fill=mat_void)

# --------------------------------------------------------------------------------------
# Moving source cloud (exercises source-motion rendering too)
# --------------------------------------------------------------------------------------
src = mcdc.Source(
    x=[-1.5, 1.5],
    y=[-1.5, 1.5],
    z=[-1.5, 1.5],
    energy=1.0e6,  # isotropic is default; avoid isotropic=True to skip zero-direction warning
    time=[0.0, 8.0],
)
src.move(
    velocities=[[0.60, 0.20, 0.10], [-0.50, 0.30, -0.10], [0.30, -0.40, 0.08], [0.0, 0.0, 0.0]],
    durations=[2.0, 2.0, 2.0, 2.0],
)

# --------------------------------------------------------------------------------------
# Visualize only
# --------------------------------------------------------------------------------------
sim = mcdc.object_.simulation.simulation
time_steps = np.linspace(0.0, 8.0, 81)
geo_viewer_3d(
    sim,
    primitive_resolution=64,
    time_steps=time_steps,
    dynamic_bounds=True,
    save_animation_path="moving_csg_stress.gif",
    animation_fps=20,
)

# Keyboard controls:
#   Left/Right or p/n to step through precomputed frames
