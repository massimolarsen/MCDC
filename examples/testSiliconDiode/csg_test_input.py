import mcdc

from mcdc.object_.tools.visualize_geometry_trimesh import geo_viewer_3d


# -----------------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------------
mat_sv = mcdc.Material(name="SV", nuclide_composition={"Si28": 1.0})
mat_a = mcdc.Material(name="MatA", nuclide_composition={"Si28": 1.0})
mat_b = mcdc.Material(name="MatB", nuclide_composition={"Si28": 1.0})
mat_c = mcdc.Material(name="MatC", nuclide_composition={"Si28": 1.0})
mat_d = mcdc.Material(name="MatD", nuclide_composition={"Si28": 1.0})
mat_e = mcdc.Material(name="MatE", nuclide_composition={"Si28": 1.0})
mat_f = mcdc.Material(name="MatF", nuclide_composition={"Si28": 1.0})
mat_void = mcdc.Material(name="Void", nuclide_composition={"H1": 0.0})


# -----------------------------------------------------------------------------
# Global bounding box
# -----------------------------------------------------------------------------
outer = 8.0
xm = mcdc.Surface.PlaneX(x=-outer, boundary_condition="vacuum")
xp = mcdc.Surface.PlaneX(x=+outer, boundary_condition="vacuum")
ym = mcdc.Surface.PlaneY(y=-outer, boundary_condition="vacuum")
yp = mcdc.Surface.PlaneY(y=+outer, boundary_condition="vacuum")
zm = mcdc.Surface.PlaneZ(z=-outer, boundary_condition="vacuum")
zp = mcdc.Surface.PlaneZ(z=+outer, boundary_condition="vacuum")


# -----------------------------------------------------------------------------
# 1) Capped X-cylinder (SV)
# -----------------------------------------------------------------------------
cyl_sv = mcdc.Surface.CylinderX(center=[0.4, -0.2], radius=0.6)
x_sv0 = mcdc.Surface.PlaneX(x=-5.6)
x_sv1 = mcdc.Surface.PlaneX(x=-2.0)
region_sv = -cyl_sv & +x_sv0 & -x_sv1
mcdc.Cell(region=region_sv, fill=mat_sv, name="sv_capped_x_cylinder")


# -----------------------------------------------------------------------------
# 2) Hollow box shell (outer box AND NOT inner box)
# -----------------------------------------------------------------------------
xo0 = mcdc.Surface.PlaneX(x=-1.9)
xo1 = mcdc.Surface.PlaneX(x=+1.9)
yo0 = mcdc.Surface.PlaneY(y=-1.9)
yo1 = mcdc.Surface.PlaneY(y=+1.9)
zo0 = mcdc.Surface.PlaneZ(z=-1.9)
zo1 = mcdc.Surface.PlaneZ(z=+1.9)
outer_box = +xo0 & -xo1 & +yo0 & -yo1 & +zo0 & -zo1

xi0 = mcdc.Surface.PlaneX(x=-1.2)
xi1 = mcdc.Surface.PlaneX(x=+1.2)
yi0 = mcdc.Surface.PlaneY(y=-1.2)
yi1 = mcdc.Surface.PlaneY(y=+1.2)
zi0 = mcdc.Surface.PlaneZ(z=-1.2)
zi1 = mcdc.Surface.PlaneZ(z=+1.2)
inner_box = +xi0 & -xi1 & +yi0 & -yi1 & +zi0 & -zi1

region_shell = outer_box & ~(inner_box)
mcdc.Cell(region=region_shell, fill=mat_a, name="hollow_box_shell")


# -----------------------------------------------------------------------------
# 3) Triple-sphere union with cylindrical drill-through
# -----------------------------------------------------------------------------
s1 = mcdc.Surface.Sphere(center=[+3.2, -2.4, 0.2], radius=1.0)
s2 = mcdc.Surface.Sphere(center=[+4.4, -2.2, 0.2], radius=1.0)
s3 = mcdc.Surface.Sphere(center=[+3.8, -1.2, 0.2], radius=1.0)
union_blob = (-s1) | (-s2) | (-s3)
drill = mcdc.Surface.CylinderX(center=[-2.0, 0.2], radius=0.35)
region_blob_drilled = union_blob & ~(-drill)
mcdc.Cell(region=region_blob_drilled, fill=mat_b, name="triple_sphere_drilled")


# -----------------------------------------------------------------------------
# 4) Sphere intersection lens clipped by a Y-slab
# -----------------------------------------------------------------------------
s4 = mcdc.Surface.Sphere(center=[+3.0, +2.2, -0.2], radius=1.5)
s5 = mcdc.Surface.Sphere(center=[+4.1, +2.2, -0.2], radius=1.5)
y_clip0 = mcdc.Surface.PlaneY(y=+1.5)
y_clip1 = mcdc.Surface.PlaneY(y=+2.9)
region_lens_clip = (-s4) & (-s5) & +y_clip0 & -y_clip1
mcdc.Cell(region=region_lens_clip, fill=mat_c, name="lens_clipped")


# -----------------------------------------------------------------------------
# 5) Central block with 3 orthogonal tunnels removed
# -----------------------------------------------------------------------------
tx0 = mcdc.Surface.PlaneX(x=-0.9)
tx1 = mcdc.Surface.PlaneX(x=+0.9)
ty0 = mcdc.Surface.PlaneY(y=-0.9)
ty1 = mcdc.Surface.PlaneY(y=+0.9)
tz0 = mcdc.Surface.PlaneZ(z=-0.9)
tz1 = mcdc.Surface.PlaneZ(z=+0.9)
core_box = +tx0 & -tx1 & +ty0 & -ty1 & +tz0 & -tz1

tun_x = mcdc.Surface.CylinderX(center=[0.0, 0.0], radius=0.35)
tun_y = mcdc.Surface.CylinderY(center=[0.0, 0.0], radius=0.35)
tun_z = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.35)
region_tunnel_cube = core_box & ~((-tun_x) | (-tun_y) | (-tun_z))
mcdc.Cell(region=region_tunnel_cube, fill=mat_d, name="cross_tunnel_cube")


# -----------------------------------------------------------------------------
# 6) Clipped spherical octant-ish piece using 3 positive halfspaces
# -----------------------------------------------------------------------------
s6 = mcdc.Surface.Sphere(center=[-2.8, +2.8, +2.3], radius=1.6)
px = mcdc.Surface.PlaneX(x=-2.8)
py = mcdc.Surface.PlaneY(y=+2.8)
pz = mcdc.Surface.PlaneZ(z=+2.3)
region_octant_piece = (-s6) & +px & +py & +pz
mcdc.Cell(region=region_octant_piece, fill=mat_e, name="clipped_sphere_octant")


# -----------------------------------------------------------------------------
# 7) Ring-like shell: big sphere minus small sphere, clipped to one side
# -----------------------------------------------------------------------------
s7 = mcdc.Surface.Sphere(center=[-3.8, -2.2, -2.0], radius=1.35)
s8 = mcdc.Surface.Sphere(center=[-3.8, -2.2, -2.0], radius=0.8)
clip_side = mcdc.Surface.PlaneX(x=-3.8)
region_shell_clip = (-s7) & ~(-s8) & +clip_side
mcdc.Cell(region=region_shell_clip, fill=mat_f, name="spherical_shell_clip")


# -----------------------------------------------------------------------------
# 8) Background void in the global box
# -----------------------------------------------------------------------------
region_outer = +xm & -xp & +ym & -yp & +zm & -zp
mcdc.Cell(region=region_outer, fill=mat_void, name="background_void")


# -----------------------------------------------------------------------------
# Diagnostics + visualization (no transport run)
# -----------------------------------------------------------------------------
sim = mcdc.object_.simulation.simulation

geo_viewer_3d(
    sim,
    primitive_resolution=56,
)
