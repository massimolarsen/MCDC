from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc
from mcdc.visualization.geometry import geo_viewer_3d

# ======================================================================================
# Time-dependent CSG showcase
# ======================================================================================
# This example builds several moving geometry features and visualizes them using the
# 3D viewer with slider + keyboard stepping.

# Materials (lightweight CE placeholders for visualization)
mat_pod = mcdc.Material(name="Pod", nuclide_composition={"Si28": 1.0})
mat_blob = mcdc.Material(name="Blob", nuclide_composition={"H1": 1.0})
mat_block = mcdc.Material(name="Block", nuclide_composition={"H1": 1.0})
mat_void = mcdc.Material(name="Void", nuclide_composition={"H1": 0.0})

# --------------------------------------------------------------------------------------
# World box
# --------------------------------------------------------------------------------------
extent = 8.0
x_min = mcdc.Surface.PlaneX(x=-extent, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=+extent, boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=-extent, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=+extent, boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=-extent, boundary_condition="vacuum")
z_max = mcdc.Surface.PlaneZ(z=+extent, boundary_condition="vacuum")
world = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max

# --------------------------------------------------------------------------------------
# Feature 1: moving capped cylinder ("pod")
# --------------------------------------------------------------------------------------
pod_cyl = mcdc.Surface.CylinderZ(center=[-2.8, 1.0], radius=0.85)
pod_bot = mcdc.Surface.PlaneZ(z=-1.3)
pod_top = mcdc.Surface.PlaneZ(z=+1.3)

pod_vel = [[0.9, 0.5, 0.0], [-1.3, 0.0, 0.0], [0.4, -0.9, 0.0]]
pod_dt = [2.0, 2.0, 2.0]
pod_cyl.move(pod_vel, pod_dt)
pod_bot.move(pod_vel, pod_dt)
pod_top.move(pod_vel, pod_dt)

region_pod = -pod_cyl & +pod_bot & -pod_top

# --------------------------------------------------------------------------------------
# Feature 2: moving twin-sphere union ("blob")
# --------------------------------------------------------------------------------------
blob_s1 = mcdc.Surface.Sphere(center=[4.2, -2.8, -0.6], radius=0.95)
blob_s2 = mcdc.Surface.Sphere(center=[5.4, -2.4, -0.2], radius=0.95)

blob_vel = [[-0.2, 1.1, 0.7], [0.0, -1.3, -0.7], [0.3, 0.2, 0.0]]
blob_dt = [2.0, 2.0, 2.0]
blob_s1.move(blob_vel, blob_dt)
blob_s2.move(blob_vel, blob_dt)

region_blob = (-blob_s1) | (-blob_s2)

# --------------------------------------------------------------------------------------
# Feature 3: static block with moving tunnel cutout
# --------------------------------------------------------------------------------------
bx0 = mcdc.Surface.PlaneX(x=-1.8)
bx1 = mcdc.Surface.PlaneX(x=+1.8)
by0 = mcdc.Surface.PlaneY(y=-1.5)
by1 = mcdc.Surface.PlaneY(y=+1.5)
bz0 = mcdc.Surface.PlaneZ(z=-1.5)
bz1 = mcdc.Surface.PlaneZ(z=+1.5)
block = +bx0 & -bx1 & +by0 & -by1 & +bz0 & -bz1

tunnel = mcdc.Surface.CylinderX(center=[0.0, 0.0], radius=0.45)
tunnel_vel = [[0.0, 0.0, 0.30], [0.0, 0.40, -0.30], [0.0, -0.40, 0.0]]
tunnel_dt = [2.0, 2.0, 2.0]
tunnel.move(tunnel_vel, tunnel_dt)

region_block = block & ~(-tunnel)

# --------------------------------------------------------------------------------------
# Cells (partitioned to avoid overlap)
# --------------------------------------------------------------------------------------
mcdc.Cell(name="moving_pod", region=region_pod, fill=mat_pod)
mcdc.Cell(name="moving_blob", region=region_blob & ~region_pod, fill=mat_blob)
mcdc.Cell(
    name="moving_tunnel_block",
    region=region_block & ~region_pod & ~region_blob,
    fill=mat_block,
)
mcdc.Cell(
    name="background_void",
    region=world & ~region_pod & ~region_blob & ~region_block,
    fill=mat_void,
)

# --------------------------------------------------------------------------------------
# Visualize only (no transport run)
# --------------------------------------------------------------------------------------
sim = mcdc.object_.simulation.simulation

# Auto timeline is inferred from Surface.move(...), but we provide denser steps for
# smoother stepping through each motion segment.
time_steps = np.linspace(0.0, 6.0, 31)
geo_viewer_3d(
    sim,
    primitive_resolution=56,
    time_steps=time_steps,
    dynamic_bounds=True,
    save_animation_path="moving_csg_showcase.gif",
    animation_fps=15,
)

# Controls in viewer:
#   - Keyboard: Left/Right or p/n to step backward/forward
