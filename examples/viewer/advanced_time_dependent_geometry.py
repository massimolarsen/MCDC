import argparse

import numpy as np

import mcdc
from mcdc.viewer import geo_viewer_3d


def bool_arg(value):
    if value.lower() in {"true", "1", "yes", "on"}:
        return True
    if value.lower() in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


parser = argparse.ArgumentParser()
parser.add_argument("--labels", type=bool_arg, default=True)
parser.add_argument("--color-by", choices=["material", "cell"], default="material")
parser.add_argument("--dynamic-bounds", type=bool_arg, default=True)
parser.add_argument("--opacity-slider", type=bool_arg, default=True)
parser.add_argument("--show-sources", type=bool_arg, default=True)
parser.add_argument("--save-animation-path")
args = parser.parse_args()

simulation = mcdc.Simulation("Advanced time-dependent viewer geometry")

aluminum = mcdc.Material.multigroup(name="Aluminum", capture=np.array([0.05]))
tungsten = mcdc.Material.multigroup(name="Tungsten shutter", capture=np.array([0.90]))
silicon = mcdc.Material.multigroup(name="Silicon payload", capture=np.array([0.08]))
poly = mcdc.Material.multigroup(name="Polyethylene", scatter=np.array([[0.55]]))


def box_surfaces(x0, x1, y0, y1, z0, z1, boundary=False):
    boundary_condition = "vacuum" if boundary else "none"
    return (
        mcdc.Surface.PlaneX(x=x0, boundary_condition=boundary_condition),
        mcdc.Surface.PlaneX(x=x1, boundary_condition=boundary_condition),
        mcdc.Surface.PlaneY(y=y0, boundary_condition=boundary_condition),
        mcdc.Surface.PlaneY(y=y1, boundary_condition=boundary_condition),
        mcdc.Surface.PlaneZ(z=z0, boundary_condition=boundary_condition),
        mcdc.Surface.PlaneZ(z=z1, boundary_condition=boundary_condition),
    )


def region_from_surfaces(surfaces):
    x0, x1, y0, y1, z0, z1 = surfaces
    return +x0 & -x1 & +y0 & -y1 & +z0 & -z1


def box_region(x0, x1, y0, y1, z0, z1, boundary=False):
    return region_from_surfaces(box_surfaces(x0, x1, y0, y1, z0, z1, boundary))


rail_a = box_region(-6.25, 6.25, -3.6, -3.2, -0.35, 0.05)
rail_b = box_region(-6.25, 6.25, 3.2, 3.6, -0.35, 0.05)
fixed_collimator = -mcdc.Surface.CylinderX(
    name="fixed round aperture", center=[0.0, 1.05], radius=0.8
) & box_region(-5.25, -4.45, -1.25, 1.25, 0.05, 2.05)

payload_surfaces = box_surfaces(-1.0, 1.0, -0.85, 0.85, 0.1, 1.35)
for surface in payload_surfaces:
    surface.move(
        velocities=[[1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.2, 0.0, 0.0]],
        durations=[2.0, 1.0, 2.0],
    )
payload = region_from_surfaces(payload_surfaces)

shutter_surfaces = box_surfaces(-0.35, 0.35, -2.9, 2.9, 1.55, 2.35)
for surface in shutter_surfaces:
    surface.move(
        velocities=[[0.0, 0.0, -0.45], [0.0, 0.0, 0.9], [0.0, 0.0, -0.45]],
        durations=[1.5, 1.5, 1.5],
    )
shutter = region_from_surfaces(shutter_surfaces)

moderator = -mcdc.Surface.CylinderZ(
    name="stationary moderator tube", center=[3.5, 0.0], radius=0.6
) & box_region(2.75, 4.25, -0.75, 0.75, 0.0, 2.5)

cells = [
    mcdc.Cell(name="lower guide rail", region=rail_a, fill=aluminum),
    mcdc.Cell(name="upper guide rail", region=rail_b, fill=aluminum),
    mcdc.Cell(name="fixed collimator", region=fixed_collimator, fill=tungsten),
    mcdc.Cell(name="moving silicon payload", region=payload, fill=silicon),
    mcdc.Cell(name="moving tungsten shutter", region=shutter, fill=tungsten),
    mcdc.Cell(name="moderator tube", region=moderator, fill=poly),
]

scanning_source = mcdc.Source(
    name="moving pencil source",
    position=[-6.0, 0.0, 1.05],
    direction=[1.0, 0.0, 0.0],
    polar_cosine=[0.95, 1.0],
    energy=0,
    time=[0.0, 5.0],
)
scanning_source.move(
    velocities=[[0.0, 0.45, 0.0], [0.0, -0.9, 0.0], [0.0, 0.45, 0.0]],
    durations=[1.5, 2.0, 1.5],
)

simulation.set_model(cells)
simulation.set_sources([scanning_source])
simulation.settings.N_particle = 100
simulation.compile()

geo_viewer_3d(
    simulation,
    time_steps=np.linspace(0.0, 5.0, 21),
    dynamic_bounds=args.dynamic_bounds,
    save_animation_path=args.save_animation_path,
    animation_fps=8,
    labels=args.labels,
    color_by=args.color_by,
    opacity_slider=args.opacity_slider,
    show_sources=args.show_sources,
)
