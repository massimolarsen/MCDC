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
parser.add_argument("--opacity-slider", type=bool_arg, default=True)
parser.add_argument("--show-sources", type=bool_arg, default=True)
args = parser.parse_args()

simulation = mcdc.Simulation("Static beamline shielding viewer")

aluminum = mcdc.Material.multigroup(name="Aluminum housing", capture=np.array([0.06]))
poly = mcdc.Material.multigroup(
    name="Polyethylene moderator", scatter=np.array([[0.55]])
)
boron = mcdc.Material.multigroup(
    name="Boron-carbide absorber", capture=np.array([1.20]), scatter=np.array([[0.02]])
)
tungsten = mcdc.Material.multigroup(
    name="Tungsten collimator", capture=np.array([0.90])
)
graphite = mcdc.Material.multigroup(name="Graphite sample", scatter=np.array([[0.40]]))
silicon = mcdc.Material.multigroup(name="Silicon detector", capture=np.array([0.08]))


def box_region(x0, x1, y0, y1, z0, z1, boundary=False):
    boundary_condition = "vacuum" if boundary else "none"
    sx0 = mcdc.Surface.PlaneX(x=x0, boundary_condition=boundary_condition)
    sx1 = mcdc.Surface.PlaneX(x=x1, boundary_condition=boundary_condition)
    sy0 = mcdc.Surface.PlaneY(y=y0, boundary_condition=boundary_condition)
    sy1 = mcdc.Surface.PlaneY(y=y1, boundary_condition=boundary_condition)
    sz0 = mcdc.Surface.PlaneZ(z=z0, boundary_condition=boundary_condition)
    sz1 = mcdc.Surface.PlaneZ(z=z1, boundary_condition=boundary_condition)
    return +sx0 & -sx1 & +sy0 & -sy1 & +sz0 & -sz1


bench = box_region(-7.5, 7.5, -2.5, 2.5, -0.55, -0.2)

beam_axis_aperture = mcdc.Surface.CylinderX(
    name="beam-axis aperture", center=[0.0, 1.0], radius=0.28
)
primary_collimator = box_region(-6.4, -5.3, -1.0, 1.0, 0.15, 1.85) & +beam_axis_aperture

front_housing = box_region(-4.8, -4.35, -1.35, 1.35, -0.05, 2.05)
poly_moderator = box_region(-4.15, -3.15, -1.25, 1.25, 0.0, 2.0)
boron_filter = box_region(-2.85, -2.35, -1.15, 1.15, 0.05, 1.95)
cleanup_collimator = (
    box_region(-2.0, -1.35, -0.85, 0.85, 0.25, 1.75) & +beam_axis_aperture
)

sample = -mcdc.Surface.Sphere(
    name="graphite scattering sample", center=[0.0, 0.0, 1.0], radius=0.65
)

axial_detector = box_region(2.4, 3.0, -0.45, 0.45, 0.45, 1.55)
upper_scatter_detector = box_region(0.7, 1.35, 1.45, 2.15, 1.55, 2.25)
lower_scatter_detector = box_region(0.7, 1.35, -2.15, -1.45, -0.25, 0.45)
beam_dump = box_region(4.3, 5.35, -0.8, 0.8, 0.1, 1.9)

cells = [
    mcdc.Cell(name="support bench", region=bench, fill=aluminum),
    mcdc.Cell(
        name="primary collimator with beam aperture",
        region=primary_collimator,
        fill=tungsten,
    ),
    mcdc.Cell(name="front detector housing", region=front_housing, fill=aluminum),
    mcdc.Cell(name="polyethylene moderator slab", region=poly_moderator, fill=poly),
    mcdc.Cell(name="boron-carbide cleanup filter", region=boron_filter, fill=boron),
    mcdc.Cell(
        name="secondary collimator with cleanup aperture",
        region=cleanup_collimator,
        fill=tungsten,
    ),
    mcdc.Cell(name="graphite scattering sample", region=sample, fill=graphite),
    mcdc.Cell(name="axial transmission detector", region=axial_detector, fill=silicon),
    mcdc.Cell(
        name="upper scatter detector", region=upper_scatter_detector, fill=silicon
    ),
    mcdc.Cell(
        name="lower scatter detector", region=lower_scatter_detector, fill=silicon
    ),
    mcdc.Cell(name="tungsten beam dump", region=beam_dump, fill=tungsten),
]

sources = [
    mcdc.Source(
        name="collimated neutron beam",
        position=[-7.0, 0.0, 1.0],
        direction=[1.0, 0.0, 0.0],
        polar_cosine=[0.97, 1.0],
        energy=0,
    ),
]

simulation.set_model(cells)
simulation.set_sources(sources)
simulation.settings.N_particle = 100
simulation.compile()

geo_viewer_3d(
    simulation,
    labels=args.labels,
    color_by=args.color_by,
    opacity_slider=args.opacity_slider,
    show_sources=args.show_sources,
)
