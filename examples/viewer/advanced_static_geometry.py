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
parser.add_argument("--sample-resolution", type=int, default=72)
args = parser.parse_args()

simulation = mcdc.Simulation("Static sampler stress beamline")

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
water = mcdc.Material.multigroup(name="Water cooling loop", scatter=np.array([[0.65]]))


def box_region(x0, x1, y0, y1, z0, z1, boundary=False):
    boundary_condition = "vacuum" if boundary else "none"
    sx0 = mcdc.Surface.PlaneX(x=x0, boundary_condition=boundary_condition)
    sx1 = mcdc.Surface.PlaneX(x=x1, boundary_condition=boundary_condition)
    sy0 = mcdc.Surface.PlaneY(y=y0, boundary_condition=boundary_condition)
    sy1 = mcdc.Surface.PlaneY(y=y1, boundary_condition=boundary_condition)
    sz0 = mcdc.Surface.PlaneZ(z=z0, boundary_condition=boundary_condition)
    sz1 = mcdc.Surface.PlaneZ(z=z1, boundary_condition=boundary_condition)
    return +sx0 & -sx1 & +sy0 & -sy1 & +sz0 & -sz1


def rotated_ellipsoid(name, center, radii, angle_deg):
    cx, cy, cz = center
    rx, ry, rz = radii
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)
    inv_rz2 = 1.0 / (rz * rz)
    A = c * c * inv_rx2 + s * s * inv_ry2
    B = s * s * inv_rx2 + c * c * inv_ry2
    C = inv_rz2
    D = 2.0 * c * s * (inv_rx2 - inv_ry2)
    G = -2.0 * A * cx - D * cy
    H = -2.0 * B * cy - D * cx
    I = -2.0 * C * cz
    J = A * cx * cx + B * cy * cy + C * cz * cz + D * cx * cy - 1.0
    return mcdc.Surface.Quadric(name=name, A=A, B=B, C=C, D=D, G=G, H=H, I=I, J=J)


def paraboloid_x(name, vertex, radius_y, radius_z):
    x0, y0, z0 = vertex
    B = 1.0 / (radius_y * radius_y)
    C = 1.0 / (radius_z * radius_z)
    return mcdc.Surface.Quadric(
        name=name,
        B=B,
        C=C,
        G=-1.0,
        H=-2.0 * B * y0,
        I=-2.0 * C * z0,
        J=x0 + B * y0 * y0 + C * z0 * z0,
    )


bench = box_region(-7.6, 6.8, -2.7, 2.7, -0.55, -0.2)

beam_axis_aperture = mcdc.Surface.CylinderX(
    name="beam-axis aperture", center=[0.0, 1.0], radius=0.28
)
wide_beam_cone = mcdc.Surface.ConeX(
    name="primary conical aperture", apex=[-6.9, 0.0, 1.0], t_sq=0.045
)
tight_beam_cone = mcdc.Surface.ConeX(
    name="secondary conical aperture", apex=[-3.1, 0.0, 1.0], t_sq=0.025
)

primary_collimator = (
    box_region(-6.7, -5.55, -1.1, 1.1, 0.0, 2.0) & +beam_axis_aperture & +wide_beam_cone
)
poly_moderator = box_region(-5.15, -4.15, -1.25, 1.25, 0.0, 2.0)
boron_filter = box_region(-3.85, -3.35, -1.15, 1.15, 0.05, 1.95)
cleanup_collimator = (
    box_region(-2.9, -2.15, -0.9, 0.9, 0.2, 1.8)
    & +beam_axis_aperture
    & +tight_beam_cone
)

upstream_ring = -mcdc.Surface.TorusX(
    name="upstream borated-poly torus", A=-5.35, B=0.0, C=1.0, R=0.78, r=0.16
)
cleanup_ring = -mcdc.Surface.TorusX(
    name="cleanup shield torus", A=-3.1, B=0.0, C=1.0, R=0.62, r=0.13
)
sample_support_ring = -mcdc.Surface.TorusZ(
    name="sample support torus", A=0.0, B=0.0, C=1.0, R=0.95, r=0.09
)
tilted_diagnostic_hoop = -mcdc.Surface.Torus(
    name="tilted diagnostic torus",
    center=[1.25, 0.0, 1.0],
    axis=[1.0, 0.25, 0.35],
    R=0.62,
    r=0.08,
)
detector_cooling_loop = -mcdc.Surface.TorusY(
    name="detector cooling torus", A=2.7, B=0.0, C=1.0, R=0.55, r=0.08
)

sample_quadric = rotated_ellipsoid(
    "tilted graphite ellipsoid",
    center=[0.0, 0.0, 1.0],
    radii=[0.72, 0.44, 0.55],
    angle_deg=25.0,
)
sample = box_region(-0.95, 0.95, -0.75, 0.75, 0.25, 1.75) & -sample_quadric

hyperboloid_quadric = mcdc.Surface.Quadric(
    name="hyperboloid scatter phantom",
    A=1.0 / (0.46 * 0.46),
    B=1.0 / (0.32 * 0.32),
    C=-1.0 / (0.58 * 0.58),
    J=-1.0,
)
hyperboloid_phantom = (
    box_region(0.65, 1.55, -0.65, 0.65, 0.3, 1.7) & -hyperboloid_quadric
)

tilted_detector_quadric = rotated_ellipsoid(
    "tilted silicon detector quadric",
    center=[2.75, 0.0, 1.0],
    radii=[0.28, 0.68, 0.46],
    angle_deg=-35.0,
)
axial_detector = box_region(2.2, 3.25, -0.9, 0.9, 0.25, 1.75) & -tilted_detector_quadric

upper_detector_quadric = rotated_ellipsoid(
    "upper scatter detector quadric",
    center=[0.9, 1.75, 1.92],
    radii=[0.34, 0.28, 0.42],
    angle_deg=18.0,
)
lower_detector_quadric = rotated_ellipsoid(
    "lower scatter detector quadric",
    center=[0.9, -1.75, 0.08],
    radii=[0.34, 0.28, 0.42],
    angle_deg=-18.0,
)
upper_scatter_detector = (
    box_region(0.35, 1.45, 1.25, 2.25, 1.25, 2.45) & -upper_detector_quadric
)
lower_scatter_detector = (
    box_region(0.35, 1.45, -2.25, -1.25, -0.45, 0.75) & -lower_detector_quadric
)

parabolic_dump = paraboloid_x(
    "parabolic beam-dump face", vertex=[4.35, 0.0, 1.0], radius_y=0.95, radius_z=0.8
)
beam_dump = box_region(4.2, 5.55, -1.0, 1.0, -0.05, 2.05) & -parabolic_dump

cells = [
    mcdc.Cell(name="support bench", region=bench, fill=aluminum),
    mcdc.Cell(
        name="primary tungsten collimator with conical aperture",
        region=primary_collimator,
        fill=tungsten,
    ),
    mcdc.Cell(name="polyethylene moderator slab", region=poly_moderator, fill=poly),
    mcdc.Cell(name="boron-carbide cleanup filter", region=boron_filter, fill=boron),
    mcdc.Cell(
        name="secondary tungsten collimator with conical aperture",
        region=cleanup_collimator,
        fill=tungsten,
    ),
    mcdc.Cell(name="upstream toroidal shield", region=upstream_ring, fill=boron),
    mcdc.Cell(name="cleanup toroidal shield", region=cleanup_ring, fill=boron),
    mcdc.Cell(
        name="toroidal sample support", region=sample_support_ring, fill=aluminum
    ),
    mcdc.Cell(
        name="tilted diagnostic torus", region=tilted_diagnostic_hoop, fill=aluminum
    ),
    mcdc.Cell(name="detector cooling loop", region=detector_cooling_loop, fill=water),
    mcdc.Cell(name="tilted ellipsoid graphite sample", region=sample, fill=graphite),
    mcdc.Cell(
        name="hyperboloid scatter phantom", region=hyperboloid_phantom, fill=graphite
    ),
    mcdc.Cell(
        name="tilted ellipsoid axial detector", region=axial_detector, fill=silicon
    ),
    mcdc.Cell(
        name="upper ellipsoid scatter detector",
        region=upper_scatter_detector,
        fill=silicon,
    ),
    mcdc.Cell(
        name="lower ellipsoid scatter detector",
        region=lower_scatter_detector,
        fill=silicon,
    ),
    mcdc.Cell(name="parabolic tungsten beam dump", region=beam_dump, fill=tungsten),
]

sources = [
    mcdc.Source(
        name="collimated neutron beam",
        position=[-7.15, 0.0, 1.0],
        direction=[1.0, 0.0, 0.0],
        polar_cosine=[0.98, 1.0],
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
    sample_resolution=args.sample_resolution,
)
