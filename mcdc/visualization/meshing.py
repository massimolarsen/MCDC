"""CSG-to-mesh conversion helpers for geometry visualization."""

from __future__ import annotations

import numpy as np

from mcdc.constant import (
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_PLANE,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_SPHERE,
)
from mcdc.visualization.types import Bounds3D

BOOLEAN_ENGINE = "manifold"


def bounds_from_planes_with_shift(simulation, surface_shift_map):
    """Infer padded world bounds from axis-aligned planes."""

    # collect axis-aligned plane positions
    xs, ys, zs = [], [], []
    for surface in simulation.surfaces:
        shift = surface_shift_map.get(surface.ID, np.zeros(3, dtype=float))
        if surface.type == SURFACE_PLANE_X:
            xs.append(-surface.J + shift[0])
        elif surface.type == SURFACE_PLANE_Y:
            ys.append(-surface.J + shift[1])
        elif surface.type == SURFACE_PLANE_Z:
            zs.append(-surface.J + shift[2])

    fallback = max([1.0] + [abs(getattr(surface, "J", 0.0)) for surface in simulation.surfaces])

    def axis(values):
        if len(values) >= 2:
            lo, hi = float(np.min(values)), float(np.max(values))
        elif len(values) == 1:
            delta = max(1.0, abs(values[0]))
            lo, hi = -delta, delta
        else:
            lo, hi = -fallback, fallback
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    return Bounds3D(x=axis(xs), y=axis(ys), z=axis(zs))


def world_box(bounds, tm):
    """Create the finite world box used for clipping and complements."""

    # box transform
    extents = np.array(
        [
            bounds.x[1] - bounds.x[0],
            bounds.y[1] - bounds.y[0],
            bounds.z[1] - bounds.z[0],
        ],
        dtype=float,
    )
    center = np.array(
        [
            (bounds.x[0] + bounds.x[1]) * 0.5,
            (bounds.y[0] + bounds.y[1]) * 0.5,
            (bounds.z[0] + bounds.z[1]) * 0.5,
        ],
        dtype=float,
    )
    transform = np.eye(4)
    transform[:3, 3] = center
    return tm.creation.box(extents=extents, transform=transform)


def clean_mesh(mesh):
    """Normalize boolean results and discard empty geometry."""

    # mesh normalization
    if mesh is None:
        return None
    if hasattr(mesh, "geometry"):
        geometries = list(mesh.geometry.values())
        if not geometries:
            return None
        mesh = geometries[0]
        for geometry in geometries[1:]:
            mesh = mesh + geometry
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        return None
    return mesh.process(validate=True)


def boolean_mesh(op, mesh_a, mesh_b, tm):
    """Run a boolean operation and clean the result."""

    # boolean inputs
    mesh_a = clean_mesh(mesh_a)
    mesh_b = clean_mesh(mesh_b)
    if mesh_a is None or mesh_b is None:
        return None
    if op == "intersection":
        return clean_mesh(
            tm.boolean.intersection([mesh_a, mesh_b], engine=BOOLEAN_ENGINE)
        )
    if op == "union":
        return clean_mesh(tm.boolean.union([mesh_a, mesh_b], engine=BOOLEAN_ENGINE))
    if op == "difference":
        return clean_mesh(
            tm.boolean.difference([mesh_a, mesh_b], engine=BOOLEAN_ENGINE)
        )
    return None


def axis_plane_halfspace(surface, sense, bounds, tm, shift):
    """Construct a clipped half-space for an axis-aligned plane."""

    # axis bounds update
    x0, x1 = bounds.x
    y0, y1 = bounds.y
    z0, z1 = bounds.z
    if surface.type == SURFACE_PLANE_X:
        value = -surface.J + shift[0]
        x0, x1 = (max(x0, value), x1) if sense > 0 else (x0, min(x1, value))
    elif surface.type == SURFACE_PLANE_Y:
        value = -surface.J + shift[1]
        y0, y1 = (max(y0, value), y1) if sense > 0 else (y0, min(y1, value))
    elif surface.type == SURFACE_PLANE_Z:
        value = -surface.J + shift[2]
        z0, z1 = (max(z0, value), z1) if sense > 0 else (z0, min(z1, value))
    else:
        return None
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return None
    return world_box(Bounds3D(x=(x0, x1), y=(y0, y1), z=(z0, z1)), tm)


def general_plane_halfspace(surface, sense, bounds, tm, shift):
    """Construct a clipped half-space for a general plane."""

    # plane clipping
    normal = np.array([surface.G, surface.H, surface.I], dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 0.0:
        return None
    normal /= normal_norm
    origin = (-surface.J / normal_norm) * normal + np.asarray(shift, dtype=float)
    keep_normal = normal if sense > 0 else -normal
    clipped = world_box(bounds, tm).slice_plane(
        plane_origin=origin,
        plane_normal=keep_normal,
        cap=True,
    )
    return clean_mesh(clipped)


def sphere_mesh(center, radius, tm, resolution):
    """Create a sphere primitive for meshing."""

    subdivisions = max(1, int(np.log2(max(8, int(resolution)))) - 1)
    sphere = tm.creation.icosphere(subdivisions=subdivisions, radius=float(radius))
    sphere.apply_translation(center)
    return sphere


def cylinder_mesh(center, direction, radius, height, sections, tm):
    """Create a cylinder primitive aligned to the requested axis."""

    cylinder = tm.creation.cylinder(
        radius=float(radius),
        height=float(height),
        sections=max(16, int(sections)),
    )
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    if not np.allclose(direction, z_axis):
        axis = np.cross(z_axis, direction)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0.0:
            axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            cylinder.apply_transform(tm.transformations.rotation_matrix(angle, axis))
    cylinder.apply_translation(center)
    return cylinder


def quadric_primitive(surface, bounds, tm, resolution, shift):
    """Create the primitive mesh for a supported quadric surface."""

    # primitive sizing
    dx, dy, dz = bounds.axis_lengths()
    pad = 0.2 * max(dx, dy, dz)

    if surface.type == SURFACE_SPHERE:
        cx, cy, cz = -0.5 * surface.G, -0.5 * surface.H, -0.5 * surface.I
        radius_squared = cx * cx + cy * cy + cz * cz - surface.J
        if radius_squared <= 0.0:
            return None
        cx += shift[0]
        cy += shift[1]
        cz += shift[2]
        return sphere_mesh([cx, cy, cz], np.sqrt(radius_squared), tm, resolution)

    if surface.type == SURFACE_CYLINDER_X:
        cy, cz = -0.5 * surface.H, -0.5 * surface.I
        radius_squared = cy * cy + cz * cz - surface.J
        if radius_squared <= 0.0:
            return None
        cy += shift[1]
        cz += shift[2]
        return cylinder_mesh(
            center=[(bounds.x[0] + bounds.x[1]) * 0.5 + shift[0], cy, cz],
            direction=[1.0, 0.0, 0.0],
            radius=np.sqrt(radius_squared),
            height=dx + 2.0 * pad,
            sections=resolution,
            tm=tm,
        )

    if surface.type == SURFACE_CYLINDER_Y:
        cx, cz = -0.5 * surface.G, -0.5 * surface.I
        radius_squared = cx * cx + cz * cz - surface.J
        if radius_squared <= 0.0:
            return None
        cx += shift[0]
        cz += shift[2]
        return cylinder_mesh(
            center=[cx, (bounds.y[0] + bounds.y[1]) * 0.5 + shift[1], cz],
            direction=[0.0, 1.0, 0.0],
            radius=np.sqrt(radius_squared),
            height=dy + 2.0 * pad,
            sections=resolution,
            tm=tm,
        )

    if surface.type == SURFACE_CYLINDER_Z:
        cx, cy = -0.5 * surface.G, -0.5 * surface.H
        radius_squared = cx * cx + cy * cy - surface.J
        if radius_squared <= 0.0:
            return None
        cx += shift[0]
        cy += shift[1]
        return cylinder_mesh(
            center=[cx, cy, (bounds.z[0] + bounds.z[1]) * 0.5 + shift[2]],
            direction=[0.0, 0.0, 1.0],
            radius=np.sqrt(radius_squared),
            height=dz + 2.0 * pad,
            sections=resolution,
            tm=tm,
        )

    return None


def halfspace_mesh(region, bounds, tm, primitive_resolution, surface_shift_map):
    """Convert a half-space region node into a bounded mesh."""

    # half-space dispatch
    surface = region.A
    sense = region.B
    world = world_box(bounds, tm)
    shift = surface_shift_map.get(surface.ID, np.zeros(3, dtype=float))
    if surface.type in (SURFACE_PLANE_X, SURFACE_PLANE_Y, SURFACE_PLANE_Z):
        return axis_plane_halfspace(surface, sense, bounds, tm, shift)
    if surface.type == SURFACE_PLANE:
        return general_plane_halfspace(surface, sense, bounds, tm, shift)

    primitive = quadric_primitive(surface, bounds, tm, primitive_resolution, shift)
    if primitive is None:
        return None
    if sense < 0:
        return boolean_mesh("intersection", world, primitive, tm)
    return boolean_mesh("difference", world, primitive, tm)


def region_mesh(region, bounds, tm, primitive_resolution, surface_shift_map):
    """Recursively convert an MCDC region tree into a trimesh mesh."""

    # region dispatch
    if region.type == "all":
        return world_box(bounds, tm)
    if region.type == "halfspace":
        return halfspace_mesh(
            region,
            bounds,
            tm,
            primitive_resolution,
            surface_shift_map,
        )
    if region.type == "intersection":
        mesh_a = region_mesh(
            region.A, bounds, tm, primitive_resolution, surface_shift_map
        )
        mesh_b = region_mesh(
            region.B, bounds, tm, primitive_resolution, surface_shift_map
        )
        if mesh_a is None or mesh_b is None:
            return None
        return boolean_mesh("intersection", mesh_a, mesh_b, tm)
    if region.type == "union":
        mesh_a = region_mesh(
            region.A, bounds, tm, primitive_resolution, surface_shift_map
        )
        mesh_b = region_mesh(
            region.B, bounds, tm, primitive_resolution, surface_shift_map
        )
        if mesh_a is None:
            return mesh_b
        if mesh_b is None:
            return mesh_a
        return boolean_mesh("union", mesh_a, mesh_b, tm)
    if region.type == "complement":
        # complement meshing
        mesh = region_mesh(
            region.A, bounds, tm, primitive_resolution, surface_shift_map
        )
        if mesh is None:
            return world_box(bounds, tm)
        return boolean_mesh("difference", world_box(bounds, tm), mesh, tm)
    return None


def trimesh_to_pyvista(mesh_tm, pv):
    """Convert a trimesh mesh into a pyvista PolyData mesh."""

    # pyvista conversion
    mesh_tm = clean_mesh(mesh_tm)
    if mesh_tm is None or mesh_tm.faces.shape[0] == 0:
        return None
    faces = np.hstack(
        [
            np.full((mesh_tm.faces.shape[0], 1), 3, dtype=np.int64),
            mesh_tm.faces.astype(np.int64),
        ]
    ).ravel()
    return pv.PolyData(mesh_tm.vertices, faces)
