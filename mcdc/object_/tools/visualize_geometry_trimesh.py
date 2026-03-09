"""Trimesh + manifold3d CSG viewer for MCDC geometry.

Builds watertight triangle meshes from MCDC CSG trees using trimesh boolean ops,
then renders with PyVista.
"""

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


def _bounds_from_planes(simulation):
    xs = [-s.J for s in simulation.surfaces if s.type == SURFACE_PLANE_X]
    ys = [-s.J for s in simulation.surfaces if s.type == SURFACE_PLANE_Y]
    zs = [-s.J for s in simulation.surfaces if s.type == SURFACE_PLANE_Z]
    fallback = max([1.0] + [abs(getattr(s, "J", 0.0)) for s in simulation.surfaces])

    def axis(vals):
        if len(vals) >= 2:
            lo, hi = float(np.min(vals)), float(np.max(vals))
        elif len(vals) == 1:
            d = max(1.0, abs(vals[0]))
            lo, hi = -d, d
        else:
            lo, hi = -fallback, fallback
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    return axis(xs), axis(ys), axis(zs)


def _world_box(bounds, tm):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    ext = np.array([x1 - x0, y1 - y0, z1 - z0], dtype=float)
    cen = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=float)
    tfm = np.eye(4)
    tfm[:3, 3] = cen
    return tm.creation.box(extents=ext, transform=tfm)


def _tm_clean(mesh):
    if mesh is None:
        return None
    if hasattr(mesh, "geometry"):  # Scene
        geos = list(mesh.geometry.values())
        if not geos:
            return None
        mesh = geos[0]
        for g in geos[1:]:
            mesh = mesh + g
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        return None
    return mesh.process(validate=True)


def _bool(op, a, b, tm, engine):
    a = _tm_clean(a)
    b = _tm_clean(b)
    if a is None or b is None:
        return None
    if op == "intersection":
        return _tm_clean(tm.boolean.intersection([a, b], engine=engine))
    if op == "union":
        return _tm_clean(tm.boolean.union([a, b], engine=engine))
    if op == "difference":
        return _tm_clean(tm.boolean.difference([a, b], engine=engine))
    return None


def _axis_plane_halfspace(surface, sense, bounds, tm):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    if surface.type == SURFACE_PLANE_X:
        v = -surface.J
        x0, x1 = (max(x0, v), x1) if sense > 0 else (x0, min(x1, v))
    elif surface.type == SURFACE_PLANE_Y:
        v = -surface.J
        y0, y1 = (max(y0, v), y1) if sense > 0 else (y0, min(y1, v))
    elif surface.type == SURFACE_PLANE_Z:
        v = -surface.J
        z0, z1 = (max(z0, v), z1) if sense > 0 else (z0, min(z1, v))
    else:
        return None
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return None
    return _world_box(((x0, x1), (y0, y1), (z0, z1)), tm)


def _general_plane_halfspace(surface, sense, bounds, tm):
    n = np.array([surface.G, surface.H, surface.I], dtype=float)
    nn = np.linalg.norm(n)
    if nn <= 0.0:
        return None
    n /= nn
    # n.x + J = 0 => point on plane
    p0 = (-surface.J / nn) * n
    keep_n = n if sense > 0 else -n
    world = _world_box(bounds, tm)
    # keep side where dot((x - p0), keep_n) >= 0
    clipped = world.slice_plane(plane_origin=p0, plane_normal=keep_n, cap=True)
    return _tm_clean(clipped)


def _sphere_mesh(center, radius, tm, resolution):
    subdivisions = max(1, int(np.log2(max(8, int(resolution)))) - 1)
    sph = tm.creation.icosphere(subdivisions=subdivisions, radius=float(radius))
    sph.apply_translation(center)
    return sph


def _cylinder_mesh(center, direction, radius, height, sections, tm):
    cyl = tm.creation.cylinder(
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
            cyl.apply_transform(tm.transformations.rotation_matrix(angle, axis))
    cyl.apply_translation(center)
    return cyl


def _quadric_primitive(surface, bounds, tm, res):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    pad = 0.2 * max(dx, dy, dz)

    if surface.type == SURFACE_SPHERE:
        cx, cy, cz = -0.5 * surface.G, -0.5 * surface.H, -0.5 * surface.I
        r2 = cx * cx + cy * cy + cz * cz - surface.J
        if r2 <= 0.0:
            return None
        return _sphere_mesh([cx, cy, cz], np.sqrt(r2), tm, res)

    if surface.type == SURFACE_CYLINDER_X:
        cy, cz = -0.5 * surface.H, -0.5 * surface.I
        r2 = cy * cy + cz * cz - surface.J
        if r2 <= 0.0:
            return None
        return _cylinder_mesh(
            center=[(x0 + x1) * 0.5, cy, cz],
            direction=[1.0, 0.0, 0.0],
            radius=np.sqrt(r2),
            height=dx + 2.0 * pad,
            sections=res,
            tm=tm,
        )

    if surface.type == SURFACE_CYLINDER_Y:
        cx, cz = -0.5 * surface.G, -0.5 * surface.I
        r2 = cx * cx + cz * cz - surface.J
        if r2 <= 0.0:
            return None
        return _cylinder_mesh(
            center=[cx, (y0 + y1) * 0.5, cz],
            direction=[0.0, 1.0, 0.0],
            radius=np.sqrt(r2),
            height=dy + 2.0 * pad,
            sections=res,
            tm=tm,
        )

    if surface.type == SURFACE_CYLINDER_Z:
        cx, cy = -0.5 * surface.G, -0.5 * surface.H
        r2 = cx * cx + cy * cy - surface.J
        if r2 <= 0.0:
            return None
        return _cylinder_mesh(
            center=[cx, cy, (z0 + z1) * 0.5],
            direction=[0.0, 0.0, 1.0],
            radius=np.sqrt(r2),
            height=dz + 2.0 * pad,
            sections=res,
            tm=tm,
        )

    return None


def _halfspace_mesh(region, bounds, tm, engine, primitive_resolution):
    s = region.A
    sense = region.B
    world = _world_box(bounds, tm)

    if s.type in (SURFACE_PLANE_X, SURFACE_PLANE_Y, SURFACE_PLANE_Z):
        return _axis_plane_halfspace(s, sense, bounds, tm)
    if s.type == SURFACE_PLANE:
        return _general_plane_halfspace(s, sense, bounds, tm)

    prim = _quadric_primitive(s, bounds, tm, primitive_resolution)
    if prim is None:
        return None

    if sense < 0:
        return _bool("intersection", world, prim, tm, engine)
    return _bool("difference", world, prim, tm, engine)


def _region_mesh(region, bounds, tm, engine, primitive_resolution):
    if region.type == "all":
        return _world_box(bounds, tm)
    if region.type == "halfspace":
        return _halfspace_mesh(region, bounds, tm, engine, primitive_resolution)
    if region.type == "intersection":
        a = _region_mesh(region.A, bounds, tm, engine, primitive_resolution)
        b = _region_mesh(region.B, bounds, tm, engine, primitive_resolution)
        if a is None or b is None:
            return None
        return _bool("intersection", a, b, tm, engine)
    if region.type == "union":
        a = _region_mesh(region.A, bounds, tm, engine, primitive_resolution)
        b = _region_mesh(region.B, bounds, tm, engine, primitive_resolution)
        if a is None:
            return b
        if b is None:
            return a
        return _bool("union", a, b, tm, engine)
    if region.type == "complement":
        a = _region_mesh(region.A, bounds, tm, engine, primitive_resolution)
        if a is None:
            return _world_box(bounds, tm)
        return _bool("difference", _world_box(bounds, tm), a, tm, engine)
    return None


def _tm_to_pv(mesh_tm, pv):
    mesh_tm = _tm_clean(mesh_tm)
    if mesh_tm is None:
        return None
    if mesh_tm.faces.shape[0] == 0:
        return None
    faces = np.hstack(
        [np.full((mesh_tm.faces.shape[0], 1), 3, dtype=np.int64), mesh_tm.faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(mesh_tm.vertices, faces)


def geo_viewer_3d_trimesh(
    simulation,
    primitive_resolution=48,
    alpha=0.6,
):
    try:
        import pyvista as pv
        import trimesh as tm
    except Exception as exc:
        raise RuntimeError(
            "Trimesh viewer requires `trimesh`, `manifold3d`, and `pyvista`."
        ) from exc

    # Required for halfspace clipping: CSG needs a finite world domain.
    bounds = _bounds_from_planes(simulation)

    engine = "manifold"
    print(f"[trimesh] Using boolean engine: {engine}")

    plotter = pv.Plotter()

    palette = [
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
    ]

    legend = []
    rendered = False
    n_cells = len(simulation.cells)
    print(f"[trimesh] Meshing {n_cells} cells...")

    for i, cell in enumerate(simulation.cells):
        mat_name = getattr(cell.fill, "name", str(cell.fill))
        color = palette[i % len(palette)]
        low = str(mat_name).lower()
        opacity = min(alpha, 0.15) if ("vacuum" in low or "void" in low) else alpha

        print(f"[trimesh] Cell {i + 1}/{n_cells}: {cell.name} ({mat_name})")

        try:
            mesh_tm = _region_mesh(
                cell.region, bounds, tm, engine, primitive_resolution
            )
            mesh_pv = _tm_to_pv(mesh_tm, pv)
        except Exception as exc:
            print(f"Warning: trimesh meshing failed for cell {cell.ID}: {exc}")
            mesh_pv = None

        if mesh_pv is None or mesh_pv.n_points == 0:
            print(f"Warning: no mesh for cell {cell.ID} ({mat_name})")
            continue

        plotter.add_mesh(mesh_pv, color=color, opacity=opacity, smooth_shading=False)
        legend.append((mat_name, color))
        rendered = True

    if not rendered:
        print("No geometry could be rendered in trimesh viewer.")
        return

    seen = set()
    leg = []
    for name, color in legend:
        if name in seen:
            continue
        seen.add(name)
        leg.append((name, color))
    if leg:
        plotter.add_legend(leg, bcolor=None)

    plotter.add_axes()
    plotter.show_grid()

    plotter.show(auto_close=False)


def visualizer_3d_trimesh(*args, **kwargs):
    return geo_viewer_3d_trimesh(*args, **kwargs)
