"""Static geometry visualization helpers for MCDC."""

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


def bounds_from_planes_with_shift(simulation, shift_map):
    xs, ys, zs = [], [], []
    for s in simulation.surfaces:
        shift = shift_map.get(s.ID, np.zeros(3, dtype=float))
        if s.type == SURFACE_PLANE_X:
            xs.append(-s.J + shift[0])
        elif s.type == SURFACE_PLANE_Y:
            ys.append(-s.J + shift[1])
        elif s.type == SURFACE_PLANE_Z:
            zs.append(-s.J + shift[2])

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


def zero_shift_map(simulation):
    return {s.ID: np.zeros(3, dtype=float) for s in simulation.surfaces}


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
    if hasattr(mesh, "geometry"):
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


def _axis_plane_halfspace(surface, sense, bounds, tm, shift):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    if surface.type == SURFACE_PLANE_X:
        v = -surface.J + shift[0]
        x0, x1 = (max(x0, v), x1) if sense > 0 else (x0, min(x1, v))
    elif surface.type == SURFACE_PLANE_Y:
        v = -surface.J + shift[1]
        y0, y1 = (max(y0, v), y1) if sense > 0 else (y0, min(y1, v))
    elif surface.type == SURFACE_PLANE_Z:
        v = -surface.J + shift[2]
        z0, z1 = (max(z0, v), z1) if sense > 0 else (z0, min(z1, v))
    else:
        return None
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return None
    return _world_box(((x0, x1), (y0, y1), (z0, z1)), tm)


def _general_plane_halfspace(surface, sense, bounds, tm, shift):
    n = np.array([surface.G, surface.H, surface.I], dtype=float)
    nn = np.linalg.norm(n)
    if nn <= 0.0:
        return None
    n /= nn
    p0 = (-surface.J / nn) * n + np.asarray(shift, dtype=float)
    keep_n = n if sense > 0 else -n
    world = _world_box(bounds, tm)
    clipped = world.slice_plane(plane_origin=p0, plane_normal=keep_n, cap=True)
    return _tm_clean(clipped)


def _sphere_mesh(center, radius, tm, resolution):
    subdivisions = max(1, int(np.log2(max(8, int(resolution)))) - 1)
    sph = tm.creation.icosphere(subdivisions=subdivisions, radius=float(radius))
    sph.apply_translation(center)
    return sph


def _cylinder_mesh(center, direction, radius, height, sections, tm):
    cyl = tm.creation.cylinder(
        radius=float(radius), height=float(height), sections=max(16, int(sections))
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


def _quadric_primitive(surface, bounds, tm, res, shift):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    pad = 0.2 * max(dx, dy, dz)

    if surface.type == SURFACE_SPHERE:
        cx, cy, cz = -0.5 * surface.G, -0.5 * surface.H, -0.5 * surface.I
        r2 = cx * cx + cy * cy + cz * cz - surface.J
        if r2 <= 0.0:
            return None
        cx += shift[0]
        cy += shift[1]
        cz += shift[2]
        return _sphere_mesh([cx, cy, cz], np.sqrt(r2), tm, res)

    if surface.type == SURFACE_CYLINDER_X:
        cy, cz = -0.5 * surface.H, -0.5 * surface.I
        r2 = cy * cy + cz * cz - surface.J
        if r2 <= 0.0:
            return None
        cy += shift[1]
        cz += shift[2]
        return _cylinder_mesh(
            center=[(x0 + x1) * 0.5 + shift[0], cy, cz],
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
        cx += shift[0]
        cz += shift[2]
        return _cylinder_mesh(
            center=[cx, (y0 + y1) * 0.5 + shift[1], cz],
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
        cx += shift[0]
        cy += shift[1]
        return _cylinder_mesh(
            center=[cx, cy, (z0 + z1) * 0.5 + shift[2]],
            direction=[0.0, 0.0, 1.0],
            radius=np.sqrt(r2),
            height=dz + 2.0 * pad,
            sections=res,
            tm=tm,
        )

    return None


def _halfspace_mesh(region, bounds, tm, engine, primitive_resolution, shift_map):
    s = region.A
    sense = region.B
    world = _world_box(bounds, tm)
    shift = shift_map.get(s.ID, np.zeros(3, dtype=float))
    if s.type in (SURFACE_PLANE_X, SURFACE_PLANE_Y, SURFACE_PLANE_Z):
        return _axis_plane_halfspace(s, sense, bounds, tm, shift)
    if s.type == SURFACE_PLANE:
        return _general_plane_halfspace(s, sense, bounds, tm, shift)

    prim = _quadric_primitive(s, bounds, tm, primitive_resolution, shift)
    if prim is None:
        return None
    if sense < 0:
        return _bool("intersection", world, prim, tm, engine)
    return _bool("difference", world, prim, tm, engine)


def _region_mesh(region, bounds, tm, engine, primitive_resolution, shift_map):
    if region.type == "all":
        return _world_box(bounds, tm)
    if region.type == "halfspace":
        return _halfspace_mesh(
            region, bounds, tm, engine, primitive_resolution, shift_map
        )
    if region.type == "intersection":
        a = _region_mesh(region.A, bounds, tm, engine, primitive_resolution, shift_map)
        b = _region_mesh(region.B, bounds, tm, engine, primitive_resolution, shift_map)
        if a is None or b is None:
            return None
        return _bool("intersection", a, b, tm, engine)
    if region.type == "union":
        a = _region_mesh(region.A, bounds, tm, engine, primitive_resolution, shift_map)
        b = _region_mesh(region.B, bounds, tm, engine, primitive_resolution, shift_map)
        if a is None:
            return b
        if b is None:
            return a
        return _bool("union", a, b, tm, engine)
    if region.type == "complement":
        a = _region_mesh(region.A, bounds, tm, engine, primitive_resolution, shift_map)
        if a is None:
            return _world_box(bounds, tm)
        return _bool("difference", _world_box(bounds, tm), a, tm, engine)
    return None


def _tm_to_pv(mesh_tm, pv):
    mesh_tm = _tm_clean(mesh_tm)
    if mesh_tm is None or mesh_tm.faces.shape[0] == 0:
        return None
    faces = np.hstack(
        [
            np.full((mesh_tm.faces.shape[0], 1), 3, dtype=np.int64),
            mesh_tm.faces.astype(np.int64),
        ]
    ).ravel()
    return pv.PolyData(mesh_tm.vertices, faces)


def cell_display_props(simulation, alpha):
    palette = [
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
    ]
    out = []
    for i, cell in enumerate(simulation.cells):
        mat_name = getattr(cell.fill, "name", str(cell.fill))
        color = palette[i % len(palette)]
        low = str(mat_name).lower()
        opacity = min(alpha, 0.15) if ("vacuum" in low or "void" in low) else alpha
        out.append((mat_name, color, opacity))
    return out


def build_frame_cache_entry(
    simulation,
    bounds,
    tm,
    pv,
    engine,
    primitive_resolution,
    source_shifts,
    cell_props,
    time_label=None,
):
    cell_entries = []
    legend = []
    rendered_cell = False

    for i, cell in enumerate(simulation.cells):
        mat_name, color, opacity = cell_props[i]
        try:
            mesh_tm = _region_mesh(
                cell.region,
                bounds,
                tm,
                engine,
                primitive_resolution,
                source_shifts["surface"],
            )
            mesh_pv = _tm_to_pv(mesh_tm, pv)
        except Exception as exc:
            print(f"Warning: geometry meshing failed for cell {cell.ID}: {exc}")
            mesh_pv = None
        if mesh_pv is None or mesh_pv.n_points == 0:
            continue
        cell_entries.append(
            {
                "name": f"mcdc_cell_{i}",
                "mesh": mesh_pv,
                "color": color,
                "opacity": opacity,
            }
        )
        legend.append((mat_name, color))
        rendered_cell = True

    if not rendered_cell:
        print("Warning: no geometry rendered for one frame.")

    seen = set()
    legend_unique = []
    for name, color in legend:
        if name in seen:
            continue
        seen.add(name)
        legend_unique.append((name, color))

    source_entries = []
    world_scale = max(
        1.0e-6,
        max(
            bounds[0][1] - bounds[0][0],
            bounds[1][1] - bounds[1][0],
            bounds[2][1] - bounds[2][0],
        ),
    )
    for i, src in enumerate(getattr(simulation, "sources", [])):
        shift = source_shifts["source"].get(src.ID, np.zeros(3, dtype=float))
        src_name = f"mcdc_source_{i}"
        if getattr(src, "point_source", False):
            center = np.asarray(src.point, dtype=float) + shift
            src_mesh = pv.Sphere(radius=0.015 * world_scale, center=center.tolist())
            source_entries.append({"name": src_name, "mesh": src_mesh, "kind": "solid"})
        else:
            x = np.asarray(src.x, dtype=float) + shift[0]
            y = np.asarray(src.y, dtype=float) + shift[1]
            z = np.asarray(src.z, dtype=float) + shift[2]
            cen = ((x[0] + x[1]) * 0.5, (y[0] + y[1]) * 0.5, (z[0] + z[1]) * 0.5)
            ext = (abs(x[1] - x[0]), abs(y[1] - y[0]), abs(z[1] - z[0]))
            src_mesh = pv.Cube(
                center=cen, x_length=ext[0], y_length=ext[1], z_length=ext[2]
            )
            source_entries.append({"name": src_name, "mesh": src_mesh, "kind": "wire"})

    return {
        "cells": cell_entries,
        "sources": source_entries,
        "legend": legend_unique,
        "label": time_label,
        "has_geometry": rendered_cell,
    }


def render_frame_cache_entry(plotter, frame_entry, actor_names):
    for name in actor_names:
        plotter.remove_actor(name, reset_camera=False)
    new_actor_names = []

    for cell_entry in frame_entry["cells"]:
        plotter.add_mesh(
            cell_entry["mesh"],
            color=cell_entry["color"],
            opacity=cell_entry["opacity"],
            smooth_shading=False,
            name=cell_entry["name"],
            reset_camera=False,
        )
        new_actor_names.append(cell_entry["name"])

    for src_entry in frame_entry["sources"]:
        if src_entry["kind"] == "wire":
            plotter.add_mesh(
                src_entry["mesh"],
                style="wireframe",
                line_width=2.0,
                color=(1.0, 0.9, 0.2),
                opacity=0.9,
                name=src_entry["name"],
                reset_camera=False,
            )
        else:
            plotter.add_mesh(
                src_entry["mesh"],
                color=(1.0, 0.9, 0.2),
                opacity=0.95,
                smooth_shading=False,
                name=src_entry["name"],
                reset_camera=False,
            )
        new_actor_names.append(src_entry["name"])

    plotter.add_legend(frame_entry["legend"], bcolor=None, name="mcdc_legend")
    if frame_entry["label"] is not None:
        plotter.add_text(
            frame_entry["label"],
            position="upper_left",
            font_size=13,
            font="courier",
            name="mcdc_time_label",
        )
    return frame_entry["has_geometry"], new_actor_names


def export_frame_cache_animation(pv, frame_cache, save_animation_path, animation_fps):
    ext = (
        save_animation_path.lower().rsplit(".", 1)[-1]
        if "." in save_animation_path
        else ""
    )
    export_plotter = pv.Plotter(off_screen=True)
    export_plotter.add_axes()
    export_plotter.show_grid()
    if ext == "gif":
        try:
            export_plotter.open_gif(save_animation_path, fps=int(animation_fps))
        except TypeError:
            export_plotter.open_gif(save_animation_path)
    else:
        export_plotter.open_movie(save_animation_path, framerate=int(animation_fps))

    actor_names = []
    for frame in frame_cache:
        rendered, actor_names = render_frame_cache_entry(
            export_plotter, frame, actor_names
        )
        if rendered:
            export_plotter.write_frame()
    export_plotter.close()


def geo_viewer_3d_static(
    simulation,
    primitive_resolution=48,
    alpha=0.6,
    save_animation_path=None,
    animation_fps=12,
):
    try:
        import pyvista as pv
        import trimesh as tm
    except Exception as exc:
        raise RuntimeError("Geometry viewer requires `trimesh` and `pyvista`.") from exc

    engine = "manifold"
    print(f"[geometry] Using boolean engine: {engine}")
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid()

    surface_shifts = zero_shift_map(simulation)
    source_shifts = {
        src.ID: np.zeros(3, dtype=float) for src in getattr(simulation, "sources", [])
    }
    bounds = bounds_from_planes_with_shift(simulation, surface_shifts)
    frame_entry = build_frame_cache_entry(
        simulation=simulation,
        bounds=bounds,
        tm=tm,
        pv=pv,
        engine=engine,
        primitive_resolution=primitive_resolution,
        source_shifts={"surface": surface_shifts, "source": source_shifts},
        cell_props=cell_display_props(simulation, alpha),
        time_label=None,
    )
    rendered, _ = render_frame_cache_entry(plotter, frame_entry, actor_names=[])
    if not rendered:
        return

    if save_animation_path:
        print(f"[geometry] Saving single-frame animation to {save_animation_path}")
        export_frame_cache_animation(
            pv=pv,
            frame_cache=[frame_entry],
            save_animation_path=save_animation_path,
            animation_fps=animation_fps,
        )

    plotter.show(auto_close=False)
