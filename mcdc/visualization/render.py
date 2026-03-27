"""Frame construction and rendering for geometry visualization."""

from __future__ import annotations

import numpy as np

from mcdc.visualization.meshing import region_mesh, trimesh_to_pyvista
from mcdc.visualization.types import CellVisual, FrameEntry, RenderState, SourceVisual

DEFAULT_ALPHA = 0.6


def cell_display_properties(simulation):
    """Assign display colors and opacities to simulation cells."""

    # material display defaults
    palette = [
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
    ]
    out = []
    for index, cell in enumerate(simulation.cells):
        material_name = getattr(cell.fill, "name", str(cell.fill))
        color = palette[index % len(palette)]
        low_name = str(material_name).lower()
        opacity = (
            min(DEFAULT_ALPHA, 0.15)
            if ("vacuum" in low_name or "void" in low_name)
            else DEFAULT_ALPHA
        )
        out.append((material_name, color, opacity))
    return out


def build_frame_entry(
    simulation,
    bounds,
    tm,
    pv,
    primitive_resolution,
    shifts,
    time_label=None,
):
    """Precompute all renderable data for a single frame."""

    # cell geometry
    cell_visuals: list[CellVisual] = []
    legend_entries: list[tuple[str, tuple[float, float, float]]] = []
    rendered_geometry = False
    cell_props = cell_display_properties(simulation)
    for index, cell in enumerate(simulation.cells):
        material_name, color, opacity = cell_props[index]
        try:
            mesh_tm = region_mesh(
                cell.region,
                bounds,
                tm,
                primitive_resolution,
                shifts.surface,
            )
            mesh_pv = trimesh_to_pyvista(mesh_tm, pv)
        except Exception as exc:
            print(f"Warning: geometry meshing failed for cell {cell.ID}: {exc}")
            mesh_pv = None
        if mesh_pv is None or mesh_pv.n_points == 0:
            continue
        cell_visuals.append(
            CellVisual(
                actor_name=f"mcdc_cell_{index}",
                mesh=mesh_pv,
                color=color,
                opacity=opacity,
                label=material_name,
            )
        )
        legend_entries.append((material_name, color))
        rendered_geometry = True

    if not rendered_geometry:
        print("Warning: no geometry rendered for one frame.")

    # legend
    seen_labels = set()
    unique_legend = []
    for label, color in legend_entries:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_legend.append((label, color))

    # source markers
    source_visuals: list[SourceVisual] = []
    world_scale = max(1.0e-6, bounds.max_length())
    for index, source in enumerate(getattr(simulation, "sources", [])):
        shift = shifts.source.get(source.ID, np.zeros(3, dtype=float))
        actor_name = f"mcdc_source_{index}"
        if getattr(source, "point_source", False):
            center = np.asarray(source.point, dtype=float) + shift
            source_mesh = pv.Sphere(radius=0.015 * world_scale, center=center.tolist())
            source_visuals.append(
                SourceVisual(actor_name=actor_name, mesh=source_mesh, kind="solid")
            )
        else:
            x = np.asarray(source.x, dtype=float) + shift[0]
            y = np.asarray(source.y, dtype=float) + shift[1]
            z = np.asarray(source.z, dtype=float) + shift[2]
            center = (
                (x[0] + x[1]) * 0.5,
                (y[0] + y[1]) * 0.5,
                (z[0] + z[1]) * 0.5,
            )
            extents = (abs(x[1] - x[0]), abs(y[1] - y[0]), abs(z[1] - z[0]))
            source_mesh = pv.Cube(
                center=center,
                x_length=extents[0],
                y_length=extents[1],
                z_length=extents[2],
            )
            source_visuals.append(
                SourceVisual(actor_name=actor_name, mesh=source_mesh, kind="wire")
            )

    return FrameEntry(
        cells=cell_visuals,
        sources=source_visuals,
        legend=unique_legend,
        label=time_label,
        has_geometry=rendered_geometry,
    )


def render_frame_entry(plotter, frame_entry, actor_names):
    """Render a prepared frame entry and return the active actor names."""

    # actor replacement
    for actor_name in actor_names:
        plotter.remove_actor(actor_name, reset_camera=False)
    new_actor_names = []

    # cell actors
    for cell_visual in frame_entry.cells:
        plotter.add_mesh(
            cell_visual.mesh,
            color=cell_visual.color,
            opacity=cell_visual.opacity,
            smooth_shading=False,
            name=cell_visual.actor_name,
            reset_camera=False,
        )
        new_actor_names.append(cell_visual.actor_name)

    # source actors
    for source_visual in frame_entry.sources:
        if source_visual.kind == "wire":
            plotter.add_mesh(
                source_visual.mesh,
                style="wireframe",
                line_width=2.0,
                color=(1.0, 0.9, 0.2),
                opacity=0.9,
                name=source_visual.actor_name,
                reset_camera=False,
            )
        else:
            plotter.add_mesh(
                source_visual.mesh,
                color=(1.0, 0.9, 0.2),
                opacity=0.95,
                smooth_shading=False,
                name=source_visual.actor_name,
                reset_camera=False,
            )
        new_actor_names.append(source_visual.actor_name)

    # overlays
    plotter.add_legend(frame_entry.legend, bcolor=None, name="mcdc_legend")
    if frame_entry.label is not None:
        plotter.add_text(
            frame_entry.label,
            position="upper_left",
            font_size=13,
            font="courier",
            name="mcdc_time_label",
        )
    return frame_entry.has_geometry, new_actor_names


def show_frame(plotter, frame_cache, render_state, frame_index):
    """Render one frame from the cache into the active plotter."""

    # re-entrant guard
    if render_state.updating:
        return
    render_state.updating = True
    try:
        # frame selection
        index = int(np.clip(int(frame_index), 0, len(frame_cache) - 1))
        rendered, actor_names = render_frame_entry(
            plotter=plotter,
            frame_entry=frame_cache[index],
            actor_names=render_state.actor_names,
        )
        if rendered:
            render_state.actor_names = actor_names
            render_state.current_frame_index = index
            plotter.render()
    finally:
        render_state.updating = False


def step_frame(plotter, frame_cache, render_state, delta):
    """Advance the current render state by a signed frame delta."""

    # frame stepping
    show_frame(
        plotter=plotter,
        frame_cache=frame_cache,
        render_state=render_state,
        frame_index=render_state.current_frame_index + delta,
    )
