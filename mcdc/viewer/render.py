"""Frame construction and rendering for geometry visualization."""

from __future__ import annotations

import warnings

import numpy as np

from mcdc.viewer.meshing import region_mesh, trimesh_to_pyvista
from mcdc.viewer.types import CellVisual, FrameEntry, RenderState, SourceVisual

DEFAULT_ALPHA = 0.6
CELL_LABEL_ACTOR = "mcdc_cell_labels"
SOURCE_LABEL_ACTOR = "mcdc_source_labels"
SOURCE_COLOR = (1.0, 0.9, 0.2)
MATERIAL_PALETTE = [
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
]


def cell_background_policy(display_name, material_name):
    """Return background display policy for a cell label/material pair."""

    cell_name = str(display_name).strip().lower()
    material_name = str(material_name).strip().lower()
    low_name = f"{cell_name} {material_name}"
    skip = cell_name == "vacuum" and "vacuum" in material_name
    fade = "vacuum" in low_name or "void" in low_name
    return skip, fade


def cell_display_properties(simulation, color_by="material"):
    """Assign display colors and opacities to simulation cells."""

    if color_by not in {"cell", "material"}:
        raise ValueError("color_by must be 'cell' or 'material'.")

    cell_data = []
    visible_materials = set()
    for index, cell in enumerate(simulation.cells):
        cell_name = getattr(cell, "name", f"cell_{index}")
        material_name = getattr(cell.fill, "name", str(cell.fill))
        display_name = cell_name or material_name
        skip, fade = cell_background_policy(display_name, material_name)
        cell_data.append((display_name, material_name, skip, fade))
        if not skip:
            visible_materials.add(material_name)

    material_colors = {
        material_name: MATERIAL_PALETTE[index % len(MATERIAL_PALETTE)]
        for index, material_name in enumerate(sorted(visible_materials))
    }

    out = []
    for index, cell in enumerate(simulation.cells):
        display_name, material_name, skip, fade = cell_data[index]
        if skip:
            out.append(None)
            continue
        color = (
            material_colors[material_name]
            if color_by == "material"
            else MATERIAL_PALETTE[index % len(MATERIAL_PALETTE)]
        )
        opacity = min(DEFAULT_ALPHA, 0.15) if fade else DEFAULT_ALPHA
        legend_label = material_name if color_by == "material" else display_name
        out.append(
            {
                "label": display_name,
                "material": material_name,
                "color": color,
                "opacity": opacity,
                "legend": legend_label,
            }
        )
    return out


def build_frame_entry(
    simulation,
    bounds,
    tm,
    pv,
    primitive_resolution,
    shifts,
    time_label=None,
    color_by="material",
):
    """Precompute all renderable data for a single frame."""

    # cell geometry
    cell_visuals: list[CellVisual] = []
    legend_entries: list[tuple[str, tuple[float, float, float]]] = []
    rendered_geometry = False
    cell_props = cell_display_properties(simulation, color_by=color_by)
    for index, cell in enumerate(simulation.cells):
        props = cell_props[index]
        if props is None:
            continue
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
            warnings.warn(
                f"Geometry meshing failed for cell {cell.ID}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            mesh_pv = None
        if mesh_pv is None or mesh_pv.n_points == 0:
            continue
        cell_visuals.append(
            CellVisual(
                actor_name=f"mcdc_cell_{index}",
                mesh=mesh_pv,
                color=props["color"],
                opacity=props["opacity"],
                label=props["label"],
            )
        )
        legend_entries.append((props["legend"], props["color"]))
        rendered_geometry = True

    if not rendered_geometry:
        warnings.warn(
            "No geometry rendered for one frame.", RuntimeWarning, stacklevel=2
        )

    # source markers
    source_visuals: list[SourceVisual] = []
    world_scale = max(1.0e-6, bounds.max_length())
    for index, source in enumerate(getattr(simulation, "sources", [])):
        shift = shifts.source.get(source.ID, np.zeros(3, dtype=float))
        actor_name = f"mcdc_source_{index}"
        source_name = getattr(source, "name", f"source_{index}")
        if getattr(source, "point_source", False):
            center = np.asarray(source.point, dtype=float) + shift
            source_mesh = pv.Sphere(radius=0.025 * world_scale, center=center.tolist())
            source_visuals.append(
                SourceVisual(
                    actor_name=actor_name,
                    mesh=source_mesh,
                    kind="solid",
                    label=source_name,
                    center=center,
                )
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
                SourceVisual(
                    actor_name=actor_name,
                    mesh=source_mesh,
                    kind="wire",
                    label=source_name,
                    center=np.asarray(center, dtype=float),
                )
            )
        legend_entries.append((source_name, SOURCE_COLOR))

    # legend
    seen_labels = set()
    unique_legend = []
    for label, color in legend_entries:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_legend.append((label, color))

    return FrameEntry(
        cells=cell_visuals,
        sources=source_visuals,
        legend=unique_legend,
        label=time_label,
        has_geometry=rendered_geometry,
    )


def _mesh_center(mesh):
    if hasattr(mesh, "center"):
        return np.asarray(mesh.center, dtype=float)
    if hasattr(mesh, "bounds"):
        bounds = np.asarray(mesh.bounds, dtype=float)
        return np.array(
            [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ]
        )
    return None


def _add_cell_labels(plotter, cell_visuals):
    points = []
    labels = []
    for cell_visual in cell_visuals:
        center = _mesh_center(cell_visual.mesh)
        if center is None:
            continue
        points.append(center)
        labels.append(cell_visual.label)

    if not labels:
        return None

    plotter.add_point_labels(
        np.asarray(points),
        labels,
        font_size=10,
        point_size=0,
        shape_opacity=0.2,
        always_visible=False,
        name=CELL_LABEL_ACTOR,
    )
    return CELL_LABEL_ACTOR


def _add_source_labels(plotter, source_visuals):
    points = []
    labels = []
    for source_visual in source_visuals:
        points.append(source_visual.center)
        labels.append(source_visual.label)

    if not labels:
        return None

    plotter.add_point_labels(
        np.asarray(points),
        labels,
        font_size=11,
        point_size=0,
        text_color=SOURCE_COLOR,
        shape_opacity=0.25,
        always_visible=False,
        name=SOURCE_LABEL_ACTOR,
    )
    return SOURCE_LABEL_ACTOR


def _actor_property(actor):
    if hasattr(actor, "GetProperty"):
        return actor.GetProperty()
    if hasattr(actor, "prop"):
        return actor.prop
    return None


def _set_actor_opacity(actor, opacity):
    prop = _actor_property(actor)
    if prop is None:
        return
    if hasattr(prop, "SetOpacity"):
        prop.SetOpacity(float(opacity))
    elif hasattr(prop, "opacity"):
        prop.opacity = float(opacity)


def add_global_opacity_slider(plotter, render_state):
    """Add a single slider controlling all rendered cell actor opacities."""

    def set_opacity(value):
        render_state.opacity_scale = float(value)
        for actor, base_opacity in getattr(plotter, "_mcdc_cell_actors", []):
            _set_actor_opacity(actor, base_opacity * render_state.opacity_scale)
        plotter.render()

    plotter.add_slider_widget(
        set_opacity,
        (0.02, 1.0),
        value=render_state.opacity_scale,
        title="Cell opacity",
        pointa=(0.68, 0.08),
        pointb=(0.96, 0.08),
        color=(0.12, 0.47, 0.71),
        interaction_event="always",
        style="modern",
        title_height=0.025,
        title_color=(0.12, 0.12, 0.12),
        fmt="%.2f",
        slider_width=0.025,
        tube_width=0.006,
    )


def render_frame_entry(
    plotter,
    frame_entry,
    actor_names,
    labels=False,
    opacity_scale=1.0,
    show_sources=True,
    source_labels=True,
):
    """Render a prepared frame entry and return the active actor names."""

    # actor replacement
    for actor_name in actor_names:
        plotter.remove_actor(actor_name, reset_camera=False)
    new_actor_names = []
    cell_actor_records = []

    # cell actors
    for cell_visual in frame_entry.cells:
        actor = plotter.add_mesh(
            cell_visual.mesh,
            color=cell_visual.color,
            opacity=cell_visual.opacity * opacity_scale,
            smooth_shading=False,
            name=cell_visual.actor_name,
            reset_camera=False,
        )
        if actor is not None:
            cell_actor_records.append((actor, cell_visual.opacity))
        new_actor_names.append(cell_visual.actor_name)

    plotter._mcdc_cell_actors = cell_actor_records

    # source actors
    if show_sources:
        for source_visual in frame_entry.sources:
            if source_visual.kind == "wire":
                plotter.add_mesh(
                    source_visual.mesh,
                    style="wireframe",
                    line_width=3.0,
                    color=SOURCE_COLOR,
                    opacity=1.0,
                    name=source_visual.actor_name,
                    reset_camera=False,
                )
            else:
                plotter.add_mesh(
                    source_visual.mesh,
                    color=SOURCE_COLOR,
                    opacity=0.98,
                    smooth_shading=False,
                    name=source_visual.actor_name,
                    reset_camera=False,
                )
            new_actor_names.append(source_visual.actor_name)

    # cell labels
    if labels:
        label_actor = _add_cell_labels(plotter, frame_entry.cells)
        if label_actor is not None:
            new_actor_names.append(label_actor)
    if show_sources and source_labels:
        label_actor = _add_source_labels(plotter, frame_entry.sources)
        if label_actor is not None:
            new_actor_names.append(label_actor)

    # overlays
    legend = frame_entry.legend
    if not show_sources:
        source_legend = {
            (source_visual.label, SOURCE_COLOR) for source_visual in frame_entry.sources
        }
        legend = [entry for entry in frame_entry.legend if entry not in source_legend]
    plotter.add_legend(legend, bcolor=None, name="mcdc_legend")
    if frame_entry.label is not None:
        plotter.add_text(
            frame_entry.label,
            position="upper_left",
            font_size=13,
            font="courier",
            name="mcdc_time_label",
        )
    return frame_entry.has_geometry, new_actor_names


def show_frame(
    plotter,
    frame_cache,
    render_state,
    frame_index,
    labels=False,
    source_labels=True,
):
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
            labels=labels,
            opacity_scale=render_state.opacity_scale,
            show_sources=render_state.show_sources,
            source_labels=source_labels,
        )
        if rendered:
            render_state.actor_names = actor_names
            render_state.current_frame_index = index
            plotter.render()
    finally:
        render_state.updating = False


def step_frame(
    plotter, frame_cache, render_state, delta, labels=False, source_labels=True
):
    """Advance the current render state by a signed frame delta."""

    # frame stepping
    show_frame(
        plotter=plotter,
        frame_cache=frame_cache,
        render_state=render_state,
        frame_index=render_state.current_frame_index + delta,
        labels=labels,
        source_labels=source_labels,
    )
