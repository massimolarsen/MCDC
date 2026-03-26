"""Time-dependent geometry visualization for MCDC."""

from __future__ import annotations

import numpy as np

from mcdc.visualization.geometry_static import (
    bounds_from_planes_with_shift,
    cell_display_props,
    build_frame_cache_entry,
    render_frame_cache_entry,
    export_frame_cache_animation,
)


def _translation_at_time(object_, time_value):
    if not getattr(object_, "moving", False):
        return np.zeros(3, dtype=float)
    time_grid = np.asarray(object_.move_time_grid, dtype=float)
    translations = np.asarray(object_.move_translations, dtype=float)
    velocities = np.asarray(object_.move_velocities, dtype=float)
    idx = int(np.searchsorted(time_grid, time_value, side="right") - 1)
    idx = max(0, min(idx, len(time_grid) - 2))
    t_local = float(time_value) - float(time_grid[idx])
    return translations[idx] + velocities[idx] * t_local


def surface_shift_map(simulation, time_value):
    return {s.ID: _translation_at_time(s, time_value) for s in simulation.surfaces}


def infer_time_steps(simulation):
    times = {0.0}
    for s in simulation.surfaces:
        if getattr(s, "moving", False):
            for t in np.asarray(s.move_time_grid, dtype=float):
                if np.isfinite(t):
                    times.add(float(t))
    for src in getattr(simulation, "sources", []):
        if getattr(src, "moving", False):
            for t in np.asarray(src.move_time_grid, dtype=float):
                if np.isfinite(t):
                    times.add(float(t))
        if getattr(src, "discrete_time", True):
            times.add(float(src.time))
        else:
            times.add(float(src.time_range[0]))
            times.add(float(src.time_range[1]))
    return sorted(times)


def _format_frame_label(frame_idx, n_frames, time_value):
    width = max(2, len(str(int(n_frames))))
    return (
        f"time step: {frame_idx + 1:0{width}d}/{n_frames:0{width}d}   "
        f"t={float(time_value):07.3f}"
    )


def geo_viewer_3d_time(
    simulation,
    primitive_resolution=48,
    alpha=0.6,
    time_steps=None,
    dynamic_bounds=False,
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

    auto_time_steps = False
    if time_steps is None:
        inferred = infer_time_steps(simulation)
        if len(inferred) > 1:
            time_steps = inferred
            auto_time_steps = True

    if time_steps is None:
        raise ValueError(
            "time-dependent viewer requested without time steps; "
            "provide time_steps or define motion with Surface.move(...) or Source.move(...)."
        )

    times = list(time_steps)
    if len(times) == 0:
        raise ValueError("time_steps must contain at least one value.")
    first_surface_shifts = surface_shift_map(simulation, times[0])
    static_bounds = (
        None
        if dynamic_bounds
        else bounds_from_planes_with_shift(simulation, first_surface_shifts)
    )

    frame_cache = []
    props = cell_display_props(simulation, alpha)
    print(f"[geometry] Precomputing {len(times)} frames...")
    for idx, t in enumerate(times):
        print(f"[geometry]   frame {idx + 1}/{len(times)} (t={t})")
        surface_shifts = surface_shift_map(simulation, t)
        source_shifts = {
            src.ID: _translation_at_time(src, t)
            for src in getattr(simulation, "sources", [])
        }
        bounds = (
            bounds_from_planes_with_shift(simulation, surface_shifts)
            if dynamic_bounds
            else static_bounds
        )
        frame_cache.append(
            build_frame_cache_entry(
                simulation=simulation,
                bounds=bounds,
                tm=tm,
                pv=pv,
                engine=engine,
                primitive_resolution=primitive_resolution,
                source_shifts={"surface": surface_shifts, "source": source_shifts},
                cell_props=props,
                time_label=_format_frame_label(idx, len(times), t),
            )
        )

    if save_animation_path:
        print(f"[geometry] Saving animation to {save_animation_path}")
        export_frame_cache_animation(
            pv=pv,
            frame_cache=frame_cache,
            save_animation_path=save_animation_path,
            animation_fps=animation_fps,
        )

    state = {"idx": 0, "actor_names": [], "updating": False}

    def _show_frame(new_idx):
        if state["updating"]:
            return
        state["updating"] = True
        try:
            idx = int(np.clip(int(new_idx), 0, len(frame_cache) - 1))
            rendered, actor_names = render_frame_cache_entry(
                plotter=plotter,
                frame_entry=frame_cache[idx],
                actor_names=state["actor_names"],
            )
            if rendered:
                state["actor_names"] = actor_names
                state["idx"] = idx
                plotter.render()
        finally:
            state["updating"] = False

    def _step_next():
        _show_frame(state["idx"] + 1)

    def _step_prev():
        _show_frame(state["idx"] - 1)

    plotter.add_key_event("Right", _step_next)
    plotter.add_key_event("Left", _step_prev)
    plotter.add_key_event("n", _step_next)
    plotter.add_key_event("p", _step_prev)
    plotter.add_text(
        "Left/Right (or p/n) to step time (precomputed)",
        position="lower_left",
        name="mcdc_controls",
    )
    if auto_time_steps:
        plotter.add_text(
            "Auto timeline from Surface/Source.move(...) grids",
            position=(20, 50),
            name="mcdc_time_auto_note",
        )

    _show_frame(0)
    plotter.show(auto_close=False)
