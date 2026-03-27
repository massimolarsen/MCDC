"""Time-view orchestration for geometry visualization."""

from __future__ import annotations

from mcdc.visualization.export import export_frame_cache_animation
from mcdc.visualization.meshing import BOOLEAN_ENGINE, bounds_from_planes_with_shift
from mcdc.visualization.motion import (
    format_frame_label,
    infer_time_steps,
    shift_state_at_time,
)
from mcdc.visualization.render import build_frame_entry, show_frame, step_frame
from mcdc.visualization.types import RenderState

DEFAULT_PRIMITIVE_RESOLUTION = 48


def geo_viewer_3d_time(
    simulation,
    time_steps=None,
    dynamic_bounds=False,
    save_animation_path=None,
    animation_fps=12,
):
    """Render a time-stepped 3D geometry view."""

    try:
        import pyvista as pv
        import trimesh as tm
    except Exception as exc:
        raise RuntimeError("Geometry viewer requires `trimesh` and `pyvista`.") from exc

    # backend setup
    print(f"[geometry] Using boolean engine: {BOOLEAN_ENGINE}")
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid()

    # time selection
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

    # bounds setup
    first_shifts = shift_state_at_time(simulation, times[0])
    static_bounds = (
        None
        if dynamic_bounds
        else bounds_from_planes_with_shift(simulation, first_shifts.surface)
    )

    # frame cache
    frame_cache = []
    print(f"[geometry] Precomputing {len(times)} frames...")
    for index, time_value in enumerate(times):
        print(f"[geometry]   frame {index + 1}/{len(times)} (t={time_value})")
        shifts = shift_state_at_time(simulation, time_value)
        bounds = (
            bounds_from_planes_with_shift(simulation, shifts.surface)
            if dynamic_bounds
            else static_bounds
        )
        frame_cache.append(
            build_frame_entry(
                simulation=simulation,
                bounds=bounds,
                tm=tm,
                pv=pv,
                primitive_resolution=DEFAULT_PRIMITIVE_RESOLUTION,
                shifts=shifts,
                time_label=format_frame_label(index, len(times), time_value),
            )
        )

    # export
    if save_animation_path:
        print(f"[geometry] Saving animation to {save_animation_path}")
        export_frame_cache_animation(
            pv=pv,
            frame_cache=frame_cache,
            save_animation_path=save_animation_path,
            animation_fps=animation_fps,
        )

    render_state = RenderState()

    # interactive controls
    def step_next():
        step_frame(plotter, frame_cache, render_state, delta=1)

    def step_prev():
        step_frame(plotter, frame_cache, render_state, delta=-1)

    plotter.add_key_event("Right", step_next)
    plotter.add_key_event("Left", step_prev)
    plotter.add_key_event("n", step_next)
    plotter.add_key_event("p", step_prev)
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

    # initial render
    show_frame(plotter, frame_cache, render_state, frame_index=0)
    plotter.show(auto_close=False)
