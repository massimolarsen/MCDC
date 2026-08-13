"""Time-view orchestration for geometry visualization."""

from __future__ import annotations

from mcdc.viewer.backends import import_backends
from mcdc.viewer.controls import (
    add_camera_key_events,
    add_controls_overlay,
    add_source_toggle_key_event,
    apply_initial_view,
    apply_interaction_style,
)
from mcdc.viewer.export import export_frame_cache_animation
from mcdc.viewer.meshing import (
    BOOLEAN_ENGINE,
    DEFAULT_PRIMITIVE_RESOLUTION,
    bounds_from_planes_with_shift,
)
from mcdc.viewer.motion import (
    format_frame_label,
    infer_time_steps,
    shift_state_at_time,
)
from mcdc.viewer.render import (
    add_global_opacity_slider,
    build_frame_entry,
    show_frame,
    step_frame,
)
from mcdc.viewer.types import RenderState


def geo_viewer_3d_time(
    simulation,
    time_steps=None,
    dynamic_bounds=False,
    save_animation_path=None,
    animation_fps=12,
    labels=False,
    color_by="material",
    opacity_slider=True,
    initial_view="isometric",
    interaction_style="terrain",
    show_sources=True,
    source_labels=True,
):
    """Render a time-stepped 3D geometry view."""

    pv, tm = import_backends()

    # backend setup
    print(f"[geometry] Using boolean engine: {BOOLEAN_ENGINE}")
    plotter = pv.Plotter()
    apply_interaction_style(plotter, interaction_style)
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
                color_by=color_by,
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
            labels=labels,
            initial_view=initial_view,
            show_sources=show_sources,
            source_labels=source_labels,
        )

    render_state = RenderState(show_sources=show_sources)

    # interactive controls
    def step_next():
        step_frame(
            plotter,
            frame_cache,
            render_state,
            delta=1,
            labels=labels,
            source_labels=source_labels,
        )

    def step_prev():
        step_frame(
            plotter,
            frame_cache,
            render_state,
            delta=-1,
            labels=labels,
            source_labels=source_labels,
        )

    plotter.add_key_event("Right", step_next)
    plotter.add_key_event("Left", step_prev)
    if auto_time_steps:
        plotter.add_text(
            "Auto timeline from Surface/Source.move(...) grids",
            position=(20, 50),
            name="mcdc_time_auto_note",
        )

    # initial render
    show_frame(
        plotter,
        frame_cache,
        render_state,
        frame_index=0,
        labels=labels,
        source_labels=source_labels,
    )
    apply_initial_view(plotter, initial_view)
    add_camera_key_events(plotter, initial_view)
    add_source_toggle_key_event(
        plotter,
        render_state,
        lambda: show_frame(
            plotter,
            frame_cache,
            render_state,
            frame_index=render_state.current_frame_index,
            labels=labels,
            source_labels=source_labels,
        ),
    )
    add_controls_overlay(plotter, time_controls=True)
    if opacity_slider:
        add_global_opacity_slider(plotter, render_state)
    plotter.show(auto_close=False)
