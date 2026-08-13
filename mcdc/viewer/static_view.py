"""Static-view orchestration for geometry visualization."""

from __future__ import annotations

from mcdc.viewer.backends import import_backends
from mcdc.viewer.controls import (
    add_camera_key_events,
    add_controls_overlay,
    add_source_toggle_key_event,
    apply_initial_view,
    apply_interaction_style,
)
from mcdc.viewer.meshing import (
    BOOLEAN_ENGINE,
    DEFAULT_PRIMITIVE_RESOLUTION,
    bounds_from_planes_with_shift,
)
from mcdc.viewer.motion import zero_shift_state
from mcdc.viewer.render import (
    add_global_opacity_slider,
    build_frame_entry,
    show_frame,
)
from mcdc.viewer.types import RenderState
from mcdc.viewer.export import (
    IMAGE_EXTENSIONS,
    export_frame_cache_animation,
    export_frame_cache_image,
    output_extension,
)


def geo_viewer_3d_static(
    simulation,
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
    """Render a single-frame 3D geometry view."""

    pv, tm = import_backends()

    # backend setup
    print(f"[geometry] Using boolean engine: {BOOLEAN_ENGINE}")
    plotter = pv.Plotter()
    apply_interaction_style(plotter, interaction_style)
    plotter.add_axes()
    plotter.show_grid()

    # frame construction
    shifts = zero_shift_state(simulation)
    bounds = bounds_from_planes_with_shift(simulation, shifts.surface)
    frame_entry = build_frame_entry(
        simulation=simulation,
        bounds=bounds,
        tm=tm,
        pv=pv,
        primitive_resolution=DEFAULT_PRIMITIVE_RESOLUTION,
        shifts=shifts,
        time_label=None,
        color_by=color_by,
    )
    render_state = RenderState(show_sources=show_sources)
    frame_cache = [frame_entry]
    show_frame(
        plotter,
        frame_cache,
        render_state,
        frame_index=0,
        labels=labels,
        source_labels=source_labels,
    )
    if not frame_entry.has_geometry:
        return
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
    add_controls_overlay(plotter)
    if opacity_slider:
        add_global_opacity_slider(plotter, render_state)

    # export
    if save_animation_path:
        if output_extension(save_animation_path) in IMAGE_EXTENSIONS:
            print(f"[geometry] Saving image to {save_animation_path}")
            export_frame_cache_image(
                pv=pv,
                frame_entry=frame_entry,
                save_image_path=save_animation_path,
                labels=labels,
                initial_view=initial_view,
                show_sources=show_sources,
                source_labels=source_labels,
            )
        else:
            print(f"[geometry] Saving single-frame animation to {save_animation_path}")
            export_frame_cache_animation(
                pv=pv,
                frame_cache=[frame_entry],
                save_animation_path=save_animation_path,
                animation_fps=animation_fps,
                labels=labels,
                initial_view=initial_view,
                show_sources=show_sources,
                source_labels=source_labels,
            )

    # interactive view
    plotter.show(auto_close=False)
