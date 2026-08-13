"""Public geometry viewer API.

Thin wrapper that dispatches to either static or time-dependent implementations.
"""

from __future__ import annotations

from mcdc.viewer.motion import infer_time_steps
from mcdc.viewer.static_view import geo_viewer_3d_static
from mcdc.viewer.time_view import geo_viewer_3d_time


def geo_viewer_3d(
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
    sample_resolution=96,
    save_vtk_path=None,
    vtk_bounds=None,
):
    # mode selection
    has_motion = any(getattr(s, "moving", False) for s in simulation.surfaces) or any(
        getattr(src, "moving", False) for src in getattr(simulation, "sources", [])
    )

    auto_time_candidates = infer_time_steps(simulation) if has_motion else []

    use_time_path = time_steps is not None or (len(auto_time_candidates) > 1)

    if use_time_path:
        # time view
        return geo_viewer_3d_time(
            simulation=simulation,
            time_steps=time_steps,
            dynamic_bounds=dynamic_bounds,
            save_animation_path=save_animation_path,
            animation_fps=animation_fps,
            labels=labels,
            color_by=color_by,
            opacity_slider=opacity_slider,
            initial_view=initial_view,
            interaction_style=interaction_style,
            show_sources=show_sources,
            source_labels=source_labels,
            sample_resolution=sample_resolution,
            save_vtk_path=save_vtk_path,
            vtk_bounds=vtk_bounds,
        )

    # static view
    return geo_viewer_3d_static(
        simulation=simulation,
        save_animation_path=save_animation_path,
        animation_fps=animation_fps,
        labels=labels,
        color_by=color_by,
        opacity_slider=opacity_slider,
        initial_view=initial_view,
        interaction_style=interaction_style,
        show_sources=show_sources,
        source_labels=source_labels,
        sample_resolution=sample_resolution,
        save_vtk_path=save_vtk_path,
        vtk_bounds=vtk_bounds,
    )
