"""Public geometry viewer API.

Thin wrapper that dispatches to either static or time-dependent implementations.
"""

from __future__ import annotations

from mcdc.object_.tools.visualize_geometry_static import (
    geo_viewer_3d_static,
)
from mcdc.object_.tools.visualize_geometry_time import (
    geo_viewer_3d_time,
    infer_time_steps,
)


def geo_viewer_3d(
    simulation,
    primitive_resolution=48,
    alpha=0.6,
    time_steps=None,
    apply_time=None,
    dynamic_bounds=False,
    use_mcdc_movement=True,
    save_animation_path=None,
    animation_fps=12,
):
    has_motion = any(getattr(s, "moving", False) for s in simulation.surfaces) or any(
        getattr(src, "moving", False) for src in getattr(simulation, "sources", [])
    )

    auto_time_candidates = []
    if use_mcdc_movement and has_motion:
        auto_time_candidates = infer_time_steps(simulation)

    use_time_path = (
        time_steps is not None
        or apply_time is not None
        or (len(auto_time_candidates) > 1)
    )

    if use_time_path:
        return geo_viewer_3d_time(
            simulation=simulation,
            primitive_resolution=primitive_resolution,
            alpha=alpha,
            time_steps=time_steps,
            apply_time=apply_time,
            dynamic_bounds=dynamic_bounds,
            use_mcdc_movement=use_mcdc_movement,
            save_animation_path=save_animation_path,
            animation_fps=animation_fps,
        )

    return geo_viewer_3d_static(
        simulation=simulation,
        primitive_resolution=primitive_resolution,
        alpha=alpha,
        save_animation_path=save_animation_path,
        animation_fps=animation_fps,
    )
