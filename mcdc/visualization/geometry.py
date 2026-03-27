"""Public geometry viewer API.

Thin wrapper that dispatches to either static or time-dependent implementations.
"""

from __future__ import annotations

from mcdc.visualization.motion import infer_time_steps
from mcdc.visualization.static_view import geo_viewer_3d_static
from mcdc.visualization.time_view import geo_viewer_3d_time


def geo_viewer_3d(
    simulation,
    time_steps=None,
    dynamic_bounds=False,
    save_animation_path=None,
    animation_fps=12,
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
        )

    # static view
    return geo_viewer_3d_static(
        simulation=simulation,
        save_animation_path=save_animation_path,
        animation_fps=animation_fps,
    )
