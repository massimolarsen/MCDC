"""Public trimesh geometry viewer API.

Thin wrapper that dispatches to either static or time-dependent implementations.
"""

from __future__ import annotations

from mcdc.object_.tools.visualize_geometry_trimesh_static import (
    geo_viewer_3d_static_trimesh,
)
from mcdc.object_.tools.visualize_geometry_trimesh_time import (
    geo_viewer_3d_time_trimesh,
    infer_time_steps,
)


def geo_viewer_3d_trimesh(
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
    inferred = infer_time_steps(simulation) if use_mcdc_movement and has_motion else [0.0]
    use_time_path = (
        time_steps is not None
        or apply_time is not None
        or (use_mcdc_movement and has_motion and len(inferred) > 1)
    )

    if use_time_path:
        return geo_viewer_3d_time_trimesh(
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

    return geo_viewer_3d_static_trimesh(
        simulation=simulation,
        primitive_resolution=primitive_resolution,
        alpha=alpha,
        save_animation_path=save_animation_path,
        animation_fps=animation_fps,
    )


def visualizer_3d_trimesh(*args, **kwargs):
    return geo_viewer_3d_trimesh(*args, **kwargs)


def geo_viewer_3d(*args, **kwargs):
    return geo_viewer_3d_trimesh(*args, **kwargs)


def visualizer_3d(*args, **kwargs):
    return geo_viewer_3d_trimesh(*args, **kwargs)


def visualize_simulation(*args, **kwargs):
    return geo_viewer_3d_trimesh(*args, **kwargs)
