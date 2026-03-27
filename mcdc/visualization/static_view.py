"""Static-view orchestration for geometry visualization."""

from __future__ import annotations

from mcdc.visualization.meshing import BOOLEAN_ENGINE, bounds_from_planes_with_shift
from mcdc.visualization.motion import zero_shift_state
from mcdc.visualization.render import build_frame_entry, render_frame_entry
from mcdc.visualization.export import export_frame_cache_animation

DEFAULT_PRIMITIVE_RESOLUTION = 48


def geo_viewer_3d_static(
    simulation,
    save_animation_path=None,
    animation_fps=12,
):
    """Render a single-frame 3D geometry view."""

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
    )
    rendered, _ = render_frame_entry(plotter, frame_entry, actor_names=[])
    if not rendered:
        return

    # export
    if save_animation_path:
        print(f"[geometry] Saving single-frame animation to {save_animation_path}")
        export_frame_cache_animation(
            pv=pv,
            frame_cache=[frame_entry],
            save_animation_path=save_animation_path,
            animation_fps=animation_fps,
        )

    # interactive view
    plotter.show(auto_close=False)
