"""Static-view orchestration for geometry visualization."""

from __future__ import annotations

from mcdc.viewer.backends import import_backends
from mcdc.viewer.meshing import (
    BOOLEAN_ENGINE,
    DEFAULT_PRIMITIVE_RESOLUTION,
    bounds_from_planes_with_shift,
)
from mcdc.viewer.motion import zero_shift_state
from mcdc.viewer.render import build_frame_entry, render_frame_entry
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
):
    """Render a single-frame 3D geometry view."""

    pv, tm = import_backends()

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
        if output_extension(save_animation_path) in IMAGE_EXTENSIONS:
            print(f"[geometry] Saving image to {save_animation_path}")
            export_frame_cache_image(
                pv=pv,
                frame_entry=frame_entry,
                save_image_path=save_animation_path,
            )
        else:
            print(f"[geometry] Saving single-frame animation to {save_animation_path}")
            export_frame_cache_animation(
                pv=pv,
                frame_cache=[frame_entry],
                save_animation_path=save_animation_path,
                animation_fps=animation_fps,
            )

    # interactive view
    plotter.show(auto_close=False)
