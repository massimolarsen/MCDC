"""Animation export helpers for geometry visualization."""

from __future__ import annotations

from mcdc.visualization.render import render_frame_entry


def export_frame_cache_animation(pv, frame_cache, save_animation_path, animation_fps):
    """Write a cached frame sequence to GIF or movie output."""

    ext = (
        save_animation_path.lower().rsplit(".", 1)[-1]
        if "." in save_animation_path
        else ""
    )
    export_plotter = pv.Plotter(off_screen=True)
    export_plotter.add_axes()
    export_plotter.show_grid()
    if ext == "gif":
        try:
            export_plotter.open_gif(save_animation_path, fps=int(animation_fps))
        except TypeError:
            export_plotter.open_gif(save_animation_path)
    else:
        export_plotter.open_movie(save_animation_path, framerate=int(animation_fps))

    actor_names = []
    for frame_entry in frame_cache:
        rendered, actor_names = render_frame_entry(
            export_plotter, frame_entry, actor_names
        )
        if rendered:
            export_plotter.write_frame()
    export_plotter.close()
