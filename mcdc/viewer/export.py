"""Animation export helpers for geometry visualization."""

from __future__ import annotations

from mcdc.viewer.render import render_frame_entry

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}


def output_extension(output_path):
    """Return the lowercase filename extension without the leading dot."""

    return output_path.lower().rsplit(".", 1)[-1] if "." in output_path else ""


def export_frame_cache_image(pv, frame_entry, save_image_path):
    """Write a single cached frame to a still image."""

    export_plotter = pv.Plotter(off_screen=True, window_size=(1800, 1400))
    export_plotter.add_axes()
    export_plotter.show_grid()
    rendered, _ = render_frame_entry(export_plotter, frame_entry, actor_names=[])
    if rendered:
        export_plotter.show(auto_close=False)
        export_plotter.screenshot(save_image_path, return_img=False)
    export_plotter.close()


def export_frame_cache_animation(pv, frame_cache, save_animation_path, animation_fps):
    """Write a cached frame sequence to GIF or movie output."""

    ext = output_extension(save_animation_path)
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
