"""Interactive controls for geometry visualization."""

from __future__ import annotations

CONTROLS_TEXT = (
    "Controls\n"
    "r: reset view\n"
    "z: top (XY)  x: X view (YZ)  y: Y view (XZ)\n"
    "s: show/hide sources"
)


def apply_interaction_style(plotter, interaction_style):
    """Configure the PyVista camera interaction style."""

    if interaction_style == "terrain":
        plotter.enable_terrain_style()
    elif interaction_style == "trackball":
        plotter.enable_trackball_style()
    else:
        raise ValueError("interaction_style must be 'terrain' or 'trackball'.")


def apply_initial_view(plotter, initial_view):
    """Apply a named view or raw PyVista camera position."""

    named_views = {
        "isometric": plotter.view_isometric,
        "xy": plotter.view_xy,
        "xz": plotter.view_xz,
        "yz": plotter.view_yz,
    }
    if isinstance(initial_view, str):
        try:
            named_views[initial_view]()
        except KeyError as exc:
            raise ValueError(
                "initial_view must be 'isometric', 'xy', 'xz', 'yz', "
                "or a PyVista camera position."
            ) from exc
        plotter.reset_camera()
    else:
        plotter.camera_position = initial_view


def add_camera_key_events(plotter, initial_view):
    """Bind reset and named camera views."""

    def reset_view():
        apply_initial_view(plotter, initial_view)
        plotter.render()

    def set_xy():
        plotter.view_xy()
        plotter.reset_camera()
        plotter.render()

    def set_xz():
        plotter.view_xz()
        plotter.reset_camera()
        plotter.render()

    def set_yz():
        plotter.view_yz()
        plotter.reset_camera()
        plotter.render()

    plotter.add_key_event("r", reset_view)
    plotter.add_key_event("z", set_xy)
    plotter.add_key_event("y", set_xz)
    plotter.add_key_event("x", set_yz)


def add_source_toggle_key_event(plotter, render_state, refresh):
    """Bind source visibility toggle."""

    def toggle_sources():
        render_state.show_sources = not render_state.show_sources
        refresh()

    plotter.add_key_event("s", toggle_sources)


def add_controls_overlay(plotter, time_controls=False):
    """Add concise keyboard control text."""

    text = CONTROLS_TEXT
    if time_controls:
        text = f"{text}\nLeft/Right: step time"
    plotter.add_text(
        text,
        position="lower_left",
        name="mcdc_controls",
        font_size=10,
    )
