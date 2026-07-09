"""Motion and time helpers for geometry visualization."""

from __future__ import annotations

import numpy as np

from mcdc.viewer.types import ShiftState


def translation_at_time(object_, time_value):
    """Return the translation of a moving object at the requested time."""

    if not getattr(object_, "moving", False):
        return np.zeros(3, dtype=float)
    time_grid = np.asarray(object_.move_time_grid, dtype=float)
    translations = np.asarray(object_.move_translations, dtype=float)
    velocities = np.asarray(object_.move_velocities, dtype=float)
    idx = int(np.searchsorted(time_grid, time_value, side="right") - 1)
    idx = max(0, min(idx, len(time_grid) - 2))
    t_local = float(time_value) - float(time_grid[idx])
    return translations[idx] + velocities[idx] * t_local


def shift_state_at_time(simulation, time_value):
    """Return the complete shift state for one frame."""

    return ShiftState(
        surface={
            surface.ID: translation_at_time(surface, time_value)
            for surface in simulation.surfaces
        },
        source={
            source.ID: translation_at_time(source, time_value)
            for source in getattr(simulation, "sources", [])
        },
    )


def zero_shift_state(simulation):
    """Return a zero-motion shift state."""

    return shift_state_at_time(simulation, 0.0)


def infer_time_steps(simulation):
    """Infer a timeline from moving surfaces and sources."""

    times = {0.0}
    for surface in simulation.surfaces:
        if getattr(surface, "moving", False):
            for t in np.asarray(surface.move_time_grid, dtype=float):
                if np.isfinite(t):
                    times.add(float(t))
    for source in getattr(simulation, "sources", []):
        if getattr(source, "moving", False):
            for t in np.asarray(source.move_time_grid, dtype=float):
                if np.isfinite(t):
                    times.add(float(t))
        if getattr(source, "discrete_time", True):
            times.add(float(source.time))
        else:
            times.add(float(source.time_range[0]))
            times.add(float(source.time_range[1]))
    return sorted(times)


def format_frame_label(frame_index, frame_count, time_value):
    """Format the time-step annotation for a precomputed frame."""

    width = max(2, len(str(int(frame_count))))
    return (
        f"time step: {frame_index + 1:0{width}d}/{frame_count:0{width}d}   "
        f"t={float(time_value):07.3f}"
    )
