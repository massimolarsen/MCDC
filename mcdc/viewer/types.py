"""Shared data structures for geometry visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

ColorRGB = tuple[float, float, float]
AxisBounds = tuple[float, float]


@dataclass(frozen=True)
class Bounds3D:
    """Axis-aligned world bounds used for meshing and display."""

    x: AxisBounds
    y: AxisBounds
    z: AxisBounds

    def axis_lengths(self) -> tuple[float, float, float]:
        return (
            float(self.x[1] - self.x[0]),
            float(self.y[1] - self.y[0]),
            float(self.z[1] - self.z[0]),
        )

    def max_length(self) -> float:
        return max(self.axis_lengths())


@dataclass(frozen=True)
class ShiftState:
    """Per-object translations for a single frame."""

    surface: dict[int, NDArray[np.float64]]
    source: dict[int, NDArray[np.float64]]


@dataclass(frozen=True)
class CellVisual:
    """Renderable cell representation for a frame."""

    actor_name: str
    mesh: Any
    color: ColorRGB
    opacity: float
    label: str


@dataclass(frozen=True)
class SourceVisual:
    """Renderable source representation for a frame."""

    actor_name: str
    mesh: Any
    kind: str


@dataclass(frozen=True)
class FrameEntry:
    """Fully prepared frame data for interactive view or export."""

    cells: list[CellVisual]
    sources: list[SourceVisual]
    legend: list[tuple[str, ColorRGB]]
    label: str | None
    has_geometry: bool


@dataclass
class RenderState:
    """Mutable state for time-stepped frame display."""

    current_frame_index: int = 0
    actor_names: list[str] = field(default_factory=list)
    updating: bool = False
