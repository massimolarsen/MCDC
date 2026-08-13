"""Runtime backend loading for the optional 3D viewer."""

from __future__ import annotations


def import_backends():
    """Import optional viewer backends only when the viewer is invoked."""

    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError("Geometry viewer requires `pyvista`.") from exc
    return pv
