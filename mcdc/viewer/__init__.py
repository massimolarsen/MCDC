"""Optional 3D geometry viewer for MCDC.

This package complements the existing ``mcdc.visualize()`` 2D slice visualizer
with a PyVista-based 3D viewer. Its runtime dependencies are optional and are
imported only when the viewer is invoked.
"""

from .geometry import geo_viewer_3d

__all__ = ["geo_viewer_3d"]
