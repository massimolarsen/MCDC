"""Generic PyVista backend for MCDC geometry visualization.

This implementation is fully generic with respect to MCDC CSG regions:
- Evaluates each cell region recursively (halfspace/intersection/union/complement)
- Samples occupancy on a structured grid
- Extracts an isosurface via contouring

"""

from __future__ import annotations

import numpy as np

from mcdc.constant import SURFACE_PLANE_X, SURFACE_PLANE_Y, SURFACE_PLANE_Z


def _surface_value(surface, x, y, z):
    return (
        surface.A * x * x
        + surface.B * y * y
        + surface.C * z * z
        + surface.D * x * y
        + surface.E * x * z
        + surface.F * y * z
        + surface.G * x
        + surface.H * y
        + surface.I * z
        + surface.J
    )


def _evaluate_region_mask(region, x, y, z, tol=1.0e-9):
    if region.type == "all":
        return np.ones(x.shape, dtype=bool)

    if region.type == "halfspace":
        f = _surface_value(region.A, x, y, z)
        return f >= -tol if region.B > 0 else f <= tol

    if region.type == "intersection":
        return _evaluate_region_mask(region.A, x, y, z, tol) & _evaluate_region_mask(
            region.B, x, y, z, tol
        )

    if region.type == "union":
        return _evaluate_region_mask(region.A, x, y, z, tol) | _evaluate_region_mask(
            region.B, x, y, z, tol
        )

    if region.type == "complement":
        return ~_evaluate_region_mask(region.A, x, y, z, tol)

    return np.zeros(x.shape, dtype=bool)


def _infer_bounds_from_surfaces(simulation):
    """Simple auto-bounds.

    Prefer axis plane extents when available; otherwise use a symmetric fallback
    based on largest absolute surface constant.
    """
    xs = [-s.J for s in simulation.surfaces if s.type == SURFACE_PLANE_X]
    ys = [-s.J for s in simulation.surfaces if s.type == SURFACE_PLANE_Y]
    zs = [-s.J for s in simulation.surfaces if s.type == SURFACE_PLANE_Z]

    fallback = max([1.0] + [abs(getattr(s, "J", 0.0)) for s in simulation.surfaces])

    def axis_bounds(values):
        if len(values) >= 2:
            lo = float(np.min(values))
            hi = float(np.max(values))
        elif len(values) == 1:
            d = max(1.0, abs(values[0]))
            lo, hi = -d, d
        else:
            lo, hi = -fallback, fallback
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    return axis_bounds(xs), axis_bounds(ys), axis_bounds(zs)


def _sample_region_mesh(region, bounds, pv, resolution, tol):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    n = int(resolution)

    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    zs = np.linspace(zmin, zmax, n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    occ = _evaluate_region_mask(region, X, Y, Z, tol=tol).astype(np.float32)

    if np.all(occ < 0.5) or np.all(occ > 0.5):
        return None

    grid = pv.ImageData(dimensions=(n, n, n))
    grid.origin = (xmin, ymin, zmin)
    grid.spacing = (
        (xmax - xmin) / max(n - 1, 1),
        (ymax - ymin) / max(n - 1, 1),
        (zmax - zmin) / max(n - 1, 1),
    )
    grid.point_data["occ"] = occ.ravel(order="F")

    surf = grid.contour(isosurfaces=[0.5], scalars="occ")
    if surf is None or surf.n_points == 0:
        return None
    return surf.extract_surface(algorithm="dataset_surface").triangulate().clean()


def geo_viewer_3d(
    simulation,
    pyvista_sample_resolution=300,
    alpha=0.6,
):
    import time

    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError("PyVista backend requested but pyvista is not available.") from exc

    bounds = _infer_bounds_from_surfaces(simulation)
    plotter = pv.Plotter()

    palette = [
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
    ]

    legend = []
    rendered = False
    n_cells = len(simulation.cells)
    t0 = time.perf_counter()

    print(
        f"[pyvista] Loading geometry for {n_cells} cells "
        f"(resolution={pyvista_sample_resolution})..."
    )

    for i, cell in enumerate(simulation.cells):
        mat_name = getattr(cell.fill, "name", str(cell.fill))
        color = palette[i % len(palette)]
        low = str(mat_name).lower()
        opacity = min(alpha, 0.15) if ("vacuum" in low or "void" in low) else alpha

        print(f"[pyvista] Cell {i + 1}/{n_cells}: {cell.name} ({mat_name})")

        mesh = _sample_region_mesh(
            cell.region,
            bounds,
            pv,
            resolution=pyvista_sample_resolution,
            tol=1.0e-9,
        )

        if mesh is None:
            print(f"Warning: could not render cell {cell.ID} ({mat_name})")
            continue

        plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=False)
        legend.append((mat_name, color))
        rendered = True

        elapsed = time.perf_counter() - t0
        print(f"[pyvista]   finished ({elapsed:.2f}s elapsed)")

    if not rendered:
        print("No geometry could be rendered in PyVista backend.")
        return

    seen = set()
    leg = []
    for name, color in legend:
        if name in seen:
            continue
        seen.add(name)
        leg.append((name, color))
    if leg:
        plotter.add_legend(leg, bcolor=None)

    plotter.add_axes()
    plotter.show_grid()

    plotter.show(auto_close=False)
    total = time.perf_counter() - t0
    print(f"[pyvista] Completed in {total:.2f}s")


# Backward-compatible alias
def visualizer_3d_pyvista(*args, **kwargs):
    return geo_viewer_3d(*args, **kwargs)
