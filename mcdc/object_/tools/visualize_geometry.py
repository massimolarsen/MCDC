"""3D geometry visualizer for MCDC simulations.

Simple primitive plotting (planes/cylinders/spheres) with CSG clipping.
No voxel rendering.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mcdc.constant import (
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_PLANE,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_SPHERE,
)


def _draw_legend(ax, color_map):
    handles = []
    labels = []
    for mat_name in sorted(color_map.keys()):
        col = color_map[mat_name]
        handles.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor=col, markersize=8)
        )
        labels.append(mat_name)
    ax.legend(handles, labels, title="Materials", loc="upper right", frameon=False)


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
    """Evaluate Region tree with CSG operators."""
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
    xmins, xmaxs = [], []
    ymins, ymaxs = [], []
    zmins, zmaxs = [], []

    for s in simulation.surfaces:
        if s.type == SURFACE_PLANE_X:
            x0 = -s.J
            xmins.append(x0)
            xmaxs.append(x0)
        elif s.type == SURFACE_PLANE_Y:
            y0 = -s.J
            ymins.append(y0)
            ymaxs.append(y0)
        elif s.type == SURFACE_PLANE_Z:
            z0 = -s.J
            zmins.append(z0)
            zmaxs.append(z0)
        elif s.type == SURFACE_SPHERE:
            cx = -0.5 * s.G
            cy = -0.5 * s.H
            cz = -0.5 * s.I
            r2 = cx * cx + cy * cy + cz * cz - s.J
            if r2 > 0.0:
                r = np.sqrt(r2)
                xmins.append(cx - r)
                xmaxs.append(cx + r)
                ymins.append(cy - r)
                ymaxs.append(cy + r)
                zmins.append(cz - r)
                zmaxs.append(cz + r)
        elif s.type == SURFACE_CYLINDER_X:
            cy = -0.5 * s.H
            cz = -0.5 * s.I
            r2 = cy * cy + cz * cz - s.J
            if r2 > 0.0:
                r = np.sqrt(r2)
                ymins.append(cy - r)
                ymaxs.append(cy + r)
                zmins.append(cz - r)
                zmaxs.append(cz + r)
        elif s.type == SURFACE_CYLINDER_Y:
            cx = -0.5 * s.G
            cz = -0.5 * s.I
            r2 = cx * cx + cz * cz - s.J
            if r2 > 0.0:
                r = np.sqrt(r2)
                xmins.append(cx - r)
                xmaxs.append(cx + r)
                zmins.append(cz - r)
                zmaxs.append(cz + r)
        elif s.type == SURFACE_CYLINDER_Z:
            cx = -0.5 * s.G
            cy = -0.5 * s.H
            r2 = cx * cx + cy * cy - s.J
            if r2 > 0.0:
                r = np.sqrt(r2)
                xmins.append(cx - r)
                xmaxs.append(cx + r)
                ymins.append(cy - r)
                ymaxs.append(cy + r)

    def _bound(mins, maxs):
        if len(mins) == 0 or len(maxs) == 0:
            return -1.0, 1.0
        lo = float(np.min(mins))
        hi = float(np.max(maxs))
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            return -1.0, 1.0
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    return _bound(xmins, xmaxs), _bound(ymins, ymaxs), _bound(zmins, zmaxs)


def _plot_masked_surface(ax, X, Y, Z, mask, color_rgb, alpha):
    # Close 1-pixel cracks from grid clipping at shared edges/corners.
    if mask.ndim == 2:
        m = mask.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                m |= np.roll(np.roll(mask, di, axis=0), dj, axis=1)
        mask = m

    Xm = X.copy()
    Ym = Y.copy()
    Zm = Z.copy()
    Xm[~mask] = np.nan
    Ym[~mask] = np.nan
    Zm[~mask] = np.nan
    if np.all(np.isnan(Xm)):
        return False
    ax.plot_surface(
        Xm,
        Ym,
        Zm,
        color=color_rgb,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=False,
    )
    return True


def visualizer_3d(
    simulation,
    figsize=(10, 8),
    alpha=0.6,
    interactive=True,
    save_path=None,
    surface_resolution=100,
    resolution=None,
    boundary_eps=1.0e-5,
    bounds=None,
    backend="matplotlib",
    pyvista_notebook=False,
    pyvista_off_screen=None,
    pyvista_sample_resolution=72,
):
    """Visualize geometry with either matplotlib or PyVista backend."""
    if backend == "pyvista":
        from mcdc.object_.tools.visualize_geometry_pyvista import geo_viewer_3d

        return geo_viewer_3d(
            simulation=simulation,
            pyvista_sample_resolution=pyvista_sample_resolution,
            alpha=alpha,
        )
    if backend != "matplotlib":
        raise ValueError("backend must be 'matplotlib' or 'pyvista'")

    if not interactive and save_path is not None:
        plt.switch_backend("Agg")

    unique_materials = []
    for cell in simulation.cells:
        fill = cell.fill
        unique_materials.append(getattr(fill, "name", str(fill)))
    unique_materials = np.unique(unique_materials)

    cmap = plt.get_cmap("tab10")
    color_map = {}
    for i, m in enumerate(unique_materials):
        base = cmap(i % 10)
        color_map[m] = (base[0], base[1], base[2])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if bounds is None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = _infer_bounds_from_surfaces(simulation)
    else:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    if resolution is not None:
        surface_resolution = resolution
    n_ang = max(36, int(surface_resolution))
    n_lin = max(30, int(surface_resolution * 0.7))

    rendered_any = False

    for cell in simulation.cells:
        mat_name = getattr(cell.fill, "name", str(cell.fill))
        col_rgb = color_map.get(mat_name, (0.5, 0.5, 0.5))
        low = str(mat_name).lower()
        use_alpha = min(alpha, 0.15) if ("vacuum" in low or "void" in low) else alpha

        for s in cell.surfaces:
            try:
                if s.type == SURFACE_PLANE_X:
                    yy, zz = np.meshgrid(np.linspace(ymin, ymax, n_lin), np.linspace(zmin, zmax, n_lin))
                    xx = np.full_like(yy, -s.J)
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps)
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

                elif s.type == SURFACE_PLANE_Y:
                    xx, zz = np.meshgrid(np.linspace(xmin, xmax, n_lin), np.linspace(zmin, zmax, n_lin))
                    yy = np.full_like(xx, -s.J)
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps)
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

                elif s.type == SURFACE_PLANE_Z:
                    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n_lin), np.linspace(ymin, ymax, n_lin))
                    zz = np.full_like(xx, -s.J)
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps)
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

                elif s.type == SURFACE_PLANE:
                    n = np.array([s.G, s.H, s.I], dtype=float)
                    nn = np.linalg.norm(n)
                    if nn <= 0.0:
                        continue
                    n /= nn
                    p0 = -s.J * n / nn
                    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                    u = np.cross(n, ref)
                    u /= np.linalg.norm(u)
                    v = np.cross(n, u)
                    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
                    uu, vv = np.meshgrid(np.linspace(-span, span, n_lin), np.linspace(-span, span, n_lin))
                    xx = p0[0] + uu * u[0] + vv * v[0]
                    yy = p0[1] + uu * u[1] + vv * v[1]
                    zz = p0[2] + uu * u[2] + vv * v[2]
                    in_box = (
                        (xx >= xmin)
                        & (xx <= xmax)
                        & (yy >= ymin)
                        & (yy <= ymax)
                        & (zz >= zmin)
                        & (zz <= zmax)
                    )
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps) & in_box
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

                elif s.type == SURFACE_CYLINDER_Z:
                    cx = -0.5 * s.G
                    cy = -0.5 * s.H
                    r2 = cx * cx + cy * cy - s.J
                    if r2 <= 0.0:
                        continue
                    r = np.sqrt(r2)
                    ang, zz = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, n_ang), np.linspace(zmin, zmax, n_lin))
                    xx = cx + r * np.cos(ang)
                    yy = cy + r * np.sin(ang)
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps)
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

                    z_caps = sorted([-p.J for p in cell.surfaces if p.type == SURFACE_PLANE_Z])
                    if len(z_caps) >= 2:
                        xcap, ycap = np.meshgrid(np.linspace(cx - r, cx + r, n_lin), np.linspace(cy - r, cy + r, n_lin))
                        disk = (xcap - cx) ** 2 + (ycap - cy) ** 2 <= r * r + boundary_eps
                        for zc in (z_caps[0], z_caps[-1]):
                            zcap = np.full_like(xcap, zc)
                            mask = _evaluate_region_mask(cell.region, xcap, ycap, zcap, boundary_eps) & disk
                            rendered_any |= _plot_masked_surface(ax, xcap, ycap, zcap, mask, col_rgb, use_alpha)

                elif s.type == SURFACE_CYLINDER_X:
                    cy = -0.5 * s.H
                    cz = -0.5 * s.I
                    r2 = cy * cy + cz * cz - s.J
                    if r2 <= 0.0:
                        continue
                    r = np.sqrt(r2)
                    ang, xx = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, n_ang), np.linspace(xmin, xmax, n_lin))
                    yy = cy + r * np.cos(ang)
                    zz = cz + r * np.sin(ang)
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps)
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

                    x_caps = sorted([-p.J for p in cell.surfaces if p.type == SURFACE_PLANE_X])
                    if len(x_caps) >= 2:
                        ycap, zcap = np.meshgrid(np.linspace(cy - r, cy + r, n_lin), np.linspace(cz - r, cz + r, n_lin))
                        disk = (ycap - cy) ** 2 + (zcap - cz) ** 2 <= r * r + boundary_eps
                        for xc in (x_caps[0], x_caps[-1]):
                            xcap = np.full_like(ycap, xc)
                            mask = _evaluate_region_mask(cell.region, xcap, ycap, zcap, boundary_eps) & disk
                            rendered_any |= _plot_masked_surface(ax, xcap, ycap, zcap, mask, col_rgb, use_alpha)

                elif s.type == SURFACE_CYLINDER_Y:
                    cx = -0.5 * s.G
                    cz = -0.5 * s.I
                    r2 = cx * cx + cz * cz - s.J
                    if r2 <= 0.0:
                        continue
                    r = np.sqrt(r2)
                    ang, yy = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, n_ang), np.linspace(ymin, ymax, n_lin))
                    xx = cx + r * np.cos(ang)
                    zz = cz + r * np.sin(ang)
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps)
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

                    y_caps = sorted([-p.J for p in cell.surfaces if p.type == SURFACE_PLANE_Y])
                    if len(y_caps) >= 2:
                        xcap, zcap = np.meshgrid(np.linspace(cx - r, cx + r, n_lin), np.linspace(cz - r, cz + r, n_lin))
                        disk = (xcap - cx) ** 2 + (zcap - cz) ** 2 <= r * r + boundary_eps
                        for yc in (y_caps[0], y_caps[-1]):
                            ycap = np.full_like(xcap, yc)
                            mask = _evaluate_region_mask(cell.region, xcap, ycap, zcap, boundary_eps) & disk
                            rendered_any |= _plot_masked_surface(ax, xcap, ycap, zcap, mask, col_rgb, use_alpha)

                elif s.type == SURFACE_SPHERE:
                    cx = -0.5 * s.G
                    cy = -0.5 * s.H
                    cz = -0.5 * s.I
                    r2 = cx * cx + cy * cy + cz * cz - s.J
                    if r2 <= 0.0:
                        continue
                    r = np.sqrt(r2)
                    u, v = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, n_ang), np.linspace(0.0, np.pi, n_lin))
                    xx = cx + r * np.cos(u) * np.sin(v)
                    yy = cy + r * np.sin(u) * np.sin(v)
                    zz = cz + r * np.cos(v)
                    mask = _evaluate_region_mask(cell.region, xx, yy, zz, boundary_eps)
                    rendered_any |= _plot_masked_surface(ax, xx, yy, zz, mask, col_rgb, use_alpha)

            except Exception as exc:
                print(f"Warning: surface render failed for cell {cell.ID} surface {s.ID}: {exc}")

    if not rendered_any:
        print("No analytic geometry could be rendered.")
        return

    _draw_legend(ax, color_map)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("Geometry preview")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        if interactive:
            try:
                plt.show()
            except Exception as exc:
                print(f"Warning: Could not display plot interactively: {exc}")
                plt.close(fig)
        else:
            plt.close(fig)


def visualize_simulation(*args, **kwargs):
    """Backward-compatible alias."""
    return visualizer_3d(*args, **kwargs)


def geo_viewer_3d(*args, **kwargs):
    """Preferred alias."""
    return visualizer_3d(*args, **kwargs)
