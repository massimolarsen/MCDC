"""
3D Geometry visualizer for MCDC simulations.

Renders geometry stored in mcdc.simulation using analytic primitives (cylinders,
spheres, boxes) when detected
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from mcdc.constant import (
    SURFACE_CYLINDER_Z,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_SPHERE,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
)


def _infer_bounds(simulation):
    """Infer bounding box from linear planes in simulation surfaces."""
    xs, ys, zs = [], [], []
    for s in simulation.surfaces:
        # Use surface type constants to identify plane surfaces
        if s.type == SURFACE_PLANE_X:
            # For PlaneX: J = -x, so x = -J
            xs.append(-s.J)
        elif s.type == SURFACE_PLANE_Y:
            # For PlaneY: J = -y, so y = -J
            ys.append(-s.J)
        elif s.type == SURFACE_PLANE_Z:
            # For PlaneZ: J = -z, so z = -J
            zs.append(-s.J)
    
    xmin, xmax = (min(xs), max(xs)) if xs else (-1.0, 1.0)
    ymin, ymax = (min(ys), max(ys)) if ys else (-1.0, 1.0)
    zmin, zmax = (min(zs), max(zs)) if zs else (-1.0, 1.0)
    
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)


def _set_axes_equal(ax):
    """Set 3D axes to equal aspect ratio."""
    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()
    z_min, z_max = ax.get_zlim3d()
    
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)
    z_range = abs(z_max - z_min)
    
    x_mid = np.mean([x_min, x_max])
    y_mid = np.mean([y_min, y_max])
    z_mid = np.mean([z_min, z_max])
    
    plot_radius = 0.5 * max(x_range, y_range, z_range)
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def _draw_legend(ax, color_map):
    """Draw material legend on axes."""
    handles = []
    labels = []
    for mat_name in sorted(color_map.keys()):
        col = color_map[mat_name]
        handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor=col, markersize=8))
        labels.append(mat_name)
    ax.legend(handles, labels, title="Materials", loc="upper right")

def visualize_simulation(
    simulation,
    bounds=None,
    figsize=(10, 8),
    alpha=0.6,
    interactive=True,
    save_path=None,
):
    """Visualize 3D geometry from MCDC simulation.
    
    Parameters
    ----------
    simulation : mcdc.object_.simulation.Simulation
        Simulation object containing geometry
    bounds : dict, optional
        {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
        If None, inferred from simulation surfaces
    figsize : tuple, default (10, 8)
        Figure size (width, height) in inches
    alpha : float, default 0.6
        Transparency level for opaque materials (0-1).  Materials whose name
        contains ``vacuum`` or ``void`` are made more transparent (capped at
        0.15) automatically.
    interactive : bool, default True
        If True, display plot interactively. If False, requires save_path.
    save_path : str, optional
        Path to save PNG. If provided, plot is saved regardless of interactive flag.
        
    Examples
    --------
    >>> from mcdc.tools.visualize_geometry import visualize_simulation
    >>> import mcdc
    >>> visualize_simulation(mcdc.object_.simulation.simulation, interactive=False, save_path="geom.png")
    >>> # Interactive mode (default)
    >>> visualize_simulation(mcdc.object_.simulation.simulation)
    """
    # Use non-interactive backend if not interactive and no display available
    if not interactive and save_path is not None:
        plt.switch_backend('Agg')

    # Infer bounds if not provided
    if bounds is None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = _infer_bounds(simulation)
    else:
        xmin, xmax = bounds.get("x", (-1.0, 1.0))
        ymin, ymax = bounds.get("y", (-1.0, 1.0))
        zmin, zmax = bounds.get("z", (-1.0, 1.0))

    # Determine unique materials from cells
    unique_materials = []
    for cell in simulation.cells:
        fill = cell.fill
        name = getattr(fill, "name", str(fill))
        unique_materials.append(name)
    unique_materials = np.unique(unique_materials)

    # assign base RGB colors per material
    cmap = plt.get_cmap("tab10")
    color_map = {}
    for i, m in enumerate(unique_materials):
        base = cmap(i % 10)
        color_map[m] = (base[0], base[1], base[2])

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Attempt analytic rendering per-cell for common primitives
    rendered_any = False
    for cid, cell in enumerate(simulation.cells):
        fill = cell.fill
        mat_name = getattr(fill, "name", str(fill))
        # choose color; if not known fall back to gray
        col_rgb = color_map.get(mat_name, (0.5, 0.5, 0.5))
        # determine alpha, with vacuum/void squeezed down
        label_low = str(mat_name).lower()
        if "vacuum" in label_low or "void" in label_low:
            use_alpha = min(alpha, 0.15)
        else:
            use_alpha = alpha
        col = (*col_rgb, use_alpha)

        # Find surfaces by type
        cyl_z = None
        cyl_x = None
        cyl_y = None
        planes_z = []
        planes_x = []
        planes_y = []
        sph = None
        for s in cell.surfaces:
            if s.type == SURFACE_CYLINDER_Z:
                cyl_z = s
            if s.type == SURFACE_CYLINDER_X:
                cyl_x = s
            if s.type == SURFACE_CYLINDER_Y:
                cyl_y = s
            if s.type == SURFACE_PLANE_Z:
                planes_z.append(s)
            if s.type == SURFACE_PLANE_X:
                planes_x.append(s)
            if s.type == SURFACE_PLANE_Y:
                planes_y.append(s)
            if s.type == SURFACE_SPHERE:
                sph = s

        try:
            # Detect rectangular box bounded by planes on all three axes
            if len(planes_x) >= 2 and len(planes_y) >= 2 and len(planes_z) >= 2:
                xs = sorted([-p.J for p in planes_x])
                ys = sorted([-p.J for p in planes_y])
                zs = sorted([-p.J for p in planes_z])
                xmin_box, xmax_box = xs[0], xs[-1]
                ymin_box, ymax_box = ys[0], ys[-1]
                zmin_box, zmax_box = zs[0], zs[-1]

                # Create faces for the box
                ngrid = 2
                Xy, Zy = np.meshgrid(np.linspace(xmin_box, xmax_box, ngrid), np.linspace(zmin_box, zmax_box, ngrid))
                # face at y = ymin_box and y = ymax_box
                ax.plot_surface(Xy, np.full_like(Xy, ymin_box), Zy, color=col[:3], alpha=col[3], linewidth=0)
                ax.plot_surface(Xy, np.full_like(Xy, ymax_box), Zy, color=col[:3], alpha=col[3], linewidth=0)

                Xx, Yx = np.meshgrid(np.linspace(xmin_box, xmax_box, ngrid), np.linspace(ymin_box, ymax_box, ngrid))
                # face at z = zmin_box and z = zmax_box
                ax.plot_surface(Xx, Yx, np.full_like(Xx, zmin_box), color=col[:3], alpha=col[3], linewidth=0)
                ax.plot_surface(Xx, Yx, np.full_like(Xx, zmax_box), color=col[:3], alpha=col[3], linewidth=0)

                Yy, Zz = np.meshgrid(np.linspace(ymin_box, ymax_box, ngrid), np.linspace(zmin_box, zmax_box, ngrid))
                # face at x = xmin_box and x = xmax_box
                ax.plot_surface(np.full_like(Yy, xmin_box), Yy, Zz, color=col[:3], alpha=col[3], linewidth=0)
                ax.plot_surface(np.full_like(Yy, xmax_box), Yy, Zz, color=col[:3], alpha=col[3], linewidth=0)

                rendered_any = True
                # continue rendering other cells as well

            if cyl_z is not None:
                cx = -0.5 * cyl_z.G
                cy = -0.5 * cyl_z.H
                r = max(0.0, (cx * cx + cy * cy - cyl_z.J) ** 0.5)
                if len(planes_z) >= 2:
                    zs = sorted([-p.J for p in planes_z])
                    z0, z1 = zs[0], zs[-1]
                else:
                    z0, z1 = zmin, zmax
                theta = np.linspace(0, 2 * np.pi, 120)
                zsurf = np.linspace(z0, z1, 60)
                Theta, Zsurf = np.meshgrid(theta, zsurf)
                Xs = cx + r * np.cos(Theta)
                Ys = cy + r * np.sin(Theta)
                Zs = Zsurf
                ax.plot_surface(Xs, Ys, Zs, color=col[:3], alpha=col[3], linewidth=0, shade=True)
                # caps
                for zz in (z0, z1):
                    xc = cx + r * np.cos(theta)
                    yc = cy + r * np.sin(theta)
                    ax.plot_trisurf(xc, yc, np.full_like(xc, zz), color=col[:3], alpha=col[3], linewidth=0)
                rendered_any = True
                continue

            if cyl_x is not None:
                cy = -0.5 * cyl_x.H
                cz = -0.5 * cyl_x.I
                r = max(0.0, (cy * cy + cz * cz - cyl_x.J) ** 0.5)
                if len(planes_x) >= 2:
                    xs = sorted([-p.J for p in planes_x])
                    x0, x1 = xs[0], xs[-1]
                else:
                    x0, x1 = xmin, xmax
                theta = np.linspace(0, 2 * np.pi, 120)
                xsurf = np.linspace(x0, x1, 60)
                Theta, Xsurf = np.meshgrid(theta, xsurf)
                Ys = cy + r * np.cos(Theta)
                Zs = cz + r * np.sin(Theta)
                Xs = Xsurf
                ax.plot_surface(Xs, Ys, Zs, color=col[:3], alpha=col[3], linewidth=0, shade=True)
                rendered_any = True
                continue

            if cyl_y is not None:
                cx = -0.5 * cyl_y.G
                cz = -0.5 * cyl_y.I
                r = max(0.0, (cx * cx + cz * cz - cyl_y.J) ** 0.5)
                if len(planes_y) >= 2:
                    ys = sorted([-p.J for p in planes_y])
                    y0, y1 = ys[0], ys[-1]
                else:
                    y0, y1 = ymin, ymax
                theta = np.linspace(0, 2 * np.pi, 120)
                ysurf = np.linspace(y0, y1, 60)
                Theta, Ysurf = np.meshgrid(theta, ysurf)
                Xs = cx + r * np.cos(Theta)
                Zs = cz + r * np.sin(Theta)
                Ys = Ysurf
                ax.plot_surface(Xs, Ys, Zs, color=col[:3], alpha=col[3], linewidth=0, shade=True)
                rendered_any = True
                continue

            if sph is not None:
                cx = -0.5 * sph.G
                cy = -0.5 * sph.H
                cz = -0.5 * sph.I
                r = max(0.0, (cx * cx + cy * cy + cz * cz - sph.J) ** 0.5)
                u = np.linspace(0, 2 * np.pi, 80)
                v = np.linspace(0, np.pi, 40)
                U, V = np.meshgrid(u, v)
                Xs = cx + r * np.cos(U) * np.sin(V)
                Ys = cy + r * np.sin(U) * np.sin(V)
                Zs = cz + r * np.cos(V)
                ax.plot_surface(Xs, Ys, Zs, color=col[:3], alpha=col[3], linewidth=0, shade=True)
                rendered_any = True
                continue
        except Exception:
            pass

    if not rendered_any:
        print("No analytic geometry could be rendered (unsupported shapes or empty simulation).")
        return

    # Legend and axes
    _draw_legend(ax, color_map)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("Geometry preview")
    try:
        _set_axes_equal(ax)
    except Exception:
        pass
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        if interactive:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot interactively: {e}")
                print("To save the plot to file instead, use: visualize_simulation(..., save_path='output.png')")
                plt.close(fig)
        else:
            plt.close(fig)
    return