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

                # axial bounds
                zs = sorted([-p.J for p in planes_z])
                z0, z1 = zs[0], zs[-1]

                # parametric angle and axial grids for side surface
                angle = np.linspace(0, 2 * np.pi, 50)
                axial = np.linspace(z0, z1, 20)
                angle_mesh, axial_mesh = np.meshgrid(angle, axial)
                Xs = cx + r * np.cos(angle_mesh)
                Ys = cy + r * np.sin(angle_mesh)
                Zs = axial_mesh
                ax.plot_surface(Xs, Ys, Zs, color=col[:3], alpha=col[3], linewidth=0, shade=True)

                # end caps
                rad = np.linspace(0, r, 20)
                angle_r, rad_mesh = np.meshgrid(angle, rad)
                Xc = cx + rad_mesh * np.cos(angle_r)
                Yc = cy + rad_mesh * np.sin(angle_r)
                for zz in (z0, z1):
                    ax.plot_surface(Xc, Yc, np.full_like(Xc, zz), color=col[:3], alpha=col[3], linewidth=0)

                rendered_any = True
                continue

            if cyl_x is not None:
                cy = -0.5 * cyl_x.H
                cz = -0.5 * cyl_x.I
                r = max(0.0, (cy * cy + cz * cz - cyl_x.J) ** 0.5)

                xs = sorted([-p.J for p in planes_x])
                x0, x1 = xs[0], xs[-1]

                angle = np.linspace(0, 2 * np.pi, 50)
                axial = np.linspace(x0, x1, 20)
                angle_mesh, axial_mesh = np.meshgrid(angle, axial)
                Ys = cy + r * np.cos(angle_mesh)
                Zs = cz + r * np.sin(angle_mesh)
                Xs = axial_mesh
                ax.plot_surface(Xs, Ys, Zs, color=col[:3], alpha=col[3], linewidth=0, shade=True)

                rad = np.linspace(0, r, 20)
                angle_r, rad_mesh = np.meshgrid(angle, rad)
                Yc = cy + rad_mesh * np.cos(angle_r)
                Zc = cz + rad_mesh * np.sin(angle_r)
                for xx in (x0, x1):
                    ax.plot_surface(np.full_like(Yc, xx), Yc, Zc, color=col[:3], alpha=col[3], linewidth=0)

                rendered_any = True
                continue

            if cyl_y is not None:
                cx = -0.5 * cyl_y.G
                cz = -0.5 * cyl_y.I
                r = max(0.0, (cx * cx + cz * cz - cyl_y.J) ** 0.5)

                ys = sorted([-p.J for p in planes_y])
                y0, y1 = ys[0], ys[-1]

                angle = np.linspace(0, 2 * np.pi, 50)
                axial = np.linspace(y0, y1, 20)
                angle_mesh, axial_mesh = np.meshgrid(angle, axial)
                Xs = cx + r * np.cos(angle_mesh)
                Zs = cz + r * np.sin(angle_mesh)
                Ys = axial_mesh
                ax.plot_surface(Xs, Ys, Zs, color=col[:3], alpha=col[3], linewidth=0, shade=True)

                rad = np.linspace(0, r, 20)
                angle_r, rad_mesh = np.meshgrid(angle, rad)
                Xc = cx + rad_mesh * np.cos(angle_r)
                Zc = cz + rad_mesh * np.sin(angle_r)
                for yy in (y0, y1):
                    ax.plot_surface(Xc, np.full_like(Xc, yy), Zc, color=col[:3], alpha=col[3], linewidth=0)

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
        except Exception as e:
            # debug: report what went wrong during analytic rendering
            print(f"Warning: analytic rendering failed for cell {cid} with surfaces {[s.type for s in cell.surfaces]}: {e}")
            # continue to next cell
            continue

    if not rendered_any:
        print("No analytic geometry could be rendered (unsupported shapes or empty simulation).")
        return

    # legend and axes
    _draw_legend(ax, color_map)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("Geometry preview")

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