from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from cubesat_G4_devices import build_detector_sizes_mm, build_device_components


MATERIAL_COLORS = {
    "G4_Galactic": "white",
    "G4_Al": "lightgray",
    "G4_ALUMINUM_OXIDE": "tan",
    "G4_BAKELITE": "saddlebrown",
    "G4_Cu": "orange",
    "G4_GRAPHITE": "dimgray",
    "G4_POLYETHYLENE": "lightskyblue",
    "G4_Si": "royalblue",
    "G4_Sn": "silver",
    "LiCoO2": "crimson",
}

SCORED_COLOR = "magenta"


def _as_array(component, field):
    return np.asarray(component[field], dtype=float)


def _bounds(component):
    center = _as_array(component, "center_mm")
    half_size = 0.5 * _as_array(component, "size_mm")
    return center - half_size, center + half_size


def _region_bounds(components):
    lows, highs = zip(*(_bounds(component) for component in components))
    return np.min(lows, axis=0), np.max(highs, axis=0)


def _detector_bounds(size_mm):
    half_size = 0.5 * np.asarray(size_mm, dtype=float)
    return -half_size, half_size


def _volume_mm3(component):
    return float(np.prod(_as_array(component, "size_mm")))


def _component_depth(component, by_name):
    depth = 0
    parent = component.get("parent", "")
    while parent:
        depth += 1
        parent = by_name[parent].get("parent", "")
    return depth


def _selected_regions(devices, requested):
    if requested == "all":
        return list(devices)
    return [requested]


def _region_offsets(detector_sizes, regions, spacing_mm):
    if len(regions) == 1:
        return {regions[0]: np.zeros(3)}

    offsets = {}
    x_cursor = 0.0
    for region in regions:
        low, high = _detector_bounds(detector_sizes[region])
        width = high[0] - low[0]
        center_x = 0.5 * (low[0] + high[0])
        offsets[region] = np.array([x_cursor - center_x, 0.0, 0.0])
        x_cursor += width + spacing_mm
    return offsets


def _print_summary(devices, detector_sizes, regions):
    for region in regions:
        components = devices[region]
        scored = [component for component in components if component.get("score", False)]
        size = detector_sizes[region]
        print(
            f"{region}: detector_size_mm={size}, "
            f"{len(components)} components, {len(scored)} scored volumes"
        )
        for component in scored:
            size = component["size_mm"]
            print(
                f"  {component['name']}: size_mm={size} "
                f"volume={_volume_mm3(component):.6g} mm^3"
            )


def _add_pyvista_box(plotter, pv, component, offset, opacity, show_labels):
    center = _as_array(component, "center_mm") + offset
    size = _as_array(component, "size_mm")
    scored = bool(component.get("score", False))
    color = SCORED_COLOR if scored else MATERIAL_COLORS.get(component["material"], "lightgray")
    box = pv.Cube(
        center=tuple(center),
        x_length=float(size[0]),
        y_length=float(size[1]),
        z_length=float(size[2]),
    )
    plotter.add_mesh(
        box,
        color=color,
        opacity=0.85 if scored else opacity,
        show_edges=True,
        edge_color="black" if scored else "gray",
        line_width=2 if scored else 1,
    )
    if show_labels:
        return center, component["name"]
    return None


def _add_pyvista_envelope(plotter, pv, size_mm, offset):
    low, high = _detector_bounds(size_mm)
    low = low + offset
    high = high + offset
    center = 0.5 * (low + high)
    size = high - low
    envelope = pv.Cube(
        center=tuple(center),
        x_length=float(size[0]),
        y_length=float(size[1]),
        z_length=float(size[2]),
    )
    plotter.add_mesh(
        envelope,
        color="white",
        opacity=0.06,
        show_edges=True,
        edge_color="black",
        line_width=1,
    )


def _visualize_pyvista(args):
    try:
        import pyvista as pv
    except ImportError as exc:
        print("PyVista is not installed. Use --backend matplotlib.")
        raise SystemExit(1) from exc

    devices = build_device_components()
    detector_sizes = build_detector_sizes_mm()
    regions = _selected_regions(devices, args.region)
    offsets = _region_offsets(detector_sizes, regions, args.spacing_mm)
    plotter = pv.Plotter(off_screen=bool(args.screenshot))
    plotter.set_background("white")
    plotter.add_axes()
    plotter.show_grid()

    labels = []
    for region in regions:
        components = devices[region]
        by_name = {component["name"]: component for component in components}
        if args.envelope:
            _add_pyvista_envelope(
                plotter,
                pv,
                detector_sizes[region],
                offsets[region],
            )

        ordered = sorted(
            components,
            key=lambda component: (
                _component_depth(component, by_name),
                not component.get("score", False),
            ),
        )
        for component in ordered:
            label_mode = args.labels == "all" or (
                args.labels == "scored" and component.get("score", False)
            )
            label = _add_pyvista_box(
                plotter,
                pv,
                component,
                offsets[region],
                args.opacity,
                label_mode,
            )
            if label is not None:
                labels.append(label)

        if len(regions) > 1:
            low, high = _region_bounds(components)
            label_center = np.array([0.5 * (low[0] + high[0]), high[1], high[2]])
            labels.append((label_center + offsets[region] + np.array([0.0, 5.0, 5.0]), region))

    if labels:
        points, text = zip(*labels)
        plotter.add_point_labels(
            np.asarray(points),
            list(text),
            font_size=args.label_size,
            point_size=0,
            shape_opacity=0.15,
            always_visible=False,
        )

    plotter.add_text(
        f"CubeSat Geant4 devices: {args.region}",
        position="upper_left",
        font_size=12,
        color="black",
    )
    plotter.camera_position = "iso"

    if args.screenshot:
        output = Path(args.screenshot)
        output.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(output), auto_close=True)
        print(f"Wrote {output}")
    else:
        plotter.show()


def _box_faces(low, high):
    x0, y0, z0 = low
    x1, y1, z1 = high
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _add_matplotlib_box(ax, Poly3DCollection, component, offset, opacity, show_label):
    low, high = _bounds(component)
    low = low + offset
    high = high + offset
    scored = bool(component.get("score", False))
    color = SCORED_COLOR if scored else MATERIAL_COLORS.get(component["material"], "lightgray")
    collection = Poly3DCollection(
        _box_faces(low, high),
        facecolors=color,
        edgecolors="black" if scored else "gray",
        linewidths=1.0 if scored else 0.5,
        alpha=0.85 if scored else opacity,
    )
    ax.add_collection3d(collection)
    if show_label:
        center = 0.5 * (low + high)
        ax.text(center[0], center[1], center[2], component["name"], fontsize=7)


def _add_matplotlib_envelope(ax, Poly3DCollection, size_mm, offset):
    low, high = _detector_bounds(size_mm)
    low = low + offset
    high = high + offset
    collection = Poly3DCollection(
        _box_faces(low, high),
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.04,
    )
    ax.add_collection3d(collection)


def _set_axes_equal(ax, low, high):
    center = 0.5 * (low + high)
    radius = 0.5 * max(high - low)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")


def _visualize_matplotlib(args):
    if args.screenshot:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    devices = build_device_components()
    detector_sizes = build_detector_sizes_mm()
    regions = _selected_regions(devices, args.region)
    offsets = _region_offsets(detector_sizes, regions, args.spacing_mm)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    lows = []
    highs = []
    for region in regions:
        components = devices[region]
        by_name = {component["name"]: component for component in components}
        if args.envelope:
            _add_matplotlib_envelope(
                ax,
                Poly3DCollection,
                detector_sizes[region],
                offsets[region],
            )

        ordered = sorted(
            components,
            key=lambda component: (
                _component_depth(component, by_name),
                not component.get("score", False),
            ),
        )
        for component in ordered:
            show_label = args.labels == "all" or (
                args.labels == "scored" and component.get("score", False)
            )
            _add_matplotlib_box(
                ax,
                Poly3DCollection,
                component,
                offsets[region],
                args.opacity,
                show_label,
            )

        low, high = _detector_bounds(detector_sizes[region])
        lows.append(low + offsets[region])
        highs.append(high + offsets[region])
        if len(regions) > 1:
            label_position = np.array([0.5 * (low[0] + high[0]), high[1], high[2]])
            label_position = label_position + offsets[region] + np.array([0.0, 5.0, 5.0])
            ax.text(*label_position, region, fontsize=10, weight="bold")

    _set_axes_equal(ax, np.min(lows, axis=0), np.max(highs, axis=0))
    ax.view_init(elev=22, azim=-55)
    ax.set_title(f"CubeSat Geant4 devices: {args.region}")
    fig.tight_layout()

    if args.screenshot:
        output = Path(args.screenshot)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Wrote {output}")
    else:
        plt.show()


def visualize(args):
    if args.backend == "pyvista":
        _visualize_pyvista(args)
        return
    if args.backend == "matplotlib":
        _visualize_matplotlib(args)
        return

    try:
        import pyvista  # noqa: F401
    except ImportError:
        _visualize_matplotlib(args)
    else:
        _visualize_pyvista(args)


def parse_args():
    devices = build_device_components()
    parser = argparse.ArgumentParser(description="Visualize CubeSat Geant4 devices.")
    parser.add_argument(
        "--region",
        choices=["all", *devices.keys()],
        default="all",
        help="Device region to visualize.",
    )
    parser.add_argument(
        "--labels",
        choices=["none", "scored", "all"],
        default="scored",
        help="Which component labels to draw.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.28,
        help="Opacity for non-scored components.",
    )
    parser.add_argument(
        "--spacing-mm",
        type=float,
        default=25.0,
        help="Spacing between regions when --region all is used.",
    )
    parser.add_argument(
        "--no-envelope",
        dest="envelope",
        action="store_false",
        help="Hide the component bounding envelope.",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=10,
        help="Label font size for the PyVista backend.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "matplotlib", "pyvista"],
        default="auto",
        help="Visualization backend.",
    )
    parser.add_argument(
        "--screenshot",
        help="Write a screenshot instead of opening an interactive window.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print scored-volume summary without opening the viewer.",
    )
    parser.set_defaults(envelope=True)
    return parser.parse_args()


def main():
    args = parse_args()
    devices = build_device_components()
    detector_sizes = build_detector_sizes_mm()
    regions = _selected_regions(devices, args.region)
    if args.summary:
        _print_summary(devices, detector_sizes, regions)
        return
    visualize(args)


if __name__ == "__main__":
    main()
