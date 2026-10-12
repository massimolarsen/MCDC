"""Draw the material stack through a sensitive volume, as a labeled cross-section.

For each SV, the column of materials through the SV's center is read from the
Geant4 device geometry (the innermost component at each depth, gaps included),
from the bottom of the board up through the die, SV, metal stack, and package.
Layer heights are compressed (thickness^0.35) so 0.08 um tungsten and mm-thick
packages both show; every layer is labeled with its true thickness.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from visualize_g4_devices import GEOMETRIES, MATERIAL_COLORS, SCORED_COLOR, _bounds, _component_depth

DEFAULT_SVS = ("Comms_BasebandActive", "OBC_ProcessorActive")
HEIGHT_EXPONENT = 0.35


def find_sv(devices, name):
    for region, components in devices.items():
        for component in components:
            if component["name"] == name:
                return region, components, component
    raise SystemExit(f"No component named {name!r}")


def material_column(components, x, y):
    """Layers (z_low, z_high, component or None for a gap) through (x, y), bottom up."""
    by_name = {c["name"]: c for c in components}
    above = []
    for c in components:
        low, high = _bounds(c)
        if low[0] <= x <= high[0] and low[1] <= y <= high[1]:
            above.append((low[2], high[2], c))
    edges = sorted({z for low, high, _ in above for z in (low, high)})
    layers = []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        if z1 - z0 < 1.0e-9:
            continue
        mid = 0.5 * (z0 + z1)
        inside = [c for low, high, c in above if low <= mid <= high]
        component = max(inside, key=lambda c: _component_depth(c, by_name)) if inside else None
        if layers and layers[-1][2] is component:
            layers[-1] = (layers[-1][0], z1, component)
        else:
            layers.append((z0, z1, component))
    return layers


def thickness_text(mm):
    um = mm * 1000.0
    if um < 1.0:
        return f"{um:.2g} µm"
    if um < 1000.0:
        return f"{um:.3g} µm"
    return f"{mm:.3g} mm"


def layer_label(component, device):
    if component is None:
        return "gap (vacuum)"
    if component.get("score", False):
        return "Sensitive volume (SV)"
    name = component["name"].split("_", 1)[-1]
    if name.startswith(device):
        name = name[len(device):] or name
    name = name.replace("BEOL", "Metal stack: ")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip()


def spread(positions, gap):
    """Shift sorted label positions apart by at least gap, keeping their mean."""
    pos = np.array(positions, dtype=float)
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                shift = 0.5 * (gap - (pos[i] - pos[i - 1]))
                pos[i - 1] -= shift
                pos[i] += shift
                moved = True
        if not moved:
            break
    return pos


def draw_stack(ax, sv_name, devices):
    region, components, sv = find_sv(devices, sv_name)
    center = np.asarray(sv["center_mm"], dtype=float)
    layers = material_column(components, center[0], center[1])
    device = sv_name.split("_", 1)[-1].replace("Active", "")
    heights = [((z1 - z0) * 1000.0) ** HEIGHT_EXPONENT for z0, z1, _ in layers]
    tops = np.cumsum(heights)
    bottoms = tops - heights
    total = tops[-1]

    label_y = spread([0.5 * (b + t) for b, t in zip(bottoms, tops)], gap=0.032 * total)
    for (z0, z1, comp), b, h, ly in zip(layers, bottoms, heights, label_y):
        if comp is None:
            color, hatch = "white", "//"
        else:
            color = SCORED_COLOR if comp.get("score", False) else MATERIAL_COLORS.get(comp["material"], "lightgray")
            hatch = None
        ax.add_patch(Rectangle((0.0, b), 1.0, h, facecolor=color, edgecolor="black", lw=0.6, hatch=hatch))
        material = "" if comp is None else comp["material"].replace("G4_", "")
        text = f"{layer_label(comp, device)}  [{material}, {thickness_text(z1 - z0)}]" if comp else \
            f"{layer_label(comp, device)}  [{thickness_text(z1 - z0)}]"
        ax.annotate(text, xy=(1.0, b + 0.5 * h), xytext=(1.25, ly), va="center", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
    ax.set_xlim(0.0, 4.2)
    ax.set_ylim(-0.02 * total, 1.02 * total)
    ax.axis("off")
    flip = any("Bumps" in c["name"] and c["name"].startswith(sv_name.replace("Active", "")) for c in components)
    ax.set_title(f"{sv_name.replace('Active', '')} ({'flip-chip' if flip else 'face-up'}, {region.upper()})",
                 fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sv", nargs="+", default=list(DEFAULT_SVS))
    parser.add_argument("--geometry", choices=list(GEOMETRIES), default="1um")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "diag" / "device_stacks.png")
    args = parser.parse_args()

    devices = GEOMETRIES[args.geometry].build_device_components()
    fig, axes = plt.subplots(1, len(args.sv), figsize=(6.2 * len(args.sv), 7.5), squeeze=False)
    for ax, sv in zip(axes[0], args.sv):
        draw_stack(ax, sv, devices)
    fig.suptitle("Material stack through the sensitive volume (layer heights not to scale)", fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
