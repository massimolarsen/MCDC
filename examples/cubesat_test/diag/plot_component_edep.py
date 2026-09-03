from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIAG_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = DIAG_DIR.parent
REGIONS = ("obc", "eps", "adcs", "comms")
REGION_COLORS = {
    "obc": "tab:red",
    "eps": "gold",
    "adcs": "tab:green",
    "comms": "tab:blue",
}


def read_region_name(file):
    name = file["name"][()]
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def read_component_scores(path):
    with h5py.File(path, "r") as file:
        region = read_region_name(file)
        if "component_names" not in file:
            return region, [], np.array([]), np.array([])

        names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in file["component_names"][()]
        ]
        edep_mev = file["component_edep_mev"][()]
        dose_gy = (
            file["component_dose_gy"][()]
            if "component_dose_gy" in file
            else np.zeros_like(edep_mev)
        )
        return region, names, edep_mev, dose_gy


def component_scores(geant4_dir, pattern):
    scores = {}
    for path in sorted(geant4_dir.glob(pattern)):
        region, names, edep_mev, dose_gy = read_component_scores(path)
        scores[region] = {
            "names": names,
            "edep_mev": np.asarray(edep_mev, dtype=float),
            "dose_gy": np.asarray(dose_gy, dtype=float),
        }
    return scores


def plot_component_edep(scores, output_path):
    labels = []
    edep_values = []
    colors = []

    for region in REGIONS:
        if region not in scores:
            continue
        names = scores[region]["names"]
        edep_mev = scores[region]["edep_mev"]
        for name, edep in zip(names, edep_mev):
            labels.append(f"{region}: {name}")
            edep_values.append(float(edep))
            colors.append(REGION_COLORS[region])

    if not labels:
        raise RuntimeError("No component_edep_mev datasets found in Geant4 outputs.")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(labels))))
    ax.barh(y, edep_values, color=colors)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Component edep [MeV]")
    ax.set_title("CubeSat Geant4 Component Energy Deposition")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_component_scores(scores):
    print("Region  Component                         Edep [MeV]      Dose [Gy]")
    print("-----------------------------------------------------------------------")
    for region in REGIONS:
        if region not in scores:
            continue
        names = scores[region]["names"]
        edep_mev = scores[region]["edep_mev"]
        dose_gy = scores[region]["dose_gy"]
        for name, edep, dose in zip(names, edep_mev, dose_gy):
            print(f"{region:<7} {name:<30} {edep:12.6e} {dose:12.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geant4-dir",
        type=Path,
        default=EXAMPLE_DIR / "geant4_h5",
    )
    parser.add_argument("--geant4-glob", default="cubesat_[!d]*_geant4.h5")
    parser.add_argument(
        "--output",
        type=Path,
        default=DIAG_DIR / "component_edep.png",
    )
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    scores = component_scores(args.geant4_dir, args.geant4_glob)
    plot_component_edep(scores, args.output)
    if args.print:
        print_component_scores(scores)
    print(args.output)


if __name__ == "__main__":
    main()
