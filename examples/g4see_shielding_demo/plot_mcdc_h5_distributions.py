from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_CASES = [
    BASE / "case_C_box_wall_vol_source.h5",
    BASE / "case_D_pcb_shield.h5",
    BASE / "case_E_polyethelene_shield.h5",
]

CASE_LABELS = {
    "case_A_none": "Case A",
    "case_B": "Case B",
    "case_B_al_2mm": "Case B (Al 2 mm)",
    "case_C_box_wall_vol_source": "Case C",
    "case_D_pcb_shield": "Case D",
    "case_E_polyethelene_shield": "Case E",
}


def resolve_score_name(h5_file: h5py.File, tally_name: str) -> str:
    tally_group = h5_file[f"tallies/{tally_name}"]
    for score_name in ("flux", "net-current", "density", "collision", "capture", "fission"):
        if score_name in tally_group:
            return score_name
    available = sorted(name for name in tally_group.keys() if name != "grid")
    raise ValueError(f"Could not resolve a score for tally {tally_name!r}. Available: {available}")


def resolve_tally_names(h5_file: h5py.File) -> tuple[str, str]:
    tally_names = set(h5_file["tallies"].keys())

    if "sram_energy" in tally_names and "sram_angle" in tally_names:
        return "sram_energy", "sram_angle"
    if "handoff_energy" in tally_names and "handoff_angle" in tally_names:
        return "handoff_energy", "handoff_angle"

    raise ValueError(
        "Could not find recognized G4SEE-coupling tallies. "
        f"Available tallies: {sorted(tally_names)}"
    )


def read_case(path: Path) -> dict:
    with h5py.File(path, "r") as h5_file:
        energy_tally, angle_tally = resolve_tally_names(h5_file)
        energy_score = resolve_score_name(h5_file, energy_tally)
        angle_score = resolve_score_name(h5_file, angle_tally)

        energy_edges = np.asarray(h5_file[f"tallies/{energy_tally}/grid/energy"][...], dtype=float)
        mu_edges = np.asarray(h5_file[f"tallies/{angle_tally}/grid/mu"][...], dtype=float)

        energy_mean = np.squeeze(
            np.asarray(h5_file[f"tallies/{energy_tally}/{energy_score}/mean"][...], dtype=float)
        )
        energy_sdev = np.squeeze(
            np.asarray(h5_file[f"tallies/{energy_tally}/{energy_score}/sdev"][...], dtype=float)
        )
        angle_mean = np.squeeze(
            np.asarray(h5_file[f"tallies/{angle_tally}/{angle_score}/mean"][...], dtype=float)
        )
        angle_sdev = np.squeeze(
            np.asarray(h5_file[f"tallies/{angle_tally}/{angle_score}/sdev"][...], dtype=float)
        )

    if energy_mean.ndim != 1 or angle_mean.ndim != 1:
        raise ValueError(
            f"Expected 1D energy and angle tallies for {path.name}, "
            f"got shapes {energy_mean.shape} and {angle_mean.shape}"
        )

    return {
        "path": path,
        "label": CASE_LABELS.get(path.stem, path.stem),
        "energy_edges": energy_edges,
        "mu_edges": mu_edges,
        "energy_mean": energy_mean,
        "energy_sdev": energy_sdev,
        "angle_mean": angle_mean,
        "angle_sdev": angle_sdev,
        "energy_score": energy_score,
        "angle_score": angle_score,
        "energy_tally": energy_tally,
        "angle_tally": angle_tally,
    }


def trim_energy_range(cases: list[dict]) -> list[dict]:
    occupied = np.zeros_like(cases[0]["energy_mean"], dtype=bool)
    for case in cases:
        occupied |= case["energy_mean"] > 0.0

    if not np.any(occupied):
        return cases

    first = int(np.argmax(occupied))
    last = int(len(occupied) - np.argmax(occupied[::-1]))

    trimmed = []
    for case in cases:
        trimmed.append(
            {
                **case,
                "energy_mean": case["energy_mean"][first:last],
                "energy_sdev": case["energy_sdev"][first:last],
                "energy_edges": case["energy_edges"][first : last + 1],
            }
        )
    return trimmed


def make_energy_plot(cases: list[dict], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for case in cases:
        values = np.clip(case["energy_mean"], 1.0e-30, None)
        ax.stairs(values, case["energy_edges"], label=case["label"], linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel(cases[0]["energy_score"].replace("-", " ").title())
    ax.set_title("MC/DC HDF5 Energy Distributions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_angle_plot(cases: list[dict], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for case in cases:
        mu_centers = 0.5 * (case["mu_edges"][:-1] + case["mu_edges"][1:])
        values = np.clip(case["angle_mean"], 1.0e-30, None)
        ax.plot(mu_centers, values, marker="o", linewidth=2, label=case["label"])

    ax.set_xlabel(r"$\mu = \cos(\theta)$")
    ax.set_ylabel(cases[0]["angle_score"].replace("-", " ").title())
    ax.set_yscale("log")
    ax.set_title("MC/DC HDF5 Angular Distributions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot energy and angular distributions from MC/DC G4SEE-coupling HDF5 outputs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="HDF5 files to compare. Defaults to cases C, D, and E.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE,
        help="Directory where the PNG plots will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = args.paths if args.paths else DEFAULT_CASES
    cases = [read_case(path.resolve()) for path in paths]
    cases = trim_energy_range(cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_energy_plot(cases, args.output_dir / "mcdc_h5_energy_distributions.png")
    make_angle_plot(cases, args.output_dir / "mcdc_h5_angular_distributions.png")


if __name__ == "__main__":
    main()
