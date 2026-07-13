from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent.parent
REGIONS = ("obc", "eps", "adcs", "comms")


def latest_h5(directory):
    paths = sorted(directory.glob("*.h5"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files found in {directory}")
    return paths[-1]


def geant4_glob_for_mcdc(mcdc_path):
    if "debug" in mcdc_path.stem or mcdc_path.stem.endswith("_dbg"):
        return "cubesat_debug_*_geant4.h5"
    return "cubesat_[!d]*_geant4.h5"


def read_scalar(file, name):
    value = file[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def read_g4_outputs(g4_dir, pattern):
    outputs = {}
    for path in sorted(g4_dir.glob(pattern)):
        with h5py.File(path, "r") as file:
            name = str(read_scalar(file, "name"))
            outputs[name] = {
                "path": path,
                "source_total_weight": float(read_scalar(file, "source_total_weight")),
                "events_run": int(read_scalar(file, "events_run")),
                "total_edep_mev": float(read_scalar(file, "total_edep_mev")),
                "dose_gy": float(read_scalar(file, "dose_gy")),
                "edep_spectrum_counts": file["edep_spectrum_counts"][()],
                "edep_spectrum_underflow": int(read_scalar(file, "edep_spectrum_underflow")),
                "edep_spectrum_overflow": int(read_scalar(file, "edep_spectrum_overflow")),
            }
    return outputs


def tally_summary(file, region):
    tally_path = f"tallies/{region}_g4_source"
    current_in = file[f"{tally_path}/current-in/mean"][()]
    if f"{tally_path}/current-out/mean" in file:
        current_out = file[f"{tally_path}/current-out/mean"][()]
    else:
        current_out = np.zeros_like(current_in)
    face_labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in file[f"{tally_path}/grid/face"][()]
    ]

    current_in_total = float(current_in.sum())
    current_out_total = float(current_out.sum())
    current_in_faces = current_in.sum(axis=(0, 1, 2, 3, 5, 6))
    current_out_faces = current_out.sum(axis=(0, 1, 2, 3, 5, 6))
    dominant_face = face_labels[int(np.argmax(current_in_faces))]

    return {
        "current_in_total": current_in_total,
        "current_out_total": current_out_total,
        "nonzero_bins": int(np.count_nonzero(current_in)),
        "face_labels": face_labels,
        "current_in_faces": current_in_faces,
        "current_out_faces": current_out_faces,
        "dominant_face": dominant_face,
    }


def print_table(mcdc_file, g4_outputs):
    n_particle = int(read_scalar(mcdc_file, "settings/N_particle"))
    header = (
        "region  mcdc_current  g4_weight  rel_diff  nonzero  face  "
        "events  edep_MeV  dose_Gy  spec_bins  uf/of"
    )
    print(header)
    print("-" * len(header))

    summaries = {}
    for region in REGIONS:
        summary = tally_summary(mcdc_file, region)
        summaries[region] = summary
        g4 = g4_outputs.get(region)
        if g4 is None:
            print(
                f"{region:<6} {summary['current_in_total']:>12.6g}  {'missing':>9}  "
                f"{'--':>8}  {summary['nonzero_bins']:>7}  "
                f"{summary['dominant_face']:<5}  {'--':>6}  {'--':>8}  "
                f"{'--':>7}  {'--':>9}  {'--':>5}"
            )
            continue

        expected_weight = summary["current_in_total"] * n_particle
        rel_diff = 0.0
        if expected_weight != 0.0:
            rel_diff = (g4["source_total_weight"] - expected_weight) / expected_weight
        spectrum_bins = int(np.count_nonzero(g4["edep_spectrum_counts"]))
        print(
            f"{region:<6} {summary['current_in_total']:>12.6g}  "
            f"{g4['source_total_weight']:>9.6g}  {rel_diff:>8.2e}  "
            f"{summary['nonzero_bins']:>7}  {summary['dominant_face']:<5}  "
            f"{g4['events_run']:>6}  {g4['total_edep_mev']:>8.4g}  "
            f"{g4['dose_gy']:>7.3g}  {spectrum_bins:>9}  "
            f"{g4['edep_spectrum_underflow']}/{g4['edep_spectrum_overflow']}"
        )
    return summaries


def print_face_totals(summaries):
    print("\nMCDC current-in by face")
    for region, summary in summaries.items():
        parts = [
            f"{face}={value:.6g}"
            for face, value in zip(summary["face_labels"], summary["current_in_faces"])
        ]
        print(f"{region:<6} " + "  ".join(parts))

    print("\nMCDC current-out by face")
    for region, summary in summaries.items():
        parts = [
            f"{face}={value:.6g}"
            for face, value in zip(summary["face_labels"], summary["current_out_faces"])
        ]
        print(f"{region:<6} " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcdc",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--geant4-dir",
        type=Path,
        default=EXAMPLE_DIR / "geant4_h5",
    )
    parser.add_argument("--geant4-glob", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mcdc is None:
        if args.debug:
            args.mcdc = EXAMPLE_DIR / "mcdc_h5" / "cubesat_CE_G4_dbg.h5"
        else:
            args.mcdc = latest_h5(EXAMPLE_DIR / "mcdc_h5")
    if args.geant4_glob is None:
        args.geant4_glob = geant4_glob_for_mcdc(args.mcdc)

    g4_outputs = read_g4_outputs(args.geant4_dir, args.geant4_glob)
    with h5py.File(args.mcdc, "r") as mcdc_file:
        summaries = print_table(mcdc_file, g4_outputs)
        print_face_totals(summaries)


if __name__ == "__main__":
    main()
