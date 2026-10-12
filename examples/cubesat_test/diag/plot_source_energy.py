from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from matplotlib.colors import LogNorm


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIAG_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = DIAG_DIR.parent
REGIONS = ("obc", "eps", "adcs", "comms")
LINESTYLES = ("-", "--", "-.", ":")
REGION_COLORS = {
    "obc": "tab:red",
    "eps": "gold",
    "adcs": "tab:green",
    "comms": "tab:blue",
}


def latest_h5(directory):
    paths = sorted(directory.glob("*.h5"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files found in {directory}")
    return paths[-1]


def geant4_glob_for_mcdc(mcdc_path):
    if "debug" in mcdc_path.stem or mcdc_path.stem.endswith("_dbg"):
        return "cubesat_debug_*_geant4.h5"
    return "cubesat_[!d]*_geant4.h5"


def output_prefix(mcdc_path):
    if "debug" in mcdc_path.stem or mcdc_path.stem.endswith("_dbg"):
        return "debug_"
    return ""


SPECIES = ("neutron", "proton", "electron")


def source_tallies(file, region):
    """Return [(species or None, tally path)]: one combined source tally, or one
    per species for runs that hand several particle types to Geant4."""
    combined = f"tallies/{region}_g4_source"
    if combined in file:
        return [(None, combined)]
    return [
        (species, f"tallies/{region}_{species}_g4_source")
        for species in SPECIES
        if f"tallies/{region}_{species}_g4_source" in file
    ]


def run_species(file):
    return [species for species, _ in source_tallies(file, REGIONS[0])]


def source_energy_distribution(file, tally_path):
    mean = file[f"{tally_path}/current-in/mean"][()]
    energy_edges = file[f"{tally_path}/grid/energy"][()]
    energy_totals = mean.sum(axis=(0, 1, 3, 4, 5, 6))
    return energy_edges, energy_totals


def spatial_current(file, region):
    # summed over species: every species tally uses the same face mesh
    total = None
    for _, tally_path in source_tallies(file, region):
        mean = file[f"{tally_path}/current-in/mean"][()].sum(axis=(0, 1, 2, 3))
        total = mean if total is None else total + mean
        face_labels = [
            clean_face_label(
                label.decode("utf-8") if isinstance(label, bytes) else str(label)
            )
            for label in file[f"{tally_path}/grid/face"][()]
        ]
    return face_labels, total


def clean_face_label(label):
    return label.replace(">", "")


def read_region_name(file):
    name = file["name"][()]
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def geant4_edep_spectra(geant4_dir, pattern):
    spectra = {}
    for path in sorted(geant4_dir.glob(pattern)):
        with h5py.File(path, "r") as file:
            spectra[read_region_name(file)] = (
                file["edep_spectrum_edges_mev"][()],
                file["edep_spectrum_edep_mev"][()],
            )
    return spectra


def positive_log_edges(edges):
    plot_edges = np.asarray(edges, dtype=float).copy()
    if plot_edges[0] <= 0.0:
        ratio = plot_edges[2] / plot_edges[1]
        plot_edges[0] = plot_edges[1] / ratio
    return plot_edges


def plot_source_current(mcdc_path, output_path):
    with h5py.File(mcdc_path, "r") as file:
        species_list = run_species(file)
        # one panel per species; species energy grids differ
        fig, axes = plt.subplots(
            1, len(species_list), figsize=(8 * len(species_list), 5), squeeze=False
        )
        for ax, species in zip(axes[0], species_list):
            for region, linestyle in zip(REGIONS, LINESTYLES):
                tally_path = dict(source_tallies(file, region)).get(species)
                if tally_path is None:
                    continue
                energy_edges, energy_totals = source_energy_distribution(
                    file, tally_path
                )
                ax.stairs(
                    energy_totals,
                    energy_edges,
                    label=region,
                    linestyle=linestyle,
                    color=REGION_COLORS[region],
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Source energy [eV]")
            ax.set_ylabel("MCDC current-in [crossings/source]")
            title = "CubeSat Geant4 Handoff Source Current"
            ax.set_title(title if species is None else f"{title}: {species}s")
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_geant4_edep(geant4_dir, pattern, output_path):
    spectra = geant4_edep_spectra(geant4_dir, pattern)
    fig, ax = plt.subplots(figsize=(8, 5))
    for region, linestyle in zip(REGIONS, LINESTYLES):
        if region not in spectra:
            continue
        energy_edges, edep_mev = spectra[region]
        if len(energy_edges) == 0:
            continue
        ax.stairs(
            edep_mev,
            positive_log_edges(energy_edges),
            label=region,
            linestyle=linestyle,
            color=REGION_COLORS[region],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deposited energy [MeV]")
    ax.set_ylabel("Weighted edep [MeV]")
    ax.set_title("CubeSat Geant4 Energy Deposition Spectrum")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spatial_current(mcdc_path, output_path):
    with h5py.File(mcdc_path, "r") as file:
        face_labels, first_spatial = spatial_current(file, REGIONS[0])
        all_spatial = {REGIONS[0]: first_spatial}
        for region in REGIONS[1:]:
            _, all_spatial[region] = spatial_current(file, region)

    positive_values = np.concatenate(
        [values[values > 0.0] for values in all_spatial.values()]
    )
    if positive_values.size:
        min_current = float(positive_values.min())
        max_current = float(positive_values.max())
        if min_current == max_current:
            min_current /= 10.0
    else:
        min_current, max_current = 1.0e-12, 1.0
    fig, axes = plt.subplots(
        len(REGIONS),
        len(face_labels),
        figsize=(12, 7),
        constrained_layout=True,
    )

    image = None
    for row, region in enumerate(REGIONS):
        for col, face in enumerate(face_labels):
            ax = axes[row, col]
            image = ax.imshow(
                all_spatial[region][col].T,
                origin="lower",
                norm=LogNorm(vmin=min_current, vmax=max_current),
                aspect="equal",
            )
            if row == 0:
                ax.set_title(face)
            if col == 0:
                ax.set_ylabel(region)
                ax.yaxis.label.set_color(REGION_COLORS[region])
            ax.set_xticks(range(all_spatial[region].shape[1]))
            ax.set_yticks(range(all_spatial[region].shape[2]))

    fig.suptitle("CubeSat MCDC Spatial Current by Handoff Face (all source species)")
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.85,
        label="MCDC current-in [crossings/source]",
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcdc",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--geant4-dir",
        type=Path,
        default=EXAMPLE_DIR / "geant4_h5",
    )
    parser.add_argument("--geant4-glob", default=None)
    parser.add_argument(
        "--edep-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--spatial-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mcdc is None:
        if args.debug:
            args.mcdc = EXAMPLE_DIR / "mcdc_h5" / "cubesat_CE_G4_dbg.h5"
        else:
            args.mcdc = latest_h5(EXAMPLE_DIR / "mcdc_h5")
    if args.geant4_glob is None:
        args.geant4_glob = geant4_glob_for_mcdc(args.mcdc)
    prefix = output_prefix(args.mcdc)
    if args.output is None:
        args.output = DIAG_DIR / f"{prefix}source_current_by_energy.png"
    if args.edep_output is None:
        args.edep_output = DIAG_DIR / f"{prefix}geant4_edep_by_energy.png"
    if args.spatial_output is None:
        args.spatial_output = DIAG_DIR / f"{prefix}spatial_current_by_face.png"

    plot_source_current(args.mcdc, args.output)
    plot_geant4_edep(args.geant4_dir, args.geant4_glob, args.edep_output)
    plot_spatial_current(args.mcdc, args.spatial_output)
    print(args.output)
    print(args.edep_output)
    print(args.spatial_output)


if __name__ == "__main__":
    main()
