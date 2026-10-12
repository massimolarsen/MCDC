"""Plot per-SV ionizing-deposit spectra, split by the depositing particle species.

One figure per region: a column per sensitive volume and a row per source
species. Each curve is the number of events (per 10^6 source histories of that
species) in which that species deposited the binned ionizing energy in the SV;
the black curve is the SV's total ionizing deposit. Spectra come from the SEU
diagnostic rows, which hold every event with >= diagnostic_min_Eion_mev in
some SV, so they are complete above that threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_info

DEPOSITING = {
    "electron_positron": ("e$^\\pm$", "tab:blue"),
    "proton": ("proton", "tab:orange"),
    "alpha": ("alpha", "tab:purple"),
    "ion_recoil": ("recoil / heavier ion", "tab:brown"),
    "gamma": ("gamma (local)", "tab:green"),
}
SOURCES = {2212: "proton", 2112: "neutron"}


def plot_region(run_dir, region, histories, bins, output):
    with h5py.File(Path(run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as file:
        names = run_info.text_array(file["component_names"][()])
        rows = file["seu_diagnostics/event_sv"][()]
        threshold = float(file["diagnostic_min_Eion_mev"][()])
    sources = [pdg for pdg in SOURCES if np.any(rows["primary_pdg"] == pdg)]

    fig, axes = plt.subplots(len(sources), len(names), figsize=(3.6 * len(names), 3.2 * len(sources)),
                             sharex=True, sharey="row", squeeze=False)
    for i, pdg in enumerate(sources):
        per_million = 1.0e6 / histories[SOURCES[pdg]]
        for j, name in enumerate(names):
            ax = axes[i, j]
            sel = rows[(rows["sv_id"] == j) & (rows["primary_pdg"] == pdg)]
            for species, (label, color) in DEPOSITING.items():
                energy = sel[f"Eion_{species}_mev"]
                keep = energy >= threshold
                if not np.any(keep):
                    continue
                counts, _ = np.histogram(energy[keep], bins, weights=sel["weight"][keep] * per_million)
                ax.stairs(counts, bins, color=color, label=label)
            total = sel["Edep_ionizing_mev"]
            keep = total >= threshold
            counts, _ = np.histogram(total[keep], bins, weights=sel["weight"][keep] * per_million)
            ax.stairs(counts, bins, color="black", lw=1.5, label="total ionizing")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(name.replace(f"{region.upper()}_", "").replace("Comms_", ""), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{SOURCES[pdg]} source\nevents / 10$^6$ histories / bin")
            if i == len(sources) - 1:
                ax.set_xlabel("Ionizing energy in SV [MeV]")
    axes[0, -1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{region.upper()} sensitive-volume deposit spectra by depositing species")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--regions", nargs="+", default=list(run_info.REGIONS))
    parser.add_argument("--bins", type=int, default=28, help="log bins from 1 keV to 50 MeV")
    args = parser.parse_args()

    histories = run_info.species_histories(args.run_dir)
    bins = np.geomspace(1.0e-3, 50.0, args.bins + 1)
    for region in args.regions:
        output = Path(args.run_dir, "diag", f"{region}_sv_deposit_spectra.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_region(args.run_dir, region, histories, bins, output)
        print(output)


if __name__ == "__main__":
    main()
