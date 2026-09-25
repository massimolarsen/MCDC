"""Plot one region's SV ionizing-energy score and selected-event diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def text_array(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_h5", type=Path)
    parser.add_argument("--sv", help="Scored sensitive-volume name; defaults to the first")
    parser.add_argument("--output", type=Path, default=Path("seu_ionizing.png"))
    parser.add_argument("--max-events", type=int, default=20)
    args = parser.parse_args()

    with h5py.File(args.region_h5) as file:
        names = text_array(file["component_names"][()])
        sv_id = names.index(args.sv) if args.sv else 0
        species = text_array(file["seu_species_names"][()])
        edges = file["component_event_ionizing_edges_mev"][()]
        sumw = file["component_event_ionizing_sumw"][sv_id]
        events = (
            file["seu_diagnostics/event_sv"][()]
            if "seu_diagnostics/event_sv" in file
            else np.empty(0)
        )
        em_births = (
            file["seu_diagnostics/em_secondary_summary"][()]
            if "seu_diagnostics/em_secondary_summary" in file
            else np.empty(0)
        )
        nuclear_births = (
            len(file["seu_diagnostics/nuclear_birth"])
            if "seu_diagnostics/nuclear_birth" in file
            else 0
        )

    fig, (ax, detail) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ax.stairs(sumw[2:-1], edges, fill=True, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SV ionizing energy per Geant4 event [MeV]")
    ax.set_ylabel("Sum of source weights")
    ax.set_title(f"{names[sv_id]} (zero={sumw[0]:.3g}, underflow={sumw[1]:.3g}, overflow={sumw[-1]:.3g})")

    if len(events):
        selected = events[events["sv_id"] == sv_id]
        selected = np.sort(selected, order="Edep_ionizing_mev")[-args.max_events:]
        labels = [f"{row['run_id']}:{row['event_id']}" for row in selected]
        y = np.arange(len(selected))
        left = np.zeros(len(selected))
        for name in species:
            field = f"Eion_{name}_mev"
            values = selected[field]
            detail.barh(y, values, left=left, label=name)
            left += values
        detail.scatter(selected["Eion_secondary_mev"], y, color="black", marker="|",
                       s=90, label="secondary-track Eion")
        detail.set_yticks(y, labels)
        detail.set_xlabel("Unweighted ionizing energy [MeV]")
        detail.set_ylabel("run:event")
        detail.legend(fontsize=8, ncol=3)
    else:
        detail.text(0.5, 0.5, "No selected-event records", ha="center", va="center",
                    transform=detail.transAxes)
        detail.set_axis_off()
    detail.set_title("Selected events: depositing species and secondary-track contribution")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)
    if len(em_births):
        print("Selected-event EM secondary births:",
              {kind: int(em_births[f"{kind}_count"].sum())
               for kind in ("electron", "positron", "gamma")})
        print("Selected-event nuclear birth rows:", nuclear_births)


if __name__ == "__main__":
    main()
