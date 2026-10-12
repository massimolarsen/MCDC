"""Two Geant4 views of the energy deposited in every sensitive volume.

sv_deposition_bars.png: ionizing and non-ionizing (NIEL) energy deposited in
each SV per source particle, for neutron and proton sources.

sv_integral_deposits.png: events per 10^6 source particles depositing at least
E (ionizing) in each SV, with the equivalent collected charge (3.6 eV per
electron-hole pair in silicon) on the top axis.
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

COLORS = {"obc": "tab:red", "eps": "tab:orange", "adcs": "tab:green", "comms": "tab:blue"}
SOURCES = (("proton", 2212), ("neutron", 2112))
FC_PER_MEV = 1.0e6 / 3.6 * 1.602176634e-19 / 1.0e-15  # collected charge in silicon


def short_name(name, region):
    return name.split("_", 1)[-1].replace("Active", "")


def standard_error(total, sum_sq, n):
    return np.sqrt(np.maximum(n / (n - 1) * (sum_sq - total * total / n), 0.0))


def plot_bars(run_dir, histories, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, (species, _) in zip(axes, SOURCES):
        labels, x0 = [], 0
        for region in run_info.REGIONS:
            with h5py.File(Path(run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as f:
                names = run_info.text_array(f["component_names"][()])
                k = run_info.text_array(f["seu_species_names"][()]).index(species)
                ion = f["component_primary_species_ionizing_mev"][()][:, k]
                edep = f["component_primary_species_edep_mev"][()][:, k]
                ion_err = standard_error(ion, f["component_primary_species_ionizing_sum_sq_mev2"][()][:, k],
                                         int(f["events_run"][()]))
            scale = 1.0 / histories[species]
            x = x0 + np.arange(len(names))
            ax.bar(x - 0.2, ion * scale, 0.4, yerr=ion_err * scale, capsize=2, color=COLORS[region],
                   label=f"{region.upper()} ionizing")
            ax.bar(x + 0.2, np.maximum(edep - ion, 0.0) * scale, 0.4, color=COLORS[region], alpha=0.45,
                   hatch="//", edgecolor="white", label=f"{region.upper()} non-ionizing")
            labels += [short_name(n, region) for n in names]
            x0 += len(names) + 1
        positions = []
        x0 = 0
        for region in run_info.REGIONS:
            with h5py.File(Path(run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as f:
                n = len(f["component_names"][()])
            positions += list(x0 + np.arange(n))
            x0 += n + 1
        ax.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{species.capitalize()} source")
        ax.grid(axis="y", alpha=0.3, which="both")
    axes[0].set_ylabel("Energy deposited per source particle [MeV]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, color="gray"), plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.45, hatch="//")]
    axes[1].legend(handles, [r.upper() for r in COLORS] + ["ionizing", "non-ionizing (NIEL)"], fontsize=8, ncol=2)
    fig.suptitle("Geant4 energy deposition in each 1 µm sensitive volume")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_integral(run_dir, histories, output):
    fig, axes = plt.subplots(1, len(run_info.REGIONS), figsize=(16, 4.4), sharey=True)
    energies = np.geomspace(1.0e-3, 20.0, 200)
    for ax, region in zip(axes, run_info.REGIONS):
        with h5py.File(Path(run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as f:
            names = run_info.text_array(f["component_names"][()])
            rows = f["seu_diagnostics/event_sv"][()]
        colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(names)))
        for j, name in enumerate(names):
            for (species, pdg), style in zip(SOURCES, ("-", "--")):
                sel = rows[(rows["sv_id"] == j) & (rows["primary_pdg"] == pdg)]
                order = np.argsort(sel["Edep_ionizing_mev"])
                e = sel["Edep_ionizing_mev"][order]
                w = sel["weight"][order] * 1.0e6 / histories[species]
                above = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
                rate = above[np.searchsorted(e, energies)]
                ax.step(energies, np.where(rate > 0, rate, np.nan), style, where="post", color=colors[j],
                        label=short_name(name, region) if style == "-" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(region.upper())
        ax.set_xlabel("Ionizing energy deposited ≥ E [MeV]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")
        top = ax.secondary_xaxis("top", functions=(lambda m: m * FC_PER_MEV, lambda q: q / FC_PER_MEV))
        top.set_xlabel("Collected charge [fC]")
    axes[0].set_ylabel("Events per 10$^6$ source particles")
    axes[0].plot([], [], "k-", label="proton source")
    axes[0].plot([], [], "k--", label="neutron source")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Geant4 events depositing at least E in each sensitive volume", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    histories = run_info.species_histories(args.run_dir)
    out_dir = Path(args.run_dir, "diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_bars(args.run_dir, histories, out_dir / "sv_deposition_bars.png")
    plot_integral(args.run_dir, histories, out_dir / "sv_integral_deposits.png")
    print(out_dir / "sv_deposition_bars.png")
    print(out_dir / "sv_integral_deposits.png")


if __name__ == "__main__":
    main()
