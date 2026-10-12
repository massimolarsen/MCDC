"""Two region-level summaries of the Geant4 scoring, in on-orbit units.

region_dose.png: ionizing dose rate (rad(Si)/yr, total ionizing dose) and
non-ionizing energy per unit mass (MeV/g/yr, displacement damage dose) in each
region's sensitive volumes combined, stacked by source species.

region_edep_spectrum.png: ionizing dose rate (rad(Si)/yr) in each region's SVs
per bin of the energy one event deposits in an SV, for proton and neutron
sources; summed over bins it gives the region's total ionizing dose rate (from
events above the diagnostic threshold).

region_event_rate.png: SV events per day depositing at least a collected
charge Q (3.6 eV per electron-hole pair in silicon), per region (all SVs of
the region), for proton, neutron, and total sources.

Each source species is scaled to its environment flux: its source histories
stand for flux * source-cube area / 4 particles per second.
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

REGION_COLORS = {"obc": "tab:red", "eps": "gold", "adcs": "tab:green", "comms": "tab:blue"}
SOURCE_COLORS = {"proton": "tab:orange", "neutron": "tab:purple"}
SOURCES = (("proton", 2212), ("neutron", 2112))
SECONDS_PER_YEAR = 3.15576e7
SECONDS_PER_DAY = 86400.0
J_PER_MEV = 1.602176634e-13
FC_PER_MEV = 1.0e6 / 3.6 * 1.602176634e-19 / 1.0e-15


def standard_error(total, sum_sq, n):
    return np.sqrt(np.maximum(n / (n - 1) * (sum_sq - total * total / n), 0.0))


def source_rates(run_dir, flux):
    """Physical particles per second each source history stands for."""
    histories = run_info.species_histories(run_dir)
    cube_area = run_info.source_cube_area_cm2(run_dir)
    return {s: flux[s] * cube_area / 4.0 / histories[s] for s, _ in SOURCES}


def region_dose(run_dir, region, per_history_rate):
    """Ionizing and non-ionizing dose rates of a region's SVs, per source species."""
    with h5py.File(Path(run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as f:
        species = run_info.text_array(f["seu_species_names"][()])
        mass_kg = f["component_mass_kg"][()].sum()
        n_events = int(f["events_run"][()])
        out = {}
        for source, _ in SOURCES:
            k = species.index(source)
            ion = f["component_primary_species_ionizing_mev"][()][:, k]
            edep = f["component_primary_species_edep_mev"][()][:, k]
            ion_se = standard_error(ion, f["component_primary_species_ionizing_sum_sq_mev2"][()][:, k], n_events)
            mev_per_s = per_history_rate[source]
            tid = ion.sum() * mev_per_s * J_PER_MEV / mass_kg * 100.0 * SECONDS_PER_YEAR  # rad/yr
            tid_err = np.sqrt((ion_se ** 2).sum()) * mev_per_s * J_PER_MEV / mass_kg * 100.0 * SECONDS_PER_YEAR
            ddd = max(edep.sum() - ion.sum(), 0.0) * mev_per_s / (mass_kg * 1000.0) * SECONDS_PER_YEAR  # MeV/g/yr
            out[source] = (tid, tid_err, ddd)
    return out


def plot_dose(run_dir, per_history_rate, output):
    regions = list(run_info.REGIONS)
    doses = {r: region_dose(run_dir, r, per_history_rate) for r in regions}
    x = np.arange(len(regions))
    fig, (ax_tid, ax_ddd) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, index, ylabel, title in (
        (ax_tid, 0, "Ionizing dose rate [rad(Si)/yr]", "Total ionizing dose in the SVs"),
        (ax_ddd, 2, "Non-ionizing energy [MeV/g/yr]", "Displacement damage dose in the SVs"),
    ):
        bottom = np.zeros(len(regions))
        for source, _ in SOURCES:
            values = np.array([doses[r][source][index] for r in regions])
            errors = np.array([doses[r][source][1] for r in regions]) if index == 0 else None
            ax.bar(x, values, 0.6, bottom=bottom, color=SOURCE_COLORS[source],
                   label=f"{source} source", yerr=errors, capsize=3)
            bottom += values
        ax.set_xticks(x, [r.upper() for r in regions])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
    for r in regions:
        tid = sum(doses[r][s][0] for s, _ in SOURCES)
        ddd = sum(doses[r][s][2] for s, _ in SOURCES)
        split = ", ".join(f"{s} {doses[r][s][0] / tid:.0%}" for s, _ in SOURCES)
        print(f"  {r:5s} TID {tid:.3g} rad(Si)/yr ({split}); DDD {ddd:.3g} MeV/g/yr")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_edep_spectrum(run_dir, per_history_rate, output, n_bins=40):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    edges = None
    for region in run_info.REGIONS:
        with h5py.File(Path(run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as f:
            rows = f["seu_diagnostics/event_sv"][()]
            minimum = float(f["diagnostic_min_Eion_mev"][()])
            mass_kg = f["component_mass_kg"][()].sum()
        if edges is None:
            edges = np.geomspace(minimum, 50.0, n_bins + 1)
        to_rad_per_year = J_PER_MEV / mass_kg * 100.0 * SECONDS_PER_YEAR
        total = 0.0
        for (source, pdg), style in zip(SOURCES, ("-", ":")):
            sel = rows[(rows["primary_pdg"] == pdg) & (rows["Edep_ionizing_mev"] >= minimum)]
            energy = sel["Edep_ionizing_mev"]
            mev, _ = np.histogram(energy, edges, weights=sel["weight"] * energy)
            rate = mev * per_history_rate[source] * to_rad_per_year
            total += rate.sum()
            ax.stairs(np.where(rate > 0, rate, np.nan), edges, baseline=None, linestyle=style,
                      color=REGION_COLORS[region], lw=1.5,
                      label=region.upper() if source == "proton" else None)
        print(f"  {region:5s} summed spectrum {total:.3g} rad(Si)/yr (events >= {minimum:g} MeV)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy deposited per event [MeV]")
    ax.set_ylabel("Ionizing dose rate [rad(Si)/yr]")
    ax.set_title("Geant4 energy deposition spectrum, all SVs per region")
    ax.plot([], [], "k-", lw=1.5, label="proton source")
    ax.plot([], [], "k:", lw=1.5, label="neutron source")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_event_rate(run_dir, per_history_rate, output):
    charges_fc = np.geomspace(1.0e-3 * FC_PER_MEV, 50.0 * FC_PER_MEV, 300)
    energies = charges_fc / FC_PER_MEV
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for region in run_info.REGIONS:
        with h5py.File(Path(run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as f:
            rows = f["seu_diagnostics/event_sv"][()]
            minimum = float(f["diagnostic_min_Eion_mev"][()])
        for (source, pdg), style in zip(SOURCES, ("-", ":")):
            sel = rows[rows["primary_pdg"] == pdg]
            order = np.argsort(sel["Edep_ionizing_mev"])
            e = sel["Edep_ionizing_mev"][order]
            w = sel["weight"][order] * per_history_rate[source] * SECONDS_PER_DAY
            above = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
            rate = above[np.searchsorted(e, energies)]
            ax.step(charges_fc, np.where(rate > 0, rate, np.nan), style, where="post",
                    color=REGION_COLORS[region], lw=1.5,
                    label=region.upper() if source == "proton" else None)
    ax.axvspan(charges_fc[0], minimum * FC_PER_MEV, color="0.9", lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(minimum * FC_PER_MEV, charges_fc[-1])
    ax.set_xlabel("Charge, Q [fC]")
    ax.set_ylabel("Events per day (≥ Q)")
    ax.set_title("Event rate vs. deposited charge")
    ax.plot([], [], "k-", lw=1.5, label="proton source")
    ax.plot([], [], "k:", lw=1.5, label="neutron source")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--flux", nargs="*", default=["proton=168.5", "neutron=0.564"],
                        metavar="SPECIES=FLUX",
                        help="particles/cm^2/s over each source spectrum (default: AP9 mean "
                             "1-200 MeV and ECSS 1 eV-200 MeV at 550 km, 97 deg)")
    args = parser.parse_args()
    flux = {k: float(v) for k, v in (item.split("=") for item in args.flux)}
    per_history_rate = source_rates(args.run_dir, flux)

    out_dir = Path(args.run_dir, "diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dose(args.run_dir, per_history_rate, out_dir / "region_dose.png")
    plot_event_rate(args.run_dir, per_history_rate, out_dir / "region_event_rate.png")
    plot_edep_spectrum(args.run_dir, per_history_rate, out_dir / "region_edep_spectrum.png")
    print(out_dir / "region_dose.png")
    print(out_dir / "region_edep_spectrum.png")
    print(out_dir / "region_event_rate.png")


if __name__ == "__main__":
    main()
