"""Plot a region SEU overview, or one SV's event-energy distribution."""

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


def sum_standard_error(total, sum_sq, n_events):
    if n_events < 2:
        return np.full_like(total, np.nan, dtype=float)
    variance = n_events / (n_events - 1) * (sum_sq - total * total / n_events)
    return np.sqrt(np.maximum(variance, 0.0))


def plot_summary(file, names, species, output):
    n_events = int(file["events_run"][()])
    niel = file["component_niel_mev"][()]
    ionizing = file["component_ionizing_mev"][()]
    by_species = file["component_species_ionizing_mev"][()]
    primary = file["component_primary_ionizing_mev"][()]
    secondary = file["component_secondary_ionizing_mev"][()]
    sumw = file["component_event_ionizing_sumw"][()]
    sumw2 = file["component_event_ionizing_sumw2"][()]
    counts = file["component_event_ionizing_count"][()]
    has_species_error = (
        "component_species_ionizing_sum_sq_mev2" in file and n_events > 1
    )

    y = np.arange(len(names))
    if has_species_error:
        height = max(10, 0.6 * len(names) + 7)
    else:
        height = max(7, 0.6 * len(names) + 3)
    fig = plt.figure(figsize=(14, height), constrained_layout=True)
    title = "Geant4 sampling uncertainty (MCDC source held fixed)"
    if not has_species_error:
        title += "; species errors require replay"
    fig.suptitle(title)
    if has_species_error:
        grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 0.9))
    else:
        grid = fig.add_gridspec(2, 2)
    energy = fig.add_subplot(grid[0, 0])
    tail = fig.add_subplot(grid[0, 1])
    attribution = fig.add_subplot(grid[1, 0])
    origin = fig.add_subplot(grid[1, 1])
    uncertainty = fig.add_subplot(grid[2, :]) if has_species_error else None

    energy.barh(y, ionizing, label="Ionizing")
    energy.barh(y, niel, left=ionizing, label="NIEL")
    if "component_ionizing_sum_sq_mev2" in file and n_events > 1:
        ion_se = sum_standard_error(
            ionizing, file["component_ionizing_sum_sq_mev2"][()], n_events
        )
        energy.errorbar(ionizing, y, xerr=ion_se, fmt="none", color="black",
                        capsize=3, label=r"Ionizing $\pm 1$ SE")
    energy.set_xlabel("Weighted deposited energy [MeV]")
    energy.set_title("Total deposition = ionizing + NIEL")
    energy.set_xlim(left=0)
    energy.legend()

    weight_total = sumw.sum(axis=1)
    exceedance = sumw[:, 2:].sum(axis=1)
    fraction = np.divide(exceedance, weight_total,
                         out=np.zeros_like(weight_total), where=weight_total > 0)
    tail_se = np.zeros_like(fraction)
    if n_events > 1:
        tail_se = np.divide(
            sum_standard_error(exceedance, sumw2[:, 2:].sum(axis=1), n_events),
            weight_total, out=np.zeros_like(weight_total), where=weight_total > 0,
        )
    tail.barh(y, fraction)
    if n_events > 1:
        tail.errorbar(fraction, y, xerr=tail_se, fmt="none", color="black", capsize=3)
    for row, value in enumerate(counts[:, 2:].sum(axis=1)):
        tail.annotate(f"n={int(value)}", (fraction[row] + tail_se[row], y[row]),
                      xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    tail.set_xlabel("Fraction of source weight")
    tail.set_title(r"$E_{\mathrm{ion}} \geq 1$ keV ($\pm 1$ SE)" if n_events > 1
                   else r"$E_{\mathrm{ion}} \geq 1$ keV (SE unavailable)")
    tail.set_xlim(0, 1.2 * max(fraction + tail_se) if np.any(fraction) else 1)

    left = np.zeros(len(names))
    for index, name in enumerate(species):
        values = np.divide(by_species[:, index], ionizing,
                           out=np.zeros_like(ionizing), where=ionizing > 0)
        attribution.barh(y, values, left=left, label=name)
        left += values
    attribution.set_xlabel("Fraction of ionizing deposition")
    attribution.set_title("Depositing-track species")
    attribution.set_xlim(0, 1)
    attribution.legend(fontsize=8, ncol=4, loc="upper center",
                       bbox_to_anchor=(0.5, -0.25))

    primary_fraction = np.divide(primary, ionizing,
                                 out=np.zeros_like(ionizing), where=ionizing > 0)
    secondary_fraction = np.divide(secondary, ionizing,
                                   out=np.zeros_like(ionizing), where=ionizing > 0)
    origin.barh(y, primary_fraction, label="Primary track")
    origin.barh(y, secondary_fraction, left=primary_fraction, label="Secondary tracks")
    origin.set_xlabel("Fraction of ionizing deposition")
    origin.set_title("Depositing-track ancestry")
    origin.set_xlim(0, 1)
    origin.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    for ax in (energy, tail, attribution, origin):
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

    if has_species_error:
        species_sq = file["component_species_ionizing_sum_sq_mev2"][()]
        positive = file["component_species_positive_events"][()]
        species_se = sum_standard_error(by_species, species_sq, n_events)
        relative_se = np.divide(species_se, by_species,
                                out=np.full_like(by_species, np.nan), where=by_species > 0)
        palette = plt.get_cmap("viridis_r")
        palette.set_bad("0.9")
        image = uncertainty.imshow(np.ma.masked_invalid(relative_se), aspect="auto",
                                   vmin=0, vmax=1, cmap=palette)
        for row in range(len(names)):
            for column in range(len(species)):
                count = int(positive[row, column])
                label = (f"{count}\n{relative_se[row, column]:.0%}"
                         if count else "0\n--")
                uncertainty.text(column, row, label, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=uncertainty, label="Relative SE of species energy sum")
        zero_upper = -np.expm1(np.log(0.05) / n_events)
        uncertainty.set_title(
            f"Depositing events / Geant4 relative SE; zero count: 95% occurrence bound < {zero_upper:.2g}"
        )
        uncertainty.set_xticks(np.arange(len(species)))
        uncertainty.set_xticklabels(species, rotation=30, ha="right")
        uncertainty.set_yticks(y)
        uncertainty.set_yticklabels(names)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_sv(file, names, species, sv_id, max_events, output):
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

    if len(events):
        fig, (ax, detail) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(edges[:-1], sumw[2:-1], width=np.diff(edges), align="edge", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SV ionizing energy per Geant4 event [MeV]")
    ax.set_ylabel("Sum of source weights")
    ax.set_title(f"{names[sv_id]} (zero={sumw[0]:.3g}, underflow={sumw[1]:.3g}, overflow={sumw[-1]:.3g})")

    if len(events):
        selected = events[events["sv_id"] == sv_id]
        selected = np.sort(selected, order="Edep_ionizing_mev")[-max_events:]
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
        detail.set_title("Selected events: depositing species and secondary-track contribution")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    if len(em_births):
        print("Selected-event EM secondary births:",
              {kind: int(em_births[f"{kind}_count"].sum())
               for kind in ("electron", "positron", "gamma")})
        print("Selected-event nuclear birth rows:", nuclear_births)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_h5", type=Path)
    parser.add_argument("--sv", help="Explicitly plot one sensitive volume instead of the region overview")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--max-events", type=int, default=20)
    args = parser.parse_args()

    with h5py.File(args.region_h5, "r") as file:
        names = text_array(file["component_names"][()])
        species = text_array(file["seu_species_names"][()])
        if args.sv:
            sv_id = names.index(args.sv)
            output = args.output or Path(f"{args.region_h5.stem}_{args.sv}.png")
            plot_sv(file, names, species, sv_id, args.max_events, output)
        else:
            output = args.output or Path(f"{args.region_h5.stem}_seu_summary.png")
            plot_summary(file, names, species, output)
    print(output)


if __name__ == "__main__":
    main()
