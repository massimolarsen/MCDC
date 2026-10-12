"""Tabulate SV event rates above ionizing-energy thresholds, per source species.

For each sensitive volume and source species: the expected number of events
whose ionizing deposit in the SV reaches each threshold, per 10^6 MC/DC source
histories of that species (and per cm^2 of SV area), with the Geant4 relative
statistical error. These are per-history numbers; with --flux for a species
(omnidirectional flux in particles/cm^2/s over the source spectrum's energy
range) they are also given as events per day, using the source cube's inward
current flux * area / 4.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

import run_info

SOURCES = {2212: "proton", 2112: "neutron"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 1.0], help="MeV")
    parser.add_argument("--flux", nargs="*", default=[], metavar="SPECIES=FLUX",
                        help="e.g. proton=12.5 neutron=0.8 (particles/cm^2/s)")
    args = parser.parse_args()

    histories = run_info.species_histories(args.run_dir)
    cube_area = run_info.source_cube_area_cm2(args.run_dir)
    flux = {k: float(v) for k, v in (item.split("=") for item in args.flux)}

    rows_out = []
    for region in run_info.REGIONS:
        payload = Path(args.run_dir, "geant4_payloads", f"{region}_payload.h5")
        with h5py.File(payload, "r") as file:
            names = run_info.text_array(file["component_names"][()])
            sizes = file["component_sizes_mm"][()]
            scored = file["component_score"][()]
        area_cm2 = {n: s[0] * s[1] / 100.0 for n, s, keep in zip(names, sizes, scored) if keep}
        with h5py.File(Path(args.run_dir, "geant4_h5", f"cubesat_{region}_geant4.h5"), "r") as file:
            sv_names = run_info.text_array(file["component_names"][()])
            rows = file["seu_diagnostics/event_sv"][()]
            minimum = float(file["diagnostic_min_Eion_mev"][()])
        if min(args.thresholds) < minimum:
            raise SystemExit(f"Thresholds must be >= the diagnostic minimum {minimum} MeV")
        for j, name in enumerate(sv_names):
            for pdg, species in SOURCES.items():
                sel = rows[(rows["sv_id"] == j) & (rows["primary_pdg"] == pdg)]
                out = {"region": region, "sv": name, "area_cm2": area_cm2[name], "source": species}
                for threshold in args.thresholds:
                    hit = sel[sel["Edep_ionizing_mev"] >= threshold]
                    w = hit["weight"]
                    per_history = w.sum() / histories[species]
                    rel_err = np.sqrt((w ** 2).sum()) / w.sum() if w.sum() > 0 else np.nan
                    out[f">={threshold:g}MeV events"] = len(hit)
                    out[f">={threshold:g}MeV per 1e6 hist"] = per_history * 1e6
                    out[f">={threshold:g}MeV per 1e6 hist per cm2"] = per_history * 1e6 / area_cm2[name]
                    out[f">={threshold:g}MeV rel err"] = rel_err
                    if species in flux:
                        # source histories stand for flux * cube area / 4 particles per second
                        out[f">={threshold:g}MeV per day"] = per_history * flux[species] * cube_area / 4.0 * 86400.0
                rows_out.append(out)

    out_dir = Path(args.run_dir, "diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sv_event_rates.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    lines = []
    header = ["Region", "SV", "Area [cm²]", "Source"]
    for t in args.thresholds:
        header += [f"≥{t:g} MeV events", f"≥{t:g} MeV / 10⁶ hist", f"≥{t:g} MeV / 10⁶ hist / cm²"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows_out:
        cells = [r["region"].upper(), r["sv"].split("_", 1)[-1], f"{r['area_cm2']:.3f}", r["source"]]
        for t in args.thresholds:
            k = f">={t:g}MeV"
            if r[f"{k} events"] == 0:
                cells += ["0", "—", "—"]
                continue
            err = f" ± {100 * r[f'{k} rel err']:.0f}%"
            cells += [str(r[f"{k} events"]), f"{r[f'{k} per 1e6 hist']:.3g}{err}", f"{r[f'{k} per 1e6 hist per cm2']:.3g}"]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (out_dir / "sv_event_rates.md").write_text(table + "\n")
    print(table)
    print(f"\n{out_dir / 'sv_event_rates.csv'}\n{out_dir / 'sv_event_rates.md'}")


if __name__ == "__main__":
    main()
