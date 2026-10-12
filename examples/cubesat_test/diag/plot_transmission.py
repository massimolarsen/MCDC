"""Plot how much of the environment reaches each region box (MC/DC level).

Transmission = particles crossing into a box / the number an unshielded box
would see. The source is a cosine-weighted inward current on the source cube,
so without the spacecraft a fraction (box face area / cube area) of the
histories of a species would cross into each face, in the source spectrum.
Energies are those at the box, so slowed particles shift to lower bins.
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


def region_transmission(run_dir, file, region, particle, n_total, n_species, cube_area, spectrum):
    group = file[f"tallies/{region}_{particle}_g4_source/current-in"]
    # time bins collapsed; axes: mu, azi, energy, face, u, v
    mean = np.maximum(group["mean"][()], 0.0)[:, :, :, 0] * n_total
    var = (group["sdev"][()][:, :, :, 0] * n_total) ** 2
    edges = file[f"tallies/{region}_{particle}_g4_source/grid/energy"][()]
    areas = run_info.face_areas_cm2(run_info.box_dims_cm(run_dir, region))

    expected_energy = n_species * areas.sum() / cube_area * run_info.spectrum_bin_probability(spectrum, edges)
    by_energy = mean.sum(axis=(0, 1, 3, 4, 5))
    by_energy_err = np.sqrt(var.sum(axis=(0, 1, 3, 4, 5)))
    expected_face = n_species * areas / cube_area
    by_face = mean.sum(axis=(0, 1, 2, 4, 5))
    by_face_err = np.sqrt(var.sum(axis=(0, 1, 2, 4, 5)))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_energy = by_energy / expected_energy
        t_energy_err = by_energy_err / expected_energy
    return {
        "edges": edges,
        "t_energy": t_energy,
        "t_energy_err": t_energy_err,
        "expected_energy": expected_energy,
        "t_face": by_face / expected_face,
        "t_face_err": by_face_err / expected_face,
        "total": by_energy.sum() / expected_energy.sum(),
        "total_err": np.sqrt(var.sum()) / expected_energy.sum(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--particle", default="proton", choices=("proton", "neutron"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    n_total = run_info.n_particle(run_dir)
    n_species = run_info.species_histories(run_dir)[args.particle]
    cube_area = run_info.source_cube_area_cm2(run_dir)
    spectrum = run_info.species_spectra(run_dir)[args.particle]
    output = next(Path(run_dir, "mcdc_h5").glob("*.h5"))

    fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    with h5py.File(output, "r") as file:
        results = {r: region_transmission(run_dir, file, r, args.particle, n_total, n_species, cube_area, spectrum)
                   for r in run_info.REGIONS}

    print(f"{args.particle}: {n_species:.3g} source histories of {n_total:.3g}, source cube {cube_area:.0f} cm2")
    for region, res in results.items():
        edges = res["edges"] / 1e6
        centers = np.sqrt(edges[:-1] * edges[1:])
        # bins the source populates enough to give a meaningful ratio
        ok = res["expected_energy"] > 5.0
        ax_e.errorbar(centers[ok], res["t_energy"][ok], yerr=res["t_energy_err"][ok], marker="o", ms=4,
                      capsize=2, color=COLORS[region], label=f"{region.upper()} (total {res['total']:.3f})")
        print(f"  {region:5s} total transmission {res['total']:.4f} +/- {res['total_err']:.4f}; by face "
              + ", ".join(f"{f} {t:.3f}" for f, t in zip(run_info.FACES, res["t_face"])))
    ax_e.set_xscale("log")
    ax_e.set_xlabel(f"{args.particle.capitalize()} energy at the box [MeV]")
    ax_e.set_ylabel("Transmission (crossings / unshielded)")
    ax_e.set_title(f"{args.particle.capitalize()} transmission into each region box")
    ax_e.grid(alpha=0.3, which="both")
    ax_e.legend()

    groups = ("sides", "z- (down)", "z+ (up)")
    width = 0.8 / len(results)
    for i, (region, res) in enumerate(results.items()):
        sides = res["t_face"][:4].mean()
        sides_err = np.sqrt((res["t_face_err"][:4] ** 2).sum()) / 4
        values = [sides, res["t_face"][4], res["t_face"][5]]
        errors = [sides_err, res["t_face_err"][4], res["t_face_err"][5]]
        x = np.arange(len(groups)) + (i - (len(results) - 1) / 2) * width
        ax_f.bar(x, values, width, yerr=errors, capsize=2, color=COLORS[region], label=region.upper())
    ax_f.set_xticks(np.arange(len(groups)), ["side faces\n(mean of x±, y±)", "z− face\n(bottom)", "z+ face\n(top)"])
    ax_f.set_ylabel("Transmission (crossings / unshielded)")
    ax_f.set_title("Transmission by box face")
    ax_f.grid(axis="y", alpha=0.3)
    ax_f.set_ylim(0.0, 1.3 * ax_f.get_ylim()[1])
    ax_f.legend(loc="upper center", ncol=len(results), fontsize=9)

    fig.tight_layout()
    out = args.output or Path(run_dir, "diag", f"{args.particle}_transmission.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
