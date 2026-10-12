"""Plot the MC/DC neutron and proton spectra entering each electronics box.

Each region line is the current crossing into that region box in each tally
energy bin, per cm^2 of box surface, per source particle of that species. The
bins are log-uniform (the same width in ln E for both species). The thick gray
line is the same for an unshielded box (the source environment alone), so the
gap shows the spacecraft's shielding and its secondary production. Styled
after plot_source_energy.py's source-current plot.

With --flux for every species (omnidirectional flux in particles/cm^2/s over
its source spectrum's energy range), the curves are physical instead: source
histories stand for flux * cube area / 4 particles per second, so each line is
particles/cm^2/s per bin and the unshielded line is the environment's inward
current density, flux / 4, spread over the bins by the source spectrum.
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

LINESTYLES = ("-", "--", "-.", ":")
REGION_COLORS = {
    "obc": "tab:red",
    "eps": "gold",
    "adcs": "tab:green",
    "comms": "tab:blue",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flux", nargs="*", default=[], metavar="SPECIES=FLUX",
                        help="e.g. proton=168.5 neutron=0.572 (particles/cm^2/s)")
    args = parser.parse_args()
    flux = {k: float(v) for k, v in (item.split("=") for item in args.flux)}

    run_dir = args.run_dir
    n_total = run_info.n_particle(run_dir)
    histories = run_info.species_histories(run_dir)
    spectra = run_info.species_spectra(run_dir)
    cube_area = run_info.source_cube_area_cm2(run_dir)
    output = next(Path(run_dir, "mcdc_h5").glob("*.h5"))

    species_list = ("neutron", "proton")
    physical = all(species in flux for species in species_list)
    # per source particle, or per second when the environment flux is known
    scale = {s: flux[s] * cube_area / 4.0 if physical else 1.0 for s in species_list}
    fig, axes = plt.subplots(1, len(species_list), figsize=(8 * len(species_list), 5), squeeze=False)
    with h5py.File(output, "r") as file:
        for ax, species in zip(axes[0], species_list):
            # unshielded box: the source spectrum in each bin, spread over the source cube
            edges = file[f"tallies/{run_info.REGIONS[0]}_{species}_g4_source/grid/energy"][()]
            unshielded = run_info.spectrum_bin_probability(spectra[species], edges) / cube_area
            # a line through the bin centers, to read as the reference curve
            centers = np.sqrt(edges[:-1] * edges[1:])
            ax.plot(centers, np.where(unshielded > 0, unshielded * scale[species], np.nan),
                    color="0.5", lw=3, alpha=0.6, label="unshielded")
            for region, linestyle in zip(run_info.REGIONS, LINESTYLES):
                tally = file[f"tallies/{region}_{species}_g4_source"]
                mean = np.maximum(tally["current-in/mean"][()], 0.0)
                crossings = mean.sum(axis=(0, 1, 3, 4, 5, 6)) * n_total / histories[species]
                edges = tally["grid/energy"][()]
                area = run_info.face_areas_cm2(run_info.box_dims_cm(run_dir, region)).sum()
                # no baseline drops and no lines down to empty bins on the log axis
                current = np.where(crossings > 0, crossings / area * scale[species], np.nan)
                ax.stairs(current, edges, baseline=None, label=region, linestyle=linestyle,
                          color=REGION_COLORS[region])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Energy [eV]")
            ax.set_ylabel("MCDC current-in [particles/cm$^2$/s]" if physical
                          else "MCDC current-in [crossings/cm$^2$/source]")
            ax.set_title(f"CubeSat Current into Region Boxes: {species}s")
            ax.legend(frameon=False)
    fig.tight_layout()
    name = "box_spectra_flux.png" if physical else "box_spectra.png"
    out = args.output or Path(run_dir, "diag", name)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
