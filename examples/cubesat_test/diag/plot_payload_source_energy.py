"""Render the standard source plots from retained Geant4 distribution payloads."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import numpy as np

from plot_source_energy import REGIONS, plot_source_current, plot_spatial_current


def load_payload_tallies(directory, n_particle):
    """Adapt payloads to the original plotter's layout without writing a fake output."""
    if n_particle <= 0:
        raise ValueError("MC/DC source particle count must be positive")
    tallies = {}
    for region in REGIONS:
        with h5py.File(directory / f"{region}_payload.h5", "r") as payload:
            if payload.attrs["source_mode"] != "distribution":
                raise ValueError(f"{region}: expected distribution payload")
            energy = payload["energy_edges_mev"][()] * 1e6
            shape = (
                len(payload["mu_edges"]) - 1,
                len(payload["azi_edges"]) - 1,
                len(energy) - 1,
                1, 6, int(payload.attrs["Nu"]), int(payload.attrs["Nv"]),
            )
            mean = payload["weights"][()].reshape(shape) / n_particle
            if not np.all(np.isfinite(mean)) or np.any(mean < 0):
                raise ValueError(f"{region}: invalid source weights")
            prefix = f"tallies/{region}_g4_source"
            tallies[f"{prefix}/current-in/mean"] = mean
            tallies[f"{prefix}/grid/energy"] = energy
            tallies[f"{prefix}/grid/face"] = np.asarray(
                ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"], dtype="S4"
            )
    return tallies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--n-particle", required=True, type=int,
                        help="Original MC/DC source count, NOT Geant4 n_events")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to the payload directory's sibling diag directory")
    args = parser.parse_args()
    tallies = load_payload_tallies(args.payload_dir, args.n_particle)
    output_dir = args.output_dir or args.payload_dir.parent / "diag"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep this HDF5 adapter in memory; the original plotting
    # functions are reused unchanged, including styles, units and axis reductions.
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        for name, values in tallies.items():
            file.create_dataset(name, data=values)
    for plot, name in (
            (plot_source_current, "source_current_by_energy.png"),
            (plot_spatial_current, "spatial_current_by_face.png"),
        ):
        path = output_dir / name
        buffer.seek(0)
        plot(buffer, path)
        print(path)


if __name__ == "__main__":
    main()
