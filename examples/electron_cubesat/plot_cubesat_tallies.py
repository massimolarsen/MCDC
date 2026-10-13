"""
Plot flux and energy deposition from the electron CubeSat output, after the
photon CubeSat plot script (Plot_cubesat_tallies_avg.py).

  1. body_flux / body_edep: mesh maps, shown as axis-averaged XY, XZ, YZ
     projections of the flux and the energy-deposition rate density.
  2. <sv>_sv_flux / <sv>_sv_edep: per-SV flux and ionizing dose rate.

Tallies are per source history. They are scaled to the AE9 environment with
R = Phi_e * A / 4 source electrons per second (boundary-cube area A, inward
cosine source), and divided by voxel or SV volume (track length -> flux).

Usage:
    python plot_cubesat_tallies.py [runs/<jobid>/mcdc_h5/cubesat_CE_electron.h5]
"""

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Source normalization (input_CubeSat_electron.py): AE9 omnidirectional flux
# over 0.04-10 MeV and the 15 cm boundary cube.
PHI_E = 3.06e5  # electrons / cm^2 / s
AREA = 6 * 15.0**2  # cm^2
RATE = PHI_E * AREA / 4.0  # source electrons / s

MEV_PER_EV = 1.0e-6
J_PER_MEV = 1.602176634e-13
SECONDS_PER_YEAR = 3.15576e7

# SV bounds (cm) and densities (g/cm^3), as in input_CubeSat_electron.py
SVS = {
    "obc": ((3.2, 6.8, 3.65, 6.35, 2.445, 3.045), 2.329, "Si"),
    "eps": ((3.75, 6.25, 1.75, 8.25, 3.945, 4.175), 5.05, "LiCoO2"),
    "adcs": ((2.75, 7.25, 2.75, 7.25, 5.945, 7.745), 2.329, "Si"),
    "comms": ((4.45, 5.55, 4.515, 5.485, 8.445, 8.715), 2.329, "Si"),
}


def box_volume(bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    return (x1 - x0) * (y1 - y0) * (z1 - z0)


def read_score(group, score):
    return np.asarray(group[f"{score}/mean"]), np.asarray(group[f"{score}/sdev"])


def load(path):
    with h5py.File(path, "r") as f:
        tallies = f["tallies"]
        flux = tallies["body_flux"]
        x, y, z = (np.asarray(flux[f"grid/{axis}"]) for axis in "xyz")
        tl, tl_sd = read_score(flux, "flux")
        ed, ed_sd = read_score(tallies["body_edep"], "energy_deposition")
        mesh = dict(x=x, y=y, z=z, tl=tl, tl_sd=tl_sd, ed=ed, ed_sd=ed_sd)

        svs = {}
        for name in SVS:
            tl, tl_sd = read_score(tallies[f"{name}_sv_flux"], "flux")
            ed, ed_sd = read_score(tallies[f"{name}_sv_edep"], "energy_deposition")
            svs[name] = dict(tl=float(tl), tl_sd=float(tl_sd), ed=float(ed), ed_sd=float(ed_sd))
        n_history = int(f["performance/N_history"][()]) if "performance/N_history" in f else None
    return mesh, svs, n_history


def voxel_volumes(mesh):
    dx, dy, dz = (np.diff(mesh[a]) for a in "xyz")
    return dx[:, None, None] * dy[None, :, None] * dz[None, None, :]


def plot_mesh(mesh, label):
    vol = voxel_volumes(mesh)
    # Physical densities per voxel
    flux = mesh["tl"] / vol * RATE  # e / cm^2 / s
    edep = mesh["ed"] * MEV_PER_EV / vol * RATE  # MeV / cm^3 / s
    x, y, z = mesh["x"], mesh["y"], mesh["z"]
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)

    def project(q, axis, widths):
        # Width-weighted average along the axis (voxels are not all 0.5 cm)
        shape = [1, 1, 1]
        shape[axis] = -1
        return (q * widths.reshape(shape)).sum(axis=axis) / widths.sum()

    views = [
        ("XY, z-avg", 2, dz, x, y, "x [cm]", "y [cm]"),
        ("XZ, y-avg", 1, dy, x, z, "x [cm]", "z [cm]"),
        ("YZ, x-avg", 0, dx, y, z, "y [cm]", "z [cm]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{label}: body mesh (axis-averaged projections)", fontsize=13)
    rows = [
        (flux, "Flux", "flux [e cm$^{-2}$ s$^{-1}$]", "viridis"),
        (edep, "Energy deposition", "deposition [MeV cm$^{-3}$ s$^{-1}$]", "magma"),
    ]
    for row, (q, title, unit, cmap) in enumerate(rows):
        for col, (view, axis, widths, gx, gy, xlabel, ylabel) in enumerate(views):
            p = project(q, axis, widths).T
            positive = p[p > 0]
            norm = LogNorm(positive.min(), positive.max()) if positive.size else None
            ax = axes[row, col]
            im = ax.pcolormesh(gx, gy, np.ma.masked_less_equal(p, 0), shading="flat",
                               cmap=cmap, norm=norm)
            ax.set_title(f"{title}: {view}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, label=unit)
    fig.tight_layout()
    return fig


def sv_physical(svs):
    rows = {}
    for name, (bounds, density, material) in SVS.items():
        vol = box_volume(bounds)
        mass_kg = vol * density * 1.0e-3
        s = svs[name]
        to_flux = RATE / vol
        to_dose = MEV_PER_EV * J_PER_MEV / mass_kg * 100.0 * SECONDS_PER_YEAR * RATE
        rows[name] = dict(
            material=material,
            flux=s["tl"] * to_flux, flux_sd=s["tl_sd"] * to_flux,
            dose=s["ed"] * to_dose, dose_sd=s["ed_sd"] * to_dose,
        )
    return rows


def plot_svs(rows, label):
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{label}: sensitive volumes", fontsize=13)
    bars = [
        ("flux", "Mean flux", "flux [e cm$^{-2}$ s$^{-1}$]", "steelblue"),
        ("dose", "Ionizing dose rate", "dose rate [rad yr$^{-1}$]", "firebrick"),
    ]
    for ax, (key, title, unit, color) in zip(axes, bars):
        ax.bar([f"{n.upper()}\n({rows[n]['material']})" for n in names],
               [rows[n][key] for n in names],
               yerr=[rows[n][f"{key}_sd"] for n in names], capsize=5, color=color)
        ax.set_title(title)
        ax.set_ylabel(unit)
    fig.tight_layout()
    return fig


def main():
    here = Path(__file__).resolve().parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "mcdc_h5" / "cubesat_CE_electron.h5"
    if not path.is_file():
        sys.exit(f"No output file at {path}")

    mesh, svs, n_history = load(path)
    label = "Electron CubeSat (AE9)"
    if n_history:
        label += f", {n_history:,} histories"

    rows = sv_physical(svs)
    print(f"{'SV':6s} {'flux [e/cm2/s]':>22s} {'dose [rad/yr]':>22s}")
    for name, r in rows.items():
        rel_f = r["flux_sd"] / r["flux"] if r["flux"] else float("nan")
        rel_d = r["dose_sd"] / r["dose"] if r["dose"] else float("nan")
        print(f"{name:6s} {r['flux']:12.3e} ({rel_f:5.2f}) {r['dose']:12.3e} ({rel_d:5.2f})")

    out_dir = path.parent
    for i, fig in enumerate([plot_mesh(mesh, label), plot_svs(rows, label)]):
        out = out_dir / f"cubesat_CE_electron_tallies_{i}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
