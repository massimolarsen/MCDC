import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
import h5py

matplotlib.use("Agg")
import matplotlib.pyplot as plt


H5_FILE = "case_E_polyethelene_shield.h5"
PLOT_DIR = "case_E_plots"


os.makedirs(PLOT_DIR, exist_ok=True)


def save_curve_plot(x, mean, sdev, xlabel, ylabel, output_name, logx=False):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    mask_sd = sdev > 0.0

    ax.plot(x, mean, "-b", label="MC")
    ax.fill_between(
        x[mask_sd],
        np.clip((mean - sdev)[mask_sd], 0.0, None),
        np.clip((mean + sdev)[mask_sd], 0.0, None),
        alpha=0.2,
        color="b",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{output_name}", dpi=200, bbox_inches="tight")
    plt.show()


# Load result and grid
with h5py.File(H5_FILE, "r") as f:
    energy = f["tallies/sram_energy/grid/energy"][:]
    energy_mid = 0.5 * (energy[:-1] + energy[1:])
    mu = f["tallies/sram_angle/grid/mu"][:]
    mu_mid = 0.5 * (mu[:-1] + mu[1:])

    cell_energy = f["tallies/sram_energy/flux/mean"][:]
    cell_energy_sd = f["tallies/sram_energy/flux/sdev"][:]
    cell_angle = f["tallies/sram_angle/flux/mean"][:]
    cell_angle_sd = f["tallies/sram_angle/flux/sdev"][:]

    face_energy = f["tallies/face_energy/flux/mean"][:]
    face_energy_sd = f["tallies/face_energy/flux/sdev"][:]
    face_angle = f["tallies/face_angle/net-current/mean"][:]
    face_angle_sd = f["tallies/face_angle/net-current/sdev"][:]


save_curve_plot(
    energy_mid,
    cell_energy,
    cell_energy_sd,
    "E (MeV)",
    r"$\phi(E)$",
    "energy_flux.png",
    logx=True,
)

save_curve_plot(
    mu_mid,
    cell_angle,
    cell_angle_sd,
    r"$\mu$",
    r"$\phi(\mu)$",
    "angular_flux.png",
)

save_curve_plot(
    energy_mid,
    face_energy,
    face_energy_sd,
    "E (MeV)",
    r"$\phi(E)$",
    "handoff_face_energy_flux.png",
    logx=True,
)

save_curve_plot(
    mu_mid,
    face_angle,
    face_angle_sd,
    r"$\mu$",
    r"$\phi(\mu)$",
    "handoff_face_angular_flux.png",
)
