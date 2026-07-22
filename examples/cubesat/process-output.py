import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
import h5py

matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5_FILE = "cubesat_CE.h5"
PLOT_DIR = "cubesat_plots"

TALLY_NAMES = [
    "OBC SV energy deposition",
    "EPS SV energy deposition",
    "ADCS SV energy deposition",
    "Comms SV energy deposition",
]

os.makedirs(PLOT_DIR, exist_ok=True)


def save_curve_plot(results, xlabel, ylabel, output_name, logx=False):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    for tally_name, energy_mid, mean, sdev in results:
        label = tally_name.replace(" SV energy deposition", "")
        (line,) = ax.plot(energy_mid, mean, label=label)
        mask_sd = sdev > 0.0
        ax.fill_between(
            energy_mid[mask_sd],
            np.clip((mean - sdev)[mask_sd], 0.0, None),
            np.clip((mean + sdev)[mask_sd], 0.0, None),
            alpha=0.2,
            color=line.get_color(),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{output_name}", dpi=200, bbox_inches="tight")
    plt.close(fig)


results = []
with h5py.File(H5_FILE, "r") as file:
    for tally_name in TALLY_NAMES:
        energy = file[f"tallies/{tally_name}/grid/energy"][:]
        energy_mid = 0.5 * (energy[:-1] + energy[1:]) / 1.0e6
        mean = np.atleast_1d(file[f"tallies/{tally_name}/energy_deposition/mean"][()])
        sdev = np.atleast_1d(file[f"tallies/{tally_name}/energy_deposition/sdev"][()])

        results.append((tally_name, energy_mid, mean, sdev))

print("\nSensitive-volume energy deposition")
print("-----------------------------------")
print(f"{'Tally':<30} {'Mean [eV/source]':>18} {'Sdev [eV/source]':>18}")

for tally_name, energy_mid, mean, sdev in results:
    print(f"{tally_name:<30} {np.sum(mean):18.6e} {np.linalg.norm(sdev):18.6e}")

save_curve_plot(
    results,
    "Energy [MeV]",
    "energy deposition [eV/source]",
    "sensitive_volume_energy_deposition_vs_energy.png",
    logx=True,
)

print()
