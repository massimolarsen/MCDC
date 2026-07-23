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
CURRENT_TALLY_NAMES = [
    "OBC SV current-in",
    "EPS SV current-in",
    "ADCS SV current-in",
    "Comms SV current-in",
]

os.makedirs(PLOT_DIR, exist_ok=True)


def save_curve_plot(results, xlabel, ylabel, output_name, logx=False):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    for tally_name, energy_mid, mean, sdev in results:
        label = tally_name.replace(" SV energy deposition", "")
        (line,) = ax.plot(energy_mid, mean, label=label)
        mask_sd = sdev > 0.0
        #ax.fill_between(
        #    energy_edges[mask_sd],
        #    np.clip((step_mean - step_sdev)[mask_sd], 0.0, None),
        #    np.clip((step_mean + step_sdev)[mask_sd], 0.0, None),
        #    step="post",
        #    alpha=0.2,
        #    color=line.get_color(),
        #)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_yscale('log')
    if logx:
        ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{output_name}", dpi=200, bbox_inches="tight")
    plt.close(fig)


results = []
current_results = []
with h5py.File(H5_FILE, "r") as file:
    n_particle = int(file["settings/N_particle"][()])

    for tally_name in TALLY_NAMES:
        energy = file[f"tallies/{tally_name}/grid/energy"][:]
        energy_mid = 0.5 * (energy[:-1] + energy[1:]) / 1.0e6
        mean = (
            np.atleast_1d(file[f"tallies/{tally_name}/energy_deposition/mean"][()])
            * n_particle  / 1.0e6
        )
        sdev = (
            np.atleast_1d(file[f"tallies/{tally_name}/energy_deposition/sdev"][()])
            * n_particle  / 1.0e6
        )

        results.append((tally_name, energy_mid, mean, sdev))

    for tally_name in CURRENT_TALLY_NAMES:
        mean = np.atleast_1d(file[f"tallies/{tally_name}/current-in/mean"][()])

        current_results.append((tally_name, mean))

print("\nSensitive-volume totals")
print("-----------------------")
print(f"{'Volume':<10} {'Total edep [MeV]':>20} {'Total current-in':>20}")

for edep_result, current_result in zip(results, current_results):
    tally_name, _, mean, _ = edep_result
    _, current = current_result
    volume = tally_name.replace(" SV energy deposition", "")
    print(
        f"{volume:<10} {np.sum(mean):20.6e} "
        f"{np.sum(current) * n_particle:20.6e}"
    )

save_curve_plot(
    results,
    "Energy [MeV]",
    "total energy deposition [MeV]",
    "sensitive_volume_energy_deposition_vs_energy.png",
    logx=True,
)

print()
