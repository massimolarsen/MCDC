from pathlib import Path
import os

import h5py
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
CASE_A = BASE / "case_A_none.h5"
CASE_B = BASE / "case_B.h5"


def read_tally(path: Path, tally_name: str):
    with h5py.File(path, "r") as h5_file:
        mean = h5_file[f"tallies/{tally_name}/net-current/mean"][...]
        grid = h5_file[f"tallies/{tally_name}/grid"]
        energy = grid["energy"][...]
        mu = grid["mu"][...]
    return mean, energy, mu


def make_energy_plot():
    mean_a, energy_a, _ = read_tally(CASE_A, "handoff_energy")
    mean_b, energy_b, _ = read_tally(CASE_B, "handoff_energy")
    occupied = (mean_a > 0.0) | (mean_b > 0.0)
    first = int(np.argmax(occupied))
    last = int(len(occupied) - np.argmax(occupied[::-1]))
    mean_a = mean_a[first:last]
    mean_b = mean_b[first:last]
    energy_a = energy_a[first : last + 1]
    energy_b = energy_b[first : last + 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(np.clip(mean_a, 1.0e-16, None), energy_a, label="Case A", linewidth=2)
    ax.stairs(np.clip(mean_b, 1.0e-16, None), energy_b, label="Case B", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Net Current")
    ax.set_title("Case A vs Case B Handoff Energy Current")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / "case_A_B_energy_current.png", dpi=200)
    plt.close(fig)


def make_angle_plot():
    mean_a, _, mu_a = read_tally(CASE_A, "handoff_angle")
    mean_b, _, mu_b = read_tally(CASE_B, "handoff_angle")

    mu_centers_a = 0.5 * (mu_a[:-1] + mu_a[1:])
    mu_centers_b = 0.5 * (mu_b[:-1] + mu_b[1:])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu_centers_a, np.clip(mean_a, 1.0e-16, None), marker="o", label="Case A", linewidth=2)
    ax.plot(mu_centers_b, np.clip(mean_b, 1.0e-16, None), marker="s", label="Case B", linewidth=2)
    ax.set_xlabel("Mu")
    ax.set_ylabel("Net Current")
    ax.set_yscale("log")
    ax.set_title("Case A vs Case B Handoff Angular Current")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / "case_A_B_angular_current.png", dpi=200)
    plt.close(fig)


def main():
    make_energy_plot()
    make_angle_plot()


if __name__ == "__main__":
    main()
