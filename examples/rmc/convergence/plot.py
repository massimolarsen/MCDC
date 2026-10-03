"""
Residual-error convergence of the chapter-4 problems: [D] Figs. 4.9-4.10.

Reads ../o16/output.h5 and ../fuel_rod/output.h5 (run their input.py first):
  - epsilon_iterations.png   ||eps~||_2 per iteration
  - epsilon_histories.png    ||eps~||_2 vs cumulative histories
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
RUNS = (
    ("../o16/output.h5", "fast", r"$^{16}$O fast region", "o"),
    ("../o16/output.h5", "resonances", r"$^{16}$O resonances", "o"),
    ("../o16/output.h5", "isolated", r"$^{16}$O isolated resonance", "^"),
    ("../fuel_rod/output.h5", "thermal", "Thermal fuel rod", "s"),
    ("../fuel_rod/output.h5", "fast", "Fast fuel rod", "D"),
)

for x_axis, path in (
    ("iterations", "epsilon_iterations.png"),
    ("histories", "epsilon_histories.png"),
):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, (file, name, label, marker) in enumerate(RUNS):
        if not os.path.exists(file):
            continue
        with h5py.File(file, "r") as f:
            if f"rmc/{name}" not in f:
                continue
            group = f[f"rmc/{name}"]
            epsilon = group["epsilon_norm"][()]
            n = np.arange(1, len(epsilon) + 1)
            x = n * group.attrs["N_history"] if x_axis == "histories" else n
        ax.plot(
            x,
            epsilon,
            color=common.SERIES[i],
            marker=marker,
            ms=5,
            ls=(0, (4, 2)),
            label=label,
        )
    ax.set_yscale("log")
    if x_axis == "histories":
        ax.set_xscale("log")
        ax.set_xlabel("Particle histories")
    else:
        ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$L_2$ of residual error ($\tilde\epsilon$)")
    ax.legend()
    fig.savefig(path)
