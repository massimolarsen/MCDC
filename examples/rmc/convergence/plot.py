"""
Residual-error convergence of the chapter-4 problems: [D] Figs. 4.9-4.10.

Reads ../o16/output.h5 and ../fuel_rod/output.h5 (run their input.py first), and the
linear-energy-basis runs where they exist (../o16/output_linear.h5 from linear.py):
  - epsilon_iterations.png   ||eps~||_2 per iteration
  - epsilon_histories.png    ||eps~||_2 vs cumulative histories
Each problem has one color: the piecewise-constant energy basis dashed with open
markers, the linear discontinuous basis solid with filled markers.
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
# (energy basis, file suffix, line style, filled markers)
BASES = (
    ("constant", "", (0, (4, 2)), False),
    ("linear", "_linear", "-", True),
)

for x_axis, path in (
    ("iterations", "epsilon_iterations.png"),
    ("histories", "epsilon_histories.png"),
):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, (file, name, label, marker) in enumerate(RUNS):
        for basis, suffix, ls, filled in BASES:
            path_basis = file.replace(".h5", f"{suffix}.h5")
            if not os.path.exists(path_basis):
                continue
            with h5py.File(path_basis, "r") as f:
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
                mfc=common.SERIES[i] if filled else "none",
                ls=ls,
                label=f"{label} ({basis})",
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
