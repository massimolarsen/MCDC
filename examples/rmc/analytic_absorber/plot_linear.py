"""
Constant vs linear discontinuous energy trial space on the A = 1 absorber (input.py's
output.h5 and linear.py's output_linear.h5, same sweep):
  - convergence_iterations_linear.png   L-infinity error of the bin-averaged flux
                                        per iteration
  - convergence_histories_linear.png    the same vs cumulative histories, with SMC
One panel per P/B/I; color: G; the previous piecewise-constant runs dashed with open
markers, the linear runs solid with filled markers.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import GRIDS, PARTICLES

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
constant = h5py.File("output.h5", "r")
linear = h5py.File("output_linear.h5", "r")
# (label, file, line style, filled markers)
BASES = (
    ("constant (previous)", constant, (0, (4, 2)), False),
    ("linear", linear, "-", True),
)


def error(group):
    """L-infinity error of the collided bin averages (the source bin excluded)."""
    reference = constant[f"reference/G{group.attrs['G']}"][()][..., :-1]
    phi = common.scalar_flux(group["psi_history"][()])[:, 0]
    return common.linf(phi[..., :-1], reference)


for x_axis in ("iterations", "histories"):
    fig, axes = plt.subplots(
        1, len(PARTICLES), figsize=(6.0 * len(PARTICLES), 5.0), sharey=True
    )
    for ax, P in zip(axes, PARTICLES):
        if x_axis == "histories":
            smc = [g for g in constant["smc"].values() if "G" in g.attrs]
            smc = sorted(smc, key=lambda g: g.attrs["N_particle"])
            reference = constant["reference/G100"][()][..., :-1]
            ax.plot(
                [g.attrs["N_particle"] for g in smc],
                [
                    common.linf(common.scalar_flux(g["psi"][()])[0][:-1], reference)
                    for g in smc
                ],
                color=common.MUTED,
                marker="o",
                ms=4,
                label="SMC: G = 100",
            )
        for i, G in enumerate(GRIDS):
            for basis, f, ls, filled in BASES:
                group = f[f"rmc/G{G}_P{P}"]
                values = error(group)
                n = np.arange(1, len(values) + 1)
                x = n * group.attrs["N_history"] if x_axis == "histories" else n
                ax.plot(
                    x,
                    values,
                    color=common.SERIES[i],
                    ls=ls,
                    marker="o",
                    ms=3,
                    mfc=common.SERIES[i] if filled else "none",
                    label=f"G = {G}, {basis}",
                )
        ax.set_yscale("log")
        if x_axis == "histories":
            ax.set_xscale("log")
            ax.set_xlabel("Particle histories")
        else:
            ax.set_xlabel("Iterations")
        ax.set_title(f"A = 1 absorber, {P} P/B/I")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel(r"$L_\infty$ norm")
    fig.tight_layout()
    fig.savefig(f"convergence_{x_axis}_linear.png")
