"""Figures of the sigma ~ E^-1/2 problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import E0

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")


def collided(group_or_array):
    """Scalar flux in the collided bins (the narrow source bin excluded)."""
    return np.asarray(group_or_array)[..., :-1]


def rmc_error(group):
    reference = collided(f[f"reference/G{group.attrs['G']}"][()])
    phi = common.scalar_flux(group["psi_history"][()])[:, 0]
    return common.linf(collided(phi), reference)


def smc_error(group):
    reference = collided(f["reference/G100"][()])
    return common.linf(collided(common.scalar_flux(group["psi"][()])[0]), reference)


# Flux: [N] Fig. 5, [D] Fig. 3.1
run = f["rmc/G100_P100"]
E = collided(run["E_edges"][()])
fig, ax = plt.subplots(figsize=(7.5, 5.0))
fine = np.logspace(0.0, np.log10(E[-1]), 400)
ax.plot(fine, fine**-0.5, color=common.INK_2, ls=(0, (4, 2)), label="Analytic")
phi = common.scalar_flux(run["psi_history"][19])[0]
ax.plot(
    *common.steps(E, collided(phi)),
    color=common.SERIES[0],
    zorder=3,
    label="RMC: 100 P/B/I, 20 iterations",
)
for N, color in ((100000, common.SERIES[1]),):
    smc = f[f"smc/G100_N{N}"]
    phi = collided(common.scalar_flux(smc["psi"][()])[0])
    ax.plot(
        *common.steps(E, phi),
        color=color,
        lw=1.0,
        label=f"SMC: $10^{int(np.log10(N))}$ particles",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E (eV)")
ax.set_ylabel(r"$\phi(E)$")
ax.legend()
fig.savefig("flux.png")

# Convergence: [N] Fig. 6, [D] Fig. 3.2
fig, _ = common.plot_sweep(f, rmc_error, "histories", smc_error)
fig.savefig("convergence_histories.png")
fig, _ = common.plot_sweep(f, rmc_error, "iterations")
fig.savefig("convergence_iterations.png")
