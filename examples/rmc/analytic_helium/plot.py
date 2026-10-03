"""Figures of the helium-like (A = 4) problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import ALPHA, E0

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")


def collided(values):
    return np.asarray(values)[..., :-1]


def rmc_error(group):
    reference = collided(f[f"reference/G{group.attrs['G']}"][()])
    return common.linf(
        collided(common.scalar_flux(group["psi_history"][()])[:, 0]), reference
    )


def smc_error(group):
    reference = collided(f["reference/G100"][()])
    return common.linf(collided(common.scalar_flux(group["psi"][()])[0]), reference)


# Flux: [N] Fig. 10
run = f["rmc/flux_G100_P100"]
E = collided(run["E_edges"][()])
fig, ax = plt.subplots(figsize=(7.5, 5.0))
fine = np.logspace(0.0, np.log10(E[-1]), 400)
xi = 1.0 + ALPHA * np.log(ALPHA) / (1.0 - ALPHA)
ax.plot(
    fine,
    1.0 / (xi * fine),
    color=common.INK_2,
    ls=(0, (4, 2)),
    label=r"Analytic: $u > \ln(1/\alpha)$",
)
ax.plot(
    fine,
    (E0 / fine) ** (ALPHA / (1.0 - ALPHA)) / ((1.0 - ALPHA) * fine),
    color=common.MUTED,
    ls=(0, (1, 1.5)),
    label=r"Analytic: $u < \ln(1/\alpha)$",
)
phi = collided(common.scalar_flux(run["psi_history"][4])[0])
ax.plot(
    *common.steps(E, phi),
    color=common.SERIES[0],
    zorder=3,
    label="RMC: 100 P/B/I, 5 iterations",
)
smc = f["smc/G100_N100000"]
ax.plot(
    *common.steps(E, collided(common.scalar_flux(smc["psi"][()])[0])),
    color=common.SERIES[1],
    lw=1.0,
    label="SMC: $10^5$ particles",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E (eV)")
ax.set_ylabel(r"$\phi(E)$")
ax.legend()
fig.savefig("flux.png")

# Convergence: [N] Figs. 11-12
fig, _ = common.plot_sweep(f, rmc_error, "histories", smc_error)
fig.savefig("convergence_histories.png")
fig, _ = common.plot_sweep(f, rmc_error, "iterations")
fig.savefig("convergence_iterations.png")
