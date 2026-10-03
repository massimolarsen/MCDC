"""Figures of the Al-27 problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")


def case(name):
    """Energy edges [eV], RMC and SMC flux [per eV], mapped back from the energy scale."""
    rmc, smc = f[f"rmc/{name}"], f[f"smc/{name}"]
    scale = rmc.attrs["energy_scale"]
    E = rmc["E_edges"][()] / scale
    phi_rmc = rmc["scalar_flux"][0] * scale
    phi_smc = common.scalar_flux(smc["psi"][()])[0] * scale
    return E, phi_rmc, phi_smc, smc.attrs["N_particle"]


def flux_figure(name, path, xlim=None):
    E, phi_rmc, phi_smc, N = case(name)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        *common.steps(E, phi_smc),
        color=common.SERIES[0],
        lw=1.0,
        label=f"SMC: $10^{{{int(np.log10(N))}}}$ particles",
    )
    ax.plot(
        *common.steps(E, phi_rmc),
        color=common.SERIES[1],
        ls=(0, (3, 1.5)),
        zorder=3,
        label="RMC: 100 P/B/I, 20 iterations",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        mask = (E[:-1] >= xlim[0]) & (E[1:] <= xlim[1])
        values = np.concatenate((phi_rmc[mask], phi_smc[mask]))
        ax.set_xlim(*xlim)
        ax.set_ylim(0.97 * values.min(), 1.03 * values.max())
    ax.set_xlabel("E (eV)")
    ax.set_ylabel(r"$\phi(E)$ [per eV]")
    ax.legend()
    fig.savefig(path)


flux_figure("low", "flux_low.png")  # [D] Fig. 3.3
flux_figure("high", "flux_high.png")  # [D] Fig. 3.4
flux_figure("high", "flux_high_zoom.png", (7.0e6, 1.0e7))  # [D] Fig. 3.5

# Convergence of the high-energy case: [D] Fig. 3.6
rmc, smc = f["rmc/high"], f["smc/high"]
phi = common.scalar_flux(rmc["psi_history"][()])[:, 0]
n = np.arange(1, len(phi) + 1)
fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.plot(
    n,
    rmc["epsilon_norm"][()],
    color=common.SERIES[0],
    marker="o",
    ms=4,
    ls=(0, (1, 1.5)),
    label=r"$L_2$ of residual error ($\tilde\epsilon$)",
)
ax.plot(
    n,
    common.linf(phi, common.scalar_flux(smc["psi"][()])[0]),
    color=common.SERIES[1],
    marker="^",
    ms=4,
    ls=(0, (4, 2)),
    label=r"$L_\infty$ to reference SMC solution",
)
ax.plot(
    n,
    common.linf(phi, f["reference/high"][()]),
    color=common.SERIES[2],
    marker="s",
    ms=4,
    ls=(0, (4, 2)),
    label=r"$L_\infty$ to deterministic reference",
)
ax.set_yscale("log")
ax.set_xlabel("Iterations")
ax.set_ylabel("Error")
ax.legend()
fig.savefig("convergence.png")
