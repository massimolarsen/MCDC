"""Figures of corrections.py: correction passes with the two r_s samplers."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
reference = h5py.File("output.h5", "r")
f = h5py.File("output_corrections.h5", "r")
smc = reference["smc/fast"]
phi_smc = common.scalar_flux(smc["psi"][()])[0]
sd_smc = np.sqrt(np.sum(smc["psi_sdev"][()][0] ** 2, axis=-1))
E = f["integrated/E_edges"][()]
E_MeV = E * 1e-6
dmu = np.diff(common.MU_EDGES)
labels = {"integrated": "integrated $E_{in}$", "pointwise": "pointwise $E_{in}$"}


def corrected(group, n):
    """Fixed point plus the mean of the first n correction passes."""
    fixed = group["psi_fixed_point"][()]
    return fixed + np.mean(group["corrections"][:n], axis=0)


# Relative error per bin vs SMC: fixed point and corrected
fig, ax = plt.subplots(figsize=(8.0, 5.0))
x, band = common.steps(E_MeV, sd_smc / phi_smc)
ax.fill_between(x, -band, band, color=common.GRID, lw=0, label=r"SMC $\pm1\sigma$")
fixed = common.scalar_flux(f["integrated/psi_fixed_point"][()])[0]
ax.plot(*common.steps(E_MeV, fixed / phi_smc - 1), color=common.MUTED, label="RMC fixed point (no corrections)")
for i, name in enumerate(("pointwise", "integrated")):
    group = f[name]
    phi = common.scalar_flux(group["psi"][()])[0]
    ax.plot(
        *common.steps(E_MeV, phi / phi_smc - 1),
        color=common.SERIES[i],
        zorder=3 + i,
        label=f"+ {len(group['corrections'])} correction passes, {labels[name]}",
    )
ax.axhline(0.0, color=common.INK_2, lw=0.8)
ax.set_xscale("log")
ax.set_ylim(-0.35, 0.35)
ax.set_xlabel("E (MeV)")
ax.set_ylabel("RMC / SMC $-$ 1")
ax.legend(loc="lower right")
fig.savefig("relative_error_corrections.png")

# Flux, as flux.png, with the corrected solution
fig, ax = plt.subplots(figsize=(7.5, 5.0))
phi = common.scalar_flux(f["integrated/psi"][()])[0]
ax.plot(*common.steps(E_MeV, fixed * 1e6), color=common.MUTED, label="RMC fixed point")
ax.plot(*common.steps(E_MeV, phi * 1e6), color=common.SERIES[0], zorder=3, label="RMC + corrections (integrated $E_{in}$)")
ax.plot(*common.steps(E_MeV, phi_smc * 1e6), color=common.SERIES[1], ls=(0, (3, 1.5)), label=f"SMC: $10^{{{int(np.log10(smc.attrs['N_particle']))}}}$ particles")
ax.set_xscale("log")
ax.set_xlabel("E (MeV)")
ax.set_ylabel(r"$\phi(E)$ [per MeV]")
ax.legend()
fig.savefig("flux_corrections.png")

# Error vs number of averaged correction passes
fig, ax = plt.subplots(figsize=(7.5, 5.0))
for i, name in enumerate(("pointwise", "integrated")):
    group = f[name]
    N = len(group["corrections"])
    rms = [
        np.sqrt(np.mean((common.scalar_flux(corrected(group, n))[0] / phi_smc - 1) ** 2))
        for n in range(1, N + 1)
    ]
    ax.plot(np.arange(1, N + 1), rms, color=common.SERIES[i], marker="o", ms=4, label=labels[name])
ax.axhline(np.sqrt(np.mean((fixed / phi_smc - 1) ** 2)), color=common.MUTED, ls=(0, (4, 2)), label="fixed point")
ax.axhline(np.sqrt(np.mean((sd_smc / phi_smc) ** 2)), color=common.INK_2, ls=(0, (1, 1.5)), label=r"SMC $1\sigma$ (rms)")
ax.set_yscale("log")
ax.set_xlabel("Correction passes averaged")
ax.set_ylabel("rms relative error vs SMC")
ax.legend()
fig.savefig("convergence_corrections.png")
