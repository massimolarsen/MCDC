"""Figures of the 1D fuel rod problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")


def power_label(value):
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10**exponent
    return f"$10^{{{exponent}}}$" if np.isclose(mantissa, 1.0) else f"{value:g}"


def case(name):
    rmc, smc = f[f"rmc/{name}"], f[f"smc/{name}"]
    scale = rmc.attrs["energy_scale"]
    E = rmc["E_edges"][()] / scale
    z = rmc["z_edges"][()]
    phi_rmc = rmc["scalar_flux"][()] * scale  # [k, g] per cm per eV
    phi_smc = common.scalar_flux(smc["psi"][()]) * scale
    labels = (
        f"RMC: {power_label(rmc.attrs['P'])} P/B/I, {len(rmc['epsilon_norm'])} iterations",
        f"SMC: {power_label(smc.attrs['N_particle'])} particles",
    )
    return E, z, phi_rmc, phi_smc, labels


def space_figure(name, path):
    """Energy-integrated flux per z cell: [D] Figs. 4.5 and 4.7."""
    E, z, phi_rmc, phi_smc, labels = case(name)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for (low, high), region in zip(((0, 1), (1, 2), (2, 3)), ("H2O", "UO2", "H2O")):
        ax.axvspan(low, high, color=common.REGION[region], lw=0, zorder=0)
    dE = np.diff(E)
    ax.plot(
        *common.steps(z, phi_rmc @ dE),
        color=common.SERIES[0],
        zorder=3,
        label=labels[0],
    )
    ax.plot(
        *common.steps(z, phi_smc @ dE),
        color=common.SERIES[1],
        ls=(0, (3, 1.5)),
        label=labels[1],
    )
    ax.text(
        0.5,
        0.02,
        r"H$_2$O",
        transform=ax.get_xaxis_transform(),
        ha="center",
        color=common.INK_2,
    )
    ax.text(
        1.5,
        0.02,
        r"UO$_2$",
        transform=ax.get_xaxis_transform(),
        ha="center",
        color=common.INK_2,
    )
    ax.text(
        2.5,
        0.02,
        r"H$_2$O",
        transform=ax.get_xaxis_transform(),
        ha="center",
        color=common.INK_2,
    )
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"$\phi(z)$")
    ax.legend(loc="upper right")
    fig.savefig(path)


def energy_figure(name, path):
    """Flux spectrum per region, averaged over its cells: [D] Figs. 4.6 and 4.8."""
    E, z, phi_rmc, phi_smc, labels = case(name)
    centers = 0.5 * (z[:-1] + z[1:])
    fuel = (centers > 1.0) & (centers < 2.0)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    for ax, mask, title in (
        (axes[0], ~fuel, r"H$_2$O moderator"),
        (axes[1], fuel, r"UO$_2$ fuel"),
    ):
        E_MeV = E * 1e-6
        ax.plot(
            *common.steps(E_MeV, phi_rmc[mask].mean(axis=0) * 1e6),
            color=common.SERIES[0],
            zorder=3,
            label=labels[0],
        )
        ax.plot(
            *common.steps(E_MeV, phi_smc[mask].mean(axis=0) * 1e6),
            color=common.SERIES[1],
            ls=(0, (3, 1.5)),
            label=labels[1],
        )
        ax.set_title(title)
        ax.set_ylabel(r"$\phi(E)$ [per cm per MeV]")
        ax.set_xscale("log")
    axes[0].legend()
    axes[1].set_xlabel("E (MeV)")
    fig.savefig(path)


for name in ("thermal", "fast"):
    if f"rmc/{name}" in f:
        space_figure(name, f"flux_space_{name}.png")
        energy_figure(name, f"flux_energy_{name}.png")
