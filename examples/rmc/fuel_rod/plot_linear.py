"""
Constant vs all-linear trial space on the 1D fuel rod (input.py's output.h5 and
linear.py's output_linear.h5), per case (thermal, fast) present in both files:
  - flux_space_linear_<case>.png    energy-integrated flux in z ([D] Figs. 4.5, 4.7):
                                    SMC and constant per cell, linear through its
                                    nodal values
  - flux_energy_linear_<case>.png   flux spectrum per region ([D] Figs. 4.6, 4.8):
                                    linear drawn as its per-bin segments
  - convergence_linear.png          ||eps~|| per iteration and vs cumulative
                                    histories ([D] Figs. 4.9-4.10)
The linear ||eps~|| is the function norm and includes the slope coefficients.
../convergence/plot.py also overlays the linear fuel rod runs on the chapter-4 figure.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
constant = h5py.File("output.h5", "r")
linear = h5py.File("output_linear.h5", "r")
CASES = [
    name
    for name in ("thermal", "fast")
    if f"rmc/{name}" in constant and f"smc/{name}" in constant and f"rmc/{name}" in linear
]
TITLES = {"thermal": "Thermal fuel rod", "fast": "Fast fuel rod"}
REGIONS = (((0, 1), "H2O", r"H$_2$O"), ((1, 2), "UO2", r"UO$_2$"), ((2, 3), "H2O", r"H$_2$O"))


def power_label(value):
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10**exponent
    return f"$10^{{{exponent}}}$" if np.isclose(mantissa, 1.0) else f"{value:g}"


def case(name):
    """Fluxes per cm per eV, with the thermal energy scaling undone."""
    rmc_c, rmc_l, smc = constant[f"rmc/{name}"], linear[f"rmc/{name}"], constant[f"smc/{name}"]
    scale = rmc_c.attrs["energy_scale"]
    E = rmc_c["E_edges"][()] / scale
    z = rmc_c["z_edges"][()]
    nodal = common.scalar_flux(rmc_l["psi"][()]) * scale  # [K + 1, G]
    slope = rmc_l["psi_slope"][()]  # nodal energy P_1 coefficients [K + 1, G, J]
    return dict(
        E=E,
        z=z,
        smc=common.scalar_flux(smc["psi"][()]) * scale,
        constant=rmc_c["scalar_flux"][()] * scale,
        linear_nodal=nodal,
        linear_cell=rmc_l["scalar_flux"][()] * scale,
        linear_slope=common.scalar_flux(0.5 * (slope[:-1] + slope[1:])) * scale,
        labels=(
            f"SMC: {power_label(smc.attrs['N_particle'])} particles",
            f"RMC constant: {power_label(rmc_c.attrs['P'])} P/B/I",
            f"RMC linear (E, z, $\\mu$): {power_label(rmc_l.attrs['P'])} P/B/I",
        ),
        iterations=len(rmc_l["epsilon_norm"]),
    )


def shade_regions(ax):
    for (low, high), region, label in REGIONS:
        ax.axvspan(low, high, color=common.REGION[region], lw=0, zorder=0)
        ax.text(
            0.5 * (low + high),
            0.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            color=common.INK_2,
        )


def space_figure(name):
    d = case(name)
    dE = np.diff(d["E"])
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    shade_regions(ax)
    ax.plot(*common.steps(d["z"], d["smc"] @ dE), color=common.MUTED, label=d["labels"][0])
    ax.plot(
        *common.steps(d["z"], d["constant"] @ dE),
        color=common.SERIES[0],
        ls=(0, (4, 2)),
        label=d["labels"][1],
    )
    ax.plot(
        d["z"],
        d["linear_nodal"] @ dE,
        color=common.SERIES[1],
        marker="o",
        ms=3,
        zorder=3,
        label=d["labels"][2],
    )
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"$\phi(z)$")
    ax.set_title(f"{TITLES[name]}, {d['iterations']} iterations")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(f"flux_space_linear_{name}.png")


def energy_figure(name):
    d = case(name)
    centers = 0.5 * (d["z"][:-1] + d["z"][1:])
    fuel = (centers > 1.0) & (centers < 2.0)
    E_MeV = d["E"] * 1e-6
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    for ax, mask, title in (
        (axes[0], ~fuel, r"H$_2$O moderator"),
        (axes[1], fuel, r"UO$_2$ fuel"),
    ):
        ax.plot(
            *common.steps(E_MeV, d["smc"][mask].mean(axis=0) * 1e6),
            color=common.MUTED,
            label=d["labels"][0],
        )
        ax.plot(
            *common.steps(E_MeV, d["constant"][mask].mean(axis=0) * 1e6),
            color=common.SERIES[0],
            ls=(0, (4, 2)),
            label=d["labels"][1],
        )
        average = d["linear_cell"][mask].mean(axis=0) * 1e6
        slope = d["linear_slope"][mask].mean(axis=0) * 1e6
        for g in range(len(E_MeV) - 1):
            ax.plot(
                E_MeV[g : g + 2],
                (average[g] - slope[g], average[g] + slope[g]),
                color=common.SERIES[1],
                zorder=3,
                label=d["labels"][2] + " (per-bin segments)" if g == 0 else None,
            )
        ax.set_title(title)
        ax.set_ylabel(r"$\phi(E)$ [per cm per MeV]")
        ax.set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("E (MeV)")
    fig.savefig(f"flux_energy_linear_{name}.png")


def convergence_figure():
    fig, axes = plt.subplots(len(CASES), 2, figsize=(11.0, 3.8 * len(CASES)), squeeze=False)
    for row, name in enumerate(CASES):
        for i, (basis, f, ls, filled) in enumerate(
            (("constant", constant, (0, (4, 2)), False), ("linear (E, z, $\\mu$)", linear, "-", True))
        ):
            group = f[f"rmc/{name}"]
            epsilon = group["epsilon_norm"][()]
            n = np.arange(1, len(epsilon) + 1)
            style = dict(
                color=common.SERIES[i],
                marker="o",
                ms=5,
                mfc=common.SERIES[i] if filled else "none",
                ls=ls,
                label=f"{basis}, {power_label(group.attrs['P'])} P/B/I",
            )
            axes[row, 0].plot(n, epsilon, **style)
            axes[row, 1].plot(n * group.attrs["N_history"], epsilon, **style)
        for ax in axes[row]:
            ax.set_yscale("log")
            ax.set_ylabel(r"$L_2$ of residual error ($\tilde\epsilon$)")
            ax.set_title(TITLES[name])
            ax.legend(fontsize=8)
        axes[row, 0].set_xlabel("Iterations")
        axes[row, 1].set_xscale("log")
        axes[row, 1].set_xlabel("Particle histories")
    fig.tight_layout()
    fig.savefig("convergence_linear.png")


for name in CASES:
    space_figure(name)
    energy_figure(name)
if CASES:
    convergence_figure()
else:
    print("No case found in both output.h5 (rmc and smc) and output_linear.h5")
