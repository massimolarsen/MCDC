"""
Constant vs linear discontinuous energy trial space on O-16 (input.py's output.h5 and
linear.py's output_linear.h5, same schedule and histories):
  - convergence_linear.png   per window: ||eps~|| per iteration (left) and the rms
                             relative difference of the bin-averaged flux to SMC per
                             iteration (right; the dotted line is the SMC noise level)
  - flux_linear.png          the flux of every window: SMC, constant, and linear
                             (drawn as its linear segments), ratios to SMC below
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
CASES = (
    ("fast", "1-10 MeV, G = 100"),
    ("isolated", "2-3 MeV, G = 80"),
    ("resonances", "1.55-2.05 MeV, G = 80"),
)
# (label, file, marker, line style, filled marker)
BASES = (
    ("constant (previous)", constant, "o", (0, (4, 2)), False),
    ("linear", linear, "o", "-", True),
)


def rms_relative(phi, reference):
    return np.sqrt(np.mean((phi / reference - 1.0) ** 2))


fig, axes = plt.subplots(len(CASES), 2, figsize=(11.0, 3.6 * len(CASES)))
for row, (name, title) in enumerate(CASES):
    smc = constant[f"smc/{name}"]
    phi_smc = common.scalar_flux(smc["psi"][()])[0]
    sdev_smc = np.sqrt(
        np.einsum("kgj,j->kg", smc["psi_sdev"][()] ** 2, np.diff(common.MU_EDGES) ** 2)
    )[0]
    for i, (basis, f, marker, ls, filled) in enumerate(BASES):
        group = f[f"rmc/{name}"]
        epsilon = group["epsilon_norm"][()]
        n = np.arange(1, len(epsilon) + 1)
        P = group.attrs["N_history"] // (len(group["E_edges"]) - 1)
        error = [
            rms_relative(common.scalar_flux(psi)[0], phi_smc)
            for psi in group["psi_history"][()]
        ]
        style = dict(
            color=common.SERIES[i],
            marker=marker,
            ms=5,
            mfc=common.SERIES[i] if filled else "none",
            ls=ls,
            label=f"$^{{16}}$O {name}, {basis}, {P} P/B/I",
        )
        axes[row, 0].plot(n, epsilon, **style)
        axes[row, 1].plot(n, error, **style)
    axes[row, 1].axhline(
        np.sqrt(np.mean((sdev_smc / phi_smc) ** 2)),
        color=common.MUTED,
        ls=(0, (1, 2)),
        label="SMC noise",
    )
    for ax in axes[row]:
        ax.set_yscale("log")
        ax.set_xlabel("Iterations")
    axes[row, 0].set_ylabel(r"$\|\tilde\epsilon\|$")
    axes[row, 1].set_ylabel(r"rms rel. difference to SMC")
    axes[row, 0].set_title(f"$^{{16}}$O {title}")
    axes[row, 1].set_title(f"$^{{16}}$O {title}")
    axes[row, 0].legend()
    axes[row, 1].legend()
fig.tight_layout()
fig.savefig("convergence_linear.png")

# Flux per window: linear drawn as its segments psi_0 + psi_1 x on each bin, with the
#   ratio of the bin averages to SMC below
fig, axes = plt.subplots(
    2,
    len(CASES),
    figsize=(6.0 * len(CASES), 7.0),
    sharex="col",
    gridspec_kw=dict(height_ratios=(3, 1.4)),
)
for col, (name, title) in enumerate(CASES):
    ax, ax_ratio = axes[0, col], axes[1, col]
    group_c, group_l = constant[f"rmc/{name}"], linear[f"rmc/{name}"]
    smc = constant[f"smc/{name}"]
    E_MeV = group_l["E_edges"][()] * 1e-6
    G = len(E_MeV) - 1
    phi_smc = common.scalar_flux(smc["psi"][()])[0]
    phi_c = common.scalar_flux(group_c["psi"][()])[0]
    phi_l = common.scalar_flux(group_l["psi"][()])[0]
    slope_l = common.scalar_flux(group_l["psi_slope"][()])[0]
    P_c = group_c.attrs["N_history"] // G
    P_l = group_l.attrs["N_history"] // G
    ax.plot(
        *common.steps(E_MeV, phi_smc * 1e6),
        color=common.MUTED,
        label=f"SMC: $10^{{{int(np.log10(smc.attrs['N_particle']))}}}$ particles",
    )
    ax.plot(
        *common.steps(E_MeV, phi_c * 1e6),
        color=common.SERIES[0],
        ls=(0, (4, 2)),
        label=f"RMC constant (previous), {P_c} P/B/I",
    )
    for g in range(G):
        ax.plot(
            E_MeV[g : g + 2],
            np.array((phi_l[g] - slope_l[g], phi_l[g] + slope_l[g])) * 1e6,
            color=common.SERIES[1],
            label=f"RMC linear (per-bin segments), {P_l} P/B/I" if g == 0 else None,
        )
    ax.set_xscale("log")
    ax.set_title(f"$^{{16}}$O {title}, 10 iterations")
    ax.legend(fontsize=8)
    ax_ratio.axhline(1.0, color=common.MUTED, lw=0.8)
    for phi, color, ls, label in (
        (phi_c, common.SERIES[0], (0, (4, 2)), "constant / SMC"),
        (phi_l, common.SERIES[1], "-", "linear / SMC"),
    ):
        ax_ratio.plot(
            *common.steps(E_MeV, phi / phi_smc),
            color=color,
            ls=ls,
            label=f"{label} (rms {np.sqrt(np.mean((phi / phi_smc - 1) ** 2)):.1%})",
        )
    ax_ratio.set_xlabel("E (MeV)")
    ax_ratio.legend(fontsize=8)
axes[0, 0].set_ylabel(r"$\phi(E)$ [per MeV]")
axes[1, 0].set_ylabel("bin average / SMC")
fig.tight_layout()
fig.savefig("flux_linear.png")
