"""Figures of the resonance-absorber problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import E0, E_R, sigma_a

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")


def lowest(values):
    """The lowest 40% of the collided bins (below the resonance), as in [N]."""
    values = np.asarray(values)[..., :-1]
    return values[..., : int(0.4 * values.shape[-1])]


def rmc_error(group):
    reference = lowest(f[f"reference/G{group.attrs['G']}"][()])
    return common.linf(
        lowest(common.scalar_flux(group["psi_history"][()])[:, 0]), reference
    )


def smc_error(group):
    reference = lowest(f["reference/G100"][()])
    return common.linf(lowest(common.scalar_flux(group["psi"][()])[0]), reference)


# Flux and sigma_t: [N] Fig. 7
run = f["rmc/G100_P20"]
E = run["E_edges"][:-1]
fig, (ax, ax_xs) = plt.subplots(
    2, 1, figsize=(7.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
fine = np.logspace(0.0, np.log10(E[-1]), 400)
escape = np.exp(
    -quad(
        lambda x: sigma_a(x) / ((1.0 + sigma_a(x)) * x),
        1.0,
        E0,
        points=[E_R],
        limit=200,
    )[0]
)
ax.plot(
    fine, 1.0 / fine, color=common.INK_2, ls=(0, (4, 2)), label=r"Analytic: $E > E_r$"
)
ax.plot(
    fine,
    escape / fine,
    color=common.MUTED,
    ls=(0, (1, 1.5)),
    label=r"Analytic: $E < E_r$",
)
phi = common.scalar_flux(run["psi_history"][19])[0, :-1]
ax.plot(
    *common.steps(E, phi),
    color=common.SERIES[0],
    zorder=3,
    label="RMC: 20 P/B/I, 20 iterations",
)
smc = f["smc/G100_N100000"]
ax.plot(
    *common.steps(E, common.scalar_flux(smc["psi"][()])[0, :-1]),
    color=common.SERIES[1],
    lw=1.0,
    label="SMC: $10^5$ particles",
)
ax.set_yscale("log")
ax.set_ylabel(r"$\phi(E)$")
ax.legend()
ax_xs.plot(fine, 1.0 + sigma_a(fine), color=common.MUTED)
ax_xs.set_xscale("log")
ax_xs.set_xlabel("E (eV)")
ax_xs.set_ylabel(r"$\sigma_t(E)$")
fig.savefig("flux.png")

# Convergence: [N] Figs. 8-9
fig, _ = common.plot_sweep(f, rmc_error, "histories", smc_error)
fig.savefig("convergence_histories.png")
fig, _ = common.plot_sweep(f, rmc_error, "iterations")
fig.savefig("convergence_iterations.png")
