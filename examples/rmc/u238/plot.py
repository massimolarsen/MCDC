"""Figures of the U-238 problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")
rmc = f["rmc/G2000_P100"]
E = rmc["E_edges"][()]
phi = rmc["scalar_flux"][0]

# Flux and sigma_t: [D] Fig. 3.7
fig, (ax, ax_xs) = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
ax.plot(
    *common.steps(E, phi),
    color=common.SERIES[0],
    zorder=3,
    label="RMC: 100 P/B/I, 20 iterations",
)
ax.set_yscale("log")
ax.set_ylabel(r"$\phi(E)$ [per eV]")
ax.legend(loc="upper left")
ax_xs.plot(
    f["data/energy"][()],
    f["data/sigma_t"][()],
    color=common.MUTED,
    label=r"$\sigma_t(E)$",
)
ax_xs.set_xscale("log")
ax_xs.set_yscale("log")
ax_xs.set_xlim(E[0], E[-1])
ax_xs.set_xlabel("E (eV)")
ax_xs.set_ylabel(r"U-238 $\sigma_t(E)$ [b]")
ax_xs.legend(loc="upper left")
fig.savefig("flux_xs.png")

# Close-up, 1-4 eV: [D] Fig. 3.8
fig, ax = plt.subplots(figsize=(8.0, 5.0))
for i, N in enumerate((1000000, 10000000)):
    smc = common.scalar_flux(f[f"smc/N{N}/psi"][()])[0]
    ax.plot(
        *common.steps(E, smc),
        color=common.SERIES[i],
        lw=1.0,
        label=f"SMC: $10^{int(np.log10(N))}$ particles",
    )
ax.plot(
    *common.steps(E, phi),
    color=common.INK,
    ls=(0, (3, 1.5)),
    zorder=3,
    label="RMC: 100 P/B/I, 20 iterations",
)
mask = (E[:-1] >= 1.0) & (E[1:] <= 4.0)
ax.set_xlim(1.0, 4.0)
ax.set_ylim(0.97 * phi[mask].min(), 1.03 * phi[mask].max())
ax.set_xscale("log")
ax.set_xlabel("E (eV)")
ax.set_ylabel(r"$\phi(E)$ [per eV]")
ax.legend(loc="upper left")
fig.savefig("flux_zoom.png")
