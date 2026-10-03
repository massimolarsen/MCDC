"""Figures of the O-16 problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")

# Cross sections: [D] Fig. 4.1
energy = f["data/energy"][()]
mask = (energy >= 1.0e6) & (energy <= 1.0e7)
elastic = f["data/elastic_scattering"][()]
inelastic = f["data/inelastic_scattering"][()]
absorption = f["data/capture"][()]
fig, ax = plt.subplots(figsize=(7.5, 5.0))
E_MeV = energy[mask] * 1e-6
ax.plot(
    E_MeV,
    (elastic + inelastic + absorption)[mask],
    color=common.MUTED,
    label=r"$\sigma_t$",
)
ax.plot(
    E_MeV, elastic[mask], color=common.SERIES[0], ls=(0, (4, 2)), label=r"$\sigma_{el}$"
)
ax.plot(
    E_MeV,
    np.where(inelastic[mask] > 0, inelastic[mask], np.nan),
    color=common.SERIES[1],
    ls=(0, (1, 1.5)),
    label=r"$\sigma_{inl}$",
)
ax.plot(
    E_MeV,
    np.where(absorption[mask] > 0, absorption[mask], np.nan),
    color=common.SERIES[2],
    ls=(0, (4, 1.5, 1, 1.5)),
    label=r"$\sigma_a$",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E (MeV)")
ax.set_ylabel(r"$\sigma$ (barns)")
ax.legend(loc="center left")
fig.savefig("cross_sections.png")

# Flux comparisons: [D] Figs. 4.2-4.4
for name, path in (
    ("fast", "flux.png"),
    ("isolated", "flux_isolated.png"),
    ("resonances", "flux_resonances.png"),
):
    rmc, smc = f[f"rmc/{name}"], f[f"smc/{name}"]
    E_MeV = rmc["E_edges"][()] * 1e-6
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        *common.steps(E_MeV, rmc["scalar_flux"][0] * 1e6),
        color=common.SERIES[0],
        zorder=3,
        label="RMC: 100 P/B/I, 10 iterations",
    )
    ax.plot(
        *common.steps(E_MeV, common.scalar_flux(smc["psi"][()])[0] * 1e6),
        color=common.SERIES[1],
        ls=(0, (3, 1.5)),
        label=f"SMC: $10^{{{int(np.log10(smc.attrs['N_particle']))}}}$ particles",
    )
    ax.set_xscale("log")
    ax.set_xlabel("E (MeV)")
    ax.set_ylabel(r"$\phi(E)$ [per MeV]")
    ax.legend()
    fig.savefig(path)
