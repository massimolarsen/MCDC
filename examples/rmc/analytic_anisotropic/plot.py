"""Figure of the anisotropic-source problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")

# Angular flux per unit volume in the collided bins: [N] Fig. 4
rmc, smc = f["rmc/G100_P10"], f["smc/G100"]
height = rmc.attrs["height"]
E = rmc["E_edges"][:-1]
fig, ax = plt.subplots(figsize=(7.5, 5.0))
for j, label in ((0, r"\mu < 0"), (1, r"\mu > 0")):
    color = common.SERIES[j]
    ax.plot(
        *common.steps(E, height * rmc["psi"][0, :-1, j]),
        color=color,
        label=f"RMC: ${label}$",
    )
    ax.plot(
        *common.steps(E, height * smc["psi"][0, :-1, j]),
        color=color,
        lw=1.0,
        ls=(0, (2, 1.5)),
        label=f"SMC: ${label}$",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E (eV)")
ax.set_ylabel(r"$\psi(E, \mu)$")
ax.legend()
fig.savefig("angular_flux.png")
