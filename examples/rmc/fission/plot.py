"""Figure of the fission problem from output.h5 (see input.py)."""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")

# Flux: [D] Fig. 2.3
run = f["rmc/G100_P100"]
E = run["E_edges"][()]
fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.plot(
    *common.steps(E, run["scalar_flux"][0]),
    color=common.SERIES[0],
    zorder=3,
    label="RMC: 100 P/B/I, 10 iterations",
)
for N, color in ((100000, common.SERIES[1]), (1000000, common.SERIES[2])):
    phi = common.scalar_flux(f[f"smc/G100_N{N}/psi"][()])[0]
    ax.plot(
        *common.steps(E, phi),
        color=color,
        lw=1.0,
        label=f"SMC: $10^{int(np.log10(N))}$ particles",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E (eV)")
ax.set_ylabel(r"$\phi(E)$")
ax.legend()
fig.savefig("flux.png")
