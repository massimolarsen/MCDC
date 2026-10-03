"""Figure of the U-238 / H-2 dilution problem from output.h5 (see input.py)."""

import os
import sys

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")

# Flux for three dilutions: [D] Fig. 3.9
fig, ax = plt.subplots(figsize=(8.0, 5.0))
labels = {
    0.25: r"$\rho_s = \frac{1}{4}$",
    0.5: r"$\rho_s = \frac{2}{4}$",
    0.75: r"$\rho_s = \frac{3}{4}$",
}
for i, group in enumerate(sorted(f["rmc"].values(), key=lambda g: g.attrs["rho"])):
    E = group["E_edges"][()]
    ax.plot(
        *common.steps(E, group["scalar_flux"][0]),
        color=common.SERIES[i],
        label=labels[float(group.attrs["rho"])],
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E (eV)")
ax.set_ylabel(r"$\phi(E)$ [per eV]")
ax.legend()
fig.savefig("flux.png")
