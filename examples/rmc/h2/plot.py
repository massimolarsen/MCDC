"""Figures of this directory's input.py (output.h5); see README.md."""

import os
import sys

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import CASES, LIBRARY, NUCLIDE, TEMPERATURE

# Bins whose Sigma_t varies less than this inside the bin: the trial-space bias of
# the phase-1 fixed point is below the SMC noise there
FLAT = 1.02

os.chdir(os.path.dirname(os.path.abspath(__file__)))
f = h5py.File("output.h5", "r")
energy, sigma_t = common.read_total_xs(NUCLIDE, TEMPERATURE, LIBRARY)
for name, (window, G) in CASES.items():
    title = f"{NUCLIDE}, {window[0]:g}-{window[1]:g} eV, G = {G}"
    E_edges = f[f"rmc/{name}/E_edges"][()]
    smooth = common.bin_xs_variation(energy, sigma_t, E_edges) < FLAT
    rms_z, rms_z_smooth = common.plot_rmc_vs_smc(
        f, name, title, f"flux_{name}.png", smooth
    )
    print(
        f"{name}: rms z = {rms_z:.2f} (all bins), {rms_z_smooth:.2f} "
        f"({smooth.sum()} flat-Sigma_t bins)"
    )
