import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

# Plot neutron energy spectrum from the surface tally

# Energy grid (relative to this script's folder)
E = np.loadtxt(r"energy_grid.txt")
G = len(E) - 1
E_mid = 0.5 * (E[1:] + E[:-1])
dE = E[1:] - E[:-1]

# HDF5 file (allow passing path as first arg)
h5_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\larse\source\repos\output.h5"

with h5py.File(h5_path, "r") as f:
    # Locate a surface tally in the file
    tallies = f.get("tallies", {})
    surface_key = None
    for key in tallies:
        if key.startswith("cell_tally"):
            surface_key = key
            break
    if surface_key is None:
        # fallback: pick first tally if only one exists
        keys = list(tallies.keys())
        if len(keys) == 0:
            raise RuntimeError("No tallies found in HDF5 file")
        surface_key = keys[0]

    # Read flux mean and sdev arrays
    # Path convention: tallies/{tally}/flux/mean and /sdev
    flux_path = f"tallies/{surface_key}/flux/mean"
    sdev_path = f"tallies/{surface_key}/flux/sdev"
    phi = f[flux_path][:]
    phi_sd = f[sdev_path][:]

    # Reduce over all axes except the energy axis and score axis.
    # Identify energy axis by matching its size to G.
    shape = phi.shape
    energy_axis = None
    for i, s in enumerate(shape):
        if s == G:
            energy_axis = i
            break
    if energy_axis is None:
        # if not found, assume energy is axis -2 (second last)
        energy_axis = -2

    score_axis = -1
    axes_to_sum = tuple(i for i in range(phi.ndim) if i not in (energy_axis, score_axis))
    if axes_to_sum:
        phi = phi.sum(axis=axes_to_sum)
        phi_sd = phi_sd.sum(axis=axes_to_sum)

    # Now phi should be (G, N_score) or (G,)
    if phi.ndim == 2:
        # pick flux score (usually single score)
        phi = phi[:, 0]
        phi_sd = phi_sd[:, 0]

    # Convert to E*phi(E) per bin (use bin width)
    spec = phi * E_mid / dE
    spec_sd = phi_sd * E_mid / dE

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.grid(True)
ax.set_xscale("log")
ax.set_xlabel("E (MeV)")
ax.set_ylabel(r"$\phi$(E)")
ax.plot(E_mid, spec, '-b', label='MC')
ax.fill_between(E_mid, spec - spec_sd, spec + spec_sd, color='b', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
