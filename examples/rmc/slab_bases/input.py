"""
Two-material slab with vacuum boundaries: the spatial and angular trial spaces of RMC
(piecewise-constant or continuous linear in z, piecewise-constant or linear
discontinuous in the polar cosine) against one SMC reference. See README.md.

Problem (synthetic A = 1 nuclide, sigma_s = sigma_c = 1 b, constant in energy):
  - z in [0, 4] cm, vacuum at z = 0 and z = 4, reflective in x and y;
  - material a (density 1.0, Sigma_t = 2 /cm) in [0, 2], material b (density 0.4) in
    [2, 4];
  - isotropic source, uniform in z over [0, 1] and in energy over [E0 - delta, E0];
  - trial space: 4 cells of 1 cm, 8 log-spaced energy bins below the source plus the
    source bin, 2 polar bins (mu < 0, mu > 0).

Each basis combination runs 8 collision-only iterations and 8 averaged correction
passes at 200 histories per bin; SMC runs 10^6 histories and tallies the flux and its
z-slope and polar-cosine-slope moments on the same cells, energy and polar bins.

Run (writes output.h5; then python plot.py):
    mpiexec -n 4 python input.py --mode=numba
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common

E0, DELTA = 10.0, 0.1
Z_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
E_EDGES = np.concatenate((np.logspace(0.0, np.log10(E0 - DELTA), 9), [E0]))
DENSITIES = (1.0, 0.4)  # materials a and b
CELL_MATERIALS = (0, 0, 1, 1)
N_ITERATION, N_CORRECTION, N_PER_BIN = 8, 8, 200
N_SMC = 1e6
BASES = [
    (spatial, angular)
    for spatial in ("constant", "linear")
    for angular in ("constant", "linear")
]


def run_name(spatial, angular):
    return f"z-{spatial}_mu-{angular}"


def make_model():
    import mcdc

    materials = [
        mcdc.Material(nuclide_composition={"HX": density}, temperature=0.1)
        for density in DENSITIES
    ]
    simulation = common.slab(
        [(0.0, 2.0, materials[0]), (2.0, 4.0, materials[1])],
        z_boundary=("vacuum", "vacuum"),
    )
    simulation.set_sources([common.volume_source(0.0, 1.0, (E0 - DELTA, E0))])
    return simulation, [materials[m] for m in CELL_MATERIALS]


def run_smc(N_particle):
    """
    SMC cell moments per (cell, energy bin, polar bin), per unit z, energy and polar
    cosine: the average of psi, of psi x (x = 2s - 1 in the cell) and of psi y
    (y = the polar cosine's Legendre variable in its bin).
    """
    import h5py
    import mcdc

    simulation, _ = make_model()
    settings = simulation.settings
    settings.N_particle = int(N_particle)
    settings.use_neutron_energy_window = True
    settings.neutron_energy_min = E_EDGES[0]
    settings.neutron_energy_max = E_EDGES[-1]
    settings.use_progress_bar = False
    settings.output_name = "smc-tmp"
    mesh = mcdc.MeshStructured("smc-slab", z=Z_EDGES)
    tally = mcdc.Tally(
        name="smc",
        mesh=mesh,
        scores=["flux", "flux-z-slope", "flux-mu-slope"],
        energy=E_EDGES,
        mu=common.MU_EDGES,
    )
    simulation.set_tallies([tally])
    simulation.run()
    if not common.is_master():
        return None
    K, G, J = len(Z_EDGES) - 1, len(E_EDGES) - 1, len(common.MU_EDGES) - 1
    volume = (
        np.diff(Z_EDGES)[:, None, None]
        * np.diff(E_EDGES)[None, :, None]
        * np.diff(common.MU_EDGES)[None, None, :]
    )
    out = {}
    with h5py.File("smc-tmp.h5", "r") as f:
        for score in ("flux", "flux-z-slope", "flux-mu-slope"):
            for x in ("mean", "sdev"):
                value = f[f"tallies/smc/{score}/{x}"][()].reshape(J, G, K)
                out[f"{score}/{x}"] = value.transpose(2, 1, 0) / volume
    os.remove("smc-tmp.h5")
    return out


def main():
    os.chdir(HERE)
    common.write_nuclide("data", "HX", 1.0, [1.0e-6, 1.0e3], [1.0, 1.0], [1.0, 1.0])
    common.use_library("data")
    output = common.open_output()

    smc = run_smc(N_SMC)
    for key, value in (smc or {}).items():
        output.create_dataset(f"smc/{key}", data=value)

    K, G, J = len(Z_EDGES) - 1, len(E_EDGES) - 1, len(common.MU_EDGES) - 1
    Q = np.zeros((K, G, J))
    Q[0, -1, :] = 1.0 / (1.0 * DELTA * 2.0)  # unit source per unit area, isotropic
    for spatial, angular in BASES:
        result = common.run_rmc(
            make_model,
            Z_EDGES,
            E_EDGES,
            Q,
            N_ITERATION,
            N_PER_BIN,
            N_correction=N_CORRECTION,
            boundary=("vacuum", "vacuum"),
            spatial_basis=spatial,
            angular_basis=angular,
        )
        common.save_rmc(output, run_name(spatial, angular), result)
    output.close()


if __name__ == "__main__":
    main()
