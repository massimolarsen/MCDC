"""
Infinite medium (0D) with real nuclear data: RMC vs MC/DC standard Monte Carlo.

One case per process:

    RMC_CASE=O16 python input.py

The source is uniform in energy over the window and isotropic. Results (SMC mean and
standard error, RMC iterates, timings) are written to results_<case>.npz.
"""

import os
import time

import numpy as np

import mcdc
from mcdc.rmc.driver import run

CASES = {
    "Al27": dict(
        composition={"Al27": 0.0602}, temperature=293.6, window=(1.0e6, 1.0e7), G=50
    ),
    "O16": dict(
        composition={"O16": 0.05}, temperature=0.1, window=(1.0e6, 1.0e7), G=50
    ),
    "U238": dict(
        composition={"U238": 0.05}, temperature=0.1, window=(1.0, 100.0), G=200
    ),
    "U238-H2": dict(
        composition={"U238": 0.005, "H2": 0.05},
        temperature=0.1,
        window=(1.0, 100.0),
        G=200,
    ),
}
N_PER_BIN = int(os.environ.get("RMC_N_PER_BIN", 100))
N_ITERATION = int(os.environ.get("RMC_N_ITERATION", 10))
N_SMC = int(os.environ.get("RMC_N_SMC", 1_000_000))


def reflective_box(material):
    x0 = mcdc.Surface.PlaneX(x=-0.5, boundary_condition="reflective")
    x1 = mcdc.Surface.PlaneX(x=0.5, boundary_condition="reflective")
    y0 = mcdc.Surface.PlaneY(y=-0.5, boundary_condition="reflective")
    y1 = mcdc.Surface.PlaneY(y=0.5, boundary_condition="reflective")
    z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="reflective")
    z1 = mcdc.Surface.PlaneZ(z=1.0, boundary_condition="reflective")
    return mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=material)


def model(case):
    material = mcdc.Material(
        nuclide_composition=case["composition"], temperature=case["temperature"]
    )
    E_min, E_max = case["window"]
    simulation = mcdc.Simulation("rmc-infinite-medium")
    simulation.set_model([reflective_box(material)])
    simulation.set_sources(
        [
            mcdc.Source(
                position=[0.0, 0.0, 0.5],
                isotropic=True,
                energy=np.array([[E_min, E_max], [1.0, 1.0]])
                / [[1.0], [E_max - E_min]],
            )
        ]
    )
    return simulation, material


def main():
    name = os.environ["RMC_CASE"]
    case = CASES[name]
    E_min, E_max = case["window"]
    E_edges = np.logspace(np.log10(E_min), np.log10(E_max), case["G"] + 1)
    mu_edges = np.array([-1.0, 0.0, 1.0])
    G = len(E_edges) - 1

    # Standard Monte Carlo reference (same energy window, all-prompt fission)
    simulation, material = model(case)
    settings = simulation.settings
    settings.N_particle = N_SMC
    settings.use_neutron_energy_window = True
    settings.neutron_energy_min = E_min
    settings.neutron_energy_max = E_max
    settings.neutron_fission_all_prompt = True
    settings.output_name = f"smc_{name}"
    tally = mcdc.Tally(scores=["flux"], energy=E_edges)
    simulation.set_tallies([tally])
    time_smc = time.perf_counter()
    simulation.run()
    time_smc = time.perf_counter() - time_smc

    import h5py

    with h5py.File(f"smc_{name}.h5", "r") as f:
        group = f["tallies"][list(f["tallies"])[0]]
        phi_smc = np.squeeze(group["flux/mean"][()])
        sd_smc = np.squeeze(group["flux/sdev"][()])

    # RMC (uniform isotropic source: Q per unit z, E, mu)
    simulation, material = model(case)
    Q = np.full((1, G, 2), 1.0 / ((E_max - E_min) * 2.0))
    time_rmc = time.perf_counter()
    result = run(
        simulation,
        [0.0, 1.0],
        E_edges,
        mu_edges,
        Q,
        [material],
        N_ITERATION,
        N_PER_BIN,
        cache_dir="rmc_cache",
    )
    time_rmc = time.perf_counter() - time_rmc
    phi_rmc = np.array(
        [
            np.einsum("kgj,j->kg", psi, np.diff(mu_edges))[0]
            for psi in result.psi_history
        ]
    ) * np.diff(E_edges)

    np.savez(
        f"results_{name}.npz",
        E_edges=E_edges,
        phi_smc=phi_smc,
        sd_smc=sd_smc,
        phi_rmc=phi_rmc,
        epsilon_norm=np.array(result.epsilon_norm),
        time_smc=time_smc,
        time_rmc=time_rmc,
        N_smc=N_SMC,
        N_rmc=N_PER_BIN * G * 2 * N_ITERATION,
    )


if __name__ == "__main__":
    main()
