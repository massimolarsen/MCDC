"""
Infinite, homogeneous, purely scattering helium-like material (A = 4, sigma_s = 1).

Reproduces [N] Figs. 10-12 (see ../common.py for references):
  - flux.png                    RMC (100 P/B/I, 5 iterations) vs SMC vs analytic, G = 100
  - convergence_histories.png   L-infinity norm vs histories, RMC sweep and SMC
  - convergence_iterations.png  L-infinity norm vs iterations, RMC sweep

Problem ([N] Sec. IV.D): isotropic monoenergetic source at E0 = 101, window [1, E0],
2 polar bins; alpha = 0.36. The energy mesh has an edge at alpha E0 (the transient
discontinuity) with log-spaced bins on either side in proportion to their lethargy
widths, as in [N].

Differences from the paper: as ../analytic_absorber (smeared source bin, P/B/I per
energy bin). The L-infinity norm uses the exact bin-averaged solution of the
slowing-down equation, which includes the Placzek transients that [N] Eqs. 38-39
(plotted as the analytic lines) leave out below alpha E0.

Run (writes output.h5; then python plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

E0, DELTA, E_MIN = 101.0, 0.01, 1.0
A = 4.0
ALPHA = ((A - 1.0) / (A + 1.0)) ** 2
GRIDS = (60, 100, 200)
PARTICLES = (20, 100)  # per energy bin per iteration
N_ITERATION = 30
SMC_HISTORIES = (1e3, 1e4, 1e5, 1e6)


def energy_grid(G):
    """G collided bins with an edge at alpha E0, plus the narrow source bin."""
    E_split, E_top = ALPHA * E0, E0 - DELTA
    G_high = int(round(G * np.log(E_top / E_split) / np.log(E_top / E_MIN)))
    low = np.logspace(np.log10(E_MIN), np.log10(E_split), G - G_high + 1)
    high = np.logspace(np.log10(E_split), np.log10(E_top), G_high + 1)
    return np.concatenate((low, high[1:], [E0]))


def make_model():
    import mcdc

    material = mcdc.Material(nuclide_composition={"HE": 1.0}, temperature=0.1)
    simulation = common.slab([(0.0, 1.0, material)])
    simulation.set_sources(
        [common.volume_source(0.0, 1.0, common.smc_source_energy(E0 - DELTA, E0))]
    )
    return simulation, [material]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = np.array([1.0e-3, 1.0e4])
    common.write_nuclide("data", "HE", A, data, np.ones(2), np.zeros(2))
    common.use_library("data")
    nuclide = [(A, 1.0, data, np.ones(2), np.ones(2))]
    z_edges = [0.0, 1.0]

    output = common.open_output()
    runs = [(G, P, N_ITERATION) for G in GRIDS for P in PARTICLES] + [(100, 100, 5)]
    for i, (G, P, N_iteration) in enumerate(runs):
        E_edges = energy_grid(G)
        if f"reference/G{G}" not in output:
            output.create_dataset(
                f"reference/G{G}",
                data=common.slowing_down_reference(
                    E_edges, nuclide, (E0 - DELTA, E0), h=2e-5
                ),
            )
        Q = common.uniform_Q(z_edges, E_edges, [0], (E0 - DELTA, E0))
        result = common.run_rmc(
            make_model, z_edges, E_edges, Q, N_iteration, P / 2, cache_dir="cache"
        )
        if i < len(runs) - 1:
            common.save_rmc(output, f"G{G}_P{P}", result, G=G, P=P)
        else:
            common.save_rmc(output, f"flux_G{G}_P{P}", result, G_flux=G, P_flux=P)

    E_edges = energy_grid(100)
    for i, N in enumerate(SMC_HISTORIES):
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N, seed=i + 1)
        common.save_smc(output, f"G100_N{int(N)}", mean, sdev, wall, N, G=100)
    output.close()


if __name__ == "__main__":
    main()
