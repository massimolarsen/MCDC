"""
Infinite, homogeneous hydrogen-like absorber: A = 1, sigma_s = 1, sigma_t = 3.

Reproduces [N] Figs. 1-3 and [D] Figs. 2.1-2.2 (see ../common.py for references):
  - flux.png                    RMC (10 P/B/I, 20 iterations) vs SMC vs analytic, G = 100
  - convergence_histories.png   L-infinity norm vs histories, RMC sweep and SMC
  - convergence_iterations.png  L-infinity norm vs iterations, RMC sweep

Problem ([N] Sec. IV.A): isotropic monoenergetic source at E0 = 101, energy window
[1, E0], scattering ratio 1/3, 2 polar bins, G log-spaced energy bins. Arbitrary energy
units are taken as eV.

Differences from the papers:
  - The monoenergetic source is smeared over a narrow top bin [E0 - 0.01, E0] (MC/DC
    tallies cannot resolve a delta; the bin is part of the trial space). The
    reference below is solved with the same smeared source. The SMC source starts
    3e-5 eV above the bin edge (see common.smc_source_energy).
  - The reference is the bin-averaged solution of the slowing-down equation
    (../common.py, accurate to ~1e-10), equal to [N] Eq. 32 for an unsmeared source.
  - P/B/I counts histories per energy bin per iteration, as in the papers: each
    energy-angle bin gets P/B/I / 2.

Run (writes output.h5; then python plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

E0, DELTA, E_MIN = 101.0, 0.01, 1.0
SIGMA_S, SIGMA_T = 1.0, 3.0
GRIDS = (50, 100, 200)
PARTICLES = (10, 100)  # per energy bin per iteration
N_ITERATION = 30
SMC_HISTORIES = (1e3, 1e4, 1e5, 1e6, 1e7)


def make_model():
    import mcdc

    material = mcdc.Material(nuclide_composition={"HA": 1.0}, temperature=0.1)
    simulation = common.slab([(0.0, 1.0, material)])
    simulation.set_sources(
        [common.volume_source(0.0, 1.0, common.smc_source_energy(E0 - DELTA, E0))]
    )
    return simulation, [material]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    data = np.array([1.0e-3, 1.0e4])
    common.write_nuclide(
        "data", "HA", 1.0, data, SIGMA_S * np.ones(2), (SIGMA_T - SIGMA_S) * np.ones(2)
    )
    common.use_library("data")
    nuclide = [(1.0, 1.0, data, SIGMA_S * np.ones(2), SIGMA_T * np.ones(2))]
    z_edges = [0.0, 1.0]

    output = common.open_output()
    for G in GRIDS:
        E_edges = common.narrow_source_grid(E_MIN, E0, DELTA, G)
        output.create_dataset(
            f"reference/G{G}",
            data=common.slowing_down_reference(E_edges, nuclide, (E0 - DELTA, E0)),
        )
        Q = common.uniform_Q(z_edges, E_edges, [0], (E0 - DELTA, E0))
        for P in PARTICLES:
            result = common.run_rmc(
                make_model,
                z_edges,
                E_edges,
                Q,
                N_ITERATION,
                P / 2,
                cache_dir="cache",
            )
            common.save_rmc(output, f"G{G}_P{P}", result, G=G, P=P)

    E_edges = common.narrow_source_grid(E_MIN, E0, DELTA, 100)
    for i, N in enumerate(SMC_HISTORIES):
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N, seed=i + 1)
        common.save_smc(output, f"G100_N{int(N)}", mean, sdev, wall, N, G=100)
    output.close()


if __name__ == "__main__":
    main()
