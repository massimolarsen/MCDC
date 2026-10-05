"""
Infinite, homogeneous, purely scattering A = 1 material with sigma_s = sigma_t = E^-1/2.

Reproduces [N] Figs. 5-6 and [D] Figs. 3.1-3.2 (see ../common.py for references):
  - flux.png                    RMC (100 P/B/I, 20 iterations) vs SMC vs analytic, G = 100
  - convergence_histories.png   L-infinity norm vs histories, RMC sweep and SMC
  - convergence_iterations.png  L-infinity norm vs iterations, RMC sweep

Problem ([N] Sec. IV.B): isotropic monoenergetic source at E0 = 101, energy window
[1, E0], 2 polar bins, G log-spaced energy bins; analytic flux phi = S0 / sqrt(E).

Differences from the papers: as ../analytic_absorber (smeared source bin, exact
bin-averaged reference, P/B/I per energy bin; [D] Sec. 3.4.1 defines P/B/I per
energy-angle bin). With 10 P/B/I this pure scatterer
diverges here (L-infinity grows ~1.2x per iteration), whereas it converged in [N];
the flux figure therefore shows the 100 P/B/I run instead of 10 P/B/I. The cross section is tabulated on a
fine grid and interpolated linearly by MC/DC (the reference uses the same table).

Run (writes output.h5; then python plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

E0, DELTA, E_MIN = 101.0, 0.01, 1.0
GRIDS = (50, 100, 200)
PARTICLES = (10, 100)  # per energy bin per iteration
N_ITERATION = 30
SMC_HISTORIES = (1e3, 1e4, 1e5, 1e6, 1e7)
DATA_ENERGY = np.logspace(-1.0, 3.0, 20001)


def make_model():
    import mcdc

    material = mcdc.Material(nuclide_composition={"HS": 1.0}, temperature=0.1)
    simulation = common.slab([(0.0, 1.0, material)])
    simulation.set_sources(
        [common.volume_source(0.0, 1.0, common.smc_source_energy(E0 - DELTA, E0))]
    )
    return simulation, [material]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sigma = DATA_ENERGY**-0.5
    common.write_nuclide("data", "HS", 1.0, DATA_ENERGY, sigma, np.zeros_like(sigma))
    common.use_library("data")
    nuclide = [(1.0, 1.0, DATA_ENERGY, sigma, sigma)]
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
                make_model, z_edges, E_edges, Q, N_ITERATION, P / 2, cache_dir="cache"
            )
            common.save_rmc(output, f"G{G}_P{P}", result, G=G, P=P)

    E_edges = common.narrow_source_grid(E_MIN, E0, DELTA, 100)
    for i, N in enumerate(SMC_HISTORIES):
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N, seed=i + 1)
        common.save_smc(output, f"G100_N{int(N)}", mean, sdev, wall, N, G=100)
    output.close()


if __name__ == "__main__":
    main()
