"""
A = 1 scatterer (sigma_s = 1) with a single-level Breit-Wigner absorption resonance.

Reproduces [N] Figs. 7-9 (see ../common.py for references):
  - flux.png                    RMC (20 P/B/I, 20 iterations) vs SMC vs analytic, with
                                sigma_t(E) below (the paper uses a second y-axis)
  - convergence_histories.png   L-infinity norm vs histories, RMC sweep and SMC
  - convergence_iterations.png  L-infinity norm vs iterations, RMC sweep

Problem ([N] Sec. IV.C): sigma_a(E) = 3 (Gamma/2)^2 / ((E - Er)^2 + (Gamma/2)^2) with
Er = 30, Gamma = 5; isotropic monoenergetic source at E0 = 101, window [1, E0], 2 polar
bins. The L-infinity norm is taken over the lowest 40% of the energy bins, as in [N].

Differences from the paper: as ../analytic_absorber (smeared source bin, P/B/I per
energy bin). [N] Fig. 7 used 10 P/B/I; here that run (rmc/flux_G100_P10, kept in
output.h5) diverges, as in ../analytic_sqrt, so the flux figure shows 20 P/B/I. The reference is the exact bin-averaged solution of the slowing-down
equation instead of [N] Eqs. 36-37 (which approximate the resonance escape probability
with bin-constant cross sections); the plotted analytic lines are [N] Eqs. 32 and 37
with the exact escape probability.

Run (writes output.h5; then python plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

E0, DELTA, E_MIN = 101.0, 0.01, 1.0
E_R, GAMMA, SIGMA_PEAK = 30.0, 5.0, 3.0
GRIDS = (60, 100, 200)
PARTICLES = (20, 100)  # per energy bin per iteration
N_ITERATION = 30
SMC_HISTORIES = (1e3, 1e4, 1e5, 1e6)
DATA_ENERGY = np.logspace(-1.0, 3.0, 20001)


def sigma_a(E):
    return SIGMA_PEAK * (GAMMA / 2) ** 2 / ((E - E_R) ** 2 + (GAMMA / 2) ** 2)


def make_model():
    import mcdc

    material = mcdc.Material(nuclide_composition={"HR": 1.0}, temperature=0.1)
    simulation = common.slab([(0.0, 1.0, material)])
    simulation.set_sources(
        [common.volume_source(0.0, 1.0, common.smc_source_energy(E0 - DELTA, E0))]
    )
    return simulation, [material]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sigma_s = np.ones_like(DATA_ENERGY)
    sigma_c = sigma_a(DATA_ENERGY)
    common.write_nuclide("data", "HR", 1.0, DATA_ENERGY, sigma_s, sigma_c)
    common.use_library("data")
    nuclide = [(1.0, 1.0, DATA_ENERGY, sigma_s, sigma_s + sigma_c)]
    z_edges = [0.0, 1.0]

    output = common.open_output()
    runs = [(G, P, N_ITERATION) for G in GRIDS for P in PARTICLES] + [(100, 10, 20)]
    for G, P, N_iteration in runs:
        E_edges = common.narrow_source_grid(E_MIN, E0, DELTA, G)
        if f"reference/G{G}" not in output:
            output.create_dataset(
                f"reference/G{G}",
                data=common.slowing_down_reference(E_edges, nuclide, (E0 - DELTA, E0)),
            )
        Q = common.uniform_Q(z_edges, E_edges, [0], (E0 - DELTA, E0))
        result = common.run_rmc(
            make_model, z_edges, E_edges, Q, N_iteration, P / 2, cache_dir="cache"
        )
        if P in PARTICLES:
            common.save_rmc(output, f"G{G}_P{P}", result, G=G, P=P)
        else:
            common.save_rmc(output, f"flux_G{G}_P{P}", result, G_flux=G, P_flux=P)

    E_edges = common.narrow_source_grid(E_MIN, E0, DELTA, 100)
    for i, N in enumerate(SMC_HISTORIES):
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N, seed=i + 1)
        common.save_smc(output, f"G100_N{int(N)}", mean, sdev, wall, N, G=100)
    output.close()


if __name__ == "__main__":
    main()
