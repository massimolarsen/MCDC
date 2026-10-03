"""
Anisotropic source in the infinite A = 1 absorber (sigma_s = 1, sigma_t = 3).

Reproduces [N] Fig. 4 (see ../common.py for references):
  - angular_flux.png   psi(E, mu) in the mu < 0 and mu > 0 bins, RMC vs SMC

Problem ([N] Sec. IV.A): as ../analytic_absorber, with 3/4 of the source emitted
isotropically into mu > 0 and 1/4 into mu < 0; G = 100, RMC with 10 P/B/I for 20
iterations, SMC with 10^6 histories.

Differences from the paper:
  - Physics: [N] Appendix A scatters with delta(mu - v / mu'), i.e. mu_out = mu_0 /
    mu_in without the azimuthal integration, so for A = 1 (mu_0 > 0) every collision
    keeps the sign of mu and the 3:1 split persists down to E = 1. MC/DC scatters in
    3D (the azimuthal arcsine kernel in RMC), so the anisotropy decays with each
    collision and psi(E, mu) is nearly isotropic near E = 1. RMC and SMC agree.
  - The infinite medium is a reflective slab of height 10^6 cm with the source spread
    uniformly over it. A unit reflective box would flip mu at its z faces and wash out
    the anisotropy; here wall effects cover ~1e-5 of the volume. Fluxes are per unit
    volume (psi~ per unit z times the height).
  - The monoenergetic source is smeared over [E0 - 0.01, E0], as in ../analytic_absorber.

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
HEIGHT = 1.0e6
G = 100
FORWARD = 0.75  # source fraction into mu > 0
N_ITERATION, PARTICLES = 20, 10
N_SMC = 1e6


def make_model():
    import mcdc

    material = mcdc.Material(nuclide_composition={"HA": 1.0}, temperature=0.1)
    simulation = common.slab([(0.0, HEIGHT, material)])
    simulation.set_sources(
        [
            common.volume_source(
                0.0,
                HEIGHT,
                common.smc_source_energy(E0 - DELTA, E0),
                [0.0, 1.0],
                FORWARD,
            ),
            common.volume_source(
                0.0,
                HEIGHT,
                common.smc_source_energy(E0 - DELTA, E0),
                [-1.0, 0.0],
                1.0 - FORWARD,
            ),
        ]
    )
    return simulation, [material]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = np.array([1.0e-3, 1.0e4])
    common.write_nuclide(
        "data", "HA", 1.0, data, SIGMA_S * np.ones(2), (SIGMA_T - SIGMA_S) * np.ones(2)
    )
    common.use_library("data")
    z_edges = [0.0, HEIGHT]
    E_edges = common.narrow_source_grid(E_MIN, E0, DELTA, G)
    Q = common.uniform_Q(
        z_edges, E_edges, [0], (E0 - DELTA, E0), (1.0 - FORWARD, FORWARD)
    )

    output = common.open_output()
    result = common.run_rmc(
        make_model, z_edges, E_edges, Q, N_ITERATION, PARTICLES / 2, cache_dir="cache"
    )
    common.save_rmc(output, "G100_P10", result, G=G, P=PARTICLES, height=HEIGHT)
    mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N_SMC)
    common.save_smc(output, "G100", mean, sdev, wall, N_SMC, height=HEIGHT)
    output.close()


if __name__ == "__main__":
    main()
