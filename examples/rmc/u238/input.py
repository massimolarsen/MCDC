"""
Infinite, homogeneous U-238 capture resonances with sigma_s = sigma_gamma / 4.

Reproduces [D] Figs. 3.7-3.8 (see ../common.py for references):
  - flux_xs.png     RMC flux (100 P/B/I, 20 iterations) and sigma_t(E), 1-100 eV
  - flux_zoom.png   RMC vs SMC (10^6 and 10^7 histories) for E in (1, 4) eV

Problem ([D] Sec. 3.4.3): sigma_gamma(E) of U-238 (MC/DC library, ENDF/B-VIII.1 at
0.1 K); sigma_s = sigma_gamma / 4, isotropic elastic in the COM frame (A = 236.006),
sigma_c = sigma_gamma; uniform isotropic source over [1, 100] eV; G = 2000 log-spaced
bins; 2 polar bins. Atom density 1 /b-cm.

Differences from the dissertation: ENDF/B-VIII.1 at 0.1 K ([D]: ENDF/B-VIII.0,
temperature not stated); SMC with 10^6 and 10^7 histories ([D]: 10^7 and 10^8), for
run time on the machine used; a deterministic reference (../common.py) is included.
The atom density is not stated in [D].

Run with MCDC_LIB pointing to the MC/DC library (writes output.h5; then plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

WINDOW = (1.0, 100.0)
DENSITY = 1.0
G = 2000
N_ITERATION, PARTICLES = 20, 100
SMC_HISTORIES = (1e6, 1e7)


def make_model():
    import mcdc

    material = mcdc.Material(nuclide_composition={"UG": DENSITY}, temperature=0.1)
    simulation = common.slab([(0.0, 1.0, material)])
    simulation.set_sources([common.volume_source(0.0, 1.0, WINDOW)])
    return simulation, [material]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    energy, sigma_gamma = common.read_xs("U238", 0.1, "capture")
    A = common.read_atomic_weight_ratio("U238", 0.1)
    keep = (energy > 0.5) & (energy < 200.0)
    energy, sigma_gamma = energy[keep], sigma_gamma[keep]
    common.write_nuclide("data", "UG", A, energy, sigma_gamma / 4, sigma_gamma)
    common.use_library("data")

    output = common.open_output()
    output.create_dataset("data/energy", data=energy)
    output.create_dataset("data/sigma_t", data=1.25 * sigma_gamma)
    z_edges = [0.0, 1.0]
    E_edges = np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), G + 1)
    output.create_dataset(
        "reference",
        data=common.slowing_down_reference(
            E_edges,
            [(A, DENSITY, energy, sigma_gamma / 4, 1.25 * sigma_gamma)],
            WINDOW,
            h=1e-5,
        ),
    )
    Q = common.uniform_Q(z_edges, E_edges, [0], WINDOW)
    result = common.run_rmc(
        make_model, z_edges, E_edges, Q, N_ITERATION, PARTICLES / 2, cache_dir="cache"
    )
    common.save_rmc(output, "G2000_P100", result)
    for i, N in enumerate(SMC_HISTORIES):
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N, seed=i + 1)
        common.save_smc(output, f"N{int(N)}", mean, sdev, wall, N)
    output.close()


if __name__ == "__main__":
    main()
