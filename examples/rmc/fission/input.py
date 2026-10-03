"""
Infinite, homogeneous A = 1 material with fission: sigma_s = 1, sigma_f = 1, sigma_t = 4.

Reproduces [D] Fig. 2.3 (see ../common.py for references):
  - flux.png   RMC (100 P/B/I, 10 iterations) vs SMC with 10^5 and 10^6 histories

Problem ([D] Sec. 2.4.2): Watt fission spectrum (a = 0.998 MeV, b = 2.249 /MeV), a
constant uniform source in energy, G = 100 log-spaced bins over the window of [D]
Fig. 2.3, [10^4, 10^9] eV, and 2 polar bins.

Differences from the dissertation:
  - nu = 2 (not stated in [D]; the value used for the fuel rod in [D] Sec. 4.4.2).
  - The Watt spectrum is tabulated on [10^3, 10^8] eV and sampled by MC/DC as a
    tabulated distribution; fission neutrons are all prompt.
  - The source emits one neutron in total (the normalization in [D] is not stated).

Run (writes output.h5; then python plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

E_MIN, E_MAX = 1.0e4, 1.0e9
SIGMA_S, SIGMA_F, SIGMA_T, NU = 1.0, 1.0, 4.0, 2.0
WATT_A, WATT_B = 0.998e6, 2.249e-6  # eV, 1/eV
G = 100
N_ITERATION, PARTICLES = 10, 100
SMC_HISTORIES = (1e5, 1e6)


def make_model():
    import mcdc

    material = mcdc.Material(nuclide_composition={"HF": 1.0}, temperature=0.1)
    simulation = common.slab([(0.0, 1.0, material)])
    simulation.set_sources([common.volume_source(0.0, 1.0, (E_MIN, E_MAX))])
    return simulation, [material]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = np.array([1.0e2, 1.0e10])
    common.write_nuclide(
        "data",
        "HF",
        1.0,
        data,
        SIGMA_S * np.ones(2),
        (SIGMA_T - SIGMA_S - SIGMA_F) * np.ones(2),
        fission=SIGMA_F * np.ones(2),
        nu=NU,
        spectrum=common.watt_spectrum(WATT_A, WATT_B, 1.0e3, 1.0e8),
    )
    common.use_library("data")
    z_edges = [0.0, 1.0]
    E_edges = np.logspace(np.log10(E_MIN), np.log10(E_MAX), G + 1)
    Q = common.uniform_Q(z_edges, E_edges, [0], (E_MIN, E_MAX))

    output = common.open_output()
    result = common.run_rmc(
        make_model, z_edges, E_edges, Q, N_ITERATION, PARTICLES / 2, cache_dir="cache"
    )
    common.save_rmc(output, f"G{G}_P{PARTICLES}", result)
    for i, N in enumerate(SMC_HISTORIES):
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N, seed=i + 1)
        common.save_smc(output, f"G{G}_N{int(N)}", mean, sdev, wall, N)
    output.close()


if __name__ == "__main__":
    main()
