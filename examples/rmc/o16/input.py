"""
Infinite medium of O-16 with continuous-energy ENDF data, fast energy range.

Reproduces [D] Figs. 4.1-4.4 (see ../common.py for references):
  - cross_sections.png   sigma_t, sigma_el, sigma_inl and sigma_a of O-16, 1-10 MeV
  - flux.png             1-10 MeV, G = 100: RMC (100 P/B/I, 10 iterations) vs SMC
  - flux_isolated.png    the 2.35 MeV resonance window, 2-3 MeV, G = 80
  - flux_resonances.png  the 1.65 and 1.83 MeV resonances, 1.55-2.05 MeV, G = 80
The residual-error histories of the three runs are used by ../convergence/plot.py
([D] Figs. 4.9-4.10).

Problem ([D] Sec. 4.4.1): O-16 (MC/DC library, ENDF/B-VIII.1 at 293.6 K) at
0.05 /b-cm, uniform isotropic source over each window, particles leaving the window
killed, 2 polar bins.

Differences from the dissertation:
  - Data: ENDF/B-VIII.1 at 293.6 K ([D]: ENDF/B-VII.1, temperature not stated); the
    atom density is not stated in [D].
  - SMC with 10^6 histories ([D]: 10^7), for run time on the machine used.
  - Elastic scattering uses the real anisotropic COM angular distributions ([D] used
    isotropic elastic scattering), and inelastic reactions use the laws in the data
    exactly as MC/DC samples them.
  - [D] Fig. 4.4 is captioned "around 1.4 MeV" but shows 1.55-2.05 MeV; that
    window is used. The bin counts of the two zoomed windows (80) are read off the
    figures.

Run with MCDC_LIB pointing to the MC/DC library (writes output.h5; then plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

TEMPERATURE = 293.6
DENSITY = 0.05
CASES = {
    # name: (window [eV], G)
    "fast": ((1.0e6, 1.0e7), 100),
    "isolated": ((2.0e6, 3.0e6), 80),
    "resonances": ((1.55e6, 2.05e6), 80),
}
N_ITERATION, PARTICLES = 10, 100
N_SMC = 1e6


def make_model_factory(window):
    def make_model():
        import mcdc

        material = mcdc.Material(
            nuclide_composition={"O16": DENSITY}, temperature=TEMPERATURE
        )
        simulation = common.slab([(0.0, 1.0, material)])
        simulation.set_sources([common.volume_source(0.0, 1.0, window)])
        return simulation, [material]

    return make_model


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    common.real_library()
    output = common.open_output()
    for reaction in ("elastic_scattering", "inelastic_scattering", "capture"):
        energy, xs = common.read_xs("O16", TEMPERATURE, reaction)
        output.create_dataset(f"data/{reaction}", data=xs)
    output.create_dataset("data/energy", data=energy)

    z_edges = [0.0, 1.0]
    for name, (window, G) in CASES.items():
        E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
        Q = common.uniform_Q(z_edges, E_edges, [0], window)
        make_model = make_model_factory(window)
        result = common.run_rmc(
            make_model,
            z_edges,
            E_edges,
            Q,
            N_ITERATION,
            PARTICLES / 2,
            cache_dir="cache",
        )
        common.save_rmc(output, name, result)
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N_SMC)
        common.save_smc(output, name, mean, sdev, wall, N_SMC)
    output.close()


if __name__ == "__main__":
    main()
