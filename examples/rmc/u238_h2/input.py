"""
U-238 absorber diluted with an H-2 scatterer, three dilutions.

Reproduces [D] Fig. 3.9 (see ../common.py for references):
  - flux.png   RMC flux (100 P/B/I, 20 iterations) for rho_s = 1/4, 2/4, 3/4

Problem ([D] Sec. 3.4.3): an infinitely massive absorber mixture: U-238 purely
absorbing (sigma_gamma, MC/DC library, ENDF/B-VIII.1 at 0.1 K) and H-2 purely
scattering (its elastic cross section, isotropic in the COM frame, A = 1.9968);
uniform isotropic source over [1, 100] eV; G = 2000 log-spaced bins; 2 polar bins.

Differences from the dissertation:
  - [D] does not define the dilution rho_s numerically; here rho_s is the H-2 atom
    fraction, with a total atom density of 1 /b-cm.
  - A deterministic reference (../common.py) is stored for each dilution.

Run with MCDC_LIB pointing to the MC/DC library (writes output.h5; then plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

WINDOW = (1.0, 100.0)
DILUTIONS = (0.25, 0.5, 0.75)
G = 2000
N_ITERATION, PARTICLES = 20, 100


def make_model_factory(rho):
    def make_model():
        import mcdc

        material = mcdc.Material(
            nuclide_composition={"UA": 1.0 - rho, "DS": rho}, temperature=0.1
        )
        simulation = common.slab([(0.0, 1.0, material)])
        simulation.set_sources([common.volume_source(0.0, 1.0, WINDOW)])
        return simulation, [material]

    return make_model


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    energy_U, sigma_gamma = common.read_xs("U238", 0.1, "capture")
    energy_D, sigma_s = common.read_xs("H2", 0.1, "elastic_scattering")
    A_U = common.read_atomic_weight_ratio("U238", 0.1)
    A_D = common.read_atomic_weight_ratio("H2", 0.1)
    keep = (energy_U > 0.5) & (energy_U < 200.0)
    energy_U, sigma_gamma = energy_U[keep], sigma_gamma[keep]
    keep = (energy_D > 0.5) & (energy_D < 200.0)
    energy_D, sigma_s = energy_D[keep], sigma_s[keep]
    zeros_U, zeros_D = np.zeros_like(energy_U), np.zeros_like(energy_D)
    common.write_nuclide("data", "UA", A_U, energy_U, zeros_U, sigma_gamma)
    common.write_nuclide("data", "DS", A_D, energy_D, sigma_s, zeros_D)
    common.use_library("data")

    output = common.open_output()
    z_edges = [0.0, 1.0]
    E_edges = np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), G + 1)
    Q = common.uniform_Q(z_edges, E_edges, [0], WINDOW)
    for rho in DILUTIONS:
        nuclides = [
            (A_U, 1.0 - rho, energy_U, zeros_U, sigma_gamma),
            (A_D, rho, energy_D, sigma_s, sigma_s),
        ]
        output.create_dataset(
            f"reference/rho{rho}",
            data=common.slowing_down_reference(E_edges, nuclides, WINDOW, h=1e-5),
        )
        result = common.run_rmc(
            make_model_factory(rho),
            z_edges,
            E_edges,
            Q,
            N_ITERATION,
            PARTICLES / 2,
            cache_dir="cache",
        )
        common.save_rmc(output, f"rho{rho}", result, rho=rho)
    output.close()


if __name__ == "__main__":
    main()
