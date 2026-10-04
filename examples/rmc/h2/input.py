"""
Infinite medium of H-2 (0D): end-to-end test of the inverted N-body phase-space kernel
(ENDF Law 6 / ACE Law 66, MT-16 (n,2n) from 3.34 MeV) and of A = 2 anisotropic
elastic scattering inside RMC, against MC/DC SMC on the same model. See README.md.

Window: 1-20 MeV, G = 100, uniform isotropic source, particles leaving it killed.
RMC: 10 collision-only iterations, then N_CORRECTION averaged full-residual correction
passes (unbiased, so RMC + corrections must agree with SMC within statistics).

Uses the regenerated library ../../../hdf5lib_mt5fix (H-2 with the N-body particle
count and total mass ratio). Run (writes output.h5; then python plot.py):
    mpiexec -n 4 python input.py --mode=numba
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common

LIBRARY = os.path.join(HERE, "..", "..", "..", "hdf5lib_mt5fix")
NUCLIDE = "H2"
TEMPERATURE = 293.6
DENSITY = 0.05
CASES = {
    # name: (window [eV], G)
    "fast": ((1.0e6, 2.0e7), 100),
}
N_ITERATION, PARTICLES, N_CORRECTION = 10, 100, 0
N_SMC = 1e6


def make_model_factory(window):
    def make_model():
        import mcdc

        material = mcdc.Material(
            nuclide_composition={NUCLIDE: DENSITY}, temperature=TEMPERATURE
        )
        simulation = common.slab([(0.0, 1.0, material)])
        simulation.set_sources([common.volume_source(0.0, 1.0, window)])
        return simulation, [material]

    return make_model


def main(cases=CASES, N_correction=N_CORRECTION, N_smc=N_SMC, path="output.h5"):
    os.chdir(HERE)
    common.use_library(LIBRARY)
    output = common.open_output(path)
    z_edges = [0.0, 1.0]
    for name, (window, G) in cases.items():
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
            N_correction=N_correction,
            cache_dir="cache",
        )
        common.save_rmc(output, name, result, P=PARTICLES)
        if N_smc:
            mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N_smc)
            common.save_smc(output, name, mean, sdev, wall, N_smc)
    output.close()


if __name__ == "__main__":
    main()
