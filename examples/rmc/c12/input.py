"""
Infinite medium of C-12 (0D): end-to-end test of the inverted evaporation, free-gas,
Kalbach-Mann tabulated-yield, level and anisotropic elastic kernels inside RMC, against
MC/DC SMC on the same model. See README.md.

Windows (uniform isotropic source over each window, particles leaving it killed):
  - fast:    1-30 MeV, G = 100: elastic, level inelastic (MT-51+, from 4.81 MeV),
             evaporation (MT-91 from 7.89 MeV, MT-28 from 17.3 MeV), Kalbach-Mann with a
             tabulated yield (MT-5, from 20 MeV)
  - thermal: 0.01-10 eV, G = 60: free-gas elastic scattering with upscatter (all
             energies below 400 kT = 10.1 eV at 293.6 K)

RMC: 10 collision-only iterations, then N_CORRECTION averaged full-residual correction
passes (unbiased, so RMC + corrections must agree with SMC within statistics).

Uses the regenerated library ../../../hdf5lib_mt5fix (C-12 with its MT-5 yield table).
Run (writes output.h5; then python plot.py):
    mpiexec -n 4 python input.py --mode=numba
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common

LIBRARY = os.path.join(HERE, "..", "..", "..", "hdf5lib_mt5fix")
NUCLIDE = "C12"
TEMPERATURE = 293.6
DENSITY = 0.1
CASES = {
    # name: (window [eV], G)
    "fast": ((1.0e6, 3.0e7), 100),
    "thermal": ((1.0e-2, 10.0), 60),
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
