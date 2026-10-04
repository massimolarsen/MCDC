"""
O-16, 1-10 MeV: the collision-only fixed point of input.py followed by averaged
full-residual correction passes, with the integrated (deterministic E_in) and the
pointwise scattering-correction samplers. See mcdc/rmc/NOTES.md and
mcdc/rmc/rmc_math.tex.

Writes output_corrections.h5 (input.py's output.h5 and figures are left untouched);
then python plot_corrections.py.
    python corrections.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import CASES, N_ITERATION, PARTICLES, make_model_factory

N_CORRECTION = 10
SAMPLERS = ("integrated", "pointwise")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    common.real_library()
    window, G = CASES["fast"]
    z_edges = [0.0, 1.0]
    E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
    Q = common.uniform_Q(z_edges, E_edges, [0], window)
    output = common.open_output("output_corrections.h5")
    for sampler in SAMPLERS:
        result = common.run_rmc(
            make_model_factory(window),
            z_edges,
            E_edges,
            Q,
            N_ITERATION,
            PARTICLES / 2,
            N_correction=N_CORRECTION,
            correction_sampler=sampler,
            cache_dir="cache",
        )
        common.save_rmc(output, sampler, result, sampler=sampler)
    output.close()


if __name__ == "__main__":
    main()
