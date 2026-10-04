"""
O-16, all three windows of input.py on the linear discontinuous energy trial space
(energy_basis="linear"), same schedule as input.py with 4x its histories: the P_1
coefficients are tallied with ~sqrt(3) more noise than the averages, and at input.py's
100 P/B/I the 1-10 MeV iteration is unstable (see mcdc/rmc/NOTES.md, section 7). The
constant-basis fixed point does not depend on the histories, so the constant results
and the SMC references are read from input.py's output.h5 by plot_linear.py.

Writes output_linear.h5; then python plot_linear.py.
    mpiexec -n 4 python linear.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import CASES, N_ITERATION, PARTICLES, make_model_factory

PARTICLES_LINEAR = 4 * PARTICLES


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    common.real_library()
    output = common.open_output("output_linear.h5")
    z_edges = [0.0, 1.0]
    for name, (window, G) in CASES.items():
        E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
        Q = common.uniform_Q(z_edges, E_edges, [0], window)
        result = common.run_rmc(
            make_model_factory(window),
            z_edges,
            E_edges,
            Q,
            N_ITERATION,
            PARTICLES_LINEAR / 2,
            energy_basis="linear",
            cache_dir="cache",
        )
        common.save_rmc(output, name, result, P=PARTICLES_LINEAR)
    output.close()


if __name__ == "__main__":
    main()
