"""
The A = 1 absorber sweep of input.py (same grids, histories and iterations) on the
linear discontinuous energy trial space (energy_basis="linear"). The constant-basis
sweep, the references and SMC are read from input.py's output.h5 by plot_linear.py.

Writes output_linear.h5; then python plot_linear.py.
    python linear.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import DELTA, E0, E_MIN, GRIDS, N_ITERATION, PARTICLES, make_model


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    common.use_library("data")  # written by input.py
    z_edges = [0.0, 1.0]
    output = common.open_output("output_linear.h5")
    for G in GRIDS:
        E_edges = common.narrow_source_grid(E_MIN, E0, DELTA, G)
        Q = common.uniform_Q(z_edges, E_edges, [0], (E0 - DELTA, E0))
        for P in PARTICLES:
            result = common.run_rmc(
                make_model,
                z_edges,
                E_edges,
                Q,
                N_ITERATION,
                P / 2,
                energy_basis="linear",
                cache_dir="cache",
            )
            common.save_rmc(output, f"G{G}_P{P}", result, G=G, P=P)
    output.close()


if __name__ == "__main__":
    main()
