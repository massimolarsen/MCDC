"""
1D fuel rod, both cases of input.py on the all-linear trial space: linear
discontinuous in energy (energy_basis="linear") and polar cosine
(angular_basis="linear"), continuous linear in z (spatial_basis="linear"). Same grids,
sources and iterations as input.py, with LINEAR_FACTOR x its histories: each slope
coefficient is tallied with ~sqrt(3) more noise than an average, and with 8
coefficients per bin the collision-only iteration needs more histories to stay stable
(see mcdc/rmc/NOTES.md, sections 7, 9 and 10). The constant-basis results and the SMC
references are read from input.py's output.h5 by plot_linear.py.

With the linear z basis, psi~ holds nodal values on Z_EDGES (K + 1) and psi_cell the
cell averages. The continuous-linear projection does not preserve cell averages, so
psi_cell differs from SMC's cell tallies by more than noise; that is not an error.

Writes output_linear.h5; then python plot_linear.py. See HPC.md for cluster runs.
    mpiexec -n 8 python linear.py --mode=numba
FUEL_ROD_CASES=thermal (or fast) runs one case; FUEL_ROD_OUTPUT names the output file;
FUEL_ROD_LINEAR_FACTOR overrides the history multiplier (default 4).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import (
    CASES,
    N_ITERATION,
    SCALE_THERMAL,
    Z_EDGES,
    make_model_factory,
    write_thermal_nuclides,
)

LINEAR_FACTOR = float(os.environ.get("FUEL_ROD_LINEAR_FACTOR", "4"))


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    real_library = common.real_library()
    if common.is_master():
        write_thermal_nuclides()
    common.barrier()
    output = common.open_output(os.environ.get("FUEL_ROD_OUTPUT", "output_linear.h5"))
    cases = os.environ.get("FUEL_ROD_CASES", "thermal,fast").split(",")
    for name, (window, G, P, source_cells, _) in CASES.items():
        if name not in cases:
            continue
        if name == "thermal":
            common.use_library("data")
            scale, suffix, temperature = SCALE_THERMAL, "T", 0.1
        else:
            common.use_library(real_library)
            scale, suffix, temperature = 1.0, "", 293.6
        window = (window[0] * scale, window[1] * scale)
        E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
        K, J = len(Z_EDGES) - 1, len(common.MU_EDGES) - 1
        Q = common.uniform_Q(Z_EDGES, E_edges, list(source_cells), window)
        P_linear = LINEAR_FACTOR * P
        result = common.run_rmc(
            make_model_factory(window, suffix, temperature, source_cells),
            Z_EDGES,
            E_edges,
            Q,
            N_ITERATION,
            P_linear / (K * J),
            energy_basis="linear",
            spatial_basis="linear",
            angular_basis="linear",
            cache_dir="cache",
        )
        common.save_rmc(output, name, result, energy_scale=scale, P=P_linear)
    output.close()


if __name__ == "__main__":
    main()
