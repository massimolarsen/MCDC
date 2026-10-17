"""
Problem (A): the 1D fuel rod (H2O | UO2 | H2O, ../fuel_rod/input.py) built in 3D, a
check that the 3D solver reproduces the slab results.

Geometry: z in [0, 3] cm with the fuel rod's 24 z cells (H2O in [0, 1] and [2, 3],
UO2 in [1, 2]), one cell of unit width in x and y. Reflective on all six faces, as
the slab (reflective in z, and in x and y by construction).

Cases and physics as in the 1D fuel rod:
  - thermal: window [1e-4, 1e-3] eV (scaled by 10^4), G = 100, synthetic
    target-at-rest nuclides, source uniform in energy over the whole slab;
  - fast: window [3, 10] MeV, G = 40, ENDF/B-VIII.1 at 293.6 K, all fission prompt,
    source uniform in energy in the UO2.
RMC, for each spatial basis (dg1, dg2, cg1, cg2; mcdc.rmc.space3d) and patch layout
(J = 8 octants, J = 32): 10 collision-only iterations and 1 full-residual correction
pass, with 100 histories per spatial unknown, energy bin and patch per iteration.
SMC: 10^7 histories, history statistics, scalar flux per (cell, energy bin).

Run with MCDC_LIB pointing to the real library (see ../fuel_rod/HPC.md):
    mpiexec -n 16 python -m mpi4py input.py --mode=numba --no-progress_bar
Settings (environment): FUEL_ROD_3D_CASES (thermal,fast), FUEL_ROD_3D_RUNS (rmc,smc),
FUEL_ROD_3D_BASES (dg1,dg2,cg1,cg2), FUEL_ROD_3D_LAYOUTS (8), FUEL_ROD_3D_OUTPUT
(output.h5); FUEL_ROD_3D_SMOKE=1 runs a small version (G = 10, 3 iterations, 10^4 SMC
histories) to check the script.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
import common3d
from common3d import fuel_rod, setting, setting_list

EDGES = (np.array([-0.5, 0.5]), np.array([-0.5, 0.5]), fuel_rod.Z_EDGES)
NAMES = np.array(
    [[["H2O" if z < 1.0 or z >= 2.0 else "UO2" for z in fuel_rod.Z_EDGES[:-1]]]]
)
BOUNDARY = (("reflective", "reflective"),) * 3
N_ITERATION = 10


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    smoke = setting("FUEL_ROD_3D_SMOKE", "0") == "1"
    real_library = common.real_library()
    common3d.prepare_thermal_nuclides(real_library)
    output = common.open_output(
        setting("FUEL_ROD_3D_OUTPUT", "output_smoke.h5" if smoke else "output.h5")
    )
    cases = setting_list("FUEL_ROD_3D_CASES", "thermal,fast")
    runs = setting_list("FUEL_ROD_3D_RUNS", "rmc,smc")
    bases = setting_list("FUEL_ROD_3D_BASES", "dg1,dg2,cg1,cg2")
    layouts = [int(J) for J in setting_list("FUEL_ROD_3D_LAYOUTS", "8")]
    K = len(fuel_rod.Z_EDGES) - 1
    for name in cases:
        os.environ["MCDC_LIB"] = real_library
        scale, suffix, temperature, window, G = common3d.case_setup(name)
        source_cells, N_smc = fuel_rod.CASES[name][3:5]
        N_iteration = N_ITERATION
        if smoke:
            G, N_iteration, N_smc = 10, 3, 1e4
        E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
        mask = np.zeros((1, 1, K), dtype=bool)
        mask[0, 0, list(source_cells)] = True

        def make_model():
            return common3d.box_model(
                EDGES,
                NAMES,
                common3d.materials(suffix, temperature),
                BOUNDARY,
                common3d.volume_sources(EDGES, mask, window),
                f"fuel-rod-3d-{name}",
            )

        common3d.run_case(
            output, name, make_model, EDGES, E_edges, mask, window, BOUNDARY, scale,
            N_iteration, N_smc, runs, bases, layouts,
        )
    output.close()


if __name__ == "__main__":
    main()
