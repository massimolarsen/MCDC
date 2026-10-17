"""
Problem (B): a C5G7-style fuel pin in 3D, fixed source, on the 3D Cartesian RMC
solver (mcdc.rmc.run3d) against SMC.

Geometry: square pitch 1.26 cm; a square UO2 region of side 0.957 cm at the center
(no cladding), surrounded by water; 4 cm tall. The trial mesh follows the material
boundaries: 3 x 3 in x-y (water 0.1515 | fuel 0.957 | water 0.1515 cm) and 4 cells of
1 cm in z, 36 cells.

Variants (boundary conditions):
  - lattice: reflective in x and y (an infinite lattice of pins), vacuum in z;
  - isolated: vacuum on all six faces.
Cases and physics as in the 1D fuel rod (../fuel_rod/input.py):
  - thermal: window [1e-4, 1e-3] eV (scaled by 10^4), G = 100, synthetic
    target-at-rest nuclides, source uniform over the whole pin;
  - fast: window [3, 10] MeV, G = 40, ENDF/B-VIII.1 at 293.6 K, all fission prompt,
    source in the UO2.
In both, the source is uniform in energy and isotropic. The windows keep the problem
subcritical: in the thermal one fission counts as capture, and in the fast one only
neutrons born in 3-10 MeV stay.

RMC, for each spatial basis (dg1, dg2, cg1, cg2; mcdc.rmc.space3d) and patch layout
(J = 8 octants, J = 32): 10 collision-only iterations and 1 full-residual correction
pass, with 100 histories per spatial unknown, energy bin and patch per iteration.
SMC: 10^7 histories, history statistics, scalar flux per (cell, energy bin).

Run with MCDC_LIB pointing to the real library (see ../fuel_rod/HPC.md):
    mpiexec -n 16 python -m mpi4py input.py --mode=numba --no-progress_bar
Settings (environment): PIN3D_CASES (thermal,fast), PIN3D_VARIANTS (lattice,isolated),
PIN3D_RUNS (rmc,smc), PIN3D_BASES (dg1,dg2,cg1,cg2), PIN3D_LAYOUTS (8), PIN3D_OUTPUT
(output.h5); PIN3D_SMOKE=1 runs a small version (G = 10, 3 iterations, 10^4 SMC
histories) to check the script.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
import common3d
from common3d import fuel_rod, setting, setting_list

PITCH = 1.26
FUEL = 0.957
GAP = 0.5 * (PITCH - FUEL)
XY = np.array([0.0, GAP, GAP + FUEL, PITCH])
Z = np.linspace(0.0, 4.0, 5)
EDGES = (XY, XY, Z)
NAMES = np.full((3, 3, 4), "H2O", dtype=object)
NAMES[1, 1, :] = "UO2"
VARIANTS = {
    "lattice": (("reflective", "reflective"),) * 2 + (("vacuum", "vacuum"),),
    "isolated": (("vacuum", "vacuum"),) * 3,
}
N_ITERATION = 10


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    smoke = setting("PIN3D_SMOKE", "0") == "1"
    real_library = common.real_library()
    common3d.prepare_thermal_nuclides(real_library)
    output = common.open_output(
        setting("PIN3D_OUTPUT", "output_smoke.h5" if smoke else "output.h5")
    )
    cases = setting_list("PIN3D_CASES", "thermal,fast")
    variants = setting_list("PIN3D_VARIANTS", "lattice,isolated")
    runs = setting_list("PIN3D_RUNS", "rmc,smc")
    bases = setting_list("PIN3D_BASES", "dg1,dg2,cg1,cg2")
    layouts = [int(J) for J in setting_list("PIN3D_LAYOUTS", "8")]
    for name in cases:
        for variant in variants:
            os.environ["MCDC_LIB"] = real_library
            scale, suffix, temperature, window, G = common3d.case_setup(name)
            N_smc = fuel_rod.CASES[name][4]
            N_iteration = N_ITERATION
            if smoke:
                G, N_iteration, N_smc = 10, 3, 1e4
            E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
            mask = np.ones(NAMES.shape, dtype=bool)
            if name == "fast":
                mask = NAMES == "UO2"
            boundary = VARIANTS[variant]

            def make_model():
                return common3d.box_model(
                    EDGES,
                    NAMES,
                    common3d.materials(suffix, temperature),
                    boundary,
                    common3d.volume_sources(EDGES, mask, window),
                    f"pin-cell-3d-{name}-{variant}",
                )

            common3d.run_case(
                output, f"{name}-{variant}", make_model, EDGES, E_edges, mask, window,
                boundary, scale, N_iteration, N_smc, runs, bases, layouts,
            )
    output.close()


if __name__ == "__main__":
    main()
