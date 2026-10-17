"""
Figures and figure-of-merit table of problem (A) from output.h5 (input.py), per run
(case / basis-layout): convergence_*.png, flux_*.png, compare_<case>.png and fom.md
(common3d.plot_problem). The 1D fuel rod's results are in ../fuel_rod.
    python plot.py [output.h5]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common3d
from common3d import fuel_rod

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    z = 0.5 * (fuel_rod.Z_EDGES[1:] + fuel_rod.Z_EDGES[:-1])
    water = ((z < 1.0) | (z > 2.0))[None, None, :]
    common3d.plot_problem(
        sys.argv[1] if len(sys.argv) > 1 else "output.h5",
        {"H2O": water, "UO2": ~water},
        "fuel rod 3D",
    )
