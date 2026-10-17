"""
Figures and figure-of-merit table of problem (B) from output.h5 (input.py), per run
(case-variant / basis-layout): convergence_*.png, flux_*.png, compare_<case>.png and
fom.md (common3d.plot_problem).
    python plot.py [output.h5]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common3d

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fuel = np.zeros((3, 3, 4), dtype=bool)
    fuel[1, 1] = True
    corner = np.zeros((3, 3, 4), dtype=bool)
    corner[0, 0] = True
    common3d.plot_problem(
        sys.argv[1] if len(sys.argv) > 1 else "output.h5",
        {"fuel": fuel, "corner water": corner},
        "pin cell 3D",
    )
