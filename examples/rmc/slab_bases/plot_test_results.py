"""
convergence_tests.png: ||eps~|| of the collision-only iterations recorded by the unit
tests test_slab_against_smc and test_slab_against_smc_linear_z (same problem as
input.py, 200 000 SMC histories; iterations 1-6 as printed by the tests). The other
figures need output.h5 from input.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()

# (z basis, mu basis): ||eps~|| of collision-only iterations 1-6
EPSILON = {
    ("constant", "constant"): [
        2.7537,
        0.026345,
        0.050677,
        0.020198,
        0.087667,
        0.048188,
    ],
    ("constant", "linear"): [2.8096, 0.11175, 0.059762, 0.074091, 0.11921, 0.085558],
    ("linear", "constant"): [3.8329, 0.054023, 0.064207, 0.079836, 0.049012, 0.045400],
    ("linear", "linear"): [3.8514, 0.096939, 0.13018, 0.093102, 0.083513, 0.085868],
}

fig, ax = plt.subplots(figsize=(7.0, 4.5))
for (spatial, angular), eps in EPSILON.items():
    ax.plot(
        np.arange(1, len(eps) + 1),
        eps,
        marker="o",
        ms=4,
        color=common.SERIES[0 if angular == "constant" else 1],
        ls=(0, (4, 2)) if spatial == "constant" else "-",
        label=f"z {spatial}, mu {angular}",
    )
ax.set_yscale("log")
ax.set_xlabel("Collision-only iteration")
ax.set_ylabel(r"$\|\tilde\epsilon\|$")
ax.legend()
fig.tight_layout()
fig.savefig("convergence_tests.png")
