"""Small 0D RMC run for the MPI rank-independence test (psi saved by rank 0)."""

import os, sys, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from mpi4py import MPI
from rmc.conftest import write_synthetic_nuclide

lib, out = sys.argv[1], sys.argv[2]
if MPI.COMM_WORLD.Get_rank() == 0:
    write_synthetic_nuclide(lib, "HX", 1.0, [1e-6, 1e3], [1.0, 1.0], [1.0, 1.0])
MPI.COMM_WORLD.Barrier()
os.environ["MCDC_LIB"] = lib
import mcdc
from mcdc.rmc.driver import run

mat = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
P = []
for ax, lo, hi in (("X", -0.5, 0.5), ("Y", -0.5, 0.5), ("Z", 0.0, 1.0)):
    cls = getattr(mcdc.Surface, "Plane" + ax)
    P.append(
        (
            cls(**{ax.lower(): lo}, boundary_condition="reflective"),
            cls(**{ax.lower(): hi}, boundary_condition="reflective"),
        )
    )
(x0, x1), (y0, y1), (z0, z1) = P
sim = mcdc.Simulation("mpi")
sim.set_model([mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=mat)])
sim.set_sources([mcdc.Source(position=[0, 0, 0.5], isotropic=True, energy=10.0)])
E = np.array([1.0, 3.0, 9.9, 10.0])
mu = np.array([-1.0, 0.0, 1.0])
Q = np.zeros((1, 3, 2))
Q[0, -1, :] = 1 / (0.1 * 2)
r = run(sim, [0.0, 1.0], E, mu, Q, [mat], 3, 200, seed=11)
if MPI.COMM_WORLD.Get_rank() == 0:
    np.save(out, r.psi)
