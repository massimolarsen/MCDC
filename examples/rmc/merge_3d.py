"""
Merge the per-job output files of the 3D fuel problems (<problem>/output_<case>_J<J>.h5)
into each problem's output.h5 and make its figures and fom.md (run by submit_3d.sh
after the jobs end).
    python merge_3d.py
"""

import glob
import os
import subprocess
import sys

import h5py

HERE = os.path.dirname(os.path.abspath(__file__))


def copy(source, target):
    """Copy every dataset group of source into target, creating parent groups."""
    def visit(name, obj):
        if isinstance(obj, h5py.Group) and any(
            isinstance(v, h5py.Dataset) for v in obj.values()
        ):
            parent = target.require_group(os.path.dirname(name)) if "/" in name else target
            if os.path.basename(name) not in parent:
                source.copy(obj, parent, os.path.basename(name))

    source.visititems(visit)


for problem in ("fuel_rod_3d", "pin_cell_3d"):
    directory = os.path.join(HERE, problem)
    parts = sorted(glob.glob(os.path.join(directory, "output_*_J*.h5")))
    if not parts:
        continue
    with h5py.File(os.path.join(directory, "output.h5"), "w") as out:
        for part in parts:
            with h5py.File(part, "r") as f:
                copy(f, out)
    subprocess.run([sys.executable, "plot.py", "output.h5"], cwd=directory, check=True)
