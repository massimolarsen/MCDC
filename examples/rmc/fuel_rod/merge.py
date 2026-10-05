"""
Merge the fuel rod pieces written by submit.sh's jobs, then make the figures:
  - output.h5         gets rmc/fast (output_fast.h5) and smc/fast (output_smc_fast.h5)
                      when it does not have them; the first merge keeps a copy of the
                      original as output_constant_thermal.h5
  - output_linear.h5  rebuilt from output_linear_thermal.h5 and output_linear_fast.h5
Missing pieces (a failed job) are reported and skipped. Then runs plot.py (constant vs
SMC) and plot_linear.py (constant vs linear vs SMC) for the cases present.
    python merge.py
"""

import os
import shutil
import subprocess
import sys

import h5py

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def copy_groups(source, target):
    """Copy rmc/<case> and smc/<case> groups of source into target if absent."""
    if not os.path.exists(source):
        print(f"{source}: missing, skipped")
        return
    with h5py.File(source, "r") as f, h5py.File(target, "a") as out:
        for kind in ("rmc", "smc"):
            for name in f.get(kind, {}):
                path = f"{kind}/{name}"
                if path in out:
                    print(f"{target}: {path} already present")
                    continue
                f.copy(f[path], out.require_group(kind), name)
                print(f"{target}: {path} <- {source}")


if os.path.exists("output.h5") and not os.path.exists("output_constant_thermal.h5"):
    shutil.copy2("output.h5", "output_constant_thermal.h5")
for part in ("output_fast.h5", "output_smc_fast.h5"):
    copy_groups(part, "output.h5")

if os.path.exists("output_linear.h5"):
    os.remove("output_linear.h5")
for part in ("output_linear_thermal.h5", "output_linear_fast.h5"):
    copy_groups(part, "output_linear.h5")

for script in ("plot.py", "plot_linear.py"):
    status = subprocess.run([sys.executable, script]).returncode
    print(f"{script}: {'done' if status == 0 else f'failed ({status})'}")
