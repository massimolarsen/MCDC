"""
Merge the fuel rod pieces written by submit.sh's jobs, then make the figures:
  - output.h5         gets rmc/fast (output_fast.h5) and smc/fast (output_smc_fast.h5)
                      when it does not have them; the first merge keeps a copy of the
                      original as output_constant_thermal.h5
  - output_linear.h5  rebuilt from output_linear_thermal.h5 and output_linear_fast.h5
Missing pieces (a failed job) are reported and skipped. Then runs plot.py (constant vs
SMC) and plot_linear.py (constant vs linear vs SMC) for the cases present, and writes
  - logs/timing_summary.txt   per run: ranks, nodes, wall and CPU (summed over ranks),
                              RMC precompute (and the moment files it computed),
                              iteration 1 (with compilation) and the mean of the
                              others; SMC cold pass and compilation estimate
  - logs/timing_sacct.txt     Slurm accounting of the fuel-* jobs (elapsed, CPU,
                              peak memory, nodes), when sacct is available
    python merge.py
"""

import os
import shutil
from datetime import date, timedelta
import subprocess
import sys

import h5py
import numpy as np

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


def timing_summary():
    lines = []
    for path in ("output.h5", "output_linear.h5"):
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for kind in ("rmc", "smc"):
                for name in f.get(kind, {}):
                    group = f[f"{kind}/{name}"]
                    a = dict(group.attrs)
                    basis = a.get("energy_basis", "")
                    line = f"{path:17s} {kind}/{name:8s} {basis:9s}"
                    if "wall_total" in a:
                        line += (
                            f" ranks {a['ranks']:3d}  wall {a['wall_total']:9.1f} s"
                            f"  cpu {a['cpu_total']:10.1f} s  nodes {a['nodes']}"
                        )
                    else:
                        line += " (no total timing: run before it was recorded)"
                    if kind == "rmc":
                        t = group["time_iteration"][()]
                        line += (
                            f"\n{'':37s} precompute {a['time_precompute']:8.1f} s"
                            f"  iteration 1 {t[0]:8.1f} s"
                            f"  others {np.mean(t[1:]):8.1f} s each"
                            f"  moments computed: {a.get('new_cache_files', '?') or 'none'}"
                        )
                    else:
                        line += f"\n{'':37s} N {a['N_particle']:.3g}  time {a['time']:.1f} s"
                        if "cold_wall" in a:
                            line += (
                                f"  cold pass {a['cold_wall']:.1f} s"
                                f"  compilation ~{a['jit_estimate']:.1f} s"
                            )
                    lines.append(line)
    with open("logs/timing_summary.txt", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def sacct():
    names = ",".join(
        f"fuel-{n}"
        for n in (
            "constant-thermal",
            "constant-fast",
            "smc-fast",
            "linear-thermal",
            "linear-fast",
            "merge",
        )
    )
    start = (date.today() - timedelta(days=14)).isoformat()
    command = [
        "sacct",
        f"--name={names}",
        f"--starttime={start}",
        "--parsable2",
        "--format=JobID,JobName%30,State,Start,End,Elapsed,NTasks,TotalCPU,"
        "CPUTime,MaxRSS,NodeList",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"sacct: not available ({error})")
        return
    with open("logs/timing_sacct.txt", "w") as handle:
        handle.write(result.stdout)
    print(f"logs/timing_sacct.txt: {len(result.stdout.splitlines()) - 1} rows")


os.makedirs("logs", exist_ok=True)
timing_summary()
sacct()
