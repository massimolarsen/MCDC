"""
Error against cost from the sweep results (results/<problem>/<config>/*.h5):
  - efficiency_<problem>.png   rms relative error of the bin-averaged scalar flux
                               against cost (CPU seconds = wall time x ranks) and
                               against histories; one point per seed, lines through
                               the seed means
  - summary.csv                per problem, configuration and level: mean error, its
                               spread over seeds, cost, histories, figure of merit
                               1/(error^2 x cost)
  - timing.csv                 per problem, configuration and level (seed means), in
                               seconds: the measured (warm) run's wall time, wall x
                               ranks, CPU summed over ranks, and for RMC its precompute
                               (cached moments), phase-1 iterations, correction passes,
                               per iteration and per pass; the cold pass's wall time and
                               the JIT estimate
  - timing_warmup.csv          per problem and RMC configuration: the transfer-moment
                               computation from scratch (wall, CPU, ranks)

The reference is analytic for the absorber and a high-statistics SMC run otherwise
(its own noise sets a floor on the measurable error). For the 1D fuel rod there are two
error measures: the cell averages (what a user reads; with the continuous linear z
basis they include the projection's in-cell error) and the interior hat-function
moments int phi h_i dz (preserved by the continuous projection).
"""

import csv
import glob
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common

from mcdc.rmc.space import mass_matrix, node_moments

os.chdir(HERE)
plt = common.style()
DMU = np.diff(common.MU_EDGES)
CONFIGS = {
    "smc": dict(color=common.MUTED, marker="s", label="SMC"),
    "dissertation": dict(
        color=common.SERIES[0], marker="o", label="RMC dissertation (constant, phase 1)"
    ),
    "current": dict(
        color=common.SERIES[1], marker="D", label="RMC current (linear + corrections)"
    ),
}


def scalar(psi):
    return np.einsum("kgj,j->kg", psi, DMU)


def hat_moments_cells(phi, z_edges):
    """int phi h_i dz of a piecewise-constant phi, interior nodes."""
    h = np.diff(z_edges)
    return node_moments(phi * h[:, None], np.zeros_like(phi))[1:-1]


def estimate(f, z_edges):
    """(cell averages phi[k, g], interior hat moments or None) of one result file."""
    one_dimensional = len(z_edges) > 2
    if "smc" in f:
        phi = scalar(f["smc/flux/mean"][()])
        hat = None
        if one_dimensional:
            h = np.diff(z_edges)[:, None]
            slope = scalar(f["smc/flux-z-slope/mean"][()])
            hat = node_moments(phi * h, slope * h)[1:-1]
        return phi, hat
    group = f["rmc"]
    psi = group["psi"][()]
    if group.attrs["spatial_basis"] == "linear":
        nodal = scalar(psi)
        return 0.5 * (nodal[:-1] + nodal[1:]), (mass_matrix(z_edges) @ nodal)[1:-1]
    phi = scalar(psi)
    return phi, hat_moments_cells(phi, z_edges) if one_dimensional else None


def histories(f):
    if "smc" in f:
        return float(f.attrs["N_particle"])
    group = f["rmc"]
    passes = len(group["epsilon_norm"]) + (
        len(group["corrections"]) if "corrections" in group else 0
    )
    return float(group.attrs["N_history"]) * passes


def rms_relative(x, reference):
    mask = np.abs(reference) > 0.0
    return float(np.sqrt(np.mean((x[mask] / reference[mask] - 1.0) ** 2)))


rows = []
for problem_dir in sorted(glob.glob("results/*")):
    problem = os.path.basename(problem_dir)
    files = glob.glob(f"{problem_dir}/*/n*_s*.h5")
    if not files:
        continue
    with h5py.File(files[0], "r") as f:
        group = f["rmc"] if "rmc" in f else None
        z_edges = group["z_edges"][()] if group is not None else None
    if z_edges is None:
        # SMC-only so far: take the grid from any RMC file of the problem
        continue

    # Reference
    reference_hat = None
    if problem == "absorber":
        with h5py.File(files[0], "r") as f:
            reference = f["analytic_reference"][()][None, :-1]  # source bin excluded
        crop = slice(0, -1)
    else:
        path = f"{problem_dir}/reference.h5"
        if not os.path.exists(path):
            print(f"{problem}: no reference.h5 yet, skipped")
            continue
        with h5py.File(path, "r") as f:
            reference, reference_hat = estimate(f, z_edges)
        crop = slice(None)

    measures = ["cell"] + (["hat"] if reference_hat is not None else [])
    fig, axes = plt.subplots(
        len(measures), 2, figsize=(11.0, 4.2 * len(measures)), squeeze=False
    )
    for config, style in CONFIGS.items():
        points = []
        for path in sorted(glob.glob(f"{problem_dir}/{config}/n*_s*.h5")):
            with h5py.File(path, "r") as f:
                phi, hat = estimate(f, z_edges)
                cost = float(f.attrs["wall"]) * float(f.attrs["ranks"])
                points.append(
                    dict(
                        level=float(f.attrs["level"]),
                        seed=int(f.attrs["seed"]),
                        cost=cost,
                        histories=histories(f),
                        cell=rms_relative(phi[:, crop], reference),
                        hat=(
                            rms_relative(hat, reference_hat)
                            if reference_hat is not None
                            else np.nan
                        ),
                    )
                )
        if not points:
            continue
        levels = sorted({p["level"] for p in points})
        for row, measure in enumerate(measures):
            for column, x_key in enumerate(("cost", "histories")):
                ax = axes[row, column]
                ax.plot(
                    [p[x_key] for p in points],
                    [p[measure] for p in points],
                    ls="none",
                    marker=style["marker"],
                    ms=3,
                    color=style["color"],
                    alpha=0.5,
                )
                means = [
                    (
                        np.mean([p[x_key] for p in points if p["level"] == level]),
                        np.mean([p[measure] for p in points if p["level"] == level]),
                    )
                    for level in levels
                ]
                ax.plot(*zip(*means), color=style["color"], label=style["label"])
        for level in levels:
            at = [p for p in points if p["level"] == level]
            error = np.mean([p["cell"] for p in at])
            cost = np.mean([p["cost"] for p in at])
            rows.append(
                dict(
                    problem=problem,
                    config=config,
                    level=level,
                    seeds=len(at),
                    error_cell=error,
                    error_cell_spread=np.std([p["cell"] for p in at]),
                    error_hat=np.mean([p["hat"] for p in at]),
                    cost_cpu_s=cost,
                    histories=np.mean([p["histories"] for p in at]),
                    fom=1.0 / (error**2 * cost) if error > 0 and cost > 0 else np.nan,
                )
            )
    for row, measure in enumerate(measures):
        for column, x_label in enumerate(("cost (CPU s)", "histories")):
            ax = axes[row, column]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(x_label)
            ax.set_ylabel(
                "rms rel. error, "
                + ("cell averages" if measure == "cell" else "interior hat moments")
            )
            ax.set_title(problem, fontsize=9)
        axes[row, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"efficiency_{problem}.png")
    print(f"{problem}: efficiency_{problem}.png")

with open("summary.csv", "w", newline="") as handle:
    if rows:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(f"summary.csv: {len(rows)} rows")


# ======================================================================================
# Timing tables
# ======================================================================================

TIMES = (
    "wall",
    "cpu",
    "precompute",
    "iterations",
    "corrections",
    "cold_wall",
    "cold_cpu",
    "jit_estimate",
)
timing_rows = []
for path in sorted(glob.glob("results/*/*/n*_s*.h5")):
    with h5py.File(path, "r") as f:
        if "timing" not in f:
            continue
        timing = f["timing"].attrs
        row = dict(
            problem=f.attrs["problem"],
            config=f.attrs["config"],
            level=float(f.attrs["level"]),
            ranks=int(f.attrs["ranks"]),
        )
        for key in TIMES:
            row[key] = float(timing[key]) if key in timing else np.nan
        row["wall_x_ranks"] = row["wall"] * row["ranks"]
        if "rmc" in f:
            N_it = len(f["rmc/epsilon_norm"])
            N_corr = len(f["rmc/corrections"]) if "corrections" in f["rmc"] else 0
            row["per_iteration"] = row["iterations"] / N_it
            row["per_correction"] = row["corrections"] / N_corr if N_corr else np.nan
        timing_rows.append(row)

columns = (
    ["problem", "config", "level", "ranks", "seeds"]
    + list(TIMES)
    + ["wall_x_ranks", "per_iteration", "per_correction"]
)
groups = {}
for row in timing_rows:
    groups.setdefault((row["problem"], row["config"], row["level"]), []).append(row)
with open("timing.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    for (problem, config, level), group in sorted(groups.items()):
        out = dict(
            problem=problem,
            config=config,
            level=level,
            ranks=group[0]["ranks"],
            seeds=len(group),
        )
        for key in columns[5:]:
            out[key] = float(np.nanmean([g.get(key, np.nan) for g in group]))
        writer.writerow(out)
print(f"timing.csv: {len(groups)} rows")

with open("timing_warmup.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["problem", "config", "ranks", "moments_computed", "precompute", "wall", "cpu"]
    )
    for path in sorted(glob.glob("results/*/*/warmup.h5")):
        with h5py.File(path, "r") as f:
            parts = path.split(os.sep)
            writer.writerow(
                [
                    parts[-3],
                    parts[-2],
                    int(f.attrs["ranks"]),
                    bool(f.attrs["moments_computed"]),
                    float(f.attrs["precompute"]),
                    float(f.attrs["wall"]),
                    float(f.attrs["cpu"]),
                ]
            )
print("timing_warmup.csv written")
