"""Summarize every saved matrix case, retaining losses and replicate uncertainty."""

import json
from pathlib import Path

import numpy as np


def interval(reference, candidate, seed=7):
    """Independent-run bootstrap of the empirical variance-time FOM ratio."""
    rng = np.random.default_rng(seed)
    a = np.array([r["mean"] for r in reference["runs"]])
    b = np.array([r["mean"] for r in candidate["runs"]])
    normalizer = np.array(reference["mean"]) ** 2
    ratios = []
    for _ in range(2000):
        va = a[rng.integers(len(a), size=len(a))].var(axis=0, ddof=1)
        vb = b[rng.integers(len(b), size=len(b))].var(axis=0, ddof=1)
        ratios.append(
            (va / normalizer).mean()
            * reference["time"]
            / ((vb / normalizer).mean() * candidate["time"])
        )
    return np.quantile(ratios, [0.025, 0.975])


def main():
    here = Path(__file__).resolve().parent
    lines = [
        "# Held-out 3D one-group matrix results",
        "",
        "All ratios include source construction, pilot, baseline and correction time. "
        "JIT compilation is warmed for the selected variant and recorded separately. "
        "Intervals are pointwise 95% independent-run bootstrap intervals for "
        "the empirical variance ratio at observed mean wall times; they do not "
        "include timing-system variability or simultaneous-case multiplicity.",
        "",
        "Each resolution compares RMC and SMC on identical cell responses. "
        "The refined variant uses more tally cells, so comparisons across "
        "resolutions are not a controlled change of trial space alone.",
        "",
    ]
    for path in sorted((here / "results").glob("*.json")):
        data = json.loads(path.read_text())
        if "rows" not in data:
            continue
        lines += [
            f"## {path.stem}",
            "",
            f"Settings: `{json.dumps(data['settings'], sort_keys=True)}`.",
            "",
            "| case | FOM RMC/SMC | bootstrap interval | reported-variance FOM | max agreement z | empirical/reported variance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        wins, losses = 0, 0
        for row in data["rows"]:
            if row["method"] != "rmc":
                continue
            ref = next(
                r
                for r in data["rows"]
                if r["case"] == row["case"] and r["method"] == "smc"
            )
            lo, hi = interval(ref, row)
            wins += lo > 1
            losses += hi < 1
            lines.append(
                f"| {row['case']} | {row['fom_ratio_empirical']:.3g} | {lo:.3g}–{hi:.3g} | "
                f"{row['fom_ratio_reported']:.3g} | {row['z_max']:.2f} | {row['coverage_ratio']:.2f} |"
            )
        lines += [
            "",
            f"Pointwise intervals above one: {wins}; below one: {losses}. "
            "These counts describe this matrix, not a universal efficiency claim.",
            "",
        ]
    (here / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
