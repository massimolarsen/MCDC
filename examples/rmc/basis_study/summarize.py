"""
Summary of the basis study: for each problem, the Pareto front of the residual
||r||_1/||q||_1 against the number of unknowns per original coarse cell and bin (the
trial spaces no cheaper space beats), and the effect of each axis.
    python summarize.py            (reads results_*_r*.json; writes summary.md)
"""

import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "results_*_r*.json"))):
        with open(path) as f:
            rows += [r for r in json.load(f) if r.get("study_version") == 2]
    if not rows:
        raise ValueError("No corrected study results: regenerate cases before summarizing")
    for r in rows:
        r.setdefault("refine", 1)
        # Unknowns per ORIGINAL coarse cell and bin: refinement multiplies by r^3
        r["cost"] = r["unknowns"] * r["refine"] ** 3
        r["label"] = f"x{r['refine']} {r['energy']} {r['angular']} {r['spatial']}"
    return rows


def pareto(rows):
    front = []
    for r in sorted(rows, key=lambda r: (r["cost"], r["residual"])):
        if not front or r["residual"] < front[-1]["residual"]:
            front.append(r)
    return front


def main():
    rows = load()
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["problem"]].append(r)
    lines = []
    for problem, items in by_problem.items():
        lines.append(f"## {problem}\n")
        lines.append("Pareto front (no cheaper trial space has a smaller residual):\n")
        lines.append("| trial space (mesh, energy, angle, space) | unknowns | r/q | volume | faces | (q/r)^2 |")
        lines.append("|---|---|---|---|---|---|")
        for r in pareto(items):
            lines.append(
                f"| {r['label']} | {r['cost']:.0f} | {r['residual']:.3f} | {r['volume']:.3f} "
                f"| {r['faces']:.3f} | {1 / r['residual'] ** 2:.2f} |"
            )
        best = min(items, key=lambda r: r["residual"])
        lines.append(f"\nSmallest residual: {best['label']}, r/q = {best['residual']:.3f}, "
                     f"{best['cost']:.0f} unknowns.\n")
    with open(os.path.join(HERE, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
