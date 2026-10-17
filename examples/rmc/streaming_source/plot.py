"""Show every case and algorithm variant; no favorable-case filtering."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main():
    here = Path(__file__).resolve().parent
    names = ["unfitted", "flat", "antipodal", "linear", "refined"]
    datasets = [
        json.loads((here / "results" / (n + ".json")).read_text()) for n in names
    ]
    cases = [r["case"] for r in datasets[0]["rows"] if r["method"] == "rmc"]
    values = np.array(
        [
            [
                next(
                    r["fom_ratio_empirical"]
                    for r in d["rows"]
                    if r["case"] == c and r["method"] == "rmc"
                )
                for d in datasets
            ]
            for c in cases
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(
        np.log10(values), cmap="RdBu", norm=TwoSlopeNorm(0, -1.7, 2.0), aspect="auto"
    )
    ax.set_xticks(
        range(len(names)),
        [
            "Unfitted hybrid",
            "Flat source",
            "Antipodal rays",
            "Linear source",
            "Refined mesh*",
        ],
    )
    ax.set_yticks(range(len(cases)), cases)
    for (i, j), value in np.ndenumerate(values):
        ax.text(
            j,
            i,
            f"{value:.2g}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if value < 0.1 or value > 15 else "black",
        )
    for y in (5.5, 11.5):
        ax.axhline(y, color="black", lw=1)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    ticks = np.array([0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    colorbar.set_ticks(np.log10(ticks), labels=[f"{v:g}" for v in ticks])
    colorbar.set_label("Empirical FOM / analog SMC (total cost)")
    ax.set_title(
        "3D one-group research matrix: all 18 cases\n24 independent runs per case and method"
    )
    fig.text(
        0.5,
        0.015,
        "Blue: faster at equal variance. Red: slower. Pointwise uncertainty intervals in results.md.\n"
        "*Refined scores 64 cells; other variants score 8. This is not a production MC/DC or CE benchmark.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(here / "fom_matrix.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
