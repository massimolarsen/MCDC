"""
Why do the localized-source and high-scattering cases lose? Separate the two possible
causes of a large residual-weight second moment E[w^2]:
  1. representation: a flat collision source per cell cannot follow the true one;
  2. fit noise: Q solves (I - c P) Q = c P q with a sampled first-flight matrix P, and
     errors in P are amplified by about 1/(1 - c).
For each case, E[w^2] of the correction with
  - Q_fit: the solver's fit (Sobol power as in the matrix runs, and a finer one);
  - Q_ref: Sigma_s times the cell-average flux of a long SMC run (the ideal flat source).
If Q_ref gives a small E[w^2] where Q_fit does not, the fit is the problem.
    python diagnose_source.py [--cells 2]
"""

import argparse

import numpy as np

from run import cases
from solver import first_flight_matrix, geometry, histories, smc, sobol_cells

CASES = (
    "thin-c0.8-uniform",
    "thick-c0.8-uniform",
    "thick-c0.8-corner",
    "thick-c0.97-uniform",
    "thick-c0.97-corner",
    "contrast-c0.97-corner",
)


def fitted_source(edges, sigma, scatter, q, volume, power, seed):
    P = first_flight_matrix(sobol_cells(len(q), power, seed), edges, sigma, volume)
    return np.linalg.solve(np.eye(len(q)) - scatter[:, None] * P, scatter * (P @ q))


def weight_second(edges, sigma, scatter, q, Q, volume, N, inner, seed):
    probability = volume / volume.sum()
    _, _, w2 = histories(edges, sigma, scatter, q, Q, volume, probability, N, inner, seed)
    return w2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--smc", type=int, default=2_000_000)
    args = parser.parse_args()
    for name, edges, sigma, scatter, q in cases(args.cells):
        if name not in CASES:
            continue
        edges, volume = geometry(edges)
        reference = smc(edges, sigma, scatter, q, N=args.smc, seed=7)
        Q_ref = scatter * reference["mean"]
        rows = {}
        for label, Q in (
            ("Q_fit p10", fitted_source(edges, sigma, scatter, q, volume, 10, 11)),
            ("Q_fit p14", fitted_source(edges, sigma, scatter, q, volume, 14, 12)),
            ("Q_ref", Q_ref),
            ("Q=0", np.zeros_like(q)),
        ):
            rows[label] = (
                weight_second(edges, sigma, scatter, q, Q, volume, 4000, 64, 3),
                np.max(np.abs(Q / np.maximum(Q_ref, 1e-300) - 1)),
            )
        print(name, "  ".join(f"{k}: w2 {v[0]:.4f} (max |Q/Q_ref-1| {v[1]:.3f})"
                              for k, v in rows.items()), flush=True)


if __name__ == "__main__":
    main()
