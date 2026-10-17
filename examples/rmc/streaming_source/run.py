"""Prespecified one-group 3D benchmark matrix; writes all results, including losses."""

import argparse
import hashlib
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from solver import geometry, smc, solve


def cases(cells=2):
    if cells < 2 or cells % 2:
        raise ValueError(
            "Use an even cell count so the source and material interfaces stay fixed"
        )
    edges = (np.linspace(0.0, 2.0, cells + 1),) * 3
    _, volume = geometry(edges)
    center = 0.5 * (edges[0][:-1] + edges[0][1:])
    X, Y, Z = np.meshgrid(center, center, center, indexing="ij")
    checker = (
        (X >= 1).astype(int) + (Y >= 1).astype(int) + (Z >= 1).astype(int)
    ).ravel() % 2
    for material, sigma in (
        ("thin", np.full(cells**3, 0.3)),
        ("thick", np.full(cells**3, 3.0)),
        ("contrast", np.where(checker, 5.0, 0.1)),
    ):
        for ratio in (0.2, 0.8, 0.97):
            for source in ("uniform", "corner"):
                q = (
                    np.ones(cells**3)
                    if source == "uniform"
                    else ((X < 1) & (Y < 1) & (Z < 1)).ravel().astype(float)
                )
                q /= q @ volume
                yield f"{material}-c{ratio}-{source}", edges, sigma, ratio * sigma, q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--histories", type=int, default=4000)
    parser.add_argument("--smc-histories", type=int, default=100000)
    parser.add_argument("--inner", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--power", type=int, default=8)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument(
        "--unfitted", action="store_true", help="Q=0 uncollided/collided hybrid control"
    )
    parser.add_argument(
        "--antithetic", action="store_true", help="Pair antipodal inner ray directions"
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Reconstruct limited linear collision sources",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Forward first-flight baseline alternative",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Source-directed cone baseline alternative",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Treat localized external emission separately",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=64,
        help="Pilot histories per birth cell; zero disables",
    )
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Independent-run variance needs at least two repeats")
    if sum((args.forward, args.targeted, args.split)) > 1:
        parser.error("Choose only one baseline alternative")
    solver_digest = hashlib.sha256(
        Path(__file__).with_name("solver.py").read_bytes()
    ).hexdigest()
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in (Path(__file__), Path(__file__).with_name("solver.py")):
        content = file.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        with gzip.open(
            output_dir / f"{file.stem}-{digest[:12]}.py.gz", "wb"
        ) as archive:
            archive.write(content)
    versions = {name: version(name) for name in ("numpy", "scipy", "numba")}
    # Compile both paths before any measured case; record this cost separately.
    _, edges, sigma, scatter, q = next(cases(args.cells))
    start = time.perf_counter()
    smc(edges, sigma, scatter, q, N=2)
    solve(
        edges,
        sigma,
        scatter,
        q,
        N=2,
        inner=2,
        power=2,
        repeats=2,
        forward=args.forward,
        targeted=args.targeted,
        split=args.split,
        linear=args.linear,
        pilot=args.pilot,
        antithetic=args.antithetic,
        fit_source=not args.unfitted,
    )
    compilation = time.perf_counter() - start
    rows = []
    for label, edges, sigma, scatter, q in cases(args.cells):
        for method, inner in [("smc", 0)] + [("rmc", k) for k in args.inner]:
            runs = []
            for repeat in range(args.repeats):
                seed = args.seed + repeat * 104729
                if inner:
                    result = solve(
                        edges,
                        sigma,
                        scatter,
                        q,
                        N=args.histories,
                        inner=inner,
                        power=args.power,
                        repeats=8,
                        seed=seed,
                        forward=args.forward,
                        pilot=args.pilot,
                        targeted=args.targeted,
                        split=args.split,
                        linear=args.linear,
                        antithetic=args.antithetic,
                        fit_source=not args.unfitted,
                    )
                else:
                    result = smc(
                        edges, sigma, scatter, q, N=args.smc_histories, seed=seed
                    )
                runs.append(
                    {
                        k: v.tolist() if isinstance(v, np.ndarray) else float(v)
                        for k, v in result.items()
                    }
                )
            means = np.array([r["mean"] for r in runs])
            empirical = means.var(axis=0, ddof=1)
            reported = np.mean([r["variance"] for r in runs], axis=0)
            row = dict(
                case=label,
                method=method,
                inner=inner,
                runs=runs,
                mean=means.mean(axis=0).tolist(),
                empirical_variance=empirical.tolist(),
                reported_variance=reported.tolist(),
                time=float(np.mean([r["total"] for r in runs])),
            )
            rows.append(row)
            if inner:
                reference = next(
                    r for r in rows if r["case"] == label and r["method"] == "smc"
                )
                ref_mean = np.array(reference["mean"])
                metric = lambda v: float(np.mean(np.asarray(v) / ref_mean**2))
                row["fom_ratio_empirical"] = (
                    metric(reference["empirical_variance"])
                    * reference["time"]
                    / (metric(empirical) * row["time"])
                )
                row["fom_ratio_reported"] = (
                    metric(reference["reported_variance"])
                    * reference["time"]
                    / (metric(reported) * row["time"])
                )
                se = np.sqrt(
                    (empirical + reference["empirical_variance"]) / args.repeats
                )
                row["z_max"] = float(np.max(np.abs(means.mean(axis=0) - ref_mean) / se))
                row["coverage_ratio"] = metric(empirical) / metric(reported)
                print(
                    label,
                    "K",
                    inner,
                    "FOM",
                    round(row["fom_ratio_empirical"], 3),
                    "reported",
                    round(row["fom_ratio_reported"], 3),
                    "max z",
                    round(row["z_max"], 2),
                    flush=True,
                )
            else:
                print(label, "SMC", round(row["time"], 3), "s", flush=True)
            payload = dict(
                settings=vars(args),
                compilation=compilation,
                rows=rows,
                solver_sha256=solver_digest,
                versions=versions,
            )
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
