"""
Basis study: for each problem, an S_N fine reference (sn3d.py) projected onto every
candidate trial space (project.py), with the L1 norm of the residual of each
projection relative to the source. See README.md and
mcdc/rmc/writeups/method/basis_study.tex.

    python run.py PROBLEM [REFINE [N]]   (absorber, thermal-lattice, thermal-isolated,
                                          fast-lattice; trial mesh refined REFINE times;
                                          reference with N fine cells per coarse cell)
Writes results_<problem>_r<REFINE>.json and results.md (all results so far).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import common  # noqa: E402
import data  # noqa: E402
import sn3d  # noqa: E402
from project import AngularBasis, Study, streaming_exact  # noqa: E402

QUADRATURE = (4, 8, 8)  # mu intervals, phi intervals, Gauss points: 2048 directions
L_SCATTER = 7
ENERGY = ("E0", "E1")
ANGULAR = ("J8p0", "J8p1", "J8p2", "J32p0", "J32p1", "J32p2", "P1", "P3", "P5", "P7")
SPATIAL = ("dg0", "dg1", "dg2", "cg1", "cg2")
# Mesh refinement: the trial mesh is the problem's coarse mesh split r times per axis
#   (the reference keeps its fine mesh); a smaller set of bases is swept at r > 1
REFINE_ANGULAR = ("J8p0", "J8p1", "J32p1", "P3", "P5")
REFINE_SPATIAL = ("dg0", "dg1", "dg2", "cg1")

PITCH, FUEL = 1.26, 0.957
GAP = 0.5 * (PITCH - FUEL)
PIN_XY = np.array([0.0, GAP, GAP + FUEL, PITCH])
PIN_Z = np.linspace(0.0, 4.0, 5)
N_UO2 = 10.2 * 0.602214 / (0.04 * 235.044 + 0.96 * 238.051 + 2 * 15.995)
UO2 = {"U235": 0.04 * N_UO2, "U238": 0.96 * N_UO2, "O16": 2 * N_UO2}
H2O = {"H1": 2 * 0.033428, "O16": 0.033428}


def refine_edges(edges, n):
    return np.concatenate(
        [np.linspace(edges[k], edges[k + 1], n + 1)[:-1] for k in range(len(edges) - 1)]
        + [edges[-1:]]
    )


def absorber(n=8):
    coarse = [np.array([0.0, 1.0, 2.0])] * 3
    fine = [refine_edges(e, n) for e in coarse]
    c = 0.5 * (fine[0][1:] + fine[0][:-1])
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    source = ((X < 1) & (Y < 1) & (Z < 1)).ravel() / (4.0 * np.pi)
    return dict(
        coarse=coarse, n=n, fine=fine, material=np.zeros(len(source), dtype=np.int64),
        Sigma_t=np.ones((1, 1)), T=np.zeros((1, L_SCATTER + 1, 1, 1)),
        E_fine=np.array([0.0, 1.0]), n_fine=1, q=source[:, None],
        bc=[[sn3d.REFLECTIVE, sn3d.VACUUM], [sn3d.VACUUM] * 2, [sn3d.VACUUM] * 2],
        energy=("E0",),
    )


def _nuclide_data(window, thermal):
    """(A, total, elastic) per nuclide name for the case's physics."""
    out = {}
    for name in ("H1", "O16", "U235", "U238"):
        if thermal:
            library = os.path.join(HERE, "..", "fuel_rod", "data")
            key, T = name + "T", 0.1
        else:
            library = common.real_library()
            key, T = name, 293.6
        A = common.read_atomic_weight_ratio(key, T, library)
        total = common.read_total_xs(key, T, library)
        elastic = common.read_xs(key, T, "elastic_scattering", library)
        out[name] = (A, total, elastic)
    return out


def pin(case, variant, G_coarse=8, n_fine=3, n=4):
    thermal = case == "thermal"
    window = (1.0, 10.0) if thermal else (3.0e6, 1.0e7)  # thermal: scaled by 1e4
    E_coarse = np.geomspace(*window, G_coarse + 1)
    E_fine = data.fine_groups(E_coarse, n_fine)
    nuclides = _nuclide_data(window, thermal)
    constants = []
    for composition in (H2O, UO2):
        items = [
            (nuclides[k][0], v, nuclides[k][1], nuclides[k][2])
            for k, v in composition.items()
        ]
        constants.append(data.group_constants(items, E_fine, L_SCATTER))
    Sigma_t = np.array([c[0] for c in constants])
    T = np.array([c[1] for c in constants])
    coarse = [PIN_XY, PIN_XY, PIN_Z]
    fine = [refine_edges(e, n) for e in coarse]
    cx = 0.5 * (fine[0][1:] + fine[0][:-1])
    cz = 0.5 * (fine[2][1:] + fine[2][:-1])
    X, Y, Z = np.meshgrid(cx, cx, cz, indexing="ij")
    in_fuel = ((X > GAP) & (X < GAP + FUEL) & (Y > GAP) & (Y < GAP + FUEL)).ravel()
    material = in_fuel.astype(np.int64)
    hx, hy, hz = (np.diff(f) for f in fine)
    V = (hx[:, None, None] * hy[None, :, None] * hz[None, None, :]).ravel()
    mask = np.ones_like(in_fuel) if thermal else in_fuel
    q_cell = mask / (np.sum(V[mask]) * (window[1] - window[0]) * 4.0 * np.pi)
    q = np.repeat(q_cell[:, None], len(E_fine) - 1, axis=1)
    if variant == "lattice":
        bc = [[sn3d.REFLECTIVE] * 2, [sn3d.REFLECTIVE] * 2, [sn3d.VACUUM] * 2]
    else:
        bc = [[sn3d.VACUUM] * 2] * 3
    return dict(coarse=coarse, n=n, fine=fine, material=material, Sigma_t=Sigma_t, T=T,
                E_fine=E_fine, n_fine=n_fine, q=q, bc=bc, energy=ENERGY)


PROBLEMS = {
    "absorber": absorber,
    "thermal-lattice": lambda n=4: pin("thermal", "lattice", n=n),
    "thermal-isolated": lambda n=4: pin("thermal", "isolated", n=n),
    "fast-lattice": lambda n=4: pin("fast", "lattice", n=n),
}


def run(name, refine=1, n=None):
    print(f"== {name}, trial mesh refined x{refine}", flush=True)
    p = PROBLEMS[name]() if n is None else PROBLEMS[name](n=n)
    quadrature = sn3d.product_quadrature(*QUADRATURE)
    start = time.time()
    psi, stats = sn3d.solve(
        dict(x=p["fine"][0], y=p["fine"][1], z=p["fine"][2], material=p["material"],
             Sigma_t=p["Sigma_t"], T=p["T"], E_fine=p["E_fine"], q=p["q"], bc=p["bc"]),
        quadrature, L_SCATTER, verbose=False,
    )
    print(f"  reference: {time.time() - start:.0f} s, iterations "
          f"{max(s[1] for s in stats)} max", flush=True)
    coarse = [refine_edges(e, refine) for e in p["coarse"]]
    study = Study(coarse, p["n"] // refine, psi, p["E_fine"], p["n_fine"], p["Sigma_t"],
                  p["T"], p["material"], p["q"], p["bc"], quadrature, L_SCATTER)
    rows = []
    angular_set = ANGULAR if refine == 1 else REFINE_ANGULAR
    spatial_set = SPATIAL if refine == 1 else REFINE_SPATIAL
    energy_set = p["energy"] if refine == 1 else ("E0",)
    for angular in angular_set:
        ang = AngularBasis(angular, *quadrature)
        for energy in energy_set:
            for spatial in spatial_set:
                start = time.time()
                volume, faces, unknowns = study.residual(energy, ang, spatial)
                rows.append(dict(problem=name, refine=refine, energy=energy, angular=angular,
                                 spatial=spatial, volume=volume, faces=faces,
                                 residual=volume + faces, unknowns=float(unknowns)))
                print(f"  {energy} {angular:6s} {spatial}: r/q = {volume + faces:.3f} "
                      f"(volume {volume:.3f}, faces {faces:.3f}), unknowns/cell/bin "
                      f"{unknowns:.0f}, {time.time() - start:.0f} s", flush=True)
    for row in rows:
        row.update(study_version=2, quadrature=QUADRATURE,
                   scattering_order=L_SCATTER, norm_evaluation="fine-cell-averages")
    with open(os.path.join(HERE, f"results_{name}_r{refine}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def run_sx(name, refine=1, n=None):
    """Streaming-exact trial functions (project.streaming_exact) for one problem."""
    print(f"== {name}, streaming-exact, trial mesh refined x{refine}", flush=True)
    p = PROBLEMS[name]() if n is None else PROBLEMS[name](n=n)
    quadrature = sn3d.product_quadrature(*QUADRATURE)
    psi, _ = sn3d.solve(
        dict(x=p["fine"][0], y=p["fine"][1], z=p["fine"][2], material=p["material"],
             Sigma_t=p["Sigma_t"], T=p["T"], E_fine=p["E_fine"], q=p["q"], bc=p["bc"]),
        quadrature, L_SCATTER, verbose=False,
    )
    coarse = [refine_edges(e, refine) for e in p["coarse"]]
    study = Study(coarse, p["n"] // refine, psi, p["E_fine"], p["n_fine"], p["Sigma_t"],
                  p["T"], p["material"], p["q"], p["bc"], quadrature, L_SCATTER)
    rows = []
    for N in (0, 1, 3):
        for spatial in ("dg0", "dg1", "dg2", "cg1"):
            start = time.time()
            e, r, unknowns = streaming_exact(study, "E0", N, spatial)
            rows.append(dict(problem=name, refine=refine, energy="E0", angular=f"SX-P{N}",
                             spatial=spatial, volume=r, faces=0.0, residual=r,
                             source_error=e, unknowns=float(unknowns)))
            print(f"  SX-P{N} {spatial}: r/q = {r:.3f} (source error {e:.3f}), "
                  f"unknowns/cell/bin {unknowns:.0f}, {time.time() - start:.0f} s", flush=True)
    for row in rows:
        row.update(study_version=2, quadrature=QUADRATURE,
                   scattering_order=L_SCATTER, residual_evaluation="direct-trial")
    with open(os.path.join(HERE, f"results_{name}_sx_r{refine}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def table(rows):
    lines = ["| problem | mesh x | energy | angle | space | ||r||/||q|| | volume | faces | gain (q/r)^2 | unknowns |",
             "|" + "---|" * 10]
    for r in sorted(rows, key=lambda r: (r["problem"], r["residual"])):
        lines.append(
            f"| {r['problem']} | {r.get('refine', 1)} | {r['energy']} | {r['angular']} | {r['spatial']} "
            f"| {r['residual']:.3f} | {r['volume']:.3f} | {r['faces']:.3f} "
            f"| {1.0 / r['residual'] ** 2:.2f} | {r['unknowns']:.0f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    os.environ.setdefault("MCDC_LIB", os.path.join(HERE, "..", "..", "..", "hdf5lib_mt5fix"))
    # python run.py NAME [REFINE [N]]: the reference has N fine cells per coarse cell
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0]
    refine = int(args[1]) if len(args) > 1 else 1
    n = int(args[2]) if len(args) > 2 else None
    if "--sx" in sys.argv:
        run_sx(name, refine, n)
    else:
        run(name, refine, n)
    all_rows = []
    import glob

    for path in sorted(glob.glob(os.path.join(HERE, "results_*_r*.json"))):
        with open(path) as f:
            all_rows += [r for r in json.load(f) if r.get("study_version") == 2]
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write(table(all_rows) + "\n")
