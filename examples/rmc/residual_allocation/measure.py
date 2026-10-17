"""
Correction-pass variance with a fixed history split (collision_fraction = 0.9) and with
the split proportional to the residual terms' L1 masses (writeups/residual_allocation.tex).

A = 1 slowing down with absorption (sigma_s = sigma_c = 1 b, c = 1/2), monoenergetic
source at E0 = 101 smeared over a narrow top bin, reflective box (an infinite medium),
G = 20 collided bins: the problem of test_analytic_slowing_down_A1, with its analytic
flux. One RMC solver runs the collision-only iterations to the binned fixed point; then,
from that same fixed point, N_PASS correction passes are run with each allocation. For
each allocation: the per-bin variance of the scalar-flux correction over the passes
(summed over bins), the error of fixed point + mean correction against the analytic
flux, and the histories each term received.
Writes results.txt.
    python measure.py --mode=numba --no-progress_bar
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "test", "unit", "rmc"))
from conftest import write_synthetic_nuclide

N_ITERATION, N_PASS, N_PER_BIN = 8, 30, 500


def main():
    os.chdir(HERE)
    os.makedirs("data", exist_ok=True)
    E0, delta, sigma_s, sigma_c = 101.0, 0.1, 1.0, 1.0
    write_synthetic_nuclide(
        "data", "HX", 1.0, [1.0e-5, 1.0e3], [sigma_s] * 2, [sigma_c] * 2
    )
    os.environ["MCDC_LIB"] = os.path.join(HERE, "data")

    import mcdc
    from mcdc.rmc.driver import RMCSolver

    def reflective_box(material):
        """Unit cell, reflective on all six faces: an infinite medium."""
        bc = "reflective"
        x0, x1 = mcdc.Surface.PlaneX(
            x=-0.5, boundary_condition=bc
        ), mcdc.Surface.PlaneX(x=0.5, boundary_condition=bc)
        y0, y1 = mcdc.Surface.PlaneY(
            y=-0.5, boundary_condition=bc
        ), mcdc.Surface.PlaneY(y=0.5, boundary_condition=bc)
        z0, z1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition=bc), mcdc.Surface.PlaneZ(
            z=1.0, boundary_condition=bc
        )
        return mcdc.Cell(+x0 & -x1 & +y0 & -y1 & +z0 & -z1, fill=material)

    material = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1)
    simulation = mcdc.Simulation("residual-allocation")
    simulation.set_model([reflective_box(material)])
    simulation.set_sources(
        [mcdc.Source(position=[0.0, 0.0, 0.5], isotropic=True, energy=E0)]
    )
    E_edges = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 21), [E0]))
    mu_edges = np.array([-1.0, 0.0, 1.0])
    G = len(E_edges) - 1
    Q = np.zeros((1, G, 2))
    Q[0, -1, :] = 1.0 / (delta * 2.0)
    solver = RMCSolver(
        simulation,
        [0.0, 1.0],
        E_edges,
        mu_edges,
        Q,
        [material],
        N_PER_BIN,
        cache_dir="cache",
    )

    Sigma_t, c = sigma_s + sigma_c, sigma_s / (sigma_s + sigma_c)
    a, b = E_edges[:-2], E_edges[1:-1]
    exact = (
        c / (Sigma_t * E0) * E0**c * (b ** (1 - c) - a ** (1 - c)) / (1 - c) / (b - a)
    )
    dmu = np.diff(mu_edges)

    def flux(psi):
        return np.einsum("kgj,j->kg", psi[..., 0], dmu)[0, :-1]

    psi = np.zeros(solver.shape)
    for n in range(N_ITERATION):
        psi = psi + solver.iterate(psi, n, corrections=False)
    lines = [
        f"A = 1, c = 1/2, G = {G - 1} collided bins, {solver.N_total} histories per pass, "
        f"{N_PASS} correction passes per allocation, from the same collision-only fixed point",
        f"fixed point: max relative error vs analytic {np.max(np.abs(flux(psi) / exact - 1)):.2e}",
        "",
    ]
    results = {}
    for allocation in ("fixed", "mass"):
        solver.correction_allocation = allocation
        passes, splits = [], []
        for m in range(N_PASS):
            passes.append(flux(solver.iterate(psi, 1000 + m)))
            splits.append(solver.last_allocation)
        passes = np.array(passes)
        mean = passes.mean(axis=0)
        variance = passes.var(axis=0, ddof=1)
        final = flux(psi) + mean
        results[allocation] = (mean, variance)
        split = np.mean(splits, axis=0)
        lines.append(
            f"{allocation:5s}: histories r_c+r_e / r_s / r_f = "
            f"{split[0]:.0f} / {split[1]:.0f} / {split[2]:.0f};  "
            f"sum of per-bin variance {variance.sum():.3e};  "
            f"rms relative error of fixed point + mean correction "
            f"{np.sqrt(np.mean((final / exact - 1) ** 2)):.2e}"
        )
    (m_f, v_f), (m_m, v_m) = results["fixed"], results["mass"]
    z = (m_f - m_m) / np.sqrt((v_f + v_m) / N_PASS)
    lines += [
        "",
        f"variance ratio fixed / mass (summed over bins): {v_f.sum() / v_m.sum():.2f}",
        f"per-bin variance ratio: median {np.median(v_f / v_m):.2f}, "
        f"min {np.min(v_f / v_m):.2f}, max {np.max(v_f / v_m):.2f}",
        f"means agree: max |z| = {np.max(np.abs(z)):.2f} over {len(z)} bins",
    ]
    text = "\n".join(lines)
    print(text)
    with open("results.txt", "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
