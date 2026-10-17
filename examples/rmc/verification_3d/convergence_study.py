"""
Convergence of 3D RMC and SMC against analytic solutions, as functions of iterations
and of particle histories (plot_convergence.py):

  infinite  A = 1, sigma_s / sigma_t = 1/3 (sigma_s = 1 b, sigma_c = 2 b), isotropic
            source at E0 = 101, cutoff 1, in a reflective 3D box (an infinite medium).
            psi~ is isotropic and has no face jumps, so the collision-only iteration
            converges exponentially to its binned fixed point (G-dependent floor).
            RMC: 100 P/B/I (histories per energy bin per iteration, over the 8
            octant patches), G = 100; reference: the analytic flux.
  absorber  pure absorber in [0, 2]^3 (2 x 2 x 2 mesh, octant patches), source in the
            corner cell, reflective on x = 0 (input.py's `absorber`). The residual
            is exact, but the constant trial space cannot represent the in-cell and
            face shape of the streaming residual, so after one iteration psi~ sits on a
            noise plateau about Pi psi; the mean of the iterates then converges like
            1/sqrt(N). RMC: 1000 histories per (cell, patch) bin; reference: the
            exact ray integral (input.reference_absorber).

One RMC run per problem; SMC: MC/DC at the cumulative histories of that run after 1,
3, 10 and 30 iterations. Error: the L-infinity norm against the
reference over all bins (absolute).
Writes convergence.h5:
    mpiexec -n 4 python convergence_study.py --mode=numba --no-progress_bar
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import MU, PHI, J, N_THETA, box, reference_absorber, smc_tally

HERE = os.path.dirname(os.path.abspath(__file__))
N_ITERATION = 30
# SMC at the cumulative histories of the RMC run after these iterations
MATCHED_ITERATIONS = (1, 3, 10, 30)

# Infinite medium
E0, DELTA = 101.0, 0.1
SIGMA_S, SIGMA_C = 1.0, 2.0
SIGMA_T, C = SIGMA_S + SIGMA_C, SIGMA_S / (SIGMA_S + SIGMA_C)
BOUNDS_INFINITE = ((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0))
GRIDS = (100,)
PBI = (100,)

# Absorber box
EDGES_ABSORBER = (np.array([0.0, 1.0, 2.0]),) * 3
BOUNDARY_ABSORBER = (
    ("reflective", "vacuum"),
    ("vacuum", "vacuum"),
    ("vacuum", "vacuum"),
)
E_ABSORBER = np.array([9.9, 10.0])
N_PER_BIN_ABSORBER = (1000,)


def grid(G):
    return np.concatenate((np.logspace(0.0, np.log10(E0 - DELTA), G + 1), [E0]))


def analytic_infinite(E_edges):
    """Bin averages of phi(E) = c S0 / (Sigma_t E0) (E0 / E)^c (collided bins)."""
    a, b = E_edges[:-2], E_edges[1:-1]
    return (
        C / (SIGMA_T * E0) * E0**C * (b ** (1 - C) - a ** (1 - C)) / (1 - C) / (b - a)
    )


def model_infinite():
    import mcdc

    material = mcdc.Material(nuclide_composition={"HY": 1.0}, temperature=0.1)
    simulation = mcdc.Simulation("convergence-infinite")
    simulation.set_model([box(material, BOUNDS_INFINITE, (("reflective",) * 2,) * 3)])
    simulation.set_sources(
        [mcdc.Source(position=[0.0, 0.0, 0.5], isotropic=True, energy=E0)]
    )
    simulation.settings.use_progress_bar = False
    return simulation, material


def model_absorber():
    import mcdc

    material = mcdc.Material(nuclide_composition={"AX": 1.0}, temperature=0.1)
    simulation = mcdc.Simulation("convergence-absorber")
    simulation.set_model([box(material, ((0.0, 2.0),) * 3, BOUNDARY_ABSORBER)])
    simulation.set_sources(
        [
            mcdc.Source(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                z=[0.0, 1.0],
                isotropic=True,
                energy=np.array([[9.9, 10.0], [10.0, 10.0]]),
            )
        ]
    )
    simulation.settings.use_progress_bar = False
    return simulation, material


def save(output, path, **data):
    if not common.is_master():
        return
    attrs = data.pop("attrs", {})
    with output.write() as f:
        group = f.require_group(path)
        for name, value in data.items():
            group.create_dataset(name, data=value)
        group.attrs.update(attrs)


def linf(a, b):
    return float(np.max(np.abs(a - b)))


def infinite(output):
    from mcdc.rmc.driver3d import run3d

    for G in GRIDS:
        E_edges = grid(G)
        reference = analytic_infinite(E_edges)
        for pbi in PBI:
            simulation, material = model_infinite()
            Q = np.zeros((1, 1, 1, G + 1, J))
            Q[0, 0, 0, -1, :] = 1.0 / (DELTA * 4.0 * np.pi)
            result = run3d(
                simulation,
                *BOUNDS_INFINITE,
                E_edges,
                MU,
                PHI,
                Q,
                material,
                N_iteration=N_ITERATION,
                N_per_bin=pbi / J,
                cache_dir="cache",
                N_theta=N_THETA,
            )
            phi = [result.scalar_flux(p)[0, 0, 0] for p in result.psi_history]
            save(
                output,
                f"infinite/rmc/G{G}_P{pbi}",
                linf=[linf(p[:-1], reference) for p in phi],
                epsilon_norm=result.epsilon_norm,
                phi_history=phi,
                E_edges=E_edges,
                attrs=dict(G=G, P=pbi, N_history=result.N_history),
            )

    E_edges = grid(100)
    reference = analytic_infinite(E_edges)
    save(output, "infinite", analytic_G100=reference, E_edges_G100=E_edges)
    runs = [(f"smc/n{n}", n * 100 * (100 + 1), n) for n in MATCHED_ITERATIONS]
    for path, N, n in runs:
        phi, _ = smc_tally(
            lambda: model_infinite()[0],
            BOUNDS_INFINITE,
            E_edges,
            N,
            "convergence",
            angular=False,
        )
        if common.is_master():
            save(
                output,
                f"infinite/{path}",
                linf=linf(phi[0, 0, 0, :-1], reference),
                phi=phi[0, 0, 0],
                attrs=dict(N_particle=N, iteration=n or 0),
            )


def absorber(output):
    from mcdc.rmc.driver3d import run3d

    q = 1.0 / (1.0 * 0.1 * 4.0 * np.pi)  # S0 = 1 per unit volume, energy, solid angle
    reference = None
    if common.is_master():
        reference, error = reference_absorber(EDGES_ABSORBER, MU, PHI, 1.0, q)
        save(output, "absorber", analytic=reference, analytic_error=error)
    for n_bin in N_PER_BIN_ABSORBER:
        simulation, material = model_absorber()
        Q = np.zeros((2, 2, 2, 1, J))
        Q[0, 0, 0, 0, :] = q
        result = run3d(
            simulation,
            *EDGES_ABSORBER,
            E_ABSORBER,
            MU,
            PHI,
            Q,
            material,
            N_iteration=N_ITERATION,
            N_per_bin=n_bin,
            boundary=BOUNDARY_ABSORBER,
            cache_dir="cache",
            N_theta=N_THETA,
        )
        if common.is_master():
            history = np.array(result.psi_history)[..., 0, :]
            # Running mean of the iterates from the second on (the plateau)
            means = (
                np.cumsum(history[1:], axis=0)
                / np.arange(1, len(history))[:, None, None, None, None]
            )
            save(
                output,
                f"absorber/rmc/N{n_bin}",
                linf=[linf(p, reference) for p in history],
                linf_mean=[linf(m, reference) for m in means],
                epsilon_norm=result.epsilon_norm,
                attrs=dict(N_per_bin=n_bin, N_history=result.N_history),
            )

    N_iteration_matched = 1000 * 2 * 2 * 2 * J
    runs = [(f"smc/n{n}", n * N_iteration_matched, n) for n in MATCHED_ITERATIONS]
    for path, N, n in runs:
        psi, _ = smc_tally(
            lambda: model_absorber()[0], EDGES_ABSORBER, E_ABSORBER, N, "convergence"
        )
        if common.is_master():
            save(
                output,
                f"absorber/{path}",
                linf=linf(psi[..., 0, :], reference),
                attrs=dict(N_particle=N, iteration=n or 0),
            )


def main():
    os.chdir(HERE)
    if common.is_master():
        common.write_nuclide(
            "data", "HY", 1.0, [1e-5, 1e3], [SIGMA_S] * 2, [SIGMA_C] * 2
        )
        common.write_nuclide("data", "AX", 1.0, [1e-5, 1e3], [0.0, 0.0], [1.0, 1.0])
    common.barrier()
    common.use_library("data")
    output = common.open_output("convergence.h5")
    cases = os.environ.get("CONVERGENCE_CASES", "infinite,absorber").split(",")
    for case in cases:
        {"infinite": infinite, "absorber": absorber}[case](output)
    output.close()


if __name__ == "__main__":
    main()
