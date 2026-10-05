"""Run one RMC case with the two-phase schedule and save its history: A1, sqrt, slab, O16."""
import os, sys, tempfile, numpy as np
sys.path.insert(0, "/Users/massimolarsen/Documents/Repos/MCDC/test/unit")
case, out = sys.argv[1], sys.argv[2]
S = os.path.dirname(os.path.abspath(__file__))
N_CORRECTION = 4
mu = np.array([-1.0, 0.0, 1.0])

if case != "O16":
    lib = tempfile.mkdtemp()
    os.environ["MCDC_LIB"] = lib
    os.chdir(lib)
from rmc.conftest import write_synthetic_nuclide
import mcdc
from mcdc.rmc.driver import run
from rmc.test_driver import reflective_box, slab_model


def save(result, extra):
    np.savez(
        out,
        psi_history=np.array(result.psi_history),
        psi_fixed=result.psi_fixed_point,
        psi_final=result.psi,
        corrections=np.array(result.corrections),
        eps=np.array(result.epsilon_norm),
        **extra,
    )


if case in ("A1", "sqrt"):
    E0, delta = 101.0, 0.1
    if case == "A1":
        write_synthetic_nuclide(lib, "HX", 1.0, [1e-5, 1e3], [1.0, 1.0], [1.0, 1.0]); name = "HX"
    else:
        e = np.logspace(-6, 3, 2000); write_synthetic_nuclide(lib, "HY", 1.0, e, e**-0.5, np.zeros_like(e)); name = "HY"
    mat = mcdc.Material(nuclide_composition={name: 1.0}, temperature=0.1)
    sim = mcdc.Simulation("gif"); sim.set_model([reflective_box(mat)])
    sim.set_sources([mcdc.Source(position=[0, 0, 0.5], isotropic=True, energy=E0)])
    E = np.concatenate((np.logspace(0.0, np.log10(E0 - delta), 21), [E0])); G = len(E) - 1
    Q = np.zeros((1, G, 2)); Q[0, -1, :] = 1 / (delta * 2)
    r = run(sim, [0.0, 1.0], E, mu, Q, [mat], 12, 500, N_correction=N_CORRECTION)
    a, b = E[:-2], E[1:-1]
    if case == "A1":
        c, St = 0.5, 2.0
        ref = c / (St * E0) * E0**c * (b ** (1 - c) - a ** (1 - c)) / (1 - c) / (b - a)
    else:
        ref = 2 * (np.sqrt(b) - np.sqrt(a)) / (b - a)
    save(r, dict(E_edges=E, ref=ref, ref_sd=np.zeros_like(ref)))

elif case == "slab":
    old = np.load(os.path.join(S, "data", "slab.npz"))  # reuse the SMC reference
    write_synthetic_nuclide(lib, "HX", 1.0, [1e-6, 1e3], [1.0, 1.0], [1.0, 1.0])
    E0, delta = 10.0, 0.1
    z = old["z_edges"]; E = old["E_edges"]; K, G = len(z) - 1, len(E) - 1
    ma = mcdc.Material(nuclide_composition={"HX": 1.0}, temperature=0.1); mb = mcdc.Material(nuclide_composition={"HX": 0.4}, temperature=0.1)
    sim = slab_model(ma, mb, E0, delta, False)
    Q = np.zeros((K, G, 2)); Q[0, -1, :] = 1 / (delta * 2)
    r = run(sim, z, E, mu, Q, [ma, ma, mb, mb], 12, 200, N_correction=N_CORRECTION, boundary=("vacuum", "vacuum"))
    save(r, dict(E_edges=E, z_edges=z, ref=old["ref"], ref_sd=old["ref_sd"]))

else:  # O16: reuse the SMC reference and the cached transfer moments
    t8 = os.path.join(os.path.dirname(S), "t8")
    old = np.load(os.path.join(t8, "results_O16.npz"))
    os.environ["MCDC_LIB"] = "/Users/massimolarsen/Documents/Repos/MCDC/hdf5lib_mt5fix"
    os.chdir(t8)
    sys.path.insert(0, "/Users/massimolarsen/Documents/Repos/MCDC/examples/rmc/infinite_medium")
    from rmc_vs_smc import CASES, model
    spec = CASES["O16"]; E = old["E_edges"]; G = len(E) - 1
    E_min, E_max = spec["window"]
    sim, mat = model(spec)
    Q = np.full((1, G, 2), 1.0 / ((E_max - E_min) * 2.0))
    r = run(sim, [0.0, 1.0], E, mu, Q, [mat], 10, 100, N_correction=N_CORRECTION, cache_dir="rmc_cache")
    dE = np.diff(E)
    save(r, dict(E_edges=E, ref=old["phi_smc"] / dE, ref_sd=old["sd_smc"] / dE))
