"""
Efficiency sweep: error against cost for SMC, the dissertation RMC configuration, and
the current RMC configuration, over histories per bin and seeds (see README.md).

    python sweep.py list [--group 0d|fuel|reference] [--problem P]
                                                        task indices (comma list)
    python sweep.py run INDEX                           one task (SLURM array index)
    python sweep.py warmup PROBLEM CONFIG               compute and cache the transfer
                                                        moments and the Numba kernels
Run tasks under MPI (mpirun -n N python sweep.py run INDEX --mode=numba). Each task
writes results/<problem>/<config>/n<level>_s<seed>.h5; analyze.py reads them.

Environment overrides (comma-separated): SWEEP_LEVELS (histories per trial-space bin
per iteration), SWEEP_SEEDS, SWEEP_N_CORRECTION, SWEEP_N_REFERENCE.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
sys.path.insert(0, EXAMPLES)
import common


def _env_list(name, default, cast):
    value = os.environ.get(name)
    return [cast(x) for x in value.split(",")] if value else default


# Histories per trial-space bin per iteration, per problem group (SWEEP_LEVELS overrides
# both). The fuel rod has about 4800 bins, so its levels are kept low.
LEVELS = {
    "0d": _env_list("SWEEP_LEVELS", [100, 400, 1600], float),
    "fuel": _env_list("SWEEP_LEVELS", [100, 400], float),
}
SEEDS = _env_list("SWEEP_SEEDS", [1], int)
N_ITERATION = 10
N_CORRECTION = int(os.environ.get("SWEEP_N_CORRECTION", "4"))
N_REFERENCE = float(os.environ.get("SWEEP_N_REFERENCE", "1e8"))

PROBLEMS = ("absorber", "o16", "fuel_thermal", "fuel_fast")
GROUPS = {"absorber": "0d", "o16": "0d", "fuel_thermal": "fuel", "fuel_fast": "fuel"}
CONFIGS = ("dissertation", "current", "smc")


def _load(name, path):
    """Import an example's input.py under a unique module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# Problems: model, trial space, source, reference
# ======================================================================================


def setup(problem):
    """
    dict(make_model, z_edges, E_edges, Q, boundary, linear_z_allowed, energy_scale,
    reference) for one problem. Called with the working directory set to
    work/<problem>, where synthetic nuclides are written.
    """
    if problem == "absorber":
        m = _load(
            "absorber_input", os.path.join(EXAMPLES, "analytic_absorber", "input.py")
        )
        data = np.array([1.0e-3, 1.0e4])
        if common.is_master():
            common.write_nuclide(
                "data",
                "HA",
                1.0,
                data,
                m.SIGMA_S * np.ones(2),
                (m.SIGMA_T - m.SIGMA_S) * np.ones(2),
            )
        common.barrier()
        common.use_library("data")
        E_edges = common.narrow_source_grid(m.E_MIN, m.E0, m.DELTA, 100)
        z_edges = np.array([0.0, 1.0])
        nuclide = [(1.0, 1.0, data, m.SIGMA_S * np.ones(2), m.SIGMA_T * np.ones(2))]
        return dict(
            make_model=m.make_model,
            z_edges=z_edges,
            E_edges=E_edges,
            Q=common.uniform_Q(z_edges, E_edges, [0], (m.E0 - m.DELTA, m.E0)),
            boundary=("reflective", "reflective"),
            one_dimensional=False,
            energy_scale=1.0,
            reference=(
                "analytic",
                lambda: common.slowing_down_reference(
                    E_edges, nuclide, (m.E0 - m.DELTA, m.E0)
                ),
            ),
        )
    if problem == "o16":
        m = _load("o16_input", os.path.join(EXAMPLES, "o16", "input.py"))
        common.use_library(common.real_library())
        window, G = m.CASES["fast"]
        E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
        z_edges = np.array([0.0, 1.0])
        return dict(
            make_model=m.make_model_factory(window),
            z_edges=z_edges,
            E_edges=E_edges,
            Q=common.uniform_Q(z_edges, E_edges, [0], window),
            boundary=("reflective", "reflective"),
            one_dimensional=False,
            energy_scale=1.0,
            reference=("smc", None),
        )
    # Fuel rod (thermal or fast), as in fuel_rod/input.py
    m = _load("fuel_rod_input", os.path.join(EXAMPLES, "fuel_rod", "input.py"))
    case = problem.split("_")[1]
    window, G, _, source_cells, _ = m.CASES[case]
    if case == "thermal":
        if common.is_master():
            m.write_thermal_nuclides()
        common.barrier()
        common.use_library("data")
        scale, suffix, temperature = m.SCALE_THERMAL, "T", 0.1
    else:
        common.use_library(common.real_library())
        scale, suffix, temperature = 1.0, "", 293.6
    window = (window[0] * scale, window[1] * scale)
    E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
    return dict(
        make_model=m.make_model_factory(window, suffix, temperature, source_cells),
        z_edges=m.Z_EDGES,
        E_edges=E_edges,
        Q=common.uniform_Q(m.Z_EDGES, E_edges, list(source_cells), window),
        boundary=("reflective", "reflective"),
        one_dimensional=True,
        energy_scale=scale,
        reference=("smc", None),
    )


def rmc_bases(problem, config):
    """Trial-space bases and correction passes of an RMC configuration."""
    if config == "dissertation":
        return dict(
            energy_basis="constant",
            spatial_basis="constant",
            angular_basis="constant",
            N_correction=0,
        )
    one_dimensional = GROUPS[problem] == "fuel"
    return dict(
        energy_basis="linear",
        spatial_basis="linear" if one_dimensional else "constant",
        angular_basis="linear" if one_dimensional else "constant",
        N_correction=N_CORRECTION,
    )


# ======================================================================================
# Tasks
# ======================================================================================


def tasks():
    """All tasks, in a fixed order (the SLURM array index is the list index)."""
    out = []
    for problem in PROBLEMS:
        for config in CONFIGS:
            for level in LEVELS[GROUPS[problem]]:
                for seed in SEEDS:
                    out.append(
                        dict(problem=problem, config=config, level=level, seed=seed)
                    )
    for problem in PROBLEMS:
        if problem != "absorber":
            out.append(dict(problem=problem, config="reference", level=0.0, seed=1000))
    return out


def task_group(task):
    return "reference" if task["config"] == "reference" else GROUPS[task["problem"]]


def result_path(task):
    if task["config"] == "reference":
        return os.path.join(HERE, "results", task["problem"], "reference.h5")
    return os.path.join(
        HERE,
        "results",
        task["problem"],
        task["config"],
        f"n{task['level']:g}_s{task['seed']}.h5",
    )


# ======================================================================================
# Runs
# ======================================================================================


def run_smc(spec, N_particle, seed):
    """
    SMC on the trial-space bins: psi (cell averages per unit z, energy and polar
    cosine), its standard error, and for 1D problems the z-slope cell moment.
    """
    import h5py
    import mcdc

    simulation, _ = spec["make_model"]()
    settings = simulation.settings
    settings.N_particle = int(N_particle)
    settings.rng_seed = int(seed)
    settings.use_neutron_energy_window = True
    settings.neutron_energy_min = spec["E_edges"][0]
    settings.neutron_energy_max = spec["E_edges"][-1]
    settings.neutron_fission_all_prompt = True
    settings.use_progress_bar = False
    settings.output_name = f"smc-{os.getpid()}-tmp"
    scores = ["flux", "flux-z-slope"] if spec["one_dimensional"] else ["flux"]
    mesh = mcdc.MeshStructured("sweep-trial-space", z=np.asarray(spec["z_edges"]))
    tally = mcdc.Tally(
        name="smc",
        mesh=mesh,
        scores=scores,
        energy=np.asarray(spec["E_edges"]),
        mu=common.MU_EDGES,
    )
    simulation.set_tallies([tally])
    simulation.run()
    if not common.is_master():
        return None
    z_edges, E_edges = np.asarray(spec["z_edges"]), np.asarray(spec["E_edges"])
    K, G, J = len(z_edges) - 1, len(E_edges) - 1, len(common.MU_EDGES) - 1
    volume = (
        np.diff(z_edges)[:, None, None]
        * np.diff(E_edges)[None, :, None]
        * np.diff(common.MU_EDGES)[None, None, :]
    )
    out = {}
    path = settings.output_name + ".h5"
    with h5py.File(path, "r") as f:
        for score in scores:
            for x in ("mean", "sdev"):
                value = f[f"tallies/smc/{score}/{x}"][()].reshape(J, G, K)
                out[f"{score}/{x}"] = value.transpose(2, 1, 0) / volume
    os.remove(path)
    return out


def measured(function):
    """
    (result, wall [s], CPU [s] summed over all ranks) of function(), between barriers.
    CPU is process time, so MPI busy-waiting counts as CPU.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm.Barrier()
    wall_start, cpu_start = MPI.Wtime(), time.process_time()
    result = function()
    cpu = time.process_time() - cpu_start
    comm.Barrier()
    wall = MPI.Wtime() - wall_start
    return result, wall, comm.allreduce(cpu, op=MPI.SUM)


def rmc_phase_times(result):
    """Wall times of an RMC run: precompute, and every iteration and correction pass."""
    return (
        float(result.time_precompute),
        np.array(result.time_iteration, dtype=float),
        np.array(result.time_correction, dtype=float),
    )


def run_task(index):
    """
    One task, in two passes in the same process:
      - cold: Numba compiles the transport (and, for RMC, the transfer moments are
        loaded from the cache written by the warmup job): 1 phase-1 iteration and at
        most 1 correction pass at the task's histories (MC/DC sizes the source bank once
        per process, so the cold pass must not be smaller), or 100 SMC histories;
      - warm: the measured run, with everything compiled and cached.
    Both record wall time and CPU time summed over ranks; the warm RMC run also its
    precompute, per-iteration and per-correction wall times. The JIT time is estimated
    as the cold pass minus the warm cost of the same work.
    """
    import h5py
    from mpi4py import MPI

    task = tasks()[index]
    problem, config = task["problem"], task["config"]
    if os.path.exists(result_path(task)) and not os.environ.get("SWEEP_FORCE"):
        if common.is_master():
            print(f"task {index}: {result_path(task)} exists, skipped", flush=True)
        return
    work = os.path.join(HERE, "work", problem)
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    spec = setup(problem)
    z_edges, E_edges = np.asarray(spec["z_edges"]), np.asarray(spec["E_edges"])
    n_bins = (len(z_edges) - 1) * (len(E_edges) - 1) * (len(common.MU_EDGES) - 1)
    cache_dir = os.path.join(HERE, "cache", problem)
    smc = config in ("smc", "reference")

    def rmc(N_iteration, **overrides):
        bases = rmc_bases(problem, config)
        bases.update(overrides)
        return common.run_rmc(
            spec["make_model"],
            z_edges,
            E_edges,
            spec["Q"],
            N_iteration,
            task["level"],
            boundary=spec["boundary"],
            cache_dir=cache_dir,
            seed=task["seed"],
            **bases,
        )

    # Cold pass
    if smc:
        _, cold_wall, cold_cpu = measured(lambda: run_smc(spec, 100, task["seed"]))
    else:
        N_cold = min(rmc_bases(problem, config)["N_correction"], 1)
        cold, cold_wall, cold_cpu = measured(lambda: rmc(1, N_correction=N_cold))

    # Warm (measured) pass
    if smc:
        # SMC at the dissertation configuration's total histories, or the reference
        N = (
            N_REFERENCE
            if config == "reference"
            else task["level"] * n_bins * N_ITERATION
        )
        data, wall, cpu = measured(lambda: run_smc(spec, N, task["seed"]))
        result = None
    else:
        result, wall, cpu = measured(lambda: rmc(N_ITERATION))

    if not common.is_master():
        return
    path = result_path(task)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in task.items():
            f.attrs[key] = value
        f.attrs["ranks"] = MPI.COMM_WORLD.Get_size()
        f.attrs["energy_scale"] = spec["energy_scale"]
        f.attrs["n_bins"] = n_bins
        # Timing [s]: warm (measured) and cold passes, wall and CPU summed over ranks
        timing = f.create_group("timing")
        timing.attrs["wall"] = wall
        timing.attrs["cpu"] = cpu
        timing.attrs["cold_wall"] = cold_wall
        timing.attrs["cold_cpu"] = cold_cpu
        f.attrs["wall"] = wall  # used by analyze.py as the cost
        if result is not None:
            result.write(f.create_group("rmc"))
            f.attrs.update(rmc_bases(problem, config))
            precompute, iterations, corrections = rmc_phase_times(result)
            cold_precompute, cold_iterations, cold_corrections = rmc_phase_times(cold)
            timing.attrs["precompute"] = precompute
            timing.attrs["iterations"] = iterations.sum()
            timing.attrs["corrections"] = corrections.sum()
            timing.attrs["cold_precompute"] = cold_precompute
            timing.attrs["cold_iteration"] = cold_iterations[0]
            # JIT: cold minus warm cost of the same work (1 iteration, <= 1 correction)
            jit = (cold_precompute - precompute) + (
                cold_iterations[0] - iterations.mean()
            )
            if len(cold_corrections):
                timing.attrs["cold_correction"] = cold_corrections[0]
                jit += cold_corrections[0] - corrections.mean()
            timing.attrs["jit_estimate"] = jit
        else:
            f.attrs["N_particle"] = N
            # The cold pass is 100 histories: almost all of it is compilation
            timing.attrs["jit_estimate"] = cold_wall - wall * 100.0 / N
            for key, value in data.items():
                f.create_dataset(f"smc/{key}", data=value)
        if spec["reference"][0] == "analytic":
            f.create_dataset("analytic_reference", data=spec["reference"][1]())
    print(
        f"task {index} {task}: warm {wall:.1f} s wall, {cpu:.1f} s CPU; "
        f"cold {cold_wall:.1f} s -> {path}",
        flush=True,
    )


def warmup(problem, config):
    """
    One small run that computes and caches the transfer moments (MPI-parallel) and the
    RMC Numba kernels. Records the moment precompute in
    results/<problem>/<config>/warmup.h5: wall and CPU time, ranks, and whether the
    moments were computed (new cache files) or found in the cache.
    """
    import h5py
    from mpi4py import MPI

    work = os.path.join(HERE, "work", problem)
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    spec = setup(problem)
    bases = rmc_bases(problem, config)
    bases["N_correction"] = min(bases["N_correction"], 1)
    cache_dir = os.path.join(HERE, "cache", problem)
    os.makedirs(cache_dir, exist_ok=True)
    files_before = set(os.listdir(cache_dir))
    result, wall, cpu = measured(
        lambda: common.run_rmc(
            spec["make_model"],
            spec["z_edges"],
            spec["E_edges"],
            spec["Q"],
            1,
            2,
            boundary=spec["boundary"],
            cache_dir=cache_dir,
            **bases,
        )
    )
    if not common.is_master():
        return
    new_files = sorted(set(os.listdir(cache_dir)) - files_before)
    path = os.path.join(HERE, "results", problem, config, "warmup.h5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["ranks"] = MPI.COMM_WORLD.Get_size()
        f.attrs["wall"] = wall
        f.attrs["cpu"] = cpu
        f.attrs["precompute"] = float(result.time_precompute)
        f.attrs["moments_computed"] = bool(new_files)
        f.attrs["new_cache_files"] = ",".join(new_files)
    print(
        f"warmup {problem} {config}: precompute {result.time_precompute:.1f} s wall "
        f"({'computed' if new_files else 'cached'}), total {wall:.1f} s, {cpu:.1f} s CPU",
        flush=True,
    )


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--mode")]
    if not args or args[0] == "list":
        options = dict(zip(args[1::2], args[2::2]))
        group = options.get("--group")
        problem = options.get("--problem")
        all_tasks = tasks()
        indices = [
            i
            for i, t in enumerate(all_tasks)
            if group in (None, task_group(t)) and problem in (None, t["problem"])
        ]
        if group is None and problem is None:
            print(json.dumps(all_tasks, indent=1))
        print(",".join(str(i) for i in indices))
    elif args[0] == "run":
        run_task(int(args[1]))
    elif args[0] == "warmup":
        warmup(args[1], args[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
