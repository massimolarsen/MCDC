"""
Shared pieces of the 3D fuel examples (fuel_rod_3d, pin_cell_3d): the box model on a
structured mesh, the volume source and its binned density, the SMC reference tally,
the RMC runs (mcdc.rmc.run3d) over spatial bases and patch layouts, and the
figure-of-merit table and convergence figure.

Physics conventions are those of the 1D fuel rod (../fuel_rod/input.py, imported as
`fuel_rod`): the thermal window uses the target-at-rest synthetic nuclides with
energies scaled by 10^4, the fast window real ENDF/B-VIII.1 data at 293.6 K with all
fission prompt.

Runs are compared on the scalar flux per (cell, energy bin): RMC's final estimate
(last iterate plus the tallied correction pass) and its history standard error from
the solver's scalar tally, against SMC's. This is the same quantity for every basis
and patch layout, so one SMC run per case serves them all.
"""

import importlib.util
import os
import re
import time

import numpy as np

import common

# Patch layouts: octants (J = 8), and 4 polar x 8 azimuthal patches (J = 32)
LAYOUTS = {
    8: (np.array([-1.0, 0.0, 1.0]), np.linspace(-np.pi, np.pi, 5)),
    32: (np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.linspace(-np.pi, np.pi, 9)),
}
CACHE = os.path.join(common.HERE, "cache")  # moments, patch and in-scatter tables
RESOLVED = 0.5  # FOM bins: SMC relative standard error below this
# Histories per spatial unknown, energy bin and patch per iteration. At 5.2 the
#   pin-cell iteration did not contract (mcdc/rmc/NOTES.md, section 13)
N_PER_BIN = 100.0
N_CORRECTION = 1

_spec = importlib.util.spec_from_file_location(
    "fuel_rod", os.path.join(common.HERE, "fuel_rod", "input.py")
)
fuel_rod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fuel_rod)


def setting(name, default, kind=str):
    """Environment override (e.g. PIN3D_CASES=fast)."""
    return kind(os.environ.get(name, default))


def setting_list(name, default):
    """List-valued environment override, comma- or colon-separated (sbatch --export
    splits on commas, so submit_3d.sh passes colons)."""
    return [v for v in re.split("[,:]", os.environ.get(name, default)) if v]


def case_setup(name):
    """(scale, suffix, temperature, window [eV, scaled], G) of the fuel-rod case."""
    window, G = fuel_rod.CASES[name][0], fuel_rod.CASES[name][1]
    if name == "thermal":
        common.use_library("data")
        scale, suffix, temperature = fuel_rod.SCALE_THERMAL, "T", 0.1
    else:
        common.use_library(common.real_library())
        scale, suffix, temperature = 1.0, "", 293.6
    return scale, suffix, temperature, (window[0] * scale, window[1] * scale), G


def prepare_thermal_nuclides(library):
    """Write the synthetic thermal nuclides to data/ from the real library (rank 0)."""
    if common.is_master():
        common.use_library(library)
        fuel_rod.write_thermal_nuclides()
    common.barrier()


def materials(suffix, temperature):
    """{"H2O": material, "UO2": material} for the case."""
    import mcdc

    def material(composition):
        return mcdc.Material(
            nuclide_composition={k + suffix: v for k, v in composition.items()},
            temperature=temperature,
        )

    return {"H2O": material(fuel_rod.H2O), "UO2": material(fuel_rod.UO2)}


def box_model(edges, cell_names, materials_by_name, boundary, sources, name):
    """
    One MC/DC cell per mesh cell. edges: (x, y, z); cell_names[kx, ky, kz] a key of
    materials_by_name; boundary ((x_low, x_high), (y_low, y_high), (z_low, z_high)).
    Returns (simulation, cell_materials) with cell_materials the (Kx, Ky, Kz) array
    of materials that run3d takes.
    """
    import mcdc

    planes = []
    for plane, axis, e, (low, high) in zip(
        (mcdc.Surface.PlaneX, mcdc.Surface.PlaneY, mcdc.Surface.PlaneZ),
        "xyz",
        edges,
        boundary,
    ):
        row = []
        for i, value in enumerate(e):
            kwargs = {axis: value}
            if i == 0:
                kwargs["boundary_condition"] = low
            elif i == len(e) - 1:
                kwargs["boundary_condition"] = high
            row.append(plane(**kwargs))
        planes.append(row)
    shape = tuple(len(e) - 1 for e in edges)
    cells = []
    cell_materials = np.empty(shape, dtype=object)
    for kx, ky, kz in np.ndindex(shape):
        material = materials_by_name[cell_names[kx, ky, kz]]
        cell_materials[kx, ky, kz] = material
        region = (
            +planes[0][kx] & -planes[0][kx + 1]
            & +planes[1][ky] & -planes[1][ky + 1]
            & +planes[2][kz] & -planes[2][kz + 1]
        )
        cells.append(mcdc.Cell(region, fill=material))
    simulation = mcdc.Simulation(name)
    simulation.set_model(cells)
    simulation.set_sources(sources)
    simulation.settings.use_progress_bar = False
    return simulation, cell_materials


def volume_sources(edges, mask, window):
    """
    Unit source spread uniformly over the mesh cells where mask[kx, ky, kz], uniform
    in energy over the window, isotropic: one MC/DC source per cell, with probability
    in proportion to its volume.
    """
    import mcdc

    E_low, E_high = window
    pdf = np.array([[E_low, E_high], [1.0, 1.0]]) / [[1.0], [E_high - E_low]]
    sources = []
    for kx, ky, kz in zip(*np.nonzero(mask)):
        bounds = [(e[k], e[k + 1]) for e, k in zip(edges, (kx, ky, kz))]
        volume = np.prod([b - a for a, b in bounds])
        sources.append(
            mcdc.Source(
                x=list(bounds[0]),
                y=list(bounds[1]),
                z=list(bounds[2]),
                energy=pdf,
                isotropic=True,
                probability=volume,
            )
        )
    return sources


def binned_source(edges, mask, E_edges, window, J):
    """Q[kx, ky, kz, g, j] of `volume_sources` (per unit volume, energy, steradian)."""
    shape = tuple(len(e) - 1 for e in edges)
    volume = np.einsum("i,j,k->ijk", *(np.diff(e) for e in edges))
    E_low, E_high = window
    in_window = (E_edges[:-1] >= E_low * (1 - 1e-12)) & (
        E_edges[1:] <= E_high * (1 + 1e-12)
    )
    Q = np.zeros(shape + (len(E_edges) - 1, J))
    density = 1.0 / (volume[mask].sum() * (E_high - E_low) * 4.0 * np.pi)
    Q[mask] = np.where(in_window, density, 0.0)[:, None]
    return Q


def run_smc(make_model, edges, E_edges, N_particle, seed=1, name="smc3d"):
    """
    SMC with MC/DC on the same model: the scalar flux per (cell, energy bin), per
    unit volume and energy, its standard error (history statistics, N_batch = 1),
    and the wall time of the transport.
    """
    import h5py
    import mcdc

    simulation, _ = make_model()
    settings = simulation.settings
    settings.N_particle = int(N_particle)
    settings.N_batch = 1
    settings.rng_seed = seed
    settings.use_neutron_energy_window = True
    settings.neutron_energy_min = E_edges[0]
    settings.neutron_energy_max = E_edges[-1]
    settings.neutron_fission_all_prompt = True
    settings.use_progress_bar = False
    settings.output_name = f"{name}-tmp"
    mesh = mcdc.MeshStructured(f"{name}-mesh", x=edges[0], y=edges[1], z=edges[2])
    simulation.set_tallies([mcdc.Tally(mesh=mesh, scores=["flux"], energy=E_edges)])
    start = time.perf_counter()
    simulation.run()
    wall = time.perf_counter() - start
    if not common.is_master():
        return None, None, wall
    with h5py.File(f"{name}-tmp.h5", "r") as f:
        group = f["tallies"][list(f["tallies"])[0]]
        mean, sdev = group["flux/mean"][()], group["flux/sdev"][()]
    os.remove(f"{name}-tmp.h5")
    shape = tuple(len(e) - 1 for e in edges)
    G = len(E_edges) - 1
    volume = np.einsum("i,j,k->ijk", *(np.diff(e) for e in edges))
    measure = volume[..., None] * np.diff(E_edges)

    def arrange(a):
        return np.moveaxis(a.reshape((G,) + shape), 0, -1) / measure

    return arrange(mean), arrange(sdev), wall


def run_case(output, label, make_model, edges, E_edges, mask, window, boundary,
             scale, N_iteration, N_smc, runs, bases, layouts):
    """
    RMC (iterations + correction pass) for every spatial basis and patch layout,
    saved as rmc/<label>/<basis>-J<J>, and SMC, saved as smc/<label>. All RMC runs
    share one cache (CACHE): same nuclides and grids; the layouts have their own
    patch tables and moments.
    """
    from mcdc.rmc.driver3d import run3d

    if "rmc" in runs:
        for J in layouts:
            mu, phi = LAYOUTS[J]
            Q = binned_source(edges, mask, E_edges, window, J)
            for basis in bases:
                cached = common.list_files(CACHE)

                def run():
                    simulation, cell_materials = make_model()
                    return run3d(
                        simulation, *edges, E_edges, mu, phi, Q, cell_materials,
                        N_iteration=N_iteration, N_per_bin=N_PER_BIN,
                        boundary=boundary, cache_dir=CACHE,
                        N_correction=N_CORRECTION, spatial_basis=basis,
                    )

                result, timing = common.measured(run)
                timing["new_cache_files"] = common.new_files(cached, CACHE)
                common.save_rmc(
                    output, f"{label}/{basis}-J{J}", result, energy_scale=scale,
                    **timing,
                )
    if "smc" in runs:
        # Cold pass: absorbs the compilation, so wall_total is the transport alone
        _, cold = common.measured(
            lambda: run_smc(make_model, edges, E_edges, 1000, seed=2)
        )
        (mean, sdev, wall), timing = common.measured(
            lambda: run_smc(make_model, edges, E_edges, N_smc)
        )
        if common.is_master():
            with output.write() as f:
                group = f.require_group("smc").create_group(label)
                group.create_dataset("scalar", data=mean)
                group.create_dataset("scalar_sdev", data=sdev)
                group.attrs.update(
                    dict(N_particle=int(N_smc), time=wall, energy_scale=scale,
                         cold_wall=cold["wall_total"], **timing)
                )


# ======================================================================================
# Figures of merit and convergence
# ======================================================================================


def fom_rows(rmc, smc):
    """
    Figures of merit on the scalar flux per (cell, energy bin), on the bins SMC
    resolves (relative standard error below RESOLVED; elsewhere its mean is a poor
    reference). FOM = 1 / (sigma^2 T), sigma the rms over bins of the standard error
    relative to the SMC mean:
      - RMC FOM_iter: T = iterations + correction pass (iteration 1 includes the
        compilation);
      - RMC FOM_total: T also includes the precompute (model, patch table, moments)
        and the in-scatter tables of the pass;
      - SMC: T = the transport wall time (compilation excluded by a cold pass).
    Also the agreement: z = (RMC - SMC) / sqrt(sd_R^2 + sd_S^2).
    """
    phi_r, sd_r = rmc["scalar"][()], rmc["scalar_sdev"][()]
    phi_s, sd_s = smc["scalar"][()], smc["scalar_sdev"][()]
    mask = (phi_s > 0.0) & (sd_s < RESOLVED * phi_s)
    sigma_r = float(np.sqrt(np.mean((sd_r[mask] / phi_s[mask]) ** 2)))
    sigma_s = float(np.sqrt(np.mean((sd_s[mask] / phi_s[mask]) ** 2)))
    a = rmc.attrs
    T_iter = float(np.sum(rmc["time_iteration"][()]) + np.sum(rmc["time_correction"][()]))
    T_total = T_iter + float(a["time_precompute"]) + float(a["time_table"])
    T_smc = float(smc.attrs["wall_total"])
    z = (phi_r - phi_s)[mask] / np.sqrt(sd_r**2 + sd_s**2)[mask]
    relative = phi_r[mask] / phi_s[mask] - 1.0
    mass = rmc["residual_mass"][()]
    return dict(
        bins=int(mask.sum()),
        sigma_rmc=sigma_r,
        sigma_smc=sigma_s,
        T_iter=T_iter,
        T_total=T_total,
        T_smc=T_smc,
        FOM_iter=1.0 / (sigma_r**2 * T_iter),
        FOM_total=1.0 / (sigma_r**2 * T_total),
        FOM_smc=1.0 / (sigma_s**2 * T_smc),
        z_rms=float(np.sqrt(np.mean(z**2))),
        z_max=float(np.max(np.abs(z))),
        rms_relative_difference=float(np.sqrt(np.mean(relative**2))),
        residual=float(mass[-1].sum()),
        N_rmc=int(a["N_history"]),
        N_smc=int(smc.attrs["N_particle"]),
    )


def fom_table(rows):
    """Markdown table of fom_rows for each run label."""
    lines = [
        "| run | bins | N/iter | sigma RMC | sigma SMC | T iter [s] | T total [s] "
        "| T SMC [s] | FOM_iter/SMC | FOM_total/SMC | rms rel. diff | z rms | z max "
        "| final ||r||_1 |",
        "|" + "---|" * 14,
    ]
    for label, r in rows.items():
        lines.append(
            f"| {label} | {r['bins']} | {r['N_rmc']:.2e} | {r['sigma_rmc']:.2e} "
            f"| {r['sigma_smc']:.2e} | {r['T_iter']:.0f} | {r['T_total']:.0f} "
            f"| {r['T_smc']:.0f} | {r['FOM_iter'] / r['FOM_smc']:.3g} "
            f"| {r['FOM_total'] / r['FOM_smc']:.3g} "
            f"| {r['rms_relative_difference']:.2e} | {r['z_rms']:.2f} "
            f"| {r['z_max']:.1f} | {r['residual']:.3g} |"
        )
    return "\n".join(lines)


def runs(f):
    """(label, rmc group, smc group) for every RMC run in an output file."""
    out = []
    for case in f.get("rmc", {}):
        if case not in f.get("smc", {}):
            continue
        for run in f["rmc"][case]:
            out.append((f"{case}/{run}", f["rmc"][case][run], f["smc"][case]))
    return out


def scalar_history(rmc):
    """Scalar flux of each iterate (cell averages times the patch areas)."""
    mu, phi = rmc["mu_edges"][()], rmc["phi_edges"][()]
    area = np.outer(np.diff(mu), np.diff(phi)).ravel()
    return rmc["psi_history"][()] @ area


def _stall(ratio):
    """First iteration after which ||eps~||/||psi~|| stops falling by 2x or more."""
    for n in range(1, len(ratio)):
        if ratio[n] > 0.5 * ratio[n - 1]:
            return n
    return len(ratio)


def convergence_figure(ax_left, ax_right, rmc, smc, title):
    """
    Left: ||eps~|| / ||psi~|| per iteration with a geometric rate fitted before the
    stall, and the rms relative error of the scalar flux against SMC per iteration and
    after the correction pass. Right: ||r||_1 per term (volume and faces in the
    iterations and in the pass).
    """
    psi_history = rmc["psi_history"][()]
    weights = _bin_measure(rmc)
    psi_norm = np.sqrt(
        np.sum(psi_history**2 * weights, axis=tuple(range(1, psi_history.ndim)))
    )
    ratio = rmc["epsilon_norm"][()] / psi_norm
    n = np.arange(1, len(ratio) + 1)
    stall = _stall(ratio)
    ax_left.semilogy(n, ratio, "o-", label=r"$\|\tilde\epsilon\|/\|\tilde\psi\|$")
    if stall >= 3:
        slope, intercept = np.polyfit(n[:stall], np.log(ratio[:stall]), 1)
        ax_left.semilogy(
            n[:stall], np.exp(intercept + slope * n[:stall]), "k--", lw=0.8,
            label=rf"fit: $\rho$ = {np.exp(slope):.2f}",
        )
    ax_left.axvline(stall + 0.5, color="gray", ls=":", lw=0.8)
    ax_left.text(stall + 0.6, ratio.max(), "stall", color="gray", va="top", fontsize=8)

    phi_s, sd_s = smc["scalar"][()], smc["scalar_sdev"][()]
    mask = (phi_s > 0.0) & (sd_s < RESOLVED * phi_s)
    error = [np.sqrt(np.mean((p[mask] / phi_s[mask] - 1.0) ** 2)) for p in scalar_history(rmc)]
    ax_left.semilogy(n, error, "s-", ms=4, label="rms rel. error vs SMC")
    if "scalar" in rmc:
        final = np.sqrt(np.mean((rmc["scalar"][()][mask] / phi_s[mask] - 1.0) ** 2))
        ax_left.semilogy([len(n) + 1], [final], "r*", ms=10, label="after correction")
        ax_left.axvline(len(n) + 0.5, color="r", ls="--", lw=0.8)
    smc_noise = np.sqrt(np.mean((sd_s[mask] / phi_s[mask]) ** 2))
    ax_left.axhline(smc_noise, color="C2", ls="-.", lw=0.8, label="SMC rel. std. error (rms)")
    ax_left.set_xlabel("iteration")
    ax_left.set_title(title, fontsize=9)
    ax_left.legend(fontsize=7)

    mass = rmc["residual_mass"][()]
    ax_right.semilogy(n, _positive(mass[:, 0]), "o-", label="volume")
    ax_right.semilogy(n, _positive(mass[:, 1]), "s-", label="faces")
    if "correction_mass" in rmc:
        cm = rmc["correction_mass"][()]
        k = np.arange(len(n) + 1, len(n) + 1 + len(cm))
        ax_right.semilogy(k, _positive(cm[:, 0]), "r^", label="pass: volume")
        ax_right.semilogy(k, _positive(cm[:, 1]), "rv", label="pass: faces")
        ax_right.axvline(len(n) + 0.5, color="r", ls="--", lw=0.8)
    ax_right.set_xlabel("iteration / pass")
    ax_right.set_ylabel(r"$\|r\|_1$ (source = 1)")
    ax_right.legend(fontsize=7)


def _positive(values):
    """Zeros (e.g. no face jumps from psi~ = 0) left out of a log plot."""
    return np.where(values > 0.0, values, np.nan)


def _bin_measure(rmc):
    """Bin measure |V_k| dE_g |A_j| of the stored grids."""
    edges = [rmc[name][()] for name in ("x_edges", "y_edges", "z_edges")]
    volume = np.einsum("i,j,k->ijk", *(np.diff(e) for e in edges))
    dE = np.diff(rmc["E_edges"][()])
    mu, phi = rmc["mu_edges"][()], rmc["phi_edges"][()]
    area = np.outer(np.diff(mu), np.diff(phi)).ravel()
    return volume[..., None, None] * dE[:, None] * area


# ======================================================================================
# Plots of one problem
# ======================================================================================


def plot_problem(path, regions, title):
    """
    From an output file: for every RMC run, convergence_<case>_<run>.png and
    flux_<case>_<run>.png (the spectrum averaged over each region, and the energy-
    integrated flux per z cell); for every case, compare_<case>.png (rms relative
    difference to SMC and FOM_iter / FOM_SMC per run); and fom.md.
    regions: {name: boolean mask (Kx, Ky, Kz)}.
    """
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {}
    with h5py.File(path, "r") as f:
        for label, rmc, smc in runs(f):
            rows[label] = fom_rows(rmc, smc)
            tag = label.replace("/", "_")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            convergence_figure(ax1, ax2, rmc, smc, f"{title}, {label}")
            fig.tight_layout()
            fig.savefig(f"convergence_{tag}.png", dpi=150)
            plt.close(fig)

            E = rmc["E_edges"][()] / rmc.attrs["energy_scale"]
            dE = np.diff(rmc["E_edges"][()])
            z = rmc["z_edges"][()]
            curves = {
                "RMC, fixed point": scalar_history(rmc)[-1],
                "RMC, corrected": rmc["scalar"][()],
                "SMC": smc["scalar"][()],
            }
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            for name, phi in curves.items():
                for (region, mask), style in zip(regions.items(), ("-", "--", ":")):
                    ax1.step(E[:-1], phi[mask].mean(axis=0), where="post", ls=style,
                             label=f"{name}, {region}")
                ax2.plot(0.5 * (z[1:] + z[:-1]), (phi @ dE).mean(axis=(0, 1)), "o-",
                         ms=3, label=name)
            ax1.set_xscale("log")
            ax1.set_xlabel("E [eV]")
            ax1.set_ylabel("scalar flux (per scaled eV)")
            ax1.legend(fontsize=6)
            ax2.set_xlabel("z [cm]")
            ax2.set_ylabel("energy-integrated flux (x-y average)")
            ax2.legend(fontsize=7)
            fig.suptitle(f"{title}, {label}", fontsize=9)
            fig.tight_layout()
            fig.savefig(f"flux_{tag}.png", dpi=150)
            plt.close(fig)

    cases = sorted({label.split("/")[0] for label in rows})
    for case in cases:
        labels = [l for l in rows if l.startswith(case + "/")]
        names = [l.split("/")[1] for l in labels]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.bar(names, [rows[l]["rms_relative_difference"] for l in labels])
        ax1.axhline(rows[labels[0]]["sigma_smc"], color="k", ls="--",
                    label="SMC rel. std. error (rms)")
        ax1.set_yscale("log")
        ax1.set_ylabel("rms relative difference to SMC")
        ax1.legend(fontsize=7)
        ax2.bar(names, [rows[l]["FOM_iter"] / rows[l]["FOM_smc"] for l in labels],
                label="FOM_iter / FOM_SMC")
        ax2.bar(names, [rows[l]["FOM_total"] / rows[l]["FOM_smc"] for l in labels],
                alpha=0.6, label="FOM_total / FOM_SMC")
        ax2.axhline(1.0, color="k", ls="--")
        ax2.set_yscale("log")
        ax2.legend(fontsize=7)
        fig.suptitle(f"{title}, {case}", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"compare_{case}.png", dpi=150)
        plt.close(fig)

    table = fom_table(rows)
    with open("fom.md", "w") as out:
        out.write(table + "\n")
    print(table)
