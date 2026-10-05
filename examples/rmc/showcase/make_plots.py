"""
Showcase figures for CE-RMC in MC/DC, made from results that already exist (no
simulation is run). Reads the example outputs (../<problem>/output*.h5), the kernel
verification statistics (mcdc/rmc/writeups/figures/*/stats.json) and the HPC timing
tables copied into hpc_timing/. Writes, in this directory:

  architecture.png        how RMC sits on top of MC/DC
  kernel_verification.png every inverted ENDF law against MC/DC's own sampler
  convergence_gallery.png ||eps~|| per iteration, analytic and real-data problems
  linear_basis.png        constant vs linear discontinuous energy basis (A = 1 absorber)
  flux_gallery.png        RMC vs SMC flux spectra on real ENDF data
  hpc_efficiency.png      error vs cost on the OSU HPC (A = 1 absorber)
  hpc_timing.png          where the time goes: compilation, precompute, iterations,
                          correction passes
  progress_bar_fix.png    the MC/DC progress-bar fix, before and after, on the HPC

    python make_plots.py
"""

import csv
import glob
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXAMPLES))
sys.path.insert(0, EXAMPLES)
import common

os.chdir(HERE)
plt = common.style()
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_SMC = common.MUTED
C_DISS = common.SERIES[0]
C_CURRENT = common.SERIES[1]


def example(path):
    return h5py.File(os.path.join(EXAMPLES, path), "r")


def read_csv(name):
    with open(os.path.join("hpc_timing", name)) as handle:
        return list(csv.DictReader(handle))


# ======================================================================================
# Architecture
# ======================================================================================


def architecture():
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=color,
                edgecolor=common.MUTED,
                linewidth=1.0,
            )
        )
        ax.text(x + 0.15, y + h - 0.2, title, va="top", fontsize=10, weight="bold")
        ax.text(
            x + 0.15, y + h - 0.55, body, va="top", fontsize=8.5, color=common.INK_2
        )

    def arrow(a, b, text=None, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=12,
                color=common.INK_2,
                linewidth=1.1,
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        if text:
            ax.text(
                (a[0] + b[0]) / 2,
                (a[1] + b[1]) / 2 + 0.12,
                text,
                ha="center",
                fontsize=8,
                color=common.INK_2,
            )

    mcdc = "#e8eef8"
    rmc = "#fbe9e1"
    ax.text(0.1, 6.2, "MC/DC (transport unchanged)", fontsize=11, color=C_DISS)
    ax.text(0.1, 2.75, "mcdc.rmc (new package)", fontsize=11, color=C_CURRENT)

    box(
        0.1,
        3.55,
        2.9,
        2.4,
        "Nuclear data",
        "ENDF/B-VIII.1 -> MC/DC HDF5\nelastic, level, Kalbach-Mann,\nLaw 61, N-body, evaporation,\nMaxwellian, fission, free gas",
        mcdc,
    )
    box(
        3.4,
        3.55,
        3.0,
        2.4,
        "MC/DC samplers",
        "sample (E', mu0 | E)\nthe 'truth' RMC mirrors",
        mcdc,
    )
    box(
        6.8,
        3.55,
        3.0,
        2.4,
        "Fixed-source transport",
        "Numba JIT (cached), MPI\nsigned weights (sign-safe\nroulette, splitting, banks)\nenergy window, prompt fission",
        mcdc,
    )
    box(
        10.2,
        3.55,
        2.7,
        2.4,
        "Track-length tallies",
        "new moment scores:\nflux-energy-slope\nflux-z-slope, flux-mu-slope\n(and their products)",
        mcdc,
    )

    box(
        0.1,
        0.15,
        2.9,
        2.4,
        "Inverted kernels",
        "evaluate f(E -> E', mu0)\nfor every law above\nchi^2-verified against\nthe MC/DC samplers",
        rmc,
    )
    box(
        3.4,
        0.15,
        3.0,
        2.4,
        "Transfer moments M",
        "adaptive G7-K15 quadrature\nkinematic pruning\nMPI-parallel, cached on disk\nP0/P1 in E, mu (and z)",
        rmc,
    )
    box(
        6.8,
        0.15,
        3.0,
        2.4,
        "Residual source",
        "r = Q + S[psi~] - Sigma_t psi~\n  + face jumps (1D)\nsigned-weight particles\nphase 2: unbiased corrections",
        rmc,
    )
    box(
        10.2,
        0.15,
        2.7,
        2.4,
        "Update",
        "eps~ = tally projection\npsi~ <- psi~ + eps~\nbases: constant or linear\nin E, z (continuous), mu",
        rmc,
    )

    arrow((3.0, 4.75), (3.4, 4.75))
    arrow((1.55, 3.55), (1.55, 2.55), "inverted")
    arrow((4.9, 3.55), (1.9, 2.55), "chi^2 check", rad=0.0)
    arrow((3.0, 1.35), (3.4, 1.35))
    arrow((6.4, 1.35), (6.8, 1.35))
    arrow((8.3, 2.55), (8.3, 3.55), "source bank")
    arrow((9.8, 4.75), (10.2, 4.75))
    arrow((11.55, 3.55), (11.55, 2.55), "eps~")
    arrow((10.2, 0.6), (9.8, 0.6), "next iteration")
    ax.set_title(
        "CE Residual Monte Carlo in MC/DC: what is reused and what is new",
        fontsize=12,
    )
    fig.savefig("architecture.png")
    plt.close(fig)


# ======================================================================================
# Kernel verification
# ======================================================================================


LAW_NAMES = {
    "elastic": "elastic, anisotropic CM",
    "level": "discrete level (Law 3)",
    "kalbach_mann": "Kalbach-Mann (Law 44)",
    "energy_angle": "energy-angle table (Law 61)",
    "tabulated": "tabulated spectrum",
    "multi_spectrum_yield": "multi-spectrum yield",
    "n_body": "N-body phase space",
    "evaporation": "evaporation",
    "maxwellian": "Maxwellian",
    "fission": "fission spectrum + nu(E)",
    "free_gas_h1": "free gas, H-1",
    "free_gas_u238": "free gas, U-238",
}


def kernel_verification():
    rows = []
    for path in sorted(glob.glob(f"{REPO}/mcdc/rmc/writeups/figures/*/stats.json")):
        law = os.path.basename(os.path.dirname(path))
        with open(path) as handle:
            stats = json.load(handle)
        for row in stats["rows"]:
            rows.append((law, row))
    labels = [f"{LAW_NAMES.get(law, law)}, E = {row['E']:.3g} eV" for law, row in rows]
    z = [(r["chi2"] - r["dof"]) / np.sqrt(2.0 * r["dof"]) for _, r in rows]
    yield_error = [
        abs(r["yield_inverted"] / r["yield_sampled"] - 1.0) + 1e-16 for _, r in rows
    ]
    quadrature = [r["quadrature_error"] + 1e-16 for _, r in rows]
    y = np.arange(len(rows))[::-1]

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12.0, 0.32 * len(rows) + 1.6), sharey=True
    )
    ax0.axvspan(-2, 2, color=common.GRID, zorder=0)
    ax0.axvline(0.0, color=common.MUTED, lw=1)
    ax0.plot(z, y, "o", color=C_CURRENT, ms=7)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=8.5)
    ax0.set_xlim(-4, 4)
    ax0.set_xlabel("(chi^2 - dof) / sqrt(2 dof)   (shaded: +-2)")
    ax0.set_title("chi^2 of sampled histogram vs inverted density")
    ax1.plot(
        yield_error, y, "o", color=C_DISS, ms=7, label="|yield inverted / sampled - 1|"
    )
    ax1.plot(quadrature, y, "D", color=common.SERIES[2], ms=6, label="quadrature error")
    ax1.set_xscale("log")
    ax1.set_xlabel("relative error")
    ax1.set_title("yield and quadrature accuracy")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2)
    fig.suptitle(
        f"Every inverted ENDF law against MC/DC's own sampler: {len(rows)} tests, "
        "10^6 samples each, all within +-2",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig("kernel_verification.png")
    plt.close(fig)


# ======================================================================================
# Convergence gallery
# ======================================================================================

ANALYTIC = (
    ("A = 1 absorber", "analytic_absorber/output.h5", "rmc/G100_P100"),
    ("A = 1, sigma ~ E^-1/2", "analytic_sqrt/output.h5", "rmc/G100_P100"),
    ("A = 1, Breit-Wigner resonance", "analytic_resonance/output.h5", "rmc/G100_P100"),
    ("A = 1, anisotropic source", "analytic_anisotropic/output.h5", "rmc/G100_P10"),
    ("A = 1 with fission", "fission/output.h5", "rmc/G100_P100"),
)
REAL = (
    ("O-16, 1-10 MeV", "o16/output.h5", "rmc/fast"),
    ("Al-27, 1e-4 - 1e3 eV", "al27/output.h5", "rmc/low"),
    ("C-12, fast", "c12/output.h5", "rmc/fast"),
    ("C-12, thermal", "c12/output.h5", "rmc/thermal"),
    ("H-2, 1-20 MeV", "h2/output.h5", "rmc/fast"),
)


def convergence_gallery():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, cases, title in (
        (axes[0], ANALYTIC, "Synthetic nuclides (analytic references)"),
        (axes[1], REAL, "Real ENDF/B-VIII.1 data"),
    ):
        for i, (label, path, group) in enumerate(cases):
            with example(path) as f:
                eps = f[group]["epsilon_norm"][()]
            n = np.arange(1, len(eps) + 1)
            ax.plot(n, eps / eps[0], "o-", ms=3, color=common.SERIES[i], label=label)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("||eps~|| / ||eps~_1||")
    fig.suptitle(
        "Residual norm per iteration: geometric convergence; with real data it levels "
        "off at the noise of one iteration",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig("convergence_gallery.png")
    plt.close(fig)


# ======================================================================================
# Linear energy basis
# ======================================================================================


def linear_basis():
    constant = example("analytic_absorber/output.h5")
    linear = example("analytic_absorber/output_linear.h5")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    gains = []
    for i, G in enumerate((50, 100, 200)):
        reference = constant[f"reference/G{G}"][()][..., :-1]
        for f, ls, basis in ((constant, "--", "constant"), (linear, "-", "linear")):
            phi = common.scalar_flux(f[f"rmc/G{G}_P100/psi_history"][()])[:, 0]
            error = common.linf(phi[..., :-1], reference)
            ax.plot(
                np.arange(1, len(error) + 1),
                error,
                ls=ls,
                marker="o",
                ms=3,
                mfc=common.SERIES[i] if basis == "linear" else "none",
                color=common.SERIES[i],
                label=f"G = {G}, {basis}",
            )
            if basis == "constant":
                floor_constant = error[-1]
            else:
                gains.append(floor_constant / error[-1])
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("L-inf relative error of bin-averaged flux")
    ax.set_title(
        "Linear discontinuous energy basis: the Galerkin bias floor drops "
        f"{min(gains):.0f}-{max(gains):.0f}x\n"
        "(A = 1 absorber, 100 histories per bin, analytic reference)"
    )
    ax.legend(ncol=2, fontsize=8)
    fig.savefig("linear_basis.png")
    plt.close(fig)


# ======================================================================================
# Flux gallery
# ======================================================================================

FLUX = (
    ("O-16, 1-10 MeV", "o16/output.h5", "rmc/fast", "smc/fast"),
    ("Al-27, 1e-4 - 1e3 eV", "al27/output.h5", "rmc/low", "smc/low"),
    ("C-12, fast", "c12/output.h5", "rmc/fast", "smc/fast"),
    ("C-12, thermal", "c12/output.h5", "rmc/thermal", "smc/thermal"),
    ("H-2, 1-20 MeV, (n,2n)", "h2/output.h5", "rmc/fast", "smc/fast"),
    ("A = 1 with fission", "fission/output.h5", "rmc/G100_P100", "smc/G100_N1000000"),
)


def flux_gallery():
    fig = plt.figure(figsize=(15.0, 7.6))
    outer = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
    for n, (label, path, rmc_group, smc_group) in enumerate(FLUX):
        inner = outer[n // 3, n % 3].subgridspec(
            2, 1, height_ratios=(3, 1), hspace=0.05
        )
        ax = fig.add_subplot(inner[0])
        ax_r = fig.add_subplot(inner[1], sharex=ax)
        with example(path) as f:
            rmc = f[rmc_group]
            scale = float(rmc.attrs.get("energy_scale", 1.0))
            E_edges = rmc["E_edges"][()] / scale
            dmu = np.diff(rmc["mu_edges"][()])
            phi = np.einsum("gj,j->g", rmc["psi"][()][0], dmu)
            smc = f[smc_group]
            phi_s = np.einsum("gj,j->g", smc["psi"][()][0], dmu)
            sd_s = np.sqrt(np.einsum("gj,j->g", smc["psi_sdev"][()][0] ** 2, dmu**2))
        E_mid = np.sqrt(E_edges[:-1] * E_edges[1:])
        G = len(phi)
        E_mid = E_mid[:G]
        ax.stairs(E_mid * phi, E_edges[: G + 1], color=C_CURRENT, lw=1.6, label="RMC")
        shown = phi_s > 0
        lower = np.minimum(2 * sd_s, 0.999 * phi_s)  # log axes: keep bars positive
        ax.errorbar(
            E_mid[shown],
            (E_mid * phi_s)[shown],
            yerr=(E_mid * np.vstack((lower, 2 * sd_s)))[:, shown],
            fmt="o",
            ms=2.5,
            color=C_SMC,
            elinewidth=0.8,
            label="SMC (2 sigma)",
        )
        ax.set_xscale("log")
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelbottom=False)
        if n % 3 == 0:
            ax.set_ylabel("E phi(E)")
        if n == 0:
            ax.legend(fontsize=8)
        y = E_mid * phi
        if np.nanmax(y) > 100 * np.nanmin(y[y > 0]):
            ax.set_yscale("log")
            ax.set_ylim(0.3 * np.min((E_mid * phi)[phi > 0]), None)
        # ratio only where SMC has a usable estimate (relative 1 sigma < 25%)
        mask = (phi_s > 0) & (sd_s < 0.25 * np.abs(phi_s))
        ratio = np.full(G, np.nan)
        ratio[mask] = phi[mask] / phi_s[mask] - 1.0
        band = np.where(mask, 2 * sd_s / np.where(mask, phi_s, 1.0), np.nan)
        ax_r.fill_between(E_mid, -band, band, step="mid", color=common.GRID)
        ax_r.plot(E_mid, ratio, ".", ms=3, color=C_CURRENT)
        ax_r.axhline(0.0, color=common.MUTED, lw=0.8)
        ax_r.set_xlabel("E (eV)")
        if n % 3 == 0:
            ax_r.set_ylabel("RMC/SMC - 1")
    fig.suptitle(
        "Collision-only RMC (constant basis) vs standard Monte Carlo on the same MC/DC "
        "model (grey band: SMC 2 sigma; missing points: no SMC histories)",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.savefig("flux_gallery.png")
    plt.close(fig)


# ======================================================================================
# HPC: efficiency, timing, the progress-bar fix
# ======================================================================================

SHORT = {"smc": "SMC", "dissertation": "RMC constant", "current": "RMC current"}
CONFIG = {
    "smc": (C_SMC, "s", "SMC"),
    "dissertation": (C_DISS, "o", "RMC, constant basis, collision-only"),
    "current": (C_CURRENT, "D", "RMC, current (linear + corrections)"),
}


def hpc_efficiency():
    rows = read_csv("absorber_hpc_summary.csv")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for config, (color, marker, label) in CONFIG.items():
        at = sorted(
            (r for r in rows if r["config"] == config), key=lambda r: float(r["level"])
        )
        cost = [float(r["cost_cpu_s"]) for r in at]
        error = [float(r["error_cell"]) for r in at]
        ax0.plot(cost, error, marker=marker, color=color, label=label)
        for r, x, y in zip(at, cost, error):
            ax0.annotate(
                f"{float(r['level']):g}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                color=common.INK_2,
            )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("cost: wall x ranks (CPU s), compilation excluded")
    ax0.set_ylabel("rms relative error vs analytic")
    ax0.set_title("A = 1 absorber on the OSU HPC (labels: histories per bin)")
    ax0.legend(fontsize=8)

    width = 0.27
    levels = sorted({float(r["level"]) for r in rows})
    x = np.arange(len(levels))
    for k, (config, (color, _, label)) in enumerate(CONFIG.items()):
        fom = [
            float(
                next(
                    r["fom"]
                    for r in rows
                    if r["config"] == config and float(r["level"]) == L
                )
            )
            for L in levels
        ]
        ax1.bar(x + (k - 1) * width, fom, width * 0.92, color=color, label=label)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{L:g}" for L in levels])
    ax1.set_xlabel("histories per bin per iteration")
    ax1.set_ylabel("figure of merit 1 / (error^2 x cost)")
    smc = {float(r["level"]): float(r["fom"]) for r in rows if r["config"] == "smc"}
    ratios = [
        float(r["fom"]) / smc[float(r["level"])] for r in rows if r["config"] != "smc"
    ]
    ax1.set_title(
        f"RMC: {min(ratios):.1e} - {max(ratios):.1e} x the figure of merit of SMC"
    )
    fig.tight_layout()
    fig.savefig("hpc_efficiency.png")
    plt.close(fig)


def hpc_timing():
    rows = read_csv("absorber_hpc_timing.csv")
    order = [(c, L) for c in CONFIG for L in (100.0, 400.0, 1600.0)]
    rows = {(r["config"], float(r["level"])): r for r in rows}
    parts = (
        ("cold_wall", "cold pass (compile / load cached code)", common.MUTED),
        ("precompute", "transfer moments (cache load)", common.SERIES[2]),
        ("iterations", "phase-1 iterations", C_DISS),
        ("corrections", "correction passes", C_CURRENT),
        ("smc", "SMC transport", common.SERIES[4]),
    )
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    width = 0.17
    x = np.arange(len(order))
    for k, (key, label, color) in enumerate(parts):
        values = []
        for config, level in order:
            r = rows[(config, level)]
            if key == "smc":
                value = float(r["wall"]) if config == "smc" else np.nan
            else:
                value = float(r[key]) if r[key] not in ("", "nan") else np.nan
                if key in ("corrections",) and value == 0.0:
                    value = np.nan
            values.append(value)
        ax.bar(x + (k - 2) * width, values, width * 0.92, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[c]}\n{L:g}/bin" for c, L in order], fontsize=8)
    ax.set_ylabel("wall time (s), 8 ranks")
    ax.set_title(
        "Where the time goes (A = 1 absorber, OSU HPC): the integrated correction sampler "
        "dominates the current runs;\ncompilation dominates everything else"
    )
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    fig.savefig("hpc_timing.png")
    plt.close(fig)


def progress_bar_fix():
    before = {
        (r["config"], float(r["level"])): float(r["wall"])
        for r in read_csv("absorber_hpc_timing_before_fix.csv")
    }
    after = {
        (r["config"], float(r["level"])): float(r["wall"])
        for r in read_csv("absorber_hpc_timing.csv")
    }
    order = [(c, L) for c in CONFIG for L in (100.0, 400.0, 1600.0)]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    ax.bar(
        x - 0.2, [before[k] for k in order], 0.38, color=common.MUTED, label="before"
    )
    ax.bar(x + 0.2, [after[k] for k in order], 0.38, color=C_CURRENT, label="after")
    for i, k in enumerate(order):
        ax.text(
            i,
            before[k] * 1.25,
            (
                f"{before[k] / after[k]:.0f}x"
                if before[k] / after[k] >= 10
                else f"{before[k] / after[k]:.1f}x"
            ),
            ha="center",
            fontsize=9,
            color=common.INK,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[c]}\n{L:g}/bin" for c, L in order], fontsize=8)
    ax.set_ylabel("measured run, wall time (s), 8 ranks")
    ax.set_title(
        "Found on the HPC: MC/DC printed its progress bar after every history "
        "(counter passed by value)"
    )
    ax.legend()
    fig.savefig("progress_bar_fix.png")
    plt.close(fig)


if __name__ == "__main__":
    for make in (
        architecture,
        kernel_verification,
        convergence_gallery,
        linear_basis,
        flux_gallery,
        hpc_efficiency,
        hpc_timing,
        progress_bar_fix,
    ):
        make()
        print(f"{make.__name__}: done")
