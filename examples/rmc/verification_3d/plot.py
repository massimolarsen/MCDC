"""
Figures of the 3D verification problems (input.py writes output.h5):
  - verification_infinite.png   A = 1 infinite medium: RMC, SMC and the analytic flux
  - verification_absorber.png   pure absorber box: RMC, SMC and the exact ray-integral
                                reference per (cell, patch) bin, and per cell
  - verification_slab.png       two-material slab: 3D RMC, slab RMC and SMC
  - convergence.png             ||eps~|| per iteration
RMC is the collision-only iteration; where the iteration sits on its noise plateau
(absorber, slab), the mean of the iterates after the first two is shown with its
standard error.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
C_RMC, C_SMC, C_REF, C_1D = common.SERIES[1], common.MUTED, common.INK, common.SERIES[0]
START = 2  # plateau average from this iterate on


def plateau(history):
    h = np.asarray(history)[START:]
    return h.mean(axis=0), h.std(axis=0, ddof=1) / np.sqrt(len(h))


def infinite(f):
    group = f["infinite"]
    rmc = group["rmc"]
    E = rmc["E_edges"][()]
    E_mid = np.sqrt(E[:-1] * E[1:])[:-1]
    phi_rmc = rmc["scalar_flux"][()][0, 0, 0, :-1]
    phi_smc = group["smc"][()][:-1]
    sd_smc = group["smc_sdev"][()][:-1]
    analytic = group["analytic"][()][:-1]

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(7.5, 6.0), sharex=True, gridspec_kw={"height_ratios": (3, 1.3)}
    )
    ax.stairs(E_mid * analytic, E[:-1], color=C_REF, lw=2.2, label="analytic")
    ax.errorbar(
        E_mid,
        E_mid * phi_smc,
        yerr=2 * E_mid * sd_smc,
        fmt="o",
        ms=3,
        color=C_SMC,
        label="SMC (MC/DC), 10$^6$ histories, 2$\\sigma$",
    )
    ax.stairs(E_mid * phi_rmc, E[:-1], color=C_RMC, lw=1.4, ls="--", label="RMC 3D")
    ax.set_xscale("log")
    ax.set_ylabel("E $\\phi$(E)")
    ax.set_title("A = 1 infinite medium (3D reflective box, octant patches, G = 40)")
    ax.legend()
    ax_r.axhline(0.0, color=C_REF, lw=0.8)
    ax_r.fill_between(
        E_mid,
        -2 * sd_smc / analytic,
        2 * sd_smc / analytic,
        step="mid",
        color=common.GRID,
        label="SMC 2$\\sigma$",
    )
    ax_r.plot(E_mid, phi_smc / analytic - 1, "o", ms=3, color=C_SMC, label="SMC")
    ax_r.plot(E_mid, phi_rmc / analytic - 1, "D", ms=3, color=C_RMC, label="RMC")
    ax_r.set_ylabel("relative to\nanalytic")
    ax_r.set_xlabel("E (eV)")
    ax_r.legend(ncol=3, fontsize=8)
    fig.savefig("verification_infinite.png")
    plt.close(fig)


def absorber(f):
    group = f["absorber"]
    psi_rmc, se_rmc = plateau(group["rmc/psi_history"][()])
    psi_rmc, se_rmc = psi_rmc[..., 0, :], se_rmc[..., 0, :]
    psi_smc, sd_smc = group["smc"][()], group["smc_sdev"][()]
    ref, ref_err = group["analytic"][()], group["analytic_error"][()]

    reached = ref > 0
    order = np.argsort(-ref[reached])
    r, s, a = psi_rmc[reached][order], psi_smc[reached][order], ref[reached][order]
    e_r, e_s = se_rmc[reached][order], sd_smc[reached][order]
    x = np.arange(len(a))

    fig = plt.figure(figsize=(13.0, 6.0))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(2.2, 1), height_ratios=(3, 1.3), hspace=0.08
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_r = fig.add_subplot(grid[1, 0], sharex=ax)
    ax.plot(
        x, a, "_", ms=16, mew=2.2, color=C_REF, label="analytic (exact ray integral)"
    )
    ax.errorbar(
        x - 0.15,
        s,
        yerr=2 * e_s,
        fmt="o",
        ms=4,
        color=C_SMC,
        label="SMC (MC/DC), 10$^6$ histories, 2$\\sigma$",
    )
    ax.errorbar(
        x + 0.15, r, yerr=2 * e_r, fmt="D", ms=4, color=C_RMC, label="RMC 3D, 2 SE"
    )
    ax.set_yscale("log")
    ax.set_ylabel("$\\psi$ (cell, patch average)")
    ax.set_title(
        f"Pure absorber in [0, 2]$^3$: the {reached.sum()} reachable (cell, octant) bins; "
        f"the other {(~reached).sum()} are 0 in all three"
    )
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=8)
    ax_r.axhline(0.0, color=C_REF, lw=0.8)
    ax_r.errorbar(x - 0.15, s / a - 1, yerr=2 * e_s / a, fmt="o", ms=3, color=C_SMC)
    ax_r.errorbar(x + 0.15, r / a - 1, yerr=2 * e_r / a, fmt="D", ms=3, color=C_RMC)
    ax_r.set_ylabel("relative to\nanalytic")
    ax_r.set_xlabel("bin, by decreasing analytic $\\psi$")

    # Scalar flux per cell
    w = np.full(8, 2.0 * np.pi / 4.0)
    phi = lambda p: (p @ w).ravel()
    phi_err = lambda e: np.sqrt((e**2) @ (w**2)).ravel()
    ax_c = fig.add_subplot(grid[:, 1])
    cells = np.arange(8)
    ax_c.plot(cells, phi(ref), "_", ms=16, mew=2.2, color=C_REF, label="analytic")
    ax_c.errorbar(
        cells - 0.15,
        phi(psi_smc),
        yerr=2 * phi_err(sd_smc),
        fmt="o",
        ms=4,
        color=C_SMC,
        label="SMC",
    )
    ax_c.errorbar(
        cells + 0.15,
        phi(psi_rmc),
        yerr=2 * phi_err(se_rmc),
        fmt="D",
        ms=4,
        color=C_RMC,
        label="RMC 3D",
    )
    ax_c.set_yscale("log")
    ax_c.set_xticks(cells)
    ax_c.set_xticklabels(
        [f"{i}{j}{k}" for i in range(2) for j in range(2) for k in range(2)]
    )
    ax_c.set_xlabel("cell (kx ky kz)")
    ax_c.set_ylabel("scalar flux $\\phi$ (cell average)")
    ax_c.set_title("Per cell")
    ax_c.legend(fontsize=8)
    fig.savefig("verification_absorber.png")
    plt.close(fig)


def slab(f):
    group = f["slab"]
    rmc3 = group["rmc3d"]
    E = rmc3["E_edges"][()]
    z = rmc3["z_edges"][()]
    phi3, se3 = plateau([common_scalar(p) for p in rmc3["psi_history"][()]])
    phi1, se1 = plateau(group["rmc1d_phi_history"][()])
    phi_s, sd_s = group["smc"][()], group["smc_sdev"][()]
    dE = np.diff(E)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    total = lambda p: p @ dE
    total_err = lambda e: np.sqrt((e**2) @ (dE**2))
    zc = 0.5 * (z[:-1] + z[1:])
    ax.errorbar(
        zc - 0.06,
        total(phi_s),
        yerr=2 * total_err(sd_s),
        fmt="o",
        color=C_SMC,
        label="SMC (MC/DC), 2$\\sigma$",
    )
    ax.errorbar(
        zc,
        total(phi1),
        yerr=2 * total_err(se1),
        fmt="s",
        color=C_1D,
        label="RMC slab solver, 2 SE",
    )
    ax.errorbar(
        zc + 0.06,
        total(phi3),
        yerr=2 * total_err(se3),
        fmt="D",
        color=C_RMC,
        label="RMC 3D, 2 SE",
    )
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("energy-integrated flux (cell average)")
    ax.set_title("Two-material slab built in 3D (reflective x, y)")
    ax.legend(fontsize=8)
    # z-scores of every (cell, energy) bin, with the uncertainties of both sides
    ax = axes[1]
    comparisons = (
        ("RMC 3D vs SMC", phi3, se3, phi_s, sd_s, C_RMC, "D"),
        ("RMC slab vs SMC", phi1, se1, phi_s, sd_s, C_1D, "s"),
        ("RMC 3D vs RMC slab", phi3, se3, phi1, se1, C_REF, "o"),
    )
    ax.axhspan(-2.0, 2.0, color=common.GRID)
    for i, (label, a, ea, b, eb, color, marker) in enumerate(comparisons):
        z = ((a - b) / np.sqrt(ea**2 + eb**2)).ravel()
        x = np.arange(z.size) + (i - 1) * 0.25
        rms = np.sqrt(np.mean(z**2))
        ax.plot(
            x, z, marker, ms=4, color=color, ls="none", label=f"{label}: rms {rms:.2f}"
        )
    K, G = phi3.shape
    for k in range(1, K):
        ax.axvline(k * G - 0.5, color=common.MUTED, lw=0.6)
    ax.set_xticks([k * G + (G - 1) / 2 for k in range(K)])
    ax.set_xticklabels([f"cell {k}" for k in range(K)])
    ax.set_ylabel("z = difference / combined standard error")
    ax.set_title("Every (cell, energy) bin; grey band: |z| < 2")
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig("verification_slab.png")
    plt.close(fig)


def common_scalar(psi):
    """Scalar flux (z, g) of a 3D psi~ (1, 1, Kz, G, 8) on octant patches."""
    return psi[0, 0] @ np.full(psi.shape[-1], 4.0 * np.pi / psi.shape[-1])


def convergence(f):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, (name, path) in enumerate(
        (
            ("A = 1 infinite medium", "infinite/rmc"),
            ("pure absorber box", "absorber/rmc"),
            ("two-material slab", "slab/rmc3d"),
        )
    ):
        if path.split("/")[0] not in f:
            continue
        eps = f[path]["epsilon_norm"][()]
        ax.plot(
            np.arange(1, len(eps) + 1),
            eps / eps[0],
            "o-",
            ms=3,
            color=common.SERIES[i],
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("||$\\tilde\\epsilon$|| / ||$\\tilde\\epsilon_1$||")
    ax.set_title("3D RMC, collision-only iterations")
    ax.legend()
    fig.savefig("convergence.png")
    plt.close(fig)


with h5py.File("output.h5", "r") as f:
    for name, make in (("infinite", infinite), ("absorber", absorber), ("slab", slab)):
        if name in f:
            make(f)
            print(f"verification_{name}.png")
    convergence(f)
    print("convergence.png")
