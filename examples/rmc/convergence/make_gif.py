"""
Render RMC convergence GIFs: one frame per iteration.

0D cases: [phi(E) vs reference] [relative error per bin] [||eps~|| per iteration]
Slab:     2x2 small multiples of phi(E) per z-cell, plus ||eps~|| per iteration.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Reference palette (light mode)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e7e6e2"
SERIES = "#2a78d6"  # categorical slot 1: the RMC iterate
HISTORY = "#b7d3f6"  # sequential blue 150: earlier iterates (recessive)
BAND = "#e7e6e2"  # reference +-1 sigma

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def steps(edges, values):
    """Piecewise-constant line through bin edges."""
    return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)


def flux_panel(ax, edges, phi, ref, ref_sd, n, ref_label, title):
    for k in range(n):
        ax.plot(*steps(edges, phi[k]), color=HISTORY, lw=1.0)
    if np.any(ref_sd > 0):
        x, lo = steps(edges, ref - ref_sd)
        _, hi = steps(edges, ref + ref_sd)
        ax.fill_between(x, lo, hi, color=BAND, lw=0)
    ax.plot(*steps(edges, ref), color=TEXT_2, lw=2.0, ls=(0, (4, 2)), label=ref_label)
    ax.plot(*steps(edges, phi[n]), color=SERIES, lw=2.0, label=f"RMC iterate {n}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [eV]")
    ax.set_title(title, loc="left")


def error_panel(ax, edges, phi, ref, ref_sd, n, limit):
    ax.axhline(0.0, color=MUTED, lw=1.0)
    if np.any(ref_sd > 0):
        x, band = steps(edges, ref_sd / ref)
        ax.fill_between(x, -band, band, color=BAND, lw=0, label="reference ±1σ")
    ax.plot(*steps(edges, phi[n] / ref - 1.0), color=SERIES, lw=2.0)
    ax.set_xscale("log")
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("RMC / reference − 1")
    clipped = np.any(np.abs(phi[n] / ref - 1.0) > limit)
    ax.set_title(
        "Relative error per bin" + (f" (clipped at ±{100 * limit:.0f}%)" if clipped else ""),
        loc="left",
    )


def eps_panel(ax, eps, n):
    it = np.arange(len(eps))
    ax.plot(it[: n + 1], eps[: n + 1], color=SERIES, lw=2.0)
    ax.plot(it[: n + 1], eps[: n + 1], "o", color=SERIES, ms=5, mec=SURFACE, mew=1.5)
    ax.plot([n], [eps[n]], "o", color=SERIES, ms=9, mec=SURFACE, mew=2)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(eps) - 0.5)
    ax.set_ylim(eps.min() / 3, eps.max() * 3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("‖ε̃‖₂")
    ax.set_title("Residual correction per iteration", loc="left")


def frame_labels(N_iteration, N_correction):
    labels = [f"iteration {n} (collision-only)" for n in range(N_iteration)]
    labels.append(f"fixed point + mean of {N_correction} correction passes")
    return labels


def render_0d(edges, frames, ref, ref_sd, eps, labels, title, ref_label, out):
    """frames: phase-1 iterates followed by the corrected final answer."""
    errors = np.abs(frames[1:] / ref - 1.0)
    limit = min(1.5, 1.15 * errors.max()) if len(frames) > 1 else 0.5
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"wspace": 0.3})
    last = len(frames) - 1

    def frame(n):
        for ax in axes:
            ax.clear()
        flux_panel(axes[0], edges, frames, ref, ref_sd, n, ref_label, "Scalar flux φ(E)")
        axes[0].set_ylabel("φ [per eV]")
        axes[0].legend(loc="lower left")
        error_panel(axes[1], edges, frames, ref, ref_sd, n, limit)
        if n == last:
            # Fixed point (before the corrections) for comparison
            axes[1].plot(*steps(edges, frames[n - 1] / ref - 1.0), color=HISTORY, lw=1.5)
        eps_panel(axes[2], eps, min(n, len(eps) - 1))
        rms = np.sqrt(np.mean((frames[n] / ref - 1.0) ** 2))
        fig.suptitle(
            f"{title}   —   {labels[n]}   (rms error {100 * rms:.2g}%)",
            x=0.06, ha="left", color=TEXT, fontsize=13,
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.14)

    animation = FuncAnimation(fig, frame, frames=len(frames))
    animation.save(out, writer=PillowWriter(fps=1.5), dpi=100)
    plt.close(fig)


def render_slab(E, z, frames, ref, ref_sd, eps, labels, title, out):
    K = len(z) - 1
    fig = plt.figure(figsize=(15, 7.2))
    grid = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.45)
    cells = [fig.add_subplot(grid[k // 2, k % 2]) for k in range(K)]
    eps_ax = fig.add_subplot(grid[:, 2])
    material = ["a (N = 1.0)", "a (N = 1.0)", "b (N = 0.4)", "b (N = 0.4)"]

    def frame(n):
        for ax in cells + [eps_ax]:
            ax.clear()
        for k, ax in enumerate(cells):
            flux_panel(
                ax, E, frames[:, k], ref[k], ref_sd[k], n, "SMC (10⁶ histories)",
                f"z ∈ [{z[k]:g}, {z[k + 1]:g}] cm — material {material[k]}",
            )
            ax.set_ylabel("φ [per cm per eV]")
            if k == 0:
                ax.legend(loc="upper left")
        eps_panel(eps_ax, eps, min(n, len(eps) - 1))
        rms = np.sqrt(np.mean((frames[n] / ref - 1.0) ** 2))
        fig.suptitle(
            f"{title}   —   {labels[n]}   (rms error vs SMC {100 * rms:.2g}%)",
            x=0.06, ha="left", color=TEXT, fontsize=13,
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08)

    animation = FuncAnimation(fig, frame, frames=len(frames))
    animation.save(out, writer=PillowWriter(fps=1.5), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    case, source, out = sys.argv[1], sys.argv[2], sys.argv[3]
    raw = np.load(source)
    dmu = np.array([1.0, 1.0])
    history = list(raw["psi_history"]) + [raw["psi_final"]]
    labels = frame_labels(len(raw["psi_history"]), len(raw["corrections"]))
    flux = np.array([np.einsum("kgj,j->kg", psi, dmu) for psi in history])
    E = raw["E_edges"]
    if case == "slab":
        render_slab(
            E, raw["z_edges"], flux, raw["ref"], raw["ref_sd"], raw["eps"], labels,
            "1D two-material slab, vacuum boundaries", out,
        )
    elif case == "O16":
        render_0d(
            E, flux[:, 0], raw["ref"], raw["ref_sd"], raw["eps"], labels,
            "0D O-16 (ENDF), 1–10 MeV", "SMC (10⁶ histories)", out,
        )
    else:
        # Bins below the (narrow) source bin
        title = {
            "A1": "0D slowing down, A = 1, constant σ (c = ½)",
            "sqrt": "0D slowing down, A = 1 pure scatterer, σ ∝ E^-½",
        }[case]
        render_0d(
            E[:-1], flux[:, 0, :-1], raw["ref"], raw["ref_sd"], raw["eps"], labels,
            title, "analytic", out,
        )
