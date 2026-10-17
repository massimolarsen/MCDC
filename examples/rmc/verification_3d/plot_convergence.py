"""
Figures of convergence_study.py (3D RMC and SMC against analytic solutions):
  - convergence_3d.png    one panel per problem (A = 1 infinite medium, pure-absorber
                          box): L-infinity error to the analytic solution against
                          particle histories, for the RMC run and SMC at the same cumulative
                          histories, with the N^-1/2 rate fitted to the SMC runs.
  - flux_comparison.png   3D infinite medium: phi(E), analytic, RMC and SMC.
"""

import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
f = h5py.File("convergence.h5", "r")
BOX = dict(boxstyle="round", facecolor="white", edgecolor="0.7")
ANALYTIC = {
    "infinite": (
        "Analytic solution (A = 1, isotropic COM scattering):\n"
        r"$\phi(E) = \dfrac{c\,S_0}{\Sigma_t E_0}\left(\dfrac{E_0}{E}\right)^{c}$,"
        "\n"
        r"$c = \sigma_s/\sigma_t = 1/3$, $\Sigma_t = 3$ cm$^{-1}$, $S_0 = 1$, $E_0 = 101$"
        "\n(bin averages over the collided bins)"
    ),
    "absorber": (
        "Analytic solution (pure absorber):\n"
        r"$\psi(\mathbf{r},\Omega) = \dfrac{q}{\Sigma_t}"
        r"\left(e^{-\Sigma_t s_1} - e^{-\Sigma_t s_2}\right)$,"
        "\n"
        r"$[s_1, s_2]$: backward ray $\mathbf{r} - s\Omega$ inside the source"
        "\n"
        r"(reflection at $x = 0$ unfolded), $q = 1/(4\pi\,\Delta E)$, $\Sigma_t = 1$ cm$^{-1}$"
        "\n(averaged over each cell and patch)"
    ),
}


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, name, title in (
    (axes[0], "infinite", "A = 1 infinite medium, 3D reflective box (G = 100)"),
    (axes[1], "absorber", "Pure absorber, [0, 2]$^3$, 2$^3$ cells x 8 patches"),
):
    group = f[name]
    rmc = list(group["rmc"].values())[0]
    N_iteration = rmc.attrs["N_history"]
    linf = rmc["linf"][()]
    n = np.arange(1, len(linf) + 1)
    ax.plot(n * N_iteration, linf, "o-", ms=4, color="C1", label="RMC: iterate")
    if "linf_mean" in rmc:
        ax.plot(
            n[1:] * N_iteration,
            rmc["linf_mean"][()],
            "-",
            lw=2.5,
            color="C3",
            label="RMC: mean of iterates 2..n",
        )
    smc = sorted(group["smc"].values(), key=lambda g: g.attrs["N_particle"])
    N = np.array([g.attrs["N_particle"] for g in smc], dtype=float)
    L = np.array([g["linf"][()] for g in smc])
    C = np.exp(np.mean(np.log(L * np.sqrt(N))))
    ax.plot(
        N, L, "s", ms=8, color="gray", label="SMC (MC/DC), same cumulative histories"
    )
    x = n * N_iteration
    ax.plot(
        x,
        C / np.sqrt(x),
        color="gray",
        lw=1.2,
        ls=(0, (1, 2)),
        label=r"SMC rate $\propto N^{-1/2}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particle histories")
    ax.set_ylabel(r"$L_\infty$ error to the analytic solution")
    ax.set_title(title, fontsize=11)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * (2.5 if name == "absorber" else 1.0))  # room for the legend
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.03,
        0.03,
        ANALYTIC[name],
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=BOX,
    )
fig.tight_layout()
fig.savefig("convergence_3d.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# Flux comparison, 3D infinite medium
group = f["infinite"]
E = group["E_edges_G100"][()]
rmc = list(group["rmc"].values())[0]
fig, ax = plt.subplots(figsize=(9, 6))
ax.stairs(
    group["analytic_G100"][()],
    E[:-1],
    ls="--",
    color="C0",
    lw=2,
    baseline=None,
    label="Analytic",
)
ax.stairs(
    rmc["phi_history"][-1][:-1],
    E[:-1],
    color="C1",
    baseline=None,
    label=rf"RMC 3D: {rmc.attrs['P']} $P/B/I$, {len(rmc['linf'])} iterations",
)
smc = sorted(group["smc"].values(), key=lambda g: g.attrs["N_particle"])[-1]
ax.stairs(
    smc["phi"][()][:-1],
    E[:-1],
    color="C2",
    baseline=None,
    label=f"SMC: {smc.attrs['N_particle']:.2e} particles (same histories as RMC)",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E")
ax.set_ylabel(r"$\phi(E)$")
ax.set_title(r"A = 1, $\sigma_s/\sigma_t = 1/3$, 3D reflective box, G = 100")
ax.legend()
ax.text(
    0.03,
    0.03,
    ANALYTIC["infinite"],
    transform=ax.transAxes,
    fontsize=9,
    va="bottom",
    ha="left",
    bbox=BOX,
)
fig.savefig("flux_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("convergence_3d.png\nflux_comparison.png")
