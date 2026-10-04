"""
Figures of the slab basis comparison from output.h5 (see input.py, README.md):
  - convergence.png     ||eps~|| of the collision-only iterations, all four bases
  - flux_profile.png    energy-integrated scalar flux in z: SMC (cell averages and its
                        in-cell slope) and RMC + corrections for each basis
  - zscores.png         RMC + corrections against SMC per bin, on the quantities each
                        projection preserves (cell averages for the constant spatial
                        basis, interior hat-function moments for the linear one)
  - angular_slopes.png  the polar-cosine slope coefficient of the linear angular basis
                        against SMC's, per cell and energy bin (constant z)
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
from input import BASES, E_EDGES, Z_EDGES, run_name

from mcdc.rmc.space import cell_average, mass_matrix, node_moments

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt = common.style()
f = h5py.File("output.h5", "r")
dmu = np.diff(common.MU_EDGES)
dE = np.diff(E_EDGES)
h = np.diff(Z_EDGES)
K = len(h)
STYLE = {
    ("constant", "constant"): dict(color=common.SERIES[0], ls=(0, (4, 2))),
    ("constant", "linear"): dict(color=common.SERIES[1], ls=(0, (4, 2))),
    ("linear", "constant"): dict(color=common.SERIES[0], ls="-"),
    ("linear", "linear"): dict(color=common.SERIES[1], ls="-"),
}


def label(spatial, angular):
    return f"z {spatial}, mu {angular}"


# SMC: scalar-flux cell averages, z-slope moments, and their standard errors (the
#   polar bins of one history are correlated: fully correlated sum as a bound)
phi_smc = np.einsum("kgj,j->kg", f["smc/flux/mean"][()], dmu)
sd_smc = np.einsum("kgj,j->kg", f["smc/flux/sdev"][()], dmu)
zslope_smc = np.einsum("kgj,j->kg", f["smc/flux-z-slope/mean"][()], dmu)
zslope_sd = np.einsum("kgj,j->kg", f["smc/flux-z-slope/sdev"][()], dmu)

# Convergence of the collision-only iterations
fig, ax = plt.subplots(figsize=(7.0, 4.5))
for spatial, angular in BASES:
    eps = f[f"rmc/{run_name(spatial, angular)}/epsilon_norm"][()]
    ax.plot(
        np.arange(1, len(eps) + 1),
        eps,
        marker="o",
        ms=4,
        label=label(spatial, angular),
        **STYLE[(spatial, angular)],
    )
ax.set_yscale("log")
ax.set_xlabel("Collision-only iteration")
ax.set_ylabel(r"$\|\tilde\epsilon\|$")
ax.legend()
fig.tight_layout()
fig.savefig("convergence.png")

# Energy-integrated scalar flux in z
fig, ax = plt.subplots(figsize=(7.5, 4.8))
phi_z = (phi_smc * dE).sum(axis=1)  # per unit z
slope_z = 3.0 * (zslope_smc * dE).sum(axis=1)  # Legendre slope coefficient
ax.plot(*common.steps(Z_EDGES, phi_z), color=common.MUTED, lw=2.5, label="SMC cells")
for k in range(K):
    ax.plot(
        Z_EDGES[k : k + 2],
        [phi_z[k] - slope_z[k], phi_z[k] + slope_z[k]],
        color=common.MUTED,
        ls=(0, (1, 1.5)),
        label="SMC in-cell slope" if k == 0 else None,
    )
for spatial, angular in BASES:
    group = f[f"rmc/{run_name(spatial, angular)}"]
    psi = group["psi"][()]
    phi = (np.einsum("kgj,j->kg", psi, dmu) * dE).sum(axis=1)
    if spatial == "constant":
        ax.plot(
            *common.steps(Z_EDGES, phi),
            label=label(spatial, angular),
            **STYLE[(spatial, angular)],
        )
    else:
        ax.plot(
            Z_EDGES, phi, label=label(spatial, angular), **STYLE[(spatial, angular)]
        )
ax.set_xlabel("z (cm)")
ax.set_ylabel(r"$\int\phi\,dE$ (per cm)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("flux_profile.png")

# z-scores of RMC + corrections against SMC
M = mass_matrix(Z_EDGES)
fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), sharey=True)
summary = []
for ax, (spatial, angular) in zip(axes.ravel(), BASES):
    group = f[f"rmc/{run_name(spatial, angular)}"]
    corrections = group["corrections"][()]
    if spatial == "constant":
        # Cell averages are preserved by the piecewise-constant projection
        rmc = np.einsum("kgj,j->kg", group["psi"][()], dmu)
        passes = np.einsum("nkgj,j->nkg", corrections, dmu)
        reference, sd = phi_smc, sd_smc
        what = "cell averages"
    else:
        # Hat moments int phi h_i dz of the interior nodes are preserved by the
        #   continuous-linear projection
        rmc = M @ np.einsum("igj,j->ig", group["psi"][()], dmu)
        passes = np.einsum("il,nlgj,j->nig", M, corrections, dmu)
        reference = node_moments(phi_smc * h[:, None], zslope_smc * h[:, None])
        sd = node_moments((sd_smc + zslope_sd) * h[:, None], np.zeros_like(sd_smc))
        rmc, passes, reference, sd = (
            rmc[1:-1],
            passes[:, 1:-1],
            reference[1:-1],
            sd[1:-1],
        )
        what = "interior hat moments"
    se = passes.std(axis=0, ddof=1) / np.sqrt(len(passes))
    z = (rmc - reference) / np.sqrt(sd**2 + se**2)
    rms = np.sqrt(np.mean(z**2))
    summary.append((spatial, angular, rms, np.abs(z).max()))
    for index, row in enumerate(z):
        ax.plot(
            np.arange(len(row)),
            row,
            marker="o",
            ms=3,
            lw=0.8,
            label=f"{'cell' if spatial == 'constant' else 'node'} {index + (spatial != 'constant')}",
        )
    for level in (-2.0, 2.0):
        ax.axhline(level, color=common.MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.set_title(f"{label(spatial, angular)}: {what}, rms z = {rms:.2f}", fontsize=9)
    ax.set_xlabel("energy bin")
    ax.legend(fontsize=7, ncol=2)
for ax in axes[:, 0]:
    ax.set_ylabel("z (RMC - SMC)")
fig.tight_layout()
fig.savefig("zscores.png")

# Polar-cosine slope coefficients (constant z, linear angle) against SMC's
group = f[f"rmc/{run_name('constant', 'linear')}"]
mu_slope_rmc = group["psi_mu_slope"][()]  # (K, G, J)
mu_slope_smc = 3.0 * f["smc/flux-mu-slope/mean"][()]
mu_slope_sd = 3.0 * f["smc/flux-mu-slope/sdev"][()]
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
for j, ax in enumerate(axes):
    for k in range(K):
        color = common.SERIES[k]
        x = np.arange(len(dE)) + 0.1 * (k - 1.5)
        ax.errorbar(
            x,
            mu_slope_smc[k, :, j],
            yerr=2.0 * mu_slope_sd[k, :, j],
            color=color,
            fmt="none",
            lw=1.0,
        )
        ax.plot(
            x,
            mu_slope_rmc[k, :, j],
            color=color,
            marker="o",
            ms=4,
            lw=0.0,
            label=f"cell {k}",
        )
    ax.axhline(0.0, color=common.MUTED, lw=0.6)
    ax.set_title(
        f"polar bin {'mu < 0' if j == 0 else 'mu > 0'}: RMC (dots) vs SMC (2 sigma bars)",
        fontsize=9,
    )
    ax.set_xlabel("energy bin")
axes[0].set_ylabel(r"slope coefficient $\psi_{j,1}$")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("angular_slopes.png")

for spatial, angular, rms, worst in summary:
    print(f"{label(spatial, angular):26s} rms z {rms:.2f}  max |z| {worst:.1f}")
