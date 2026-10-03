"""
Sampling-vs-inverted verification of every RMC emission kernel.

For each case (a reaction of a nuclide at a few incident energies E):
  1. sample N collisions with MC/DC's own reaction samplers (sample_elastic_scattering,
     sample_inelastic_scattering, sample_fission), incident direction +z, and
     histogram every emitted neutron in (E', mu0) bins (mu0 = lab cosine to +z);
  2. integrate RMC's inverted lab kernel over the same bins: continuous laws and
     free gas through `continuous_lab_density`, two-body lines (elastic, level)
     through `delta_lab_line` (g(E') on E' pieces split where mu0*(E') crosses a
     mu edge). Expected counts per collision include the yield;
  3. chi-square test (bins with >= 5 expected counts; the rest pooled), and the
     quadrature error from repeating step 2 with doubled subdivisions.

Figures and statistics go to ../figures/<case>/ (PNG, stats.json, stats.tex).

    MCDC_LIB=<library> python verify_laws.py [case ...] --mode=numba
"""

import json
import math
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
from numba import njit
from scipy.stats import chi2

import mcdc
import mcdc.numba_types as type_
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng
from mcdc.constant import (
    NEUTRON_REACTION_ELASTIC_SCATTERING,
    NEUTRON_REACTION_FISSION,
    PARTICLE_NEUTRON,
)
from mcdc.main import prepare
from mcdc.rmc.kernel import nuclide_reactions
from mcdc.rmc.reaction import (
    KERNEL_CONTINUOUS,
    continuous_lab_density,
    delta_lab_line,
    is_free_gas,
    kernel_type,
)
from mcdc.transport.physics.neutron.native import (
    sample_elastic_scattering,
    sample_fission,
    sample_inelastic_scattering,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "..", "figures")
N_SAMPLE = 1_000_000
N_E_BIN, N_MU_BIN = 48, 24
GL_X, GL_W = np.polynomial.legendre.leggauss(8)

# ======================================================================================
# Cases
# ======================================================================================


def synthetic_maxwellian(source, path):
    """C-12 with MT-91 converted from evaporation (Law 9) to a Maxwellian (Law 7)."""
    shutil.copy(source, path)
    with h5py.File(path, "r+") as f:
        group = f["neutron_reactions/inelastic_scattering/MT-091/energy_spectrum-1"]
        group.attrs["type"] = "maxwellian"


def synthetic_multi_spectrum(source, path):
    """
    Li-7 MT-16 with two spectra (the data's and a copy compressed to half the
    outgoing energies), probabilities 0.3 / 0.7, and a tabulated yield of 1.6.
    """
    shutil.copy(source, path)
    with h5py.File(path, "r+") as f:
        group = f["neutron_reactions/inelastic_scattering/MT-016"]
        first = group["energy_spectrum-1"]
        second = group.create_group("energy_spectrum-2")
        second.attrs["type"] = first.attrs["type"]
        for key in first:
            values = first[key][()]
            if key == "value":
                values = 0.5 * values
            if key == "pdf":
                values = 2.0 * values
            second.create_dataset(key, data=values)
        del group["spectrum_probability"]
        group.create_dataset("spectrum_probability", data=np.array([[0.3, 0.7]]))
        del group["multiplicity"]
        group.create_dataset("multiplicity", data=-1)
        table = group.create_group("multiplicity_table")
        table.attrs["type"] = "tabulated"
        table.create_dataset("energy", data=np.array([1.0e-11, 30.0]))
        table.create_dataset("value", data=np.array([1.6, 1.6]))


CASES = {
    "elastic": dict(
        title="Elastic scattering, target at rest (O-16, anisotropic COM data)",
        nuclide="O16",
        temperature=293.6,
        MT=2,
        energies=[1.0e6, 5.0e6],
    ),
    "free_gas_h1": dict(
        title="Free-gas elastic scattering (H-1, 293.6 K)",
        nuclide="H1",
        temperature=293.6,
        MT=2,
        energies=[0.05, 1.0],
    ),
    "free_gas_u238": dict(
        title="Free-gas elastic scattering (U-238, 293.6 K)",
        nuclide="U238",
        temperature=293.6,
        MT=2,
        energies=[1.0],
    ),
    "level": dict(
        title="Discrete-level inelastic scattering, Law 3 (O-16 MT-51)",
        nuclide="O16",
        temperature=293.6,
        MT=51,
        energies=[8.0e6, 14.0e6],
    ),
    "kalbach_mann": dict(
        title="Kalbach-Mann correlated energy-angle, Law 44 (Al-27 MT-22)",
        nuclide="Al27",
        temperature=293.6,
        MT=22,
        energies=[12.0e6, 18.0e6],
    ),
    "energy_angle": dict(
        title="Tabulated energy-angle, Law 61 (Co-59 MT-22)",
        nuclide="Co59",
        temperature=293.6,
        MT=22,
        energies=[10.0e6, 18.0e6],
    ),
    "tabulated": dict(
        title="Tabulated lab spectrum and angle, Law 4 (Li-7 MT-16, yield 2)",
        nuclide="Li7",
        temperature=293.6,
        MT=16,
        energies=[10.0e6, 16.0e6],
    ),
    "fission": dict(
        title="Prompt fission spectrum (U-235 MT-18, all prompt)",
        nuclide="U235",
        temperature=293.6,
        MT=18,
        energies=[1.0e3, 2.0e6],
    ),
    "evaporation": dict(
        title="Evaporation spectrum, Law 9 (C-12 MT-91)",
        nuclide="C12",
        temperature=293.6,
        MT=91,
        energies=[12.0e6, 18.0e6],
    ),
    "maxwellian": dict(
        title="Simple Maxwellian spectrum, Law 7 (synthetic, from C-12 MT-91)",
        nuclide="C12",
        temperature=293.6,
        MT=91,
        energies=[12.0e6, 18.0e6],
        synthetic=synthetic_maxwellian,
    ),
    "n_body": dict(
        title="N-body phase space, Law 66 (H-2 MT-16)",
        nuclide="H2",
        temperature=293.6,
        MT=16,
        energies=[8.0e6, 18.0e6],
    ),
    "multi_spectrum_yield": dict(
        title="Two spectra with a tabulated yield (synthetic, from Li-7 MT-16)",
        nuclide="Li7",
        temperature=293.6,
        MT=16,
        energies=[12.0e6],
        synthetic=synthetic_multi_spectrum,
    ),
}

# ======================================================================================
# Model setup
# ======================================================================================


def load(case, workdir):
    library = os.environ["MCDC_LIB"]
    file_name = f"{case['nuclide']}-{case['temperature']}K.h5"
    if "synthetic" in case:
        local = os.path.join(workdir, "lib")
        os.makedirs(local, exist_ok=True)
        case["synthetic"](
            os.path.join(library, file_name), os.path.join(local, file_name)
        )
        os.environ["MCDC_LIB"] = local
    try:
        material = mcdc.Material(
            nuclide_composition={case["nuclide"]: 1.0},
            temperature=case["temperature"],
        )
        simulation = mcdc.Simulation("verify")
        simulation.set_model([mcdc.Cell(fill=material)])
        simulation.settings.neutron_fission_all_prompt = True
        simulation.compile()
        container, data = prepare(simulation)
    finally:
        os.environ["MCDC_LIB"] = library
    program = container[0]
    nuclide = program["nuclides"][0]
    for reaction, _ in nuclide_reactions(program, data, nuclide):
        if reaction["MT"] == case["MT"]:
            return program, data, nuclide, reaction
    raise ValueError(f"MT-{case['MT']} not found")


# ======================================================================================
# Sampling with MC/DC
# ======================================================================================


@njit
def sample_emissions(E, N, reaction, nuclide, program, data, out_E, out_mu, seed):
    """N collisions at E along +z; every emitted neutron's (E', mu0). Returns count."""
    particles = np.zeros(1, type_.particle)
    collision = np.zeros(1, type_.collision_data)
    bank = program["bank_active"]
    count = 0
    for i in range(N):
        particle_bank_module.set_bank_size(bank, 0)
        particles[0]["E"] = E
        particles[0]["ux"] = 0.0
        particles[0]["uy"] = 0.0
        particles[0]["uz"] = 1.0
        particles[0]["w"] = 1.0
        particles[0]["alive"] = True
        particles[0]["particle_type"] = PARTICLE_NEUTRON
        particles[0]["rng_seed"] = rng.split_seed(np.uint64(i), np.uint64(seed))
        if reaction["sub_type"] == NEUTRON_REACTION_ELASTIC_SCATTERING:
            sample_elastic_scattering(
                reaction, particles, collision, nuclide, program, data
            )
        elif reaction["sub_type"] == NEUTRON_REACTION_FISSION:
            sample_fission(reaction, particles, collision, nuclide, program, data)
        else:
            sample_inelastic_scattering(
                reaction, particles, collision, nuclide, program, data
            )
        if particles[0]["alive"]:
            out_E[count] = particles[0]["E"]
            out_mu[count] = particles[0]["uz"]
            count += 1
        for k in range(particle_bank_module.get_bank_size(bank)):
            out_E[count] = bank["particle_data"][k]["E"]
            out_mu[count] = bank["particle_data"][k]["uz"]
            count += 1
    return count


# ======================================================================================
# Inverted kernel over (E', mu0) bins
# ======================================================================================


@njit
def continuous_bins(E, E_edges, mu_edges, sub, reaction, nuclide, program, data):
    """int_bin f_L(E', mu0 | E) dE' dmu0 with sub x sub Gauss-Legendre panels per bin."""
    G = len(E_edges) - 1
    J = len(mu_edges) - 1
    out = np.zeros((G, J))
    for g in range(G):
        for j in range(J):
            total = 0.0
            hE = (E_edges[g + 1] - E_edges[g]) / sub
            hm = (mu_edges[j + 1] - mu_edges[j]) / sub
            for a in range(sub):
                for b in range(sub):
                    for qe in range(len(GL_X)):
                        E_out = E_edges[g] + hE * (a + 0.5 * (1.0 + GL_X[qe]))
                        for qm in range(len(GL_X)):
                            mu0 = mu_edges[j] + hm * (b + 0.5 * (1.0 + GL_X[qm]))
                            total += (
                                GL_W[qe]
                                * GL_W[qm]
                                * continuous_lab_density(
                                    E, E_out, mu0, reaction, nuclide, program, data
                                )
                            )
            out[g, j] = 0.25 * hE * hm * total
    return out


@njit
def _line_mu(E, E_out, reaction, nuclide, program, data):
    g, mu0 = delta_lab_line(E, E_out, reaction, nuclide, program, data)
    return g, mu0


@njit
def line_bins(E, E_edges, mu_edges, sub, reaction, nuclide, program, data):
    """
    int_{E' bin} g(E') [mu0*(E') in mu bin] dE': each E' bin is cut at sub panels and
    at the E' where mu0*(E') crosses a mu edge (bisection), then Gauss-Legendre.
    """
    G = len(E_edges) - 1
    J = len(mu_edges) - 1
    out = np.zeros((G, J))
    for g in range(G):
        h = (E_edges[g + 1] - E_edges[g]) / sub
        for a in range(sub):
            lo = E_edges[g] + a * h
            hi = lo + h
            # Cut points where mu0*(E') crosses a mu edge, found on a fine scan
            cuts = np.empty(4 * (J + 1) + 2)
            cuts[0] = lo
            N = 1
            scan = 32
            prev_E = lo
            _, prev_mu = _line_mu(E, lo, reaction, nuclide, program, data)
            for s in range(1, scan + 1):
                x = lo + (hi - lo) * s / scan
                _, mu_x = _line_mu(E, x, reaction, nuclide, program, data)
                for edge in mu_edges[1:-1]:
                    if (prev_mu - edge) * (mu_x - edge) < 0.0:
                        u, v = prev_E, x
                        for _ in range(60):
                            m = 0.5 * (u + v)
                            _, mu_m = _line_mu(E, m, reaction, nuclide, program, data)
                            if (prev_mu - edge) * (mu_m - edge) <= 0.0:
                                v = m
                            else:
                                u = m
                        if N < len(cuts) - 1:
                            cuts[N] = 0.5 * (u + v)
                            N += 1
                prev_E, prev_mu = x, mu_x
            cuts[N] = hi
            N += 1
            pieces = np.sort(cuts[:N])
            for p in range(N - 1):
                c, d = pieces[p], pieces[p + 1]
                if d <= c:
                    continue
                for q in range(len(GL_X)):
                    x = 0.5 * (c + d) + 0.5 * (d - c) * GL_X[q]
                    value, mu0 = _line_mu(E, x, reaction, nuclide, program, data)
                    if value == 0.0 or mu0 < mu_edges[0] or mu0 > mu_edges[-1]:
                        continue
                    j = 0
                    while j < J - 1 and mu0 > mu_edges[j + 1]:
                        j += 1
                    out[g, j] += 0.5 * (d - c) * GL_W[q] * value
    return out


def expected_bins(E, E_edges, mu_edges, sub, reaction, nuclide, program, data):
    ktype = kernel_type(reaction, program, data)
    if ktype == KERNEL_CONTINUOUS or is_free_gas(E, reaction, nuclide):
        return continuous_bins(
            E, E_edges, mu_edges, sub, reaction, nuclide, program, data
        )
    return line_bins(E, E_edges, mu_edges, 4 * sub, reaction, nuclide, program, data)


# ======================================================================================
# Statistics and figures
# ======================================================================================


def chi_square(observed, expected):
    """Bins with >= 5 expected counts, the remaining ones pooled into one bin."""
    keep = expected >= 5.0
    o = list(observed[keep])
    e = list(expected[keep])
    if np.any(~keep) and expected[~keep].sum() > 0.0:
        o.append(observed[~keep].sum())
        e.append(expected[~keep].sum())
    o, e = np.array(o), np.array(e)
    statistic = float(np.sum((o - e) ** 2 / e))
    dof = len(o)
    return statistic, dof, float(chi2.sf(statistic, dof))


def figure(path, title, E, E_edges, mu_edges, observed, expected, N, stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def steps(edges, values):
        return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)

    unit, scale = ("MeV", 1e-6) if E >= 1e5 else ("eV", 1.0)
    dE = np.diff(E_edges)
    dmu = np.diff(mu_edges)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    ax.plot(
        *steps(E_edges * scale, observed.sum(1) / (N * dE * scale)),
        color="#2a78d6",
        lw=1.2,
        label="MC/DC sampled",
    )
    ax.plot(
        *steps(E_edges * scale, expected.sum(1) / (N * dE * scale)),
        color="#0b0b0b",
        ls=(0, (3, 1.5)),
        lw=1.2,
        label="RMC inverted",
    )
    ax.set_xlabel(f"E' ({unit})")
    ax.set_ylabel(f"emitted per collision per {unit}")
    ax.set_title("Outgoing-energy marginal", loc="left")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(
        *steps(mu_edges, observed.sum(0) / (N * dmu)),
        color="#2a78d6",
        lw=1.2,
        label="MC/DC sampled",
    )
    ax.plot(
        *steps(mu_edges, expected.sum(0) / (N * dmu)),
        color="#0b0b0b",
        ls=(0, (3, 1.5)),
        lw=1.2,
        label="RMC inverted",
    )
    ax.set_xlabel(r"$\mu_0$")
    ax.set_ylabel("emitted per collision per unit cosine")
    ax.set_title("Lab-cosine marginal", loc="left")
    ax = axes[2]
    z = np.where(
        expected >= 5.0,
        (observed - expected) / np.sqrt(np.maximum(expected, 1e-300)),
        np.nan,
    )
    image = ax.pcolormesh(
        E_edges * scale, mu_edges, z.T, cmap="RdBu_r", vmin=-4, vmax=4
    )
    fig.colorbar(image, ax=ax, label="(sampled - inverted) / sqrt(inverted)")
    ax.set_xlabel(f"E' ({unit})")
    ax.set_ylabel(r"$\mu_0$")
    ax.set_title("Per-bin z-score (bins >= 5 counts)", loc="left")
    fig.suptitle(
        f"{title}, E = {E * scale:g} {unit}:  chi2/dof = {stats['chi2']:.1f}/{stats['dof']}"
        f" (p = {stats['p_value']:.2f}),  yield {stats['yield_sampled']:.4f} vs {stats['yield_inverted']:.4f}",
        x=0.02,
        y=1.04,
        ha="left",
        fontsize=11,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def energy_label(E):
    """File-name label without '.' or '+' (LaTeX graphicx): 1e+06 -> E1e06, 0.05 -> E0p05."""
    return "E" + f"{E:.4g}".replace(".", "p").replace("+", "")


def latex_table(name, rows):
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$E$ [eV] & samples & bins & $\chi^2/\mathrm{dof}$ & $p$ & yield (sampled / inverted) & quad. err. \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['E']:.4g} & {r['N_collision']:.0e} & {r['dof']} & "
            f"{r['chi2']:.1f}/{r['dof']} & {r['p_value']:.3f} & "
            f"{r['yield_sampled']:.5f} / {r['yield_inverted']:.5f} & "
            f"{r['quadrature_error']:.1e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ======================================================================================
# Driver
# ======================================================================================


def run_case(name, case, workdir):
    program, data, nuclide, reaction = load(case, workdir)
    directory = os.path.join(FIGURES, name)
    os.makedirs(directory, exist_ok=True)
    rows = []
    for E in case["energies"]:
        capacity = 6 * N_SAMPLE
        out_E = np.empty(capacity)
        out_mu = np.empty(capacity)
        count = sample_emissions(
            E, N_SAMPLE, reaction, nuclide, program, data, out_E, out_mu, 12345
        )
        E_s, mu_s = out_E[:count], out_mu[:count]
        E_edges = np.linspace(E_s.min(), E_s.max(), N_E_BIN + 1)
        E_edges[-1] = np.nextafter(E_edges[-1], np.inf)
        mu_edges = np.linspace(-1.0, 1.0, N_MU_BIN + 1)
        observed, _, _ = np.histogram2d(E_s, mu_s, bins=[E_edges, mu_edges])

        coarse = expected_bins(
            E, E_edges, mu_edges, 4, reaction, nuclide, program, data
        )
        fine = expected_bins(E, E_edges, mu_edges, 8, reaction, nuclide, program, data)
        expected = N_SAMPLE * fine
        used = expected >= 5.0
        quadrature_error = float(np.max(np.abs(fine[used] - coarse[used]) / fine[used]))
        statistic, dof, p_value = chi_square(observed, expected)
        stats = dict(
            E=E,
            N_collision=N_SAMPLE,
            N_emitted=int(count),
            chi2=statistic,
            dof=dof,
            p_value=p_value,
            yield_sampled=count / N_SAMPLE,
            yield_inverted=float(fine.sum()),
            quadrature_error=quadrature_error,
            max_abs_z=float(
                np.nanmax(
                    np.abs(
                        (observed - expected) / np.sqrt(np.maximum(expected, 1e-300))
                    )[used]
                )
            ),
        )
        rows.append(stats)
        figure(
            os.path.join(directory, f"{name}_{energy_label(E)}.png"),
            case["title"],
            E,
            E_edges,
            mu_edges,
            observed,
            expected,
            N_SAMPLE,
            stats,
        )
        print(
            f"{name:22s} E={E:<10.4g} chi2/dof={statistic:8.1f}/{dof:<4d} p={p_value:.3f} "
            f"yield {count / N_SAMPLE:.5f} vs {fine.sum():.5f} quad {quadrature_error:.1e}",
            flush=True,
        )
    with open(os.path.join(directory, "stats.json"), "w") as f:
        json.dump(dict(title=case["title"], rows=rows), f, indent=2)
    with open(os.path.join(directory, "stats.tex"), "w") as f:
        f.write(latex_table(name, rows))


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CASES)
    with tempfile.TemporaryDirectory() as workdir:
        for name in names:
            run_case(name, CASES[name], workdir)


if __name__ == "__main__":
    main()
