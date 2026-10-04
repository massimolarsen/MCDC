"""
Unbiasedness of the residual source samplers: per-bin sums of banked weights / N must
estimate the exact bin integrals of each residual term.
"""

import numpy as np
import pytest

import mcdc
import mcdc.mcdc_get as mcdc_get
import mcdc.transport.particle_bank as particle_bank_module
from mcdc.rmc.angular import build_angular_transfer_table
from mcdc.rmc.driver import _emission_integrals, _reaction_tables
from mcdc.rmc.residual import material_total_xs
from mcdc.rmc.kernel import nuclide_transfer_moments
from mcdc.rmc.residual import collision_residual_mass
from mcdc.rmc.source import (
    FACE_NUDGE,
    PROPOSAL_DEFENSIVE,
    PROPOSAL_UNIFORM_LETHARGY,
    binned_in_scatter,
    correction_masses,
    in_scatter_density,
    sample_collision_edge,
    sample_correction,
    sample_correction_integrated,
)

from .conftest import DATA_DIR, numba_only, requires_nuclide


def banked(simulation):
    size = particle_bank_module.get_bank_size(simulation["bank_source"])
    return simulation["bank_source"]["particle_data"][:size].copy()


def tally_bins(particles, z_edges, E_edges, mu_edges, N_total):
    """Mean and standard error of sum(w in bin) / N over the histories."""
    K, G, J = len(z_edges) - 1, len(E_edges) - 1, len(mu_edges) - 1
    k = np.clip(np.searchsorted(z_edges, particles["z"], side="right") - 1, 0, K - 1)
    g = np.clip(np.searchsorted(E_edges, particles["E"], side="right") - 1, 0, G - 1)
    j = np.clip(np.searchsorted(mu_edges, particles["uz"], side="right") - 1, 0, J - 1)
    total = np.zeros((K, G, J))
    square = np.zeros((K, G, J))
    np.add.at(total, (k, g, j), particles["w"])
    np.add.at(square, (k, g, j), particles["w"] ** 2)
    mean = total / N_total
    std = np.sqrt(np.maximum(square / N_total - mean**2, 0.0) / N_total)
    return mean, std


@pytest.mark.parametrize("P, antithetic", [(1, False), (2, False), (2, True)])
def test_collision_edge_unbiased(prepare_simulation, P, antithetic):
    """
    Constant (P = 1) and linear (P = 2) energy coefficients of psi~, c and jumps;
    independent or antithetic (mirrored-energy pair) sampling.
    """
    N = 200000

    def configure(simulation):
        simulation.settings.N_particle = N
        simulation.settings.use_source_bank = True

    container, data = prepare_simulation(configure=configure)
    simulation = container[0]

    rng = np.random.default_rng(4)
    z_edges = np.array([0.0, 1.0, 3.0])
    E_edges = np.array([1.0, 2.0, 5.0, 9.0])
    mu_edges = np.array([-1.0, 0.0, 1.0])
    K, G, J = 2, 3, 2
    psi = rng.uniform(0.5, 1.5, (K, G, J, P))
    c = rng.uniform(-1.0, 3.0, (K, G, J, P))
    jump = rng.uniform(-1.0, 1.0, (K + 1, G, J, P))
    E_grid = np.array([1.0, 1.5, 4.0, 6.0, 9.0])
    Sigma = np.array([[1.0, 3.0, 0.5, 2.0, 1.0], [2.0, 0.2, 1.0, 4.0, 0.5]])
    xs_offsets = np.array([0, 5, 10])
    xs_energy = np.concatenate((E_grid, E_grid))
    xs_total = Sigma.ravel()
    cell_material = np.array([0, 1])

    # Exact bin integrals of r_c and |r_c|, |r_e|
    exact_c = np.zeros((K, G, J))
    mass_c = np.zeros((K, G, J))
    for k in range(K):
        h = z_edges[k + 1] - z_edges[k]
        for g in range(G):
            x = np.linspace(E_edges[g], E_edges[g + 1], 200001)
            s = (2 * x - E_edges[g] - E_edges[g + 1]) / (E_edges[g + 1] - E_edges[g])
            for j in range(J):
                c_E = c[k, g, j, 0] + (c[k, g, j, 1] * s if P > 1 else 0.0)
                psi_E = psi[k, g, j, 0] + (psi[k, g, j, 1] * s if P > 1 else 0.0)
                y = c_E - psi_E * np.interp(x, E_grid, Sigma[k])
                exact_c[k, g, j] = h * 1.0 * np.trapezoid(y, x)
                mass_c[k, g, j] = h * collision_residual_mass(
                    c[k, g, j],
                    psi[k, g, j],
                    E_edges[g],
                    E_edges[g + 1],
                    E_grid,
                    Sigma[k],
                )
    mu_mean = np.array([-0.5, 0.5])  # int_j mu dmu
    dE = np.diff(E_edges)
    # Edge masses as the residual computes them (any positive masses are unbiased)
    D0 = jump[..., 0]
    D1 = jump[..., 1] if P > 1 else np.zeros_like(D0)
    mass_e = (np.abs(D0) + np.abs(D1)) * dE[None, :, None] * 0.5
    exact_e = -D0 * dE[None, :, None] * mu_mean[None, None, :]

    mass = np.concatenate((mass_c.ravel(), mass_e.ravel()))
    particle_bank_module.set_bank_size(simulation["bank_source"], 0)
    sample_collision_edge(
        0,
        N,
        N,
        N,
        np.uint64(9),
        z_edges,
        E_edges,
        mu_edges,
        psi,
        c,
        mass,
        jump,
        xs_offsets,
        xs_energy,
        xs_total,
        cell_material,
        simulation,
        antithetic,
    )
    particles = banked(simulation)
    assert len(particles) == N

    # Face particles sit within FACE_NUDGE of a face
    on_face = (
        np.min(np.abs(particles["z"][:, None] - z_edges[None, :]), axis=1)
        < 2 * FACE_NUDGE
    )
    mean, std = tally_bins(particles[~on_face], z_edges, E_edges, mu_edges, N)
    # Pair partners land in the same bin: the independent-sample standard error
    #   understates the bin-sum error by up to sqrt(2)
    inflation = np.sqrt(2.0) if antithetic else 1.0
    std *= inflation
    assert np.all(np.abs(mean - exact_c) < 5 * std + 1e-12)

    face = np.argmin(
        np.abs(particles["z"][on_face][:, None] - z_edges[None, :]), axis=1
    )
    face_particles = particles[on_face]
    g = np.searchsorted(E_edges, face_particles["E"], side="right") - 1
    j = np.searchsorted(mu_edges, face_particles["uz"], side="right") - 1
    total = np.zeros((K + 1, G, J))
    square = np.zeros((K + 1, G, J))
    np.add.at(total, (face, g, j), face_particles["w"])
    np.add.at(square, (face, g, j), face_particles["w"] ** 2)
    mean_e = total / N
    std_e = np.sqrt(np.maximum(square / N - mean_e**2, 0.0) / N)
    std_e *= inflation
    assert np.all(np.abs(mean_e - exact_e) < 5 * std_e + 1e-12)


@numba_only
@requires_nuclide("O16")
@pytest.mark.parametrize("proposal", [PROPOSAL_DEFENSIVE, PROPOSAL_UNIFORM_LETHARGY])
def test_scattering_correction_unbiased(prepare_simulation, monkeypatch, proposal):
    """With M deliberately halved, E[r_s] per outgoing bin = h sum psi (M_true - M_used)."""
    N = 40000
    monkeypatch.setenv("MCDC_LIB", DATA_DIR)
    material = mcdc.Material(nuclide_composition={"O16": 0.05})

    def configure(simulation):
        simulation.settings.N_particle = N
        simulation.settings.use_source_bank = True

    container, data = prepare_simulation(
        cells=(mcdc.Cell(fill=material),), configure=configure
    )
    simulation = container[0]
    nuclide = simulation["nuclides"][0]

    z_edges = np.array([0.0, 1.0])
    E_edges = np.logspace(6.0, np.log10(6.0e6), 5)  # elastic only (below thresholds)
    mu_edges = np.array([-1.0, 0.0, 1.0])
    G, J = 4, 2
    table = build_angular_transfer_table(mu_edges, 2049)
    M_true, _ = nuclide_transfer_moments(
        simulation, data, nuclide, E_edges, mu_edges, table
    )
    M_true *= 0.05  # atom density
    M_used = 0.5 * M_true
    psi = np.random.default_rng(2).uniform(0.5, 1.5, (1, G, J))
    psi_coefficients = psi[..., None]  # constant energy basis

    tables = _reaction_tables(simulation, data, [material.ID], False)
    xs = [
        material_total_xs(
            simulation,
            data,
            simulation["materials"][material.ID],
            E_edges[0],
            E_edges[-1],
        )
    ]
    emission = _emission_integrals(simulation, data, tables, xs, E_edges)
    particle_bank_module.set_bank_size(simulation["bank_source"], 0)
    sample_correction(
        0,
        N,
        N,
        N,
        np.uint64(3),
        proposal,
        0.5,
        z_edges,
        E_edges,
        mu_edges,
        psi_coefficients,
        M_used[None],
        emission,
        np.array([0]),
        *tables,
        simulation,
        data,
    )
    particles = banked(simulation)
    mean, std = tally_bins(particles, z_edges, E_edges, mu_edges, N)
    expected = np.einsum("pqgj,pq->gj", (M_true - M_used)[0, :, :, 0], psi[0])[None]
    assert np.all(np.abs(mean - expected) < 5 * std + 1e-12 * expected.max())
    w = particles["w"]
    std_total = np.sqrt(max(np.sum(w**2) / N - (np.sum(w) / N) ** 2, 0.0) / N)
    assert abs(mean.sum() - expected.sum()) < 5 * std_total


@numba_only
@requires_nuclide("O16")
def test_scattering_correction_integrated(prepare_simulation, monkeypatch):
    """
    Integrated-E_in sampler: (1) int_bin T dE dmu = sum M psi~ (the transfer moments),
    (2) with M deliberately halved, E[r_s] per bin = h sum psi (M_true - M_used).
    """
    N = 4000
    monkeypatch.setenv("MCDC_LIB", DATA_DIR)
    material = mcdc.Material(nuclide_composition={"O16": 0.05})

    def configure(simulation):
        simulation.settings.N_particle = N
        simulation.settings.use_source_bank = True

    container, data = prepare_simulation(
        cells=(mcdc.Cell(fill=material),), configure=configure
    )
    simulation = container[0]
    nuclide = simulation["nuclides"][0]

    z_edges = np.array([0.0, 1.0])
    E_edges = np.logspace(6.0, np.log10(6.0e6), 5)  # elastic only (below thresholds)
    mu_edges = np.array([-1.0, 0.0, 1.0])
    G, J = 4, 2
    table = build_angular_transfer_table(mu_edges, 2049)
    M_true, _ = nuclide_transfer_moments(
        simulation, data, nuclide, E_edges, mu_edges, table
    )
    M_true *= 0.05  # atom density
    psi = np.random.default_rng(2).uniform(0.5, 1.5, (1, G, J))[..., None]
    tables = _reaction_tables(simulation, data, [material.ID], False)
    cell_material = np.array([0])

    # (1) Bin integrals of T: composite Gauss-Legendre in E_out (T has kinks where the
    #     kinematic limits of incident-bin edges cross E_out), Gauss in mu_out
    S_true = binned_in_scatter(psi, M_true[None], cell_material, E_edges, mu_edges)
    x, w = np.polynomial.legendre.leggauss(8)
    for g in range(G):
        for j in range(J):
            sub = np.linspace(E_edges[g], E_edges[g + 1], 33)
            integral = 0.0
            for a, b in zip(sub[:-1], sub[1:]):
                for xe, we in zip(x, w):
                    E_out = 0.5 * (a + b) + 0.5 * (b - a) * xe
                    for xm, wm in zip(x, w):
                        mu_out = 0.5 * (mu_edges[j] + mu_edges[j + 1]) + 0.5 * xm
                        T = in_scatter_density(
                            0,
                            E_out,
                            mu_out,
                            psi,
                            E_edges,
                            mu_edges,
                            0,
                            tables[0][1],
                            *tables[1:],
                            simulation,
                            data,
                        )
                        integral += 0.25 * (b - a) * we * wm * T
            average = integral / ((E_edges[g + 1] - E_edges[g]) * 1.0)
            # The test quadrature converges as O(h^2) across those kinks (1e-6 with
            #   128 panels); 32 panels give a few 1e-4
            assert average == pytest.approx(S_true[0, g, j, 0], rel=5e-4)

    # (2) Unbiased with M halved
    S_used = 0.5 * S_true
    mass = correction_masses(
        psi,
        S_used,
        psi,
        S_used,
        False,
        z_edges,
        E_edges,
        mu_edges,
        cell_material,
        *tables,
        simulation,
        data,
    )
    particle_bank_module.set_bank_size(simulation["bank_source"], 0)
    sample_correction_integrated(
        0,
        N,
        N,
        N,
        np.uint64(3),
        z_edges,
        E_edges,
        mu_edges,
        psi,
        S_used,
        psi,
        S_used,
        False,
        mass,
        cell_material,
        *tables,
        simulation,
        data,
    )
    particles = banked(simulation)
    mean, std = tally_bins(particles, z_edges, E_edges, mu_edges, N)
    expected = 0.5 * np.einsum("pqgj,pq->gj", M_true[0, :, :, 0], psi[0, ..., 0])[None]
    assert np.all(np.abs(mean - expected) < 5 * std + 1e-12 * expected.max())
