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
from mcdc.rmc.residual import _abs_linear_integral
from mcdc.rmc.source import (
    FACE_NUDGE,
    PROPOSAL_DEFENSIVE,
    PROPOSAL_UNIFORM_LETHARGY,
    sample_collision_edge,
    sample_correction,
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


def test_collision_edge_unbiased(prepare_simulation):
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
    psi = rng.uniform(0.5, 1.5, (K, G, J))
    c = rng.uniform(-1.0, 3.0, (K, G, J))
    jump = rng.uniform(-1.0, 1.0, (K + 1, G, J))
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
            x = np.unique(np.concatenate(([E_edges[g], E_edges[g + 1]], E_grid)))
            x = x[(x >= E_edges[g]) & (x <= E_edges[g + 1])]
            for j in range(J):
                y = c[k, g, j] - psi[k, g, j] * np.interp(x, E_grid, Sigma[k])
                exact_c[k, g, j] = h * 1.0 * np.trapezoid(y, x)
                mass_c[k, g, j] = (
                    h
                    * 1.0
                    * sum(
                        _abs_linear_integral(x[i], x[i + 1], y[i], y[i + 1])
                        for i in range(len(x) - 1)
                    )
                )
    abs_mu = np.array([0.5, 0.5])
    mu_mean = np.array([-0.5, 0.5])  # int_j mu dmu
    dE = np.diff(E_edges)
    mass_e = np.abs(jump) * dE[None, :, None] * abs_mu[None, None, :]
    exact_e = -jump * dE[None, :, None] * mu_mean[None, None, :]

    mass = np.concatenate((mass_c.ravel(), mass_e.ravel()))
    particle_bank_module.set_bank_size(simulation["bank_source"], 0)
    sample_collision_edge(
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
    )
    particles = banked(simulation)
    assert len(particles) == N

    # Face particles sit within FACE_NUDGE of a face
    on_face = (
        np.min(np.abs(particles["z"][:, None] - z_edges[None, :]), axis=1)
        < 2 * FACE_NUDGE
    )
    mean, std = tally_bins(particles[~on_face], z_edges, E_edges, mu_edges, N)
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
        N,
        N,
        np.uint64(3),
        proposal,
        0.5,
        z_edges,
        E_edges,
        mu_edges,
        psi,
        M_used[None],
        emission,
        np.array([0]),
        *tables,
        simulation,
        data,
    )
    particles = banked(simulation)
    mean, std = tally_bins(particles, z_edges, E_edges, mu_edges, N)
    expected = np.einsum("pqgj,pq->gj", M_true - M_used, psi[0])[None]
    assert np.all(np.abs(mean - expected) < 5 * std + 1e-12 * expected.max())
    w = particles["w"]
    std_total = np.sqrt(max(np.sum(w**2) / N - (np.sum(w) / N) ** 2, 0.0) / N)
    assert abs(mean.sum() - expected.sum()) < 5 * std_total
