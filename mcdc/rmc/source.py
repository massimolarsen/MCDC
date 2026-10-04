"""
Residual source particles with signed weights.

Every residual term is sampled with an unbiased estimator: a particle drawn from a
proposal density q carries weight w = (N / N_term) r(x) / q(x), so that MC/DC's
per-history tally normalization (division by N, the total number of particles in the
iteration) reproduces the response to r.

  - r_c + r_e: bins by |r| mass, uniform within bins (q > 0 wherever r != 0).
  - r_s, r_f over y = (E_in, mu_in, E_out, mu_out) (and z):
      T(y) - s~(y),  T = psi~ sum_r N_n sigma_r(E_in) tau_r(y),
                     s~ = M[g', j', g, j] psi~[g', j'] / (dE_g dmu_j dE_g' dmu_j'),
    with proposals:
      * "defensive": a q_T + (1 - a) q_S, where q_T draws (k, g', j') ~ h |psi~| R,
        E_in, mu_in uniform, a reaction by emission rate, and one emitted neutron with
        MC/DC's samplers (density P/(h dE dmu) T / (psi~ Psi(E_in))), and q_S draws
        (k, g', j', g, j) ~ h |M psi~| uniformly within bins;
      * "uniform-linear" / "uniform-lethargy": z, mu uniform; E_in, E_out uniform in
        E or in ln E over the window (the thesis proposal).
"""

import math

import numpy as np

from numba import njit, uint64

####

import mcdc.numba_types as type_
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng

from mcdc.constant import (
    ANGLE_ENERGY_CORRELATED,
    BOLTZMANN_K,
    ANGLE_ISOTROPIC,
    NEUTRON_REACTION_FISSION,
    PARTICLE_NEUTRON,
    PI,
    REFERENCE_FRAME_COM,
)
from mcdc.rmc.kernel import GK_WG, GK_WK, GK_X, _pieces, _support_crossings
from mcdc.rmc.kinematics import com_to_lab, elastic_E_out, elastic_mu_lab
from mcdc.rmc.residual import _trial_value
from mcdc.rmc.reaction import (
    KERNEL_ELASTIC,
    KERNEL_LEVEL,
    _inelastic_spectrum,
    _spectrum_weight,
    emission_kernel_bin,
    fission_yield,
    inelastic_yield,
    is_free_gas,
    sample_free_gas,
)
from mcdc.transport.distribution import (
    sample_correlated_distribution_with_scale,
    sample_distribution,
    sample_distribution_with_scale,
    sample_isotropic_cosine,
)
from mcdc.transport.physics.neutron.native import reaction_micro_xs

PROPOSAL_DEFENSIVE = 0
PROPOSAL_UNIFORM_LINEAR = 1
PROPOSAL_UNIFORM_LETHARGY = 2

# Distance [cm] that face particles are placed into their downwind cell
FACE_NUDGE = 1.0e-9

# ======================================================================================
# Helpers
# ======================================================================================


@njit(cache=True)
def _bin(x, edges):
    """Bin index of x in edges (right edge included in the last bin); -1 if outside."""
    if x < edges[0] or x > edges[-1]:
        return -1
    idx = np.searchsorted(edges, x, side="right") - 1
    return min(idx, len(edges) - 2)


@njit(cache=True)
def _sample_cdf(cdf, xi):
    """Index i with cdf[i] <= xi * cdf[-1] < cdf[i + 1] (cdf[0] = 0)."""
    idx = np.searchsorted(cdf, xi * cdf[-1], side="right") - 1
    return min(max(idx, 0), len(cdf) - 2)


@njit(cache=True)
def _new_container(seed):
    container = np.zeros(1, type_.particle_data)
    container[0]["rng_seed"] = seed
    return container


@njit(cache=True)
def _bank(container, z, E, mu, w, simulation):
    particle = container[0]
    azi = 2.0 * PI * rng.lcg(container)
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    particle["x"] = 0.0
    particle["y"] = 0.0
    particle["z"] = z
    particle["t"] = 0.0
    particle["ux"] = s * math.cos(azi)
    particle["uy"] = s * math.sin(azi)
    particle["uz"] = mu
    particle["E"] = E
    particle["w"] = w
    particle["particle_type"] = PARTICLE_NEUTRON
    particle_bank_module.bank_source_particle(container, simulation)


@njit(cache=True)
def _bank_dead(container, simulation):
    """A zero-weight placeholder outside the energy window (killed at birth), so that
    every sample index banks exactly one history (MPI work slices stay aligned)."""
    _bank(container, 0.0, -1.0, 0.0, 0.0, simulation)


# ======================================================================================
# Collision and edge residual (r_c + r_e)
# ======================================================================================


@njit(cache=True)
def sample_collision_edge(
    i_start,
    i_end,
    N_term,
    N_total,
    seed,
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
    antithetic=False,
):
    """
    Bank sample indices [i_start, i_end) of the N_term particles drawn from r_c
    (mass[:K*G*J]) and r_e (mass[K*G*J:]).

    antithetic: indices 2m and 2m + 1 are drawn from the same random numbers, the
    second with its energy mirrored in the bin (x -> -x). Each particle is still a
    uniform-in-bin sample (unbiased); the pair also shares its transport random
    numbers, so the P_1 (slope) tallies of the bin-average part of the residual
    cancel instead of adding noise.

    mass is the flattened concatenation of the collision masses (k, g, j) and edge
    masses (f, g, j); xs_* hold each material's union energy grid and Sigma_t. psi, c
    and jump carry the energy coefficients of each bin in their last axis.
    """
    K, G, J = psi.shape[:3]
    N_collision = K * G * J
    cdf = np.zeros(len(mass) + 1)
    cdf[1:] = np.cumsum(mass)
    total = cdf[-1]
    if N_term == 0:
        return
    scale = N_total / N_term

    for i in range(i_start, i_end):
        pair = i // 2 if antithetic else i
        mirror = antithetic and i % 2 == 1
        container = _new_container(rng.split_seed(uint64(pair), seed))
        if total == 0.0:
            _bank_dead(container, simulation)
            continue
        idx = _sample_cdf(cdf, rng.lcg(container))

        if idx < N_collision:
            k = idx // (G * J)
            g = (idx // J) % G
            j = idx % J
            z = z_edges[k] + rng.lcg(container) * (z_edges[k + 1] - z_edges[k])
            E = E_edges[g] + rng.lcg(container) * (E_edges[g + 1] - E_edges[g])
            if mirror:
                E = E_edges[g] + E_edges[g + 1] - E
            mu = mu_edges[j] + rng.lcg(container) * (mu_edges[j + 1] - mu_edges[j])

            m = cell_material[k]
            start = xs_offsets[m]
            end = xs_offsets[m + 1]
            Sigma = np.interp(E, xs_energy[start:end], xs_total[start:end])
            E_a = E_edges[g]
            E_b = E_edges[g + 1]
            r = _trial_value(c[k, g, j], E, E_a, E_b) - Sigma * _trial_value(
                psi[k, g, j], E, E_a, E_b
            )
            volume = (
                (z_edges[k + 1] - z_edges[k])
                * (E_edges[g + 1] - E_edges[g])
                * (mu_edges[j + 1] - mu_edges[j])
            )
            w = scale * r * volume * total / mass[idx]
        else:
            idx -= N_collision
            f = idx // (G * J)
            g = (idx // J) % G
            j = idx % J
            E = E_edges[g] + rng.lcg(container) * (E_edges[g + 1] - E_edges[g])
            if mirror:
                E = E_edges[g] + E_edges[g + 1] - E
            # mu ~ |mu| within the bin (bins do not straddle 0)
            a = mu_edges[j]
            b = mu_edges[j + 1]
            xi = rng.lcg(container)
            if a >= 0.0:
                mu = math.sqrt(a * a + xi * (b * b - a * a))
            else:
                mu = -math.sqrt(b * b + xi * (a * a - b * b))
            z = z_edges[f] + math.copysign(FACE_NUDGE, mu)
            # q = mass / total / dE * |mu| / int_j |mu| dmu
            D = _trial_value(jump[f, g, j], E, E_edges[g], E_edges[g + 1])
            abs_mu = 0.5 * abs(b * abs(b) - a * abs(a))
            w = (
                scale
                * math.copysign(1.0, -mu)
                * D
                * total
                * (E_edges[g + 1] - E_edges[g])
                * abs_mu
                / mass[N_collision + idx]
            )

        _bank(container, z, E, mu, w, simulation)


@njit(cache=True)
def sample_collision_linear_z(
    i_start,
    i_end,
    N_term,
    N_total,
    seed,
    z_edges,
    E_edges,
    mu_edges,
    c_left,
    c_right,
    psi_left,
    psi_right,
    delta,
    mass,
    xs_offsets,
    xs_energy,
    xs_total,
    cell_material,
    simulation,
    antithetic=False,
):
    """
    Bank sample indices [i_start, i_end) of the N_term particles drawn from r_c of the
    continuous piecewise-linear trial space in z (`ResidualLinearZ`): bin (k, g, j)
    with probability ~ mass[k, g, j], (z, E, mu) uniform in it, weight
    (N_total / N_term) r_c / q with

        r_c = c(s, E) - Sigma_t(E) psi~(s, E) - mu delta(E),   s = (z - z_k) / h_k,

    c and psi~ interpolated linearly in s between the left and right nodal values.
    Antithetic pairs mirror the energy in the bin, as in `sample_collision_edge`.
    """
    K, G, J = mass.shape
    if N_term == 0:
        return
    scale = N_total / N_term
    cdf = np.zeros(K * G * J + 1)
    cdf[1:] = np.cumsum(mass.ravel())
    total = cdf[-1]

    for i in range(i_start, i_end):
        pair = i // 2 if antithetic else i
        mirror = antithetic and i % 2 == 1
        container = _new_container(rng.split_seed(uint64(pair), seed))
        if total == 0.0:
            _bank_dead(container, simulation)
            continue
        idx = _sample_cdf(cdf, rng.lcg(container))
        k = idx // (G * J)
        g = (idx // J) % G
        j = idx % J
        h = z_edges[k + 1] - z_edges[k]
        E_a = E_edges[g]
        E_b = E_edges[g + 1]
        s = rng.lcg(container)
        z = z_edges[k] + s * h
        E = E_a + rng.lcg(container) * (E_b - E_a)
        if mirror:
            E = E_a + E_b - E
        mu = mu_edges[j] + rng.lcg(container) * (mu_edges[j + 1] - mu_edges[j])

        m = cell_material[k]
        start = xs_offsets[m]
        end = xs_offsets[m + 1]
        Sigma = np.interp(E, xs_energy[start:end], xs_total[start:end])
        c = (1.0 - s) * _trial_value(c_left[k, g, j], E, E_a, E_b) + s * _trial_value(
            c_right[k, g, j], E, E_a, E_b
        )
        psi = (1.0 - s) * _trial_value(
            psi_left[k, g, j], E, E_a, E_b
        ) + s * _trial_value(psi_right[k, g, j], E, E_a, E_b)
        r = c - Sigma * psi - mu * _trial_value(delta[k, g, j], E, E_a, E_b)
        volume = h * (E_b - E_a) * (mu_edges[j + 1] - mu_edges[j])
        w = scale * r * volume * total / mass[k, g, j]
        _bank(container, z, E, mu, w, simulation)


# ======================================================================================
# Emission sampling (one emitted neutron, density sum_n w_n f_n / yield)
# ======================================================================================


@njit(cache=True)
def sample_emission(E_in, reaction, ktype, nuclide, simulation, data, container):
    """Lab (E_out, mu0) of one emitted neutron, with MC/DC's distribution samplers."""
    awr = nuclide["atomic_weight_ratio"]

    if ktype == KERNEL_ELASTIC:
        if is_free_gas(E_in, reaction, nuclide):
            kT = BOLTZMANN_K * nuclide["temperature"]
            return sample_free_gas(E_in, awr, kT, container)
        elastic = simulation["neutron_elastic_scattering_reactions"][reaction["sub_ID"]]
        mu_distribution = simulation["distributions"][elastic["mu_table_ID"]]
        mu_cm = sample_distribution(E_in, mu_distribution, container, simulation, data)
        return elastic_E_out(E_in, mu_cm, awr), elastic_mu_lab(mu_cm, awr)

    if reaction["sub_type"] == NEUTRON_REACTION_FISSION:
        fission = simulation["neutron_fission_reactions"][reaction["sub_ID"]]
        spectrum = simulation["distributions"][fission["spectrum_ID"]]
        angle_type = fission["angle_type"]
        mu_ID = fission["mu_ID"]
    else:
        inelastic = simulation["neutron_inelastic_scattering_reactions"][
            reaction["sub_ID"]
        ]
        angle_type = inelastic["angle_type"]
        mu_ID = inelastic["mu_ID"]

        # Spectrum n with probability proportional to its expected emissions
        N_spectrum = inelastic["N_spectrum"]
        total = 0.0
        for n in range(N_spectrum):
            total += _spectrum_weight(E_in, inelastic, n, simulation, data)
        xi = rng.lcg(container) * total
        cumulative = 0.0
        n_pick = N_spectrum - 1
        for n in range(N_spectrum):
            cumulative += _spectrum_weight(E_in, inelastic, n, simulation, data)
            if xi < cumulative:
                n_pick = n
                break
        spectrum = _inelastic_spectrum(inelastic, n_pick, simulation, data)

    if ktype == KERNEL_LEVEL:
        level = simulation["level_scattering_distributions"][spectrum["sub_ID"]]
        E_x = level["C2"] * (E_in - level["C1"])
        if angle_type == ANGLE_ISOTROPIC:
            mu_x = sample_isotropic_cosine(container)
        else:
            mu_x = sample_distribution(
                E_in, simulation["distributions"][mu_ID], container, simulation, data
            )
    elif angle_type == ANGLE_ENERGY_CORRELATED:
        E_x, mu_x = sample_correlated_distribution_with_scale(
            E_in, spectrum, container, simulation, data
        )
    else:
        if angle_type == ANGLE_ISOTROPIC:
            mu_x = sample_isotropic_cosine(container)
        else:
            mu_x = sample_distribution(
                E_in, simulation["distributions"][mu_ID], container, simulation, data
            )
        E_x = sample_distribution_with_scale(
            E_in, spectrum, container, simulation, data
        )

    if reaction["reference_frame"] == REFERENCE_FRAME_COM:
        return com_to_lab(E_in, E_x, mu_x, awr)
    return E_x, mu_x


@njit(cache=True)
def reaction_yield(E_in, reaction, ktype, nuclide, simulation, data):
    if ktype == KERNEL_ELASTIC:
        return 1.0
    if reaction["sub_type"] == NEUTRON_REACTION_FISSION:
        return fission_yield(E_in, nuclide, simulation, data)
    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    return inelastic_yield(E_in, inelastic, simulation, data)


# ======================================================================================
# Scattering / fission residual (r_s, r_f)
# ======================================================================================


@njit(cache=True)
def _emission_rate(
    E_in, start, end, rx_ID, rx_nuclide, rx_density, rx_type, simulation, data
):
    """Psi(E_in) = sum_r N_n sigma_r y_r over a material's reactions [start, end)."""
    total = 0.0
    for i in range(start, end):
        reaction = simulation["neutron_reactions"][rx_ID[i]]
        nuclide = simulation["nuclides"][rx_nuclide[i]]
        sigma = reaction_micro_xs(E_in, reaction, nuclide, data)
        if sigma > 0.0:
            total += (
                rx_density[i]
                * sigma
                * reaction_yield(E_in, reaction, rx_type[i], nuclide, simulation, data)
            )
    return total


@njit(cache=True)
def _bin_kernels(
    E_in,
    E_out,
    mu_out,
    mu_edges,
    start,
    end,
    rx_ID,
    rx_nuclide,
    rx_density,
    rx_type,
    simulation,
    data,
    out,
):
    """out[j'] = U_j' = sum_r N_n sigma_r(E_in) tau_bar_{r, j'}(E_in -> E_out, mu_out)."""
    out[:] = 0.0
    J = len(mu_edges) - 1
    for i in range(start, end):
        reaction = simulation["neutron_reactions"][rx_ID[i]]
        nuclide = simulation["nuclides"][rx_nuclide[i]]
        sigma = reaction_micro_xs(E_in, reaction, nuclide, data)
        if sigma <= 0.0:
            continue
        for jp in range(J):
            out[jp] += (
                rx_density[i]
                * sigma
                * emission_kernel_bin(
                    E_in,
                    E_out,
                    mu_out,
                    mu_edges[jp],
                    mu_edges[jp + 1],
                    reaction,
                    rx_type[i],
                    nuclide,
                    simulation,
                    data,
                )
            )


@njit(cache=True)
def sample_correction(
    i_start,
    i_end,
    N_term,
    N_total,
    seed,
    proposal,
    defensive_fraction,
    z_edges,
    E_edges,
    mu_edges,
    psi,
    M,
    emission_integral,
    cell_material,
    rx_offsets,
    rx_ID,
    rx_nuclide,
    rx_density,
    rx_type,
    simulation,
    data,
):
    """
    Bank sample indices [i_start, i_end) of the N_term particles estimating one
    correction term (scattering or fission),

        r(z, E_out, mu_out) = int dE_in [T_bar - s_bar](E_in, E_out, mu_out),
        T_bar = sum_j' psi~[g', j'] U_j',   U_j' = sum_r N_n sigma_r tau_bar_{r, j'},
        s_bar = sum_j' M[g', j', g, j] psi~[g', j'] / (dE_g dmu_j dE_g'),

    i.e. with the incident polar cosine integrated out analytically over each
    trial-space bin (which keeps T_bar bounded and smooth in mu_out). M[m] holds the
    term's transfer moments for material m, and the reaction arrays (by material, via
    rx_offsets) hold only the term's reactions. With linear energy coefficients,
    psi~[g', j'] is evaluated at E_in and s_bar is the projected (linear in E_out)
    source per unit E_in.
    """
    K, G, J, P = psi.shape
    if N_term == 0:
        return
    scale = N_total / N_term
    h = z_edges[1:] - z_edges[:-1]
    dE = E_edges[1:] - E_edges[:-1]
    dmu = mu_edges[1:] - mu_edges[:-1]
    H = z_edges[-1] - z_edges[0]
    E_min = E_edges[0]
    E_max = E_edges[-1]
    L = math.log(E_max / E_min)
    U = np.zeros(J)

    # Defensive-mixture tables
    #   q_T: (k, g') ~ h sum_j' |psi~| dmu_j' int_g' Psi,  j' | (k, g') ~ |psi~| dmu_j',
    #        with Psi the total emission rate (emission_integral[m, g'] = int_g' Psi,
    #        including emissions that leave the energy grid, so T_bar / q_T is bounded)
    #   q_S: (k, g', g, j) ~ h sum_a (2a + 1) |sum_{a', j'} M psi~|
    cdf_T = np.zeros(K * G + 1)
    P_jp = np.zeros((K, G, J))
    S_bar = np.zeros((K, G, P, G, J))
    S_mass = np.zeros((K, G, G, J))
    cdf_S = np.zeros(K * G * G * J + 1)
    if proposal == PROPOSAL_DEFENSIVE:
        for k in range(K):
            Mk = M[cell_material[k]]
            for gp in range(G):
                row = 0.0
                for jp in range(J):
                    P_jp[k, gp, jp] = np.sum(np.abs(psi[k, gp, jp])) * dmu[jp]
                    row += P_jp[k, gp, jp]
                    for b in range(P):
                        S_bar[k, gp] += Mk[b, gp, jp] * psi[k, gp, jp, b]
                if row > 0.0:
                    P_jp[k, gp] /= row
                cdf_T[k * G + gp + 1] = (
                    h[k] * row * emission_integral[cell_material[k], gp]
                )
                for g in range(G):
                    for j in range(J):
                        for a in range(P):
                            S_mass[k, gp, g, j] += (2.0 * a + 1.0) * abs(
                                S_bar[k, gp, a, g, j]
                            )
                        cdf_S[((k * G + gp) * G + g) * J + j + 1] = (
                            h[k] * S_mass[k, gp, g, j]
                        )
        Z_T = np.sum(cdf_T)
        Z_S = np.sum(cdf_S)
        if Z_T == 0.0:
            for i in range(i_start, i_end):
                _bank_dead(_new_container(rng.split_seed(uint64(i), seed)), simulation)
            return
        P_T = cdf_T[1:] / Z_T
        cdf_T = np.cumsum(cdf_T)
        cdf_S = np.cumsum(cdf_S)

    for i in range(i_start, i_end):
        container = _new_container(rng.split_seed(uint64(i), seed))

        # ==============================================================================
        # Sample y = (k, z, E_in, E_out, mu_out)
        # ==============================================================================

        if proposal == PROPOSAL_DEFENSIVE:
            use_T = rng.lcg(container) < defensive_fraction
            if use_T:
                idx = _sample_cdf(cdf_T, rng.lcg(container))
                k = idx // G
                gp = idx % G
            else:
                idx = _sample_cdf(cdf_S, rng.lcg(container))
                k = idx // (G * G * J)
                gp = (idx // (G * J)) % G
                g = (idx // J) % G
                j = idx % J
            z = z_edges[k] + rng.lcg(container) * h[k]
            E_in = E_edges[gp] + rng.lcg(container) * dE[gp]
            m = cell_material[k]
            start = rx_offsets[m]
            end = rx_offsets[m + 1]

            if use_T:
                # Incident bin, reaction by emission rate, then one emitted neutron
                xi = rng.lcg(container)
                cumulative = 0.0
                jp = J - 1
                for jj in range(J):
                    cumulative += P_jp[k, gp, jj]
                    if xi < cumulative:
                        jp = jj
                        break
                mu_in = mu_edges[jp] + rng.lcg(container) * dmu[jp]
                rate = _emission_rate(
                    E_in,
                    start,
                    end,
                    rx_ID,
                    rx_nuclide,
                    rx_density,
                    rx_type,
                    simulation,
                    data,
                )
                if rate == 0.0:
                    _bank_dead(container, simulation)
                    continue
                xi = rng.lcg(container) * rate
                cumulative = 0.0
                pick = end - 1
                for r in range(start, end):
                    reaction = simulation["neutron_reactions"][rx_ID[r]]
                    nuclide = simulation["nuclides"][rx_nuclide[r]]
                    sigma = reaction_micro_xs(E_in, reaction, nuclide, data)
                    if sigma > 0.0:
                        cumulative += (
                            rx_density[r]
                            * sigma
                            * reaction_yield(
                                E_in, reaction, rx_type[r], nuclide, simulation, data
                            )
                        )
                    if xi < cumulative:
                        pick = r
                        break
                reaction = simulation["neutron_reactions"][rx_ID[pick]]
                nuclide = simulation["nuclides"][rx_nuclide[pick]]
                E_out, mu0 = sample_emission(
                    E_in, reaction, rx_type[pick], nuclide, simulation, data, container
                )
                azi = 2.0 * PI * rng.lcg(container)
                mu_out = mu_in * mu0 + math.sqrt(max(0.0, 1.0 - mu_in * mu_in)) * (
                    math.sqrt(max(0.0, 1.0 - mu0 * mu0)) * math.cos(azi)
                )
                g = _bin(E_out, E_edges)
                j = _bin(mu_out, mu_edges)
            else:
                E_out = E_edges[g] + rng.lcg(container) * dE[g]
                mu_out = mu_edges[j] + rng.lcg(container) * dmu[j]
        else:
            z = z_edges[0] + rng.lcg(container) * H
            k = _bin(z, z_edges)
            if proposal == PROPOSAL_UNIFORM_LINEAR:
                E_in = E_min + rng.lcg(container) * (E_max - E_min)
                E_out = E_min + rng.lcg(container) * (E_max - E_min)
            else:
                E_in = E_min * math.exp(rng.lcg(container) * L)
                E_out = E_min * math.exp(rng.lcg(container) * L)
            mu_out = -1.0 + 2.0 * rng.lcg(container)
            gp = _bin(E_in, E_edges)
            g = _bin(E_out, E_edges)
            j = _bin(mu_out, mu_edges)
            m = cell_material[k]
            start = rx_offsets[m]
            end = rx_offsets[m + 1]

        # Outside the energy grid: killed at birth by the energy window
        if g == -1:
            _bank_dead(container, simulation)
            continue

        # ==============================================================================
        # Weight (T_bar - s_bar) / q
        # ==============================================================================

        _bin_kernels(
            E_in,
            E_out,
            mu_out,
            mu_edges,
            start,
            end,
            rx_ID,
            rx_nuclide,
            rx_density,
            rx_type,
            simulation,
            data,
            U,
        )
        T_bar = 0.0
        s_bar = 0.0
        x_out = (2.0 * E_out - E_edges[g] - E_edges[g + 1]) / dE[g]
        for jp in range(J):
            T_bar += (
                _trial_value(psi[k, gp, jp], E_in, E_edges[gp], E_edges[gp + 1]) * U[jp]
            )
            for b in range(P):
                for a in range(P):
                    s_bar += (
                        M[m, b, gp, jp, a, g, j]
                        * psi[k, gp, jp, b]
                        * (2.0 * a + 1.0)
                        * (x_out if a == 1 else 1.0)
                    )
        s_bar /= dE[g] * dmu[j] * dE[gp]

        if proposal == PROPOSAL_DEFENSIVE:
            rate = _emission_rate(
                E_in,
                start,
                end,
                rx_ID,
                rx_nuclide,
                rx_density,
                rx_type,
                simulation,
                data,
            )
            q_T = 0.0
            if rate > 0.0:
                angular = 0.0
                for jp in range(J):
                    angular += P_jp[k, gp, jp] / dmu[jp] * U[jp]
                q_T = P_T[k * G + gp] / (h[k] * dE[gp]) * angular / rate
            q_S = h[k] * S_mass[k, gp, g, j] / Z_S / (h[k] * dE[gp] * dE[g] * dmu[j])
            q = defensive_fraction * q_T + (1.0 - defensive_fraction) * q_S
        elif proposal == PROPOSAL_UNIFORM_LINEAR:
            q = 1.0 / (H * 2.0 * (E_max - E_min) ** 2)
        else:
            q = 1.0 / (H * 2.0 * L * L * E_in * E_out)

        if q <= 0.0:
            _bank_dead(container, simulation)
            continue
        w = scale * (T_bar - s_bar) / q
        _bank(container, z, E_out, mu_out, w, simulation)


# ======================================================================================
# Scattering / fission residual with the incident energy integrated (low variance)
# ======================================================================================
#
# The pointwise sampler above draws y = (z, E_in, E_out, mu_out) and scores
# T_bar - s_bar per unit E_in. Here only (z, E_out, mu_out) is sampled, and the
# incident-energy integral is done deterministically:
#
#   r(z, E_out, mu_out) = T(E_out, mu_out) - S_bar[k, g, j],
#   T = sum_{g', j'} psi~[k, g', j'] int_{g'} dE_in U_j'(E_in -> E_out, mu_out),
#   S_bar = sum_{g', j'} M[g', j', g, j] psi~[k, g', j'] / (dE_g dmu_j),
#
# so the weights scale with |r| instead of with the pointwise in-scatter density.

INSCATTER_TOLERANCE = 1e-6
INSCATTER_MAX_DEPTH = 16


@njit(cache=True)
def _psi_weighted_kernels(
    E_in,
    k,
    E_out,
    mu_out,
    psi,
    E_edges,
    mu_edges,
    start,
    end,
    rx_ID,
    rx_nuclide,
    rx_density,
    rx_type,
    simulation,
    data,
    U,
):
    """sum_j' psi~[k, j'](E_in) U_j'(E_in -> E_out, mu_out)."""
    gp = _bin(E_in, E_edges)
    if gp < 0:
        return 0.0
    _bin_kernels(
        E_in,
        E_out,
        mu_out,
        mu_edges,
        start,
        end,
        rx_ID,
        rx_nuclide,
        rx_density,
        rx_type,
        simulation,
        data,
        U,
    )
    value = 0.0
    for jp in range(len(U)):
        value += (
            _trial_value(psi[k, gp, jp], E_in, E_edges[gp], E_edges[gp + 1]) * U[jp]
        )
    return value


@njit(cache=True)
def in_scatter_density(
    k,
    E_out,
    mu_out,
    psi,
    E_edges,
    mu_edges,
    start,
    end,
    rx_ID,
    rx_nuclide,
    rx_density,
    rx_type,
    simulation,
    data,
):
    """
    T(E_out, mu_out): emission density (per unit z, E_out and mu_out) of a term's
    reactions [start, end) from the trial-space psi~ of cell k, integrated over
    the incident energy with adaptive G7-K15. Pieces are split at the trial-space
    energy edges (psi~ jumps) and where a reaction's emission support crosses E_out.
    A piece is accepted at INSCATTER_TOLERANCE relative to itself or to its width
    share of the total.
    """
    J = len(mu_edges) - 1
    U = np.zeros(J)
    E_low = E_edges[0]
    E_high = E_edges[-1]

    # Breakpoints
    candidates = E_edges.copy()
    target = np.array([E_out])
    for i in range(start, end):
        reaction = simulation["neutron_reactions"][rx_ID[i]]
        nuclide = simulation["nuclides"][rx_nuclide[i]]
        crossings = _support_crossings(
            E_edges, reaction, nuclide, simulation, data, target
        )
        candidates = np.concatenate((candidates, crossings))
    points = _pieces(E_low, E_high, candidates)
    N_piece = len(points) - 1

    # Coarse pass: total magnitude
    S = 0.0
    for p in range(N_piece):
        half = 0.5 * (points[p + 1] - points[p])
        center = 0.5 * (points[p + 1] + points[p])
        for q in range(15):
            f = _psi_weighted_kernels(
                center + half * GK_X[q],
                k,
                E_out,
                mu_out,
                psi,
                E_edges,
                mu_edges,
                start,
                end,
                rx_ID,
                rx_nuclide,
                rx_density,
                rx_type,
                simulation,
                data,
                U,
            )
            S += abs(half * GK_WK[q] * f)
    if S == 0.0:
        return 0.0
    width = E_high - E_low

    total = 0.0
    stack_a = np.empty(INSCATTER_MAX_DEPTH + 2)
    stack_b = np.empty(INSCATTER_MAX_DEPTH + 2)
    stack_d = np.empty(INSCATTER_MAX_DEPTH + 2, dtype=np.int64)
    for p in range(N_piece):
        stack_a[0] = points[p]
        stack_b[0] = points[p + 1]
        stack_d[0] = 0
        top = 1
        while top > 0:
            top -= 1
            a = stack_a[top]
            b = stack_b[top]
            depth = stack_d[top]
            half = 0.5 * (b - a)
            center = 0.5 * (b + a)
            K = 0.0
            Gs = 0.0
            for q in range(15):
                f = _psi_weighted_kernels(
                    center + half * GK_X[q],
                    k,
                    E_out,
                    mu_out,
                    psi,
                    E_edges,
                    mu_edges,
                    start,
                    end,
                    rx_ID,
                    rx_nuclide,
                    rx_density,
                    rx_type,
                    simulation,
                    data,
                    U,
                )
                K += half * GK_WK[q] * f
                Gs += half * GK_WG[q] * f
            error = abs(K - Gs)
            if (
                error <= INSCATTER_TOLERANCE * abs(K)
                or error <= INSCATTER_TOLERANCE * S * (b - a) / width
                or depth >= INSCATTER_MAX_DEPTH
                or K == 0.0
            ):
                total += K
            else:
                stack_a[top] = a
                stack_b[top] = center
                stack_d[top] = depth + 1
                stack_a[top + 1] = center
                stack_b[top + 1] = b
                stack_d[top + 1] = depth + 1
                top += 2
    return total


@njit(cache=True)
def binned_in_scatter(psi, M, cell_material, E_edges, mu_edges):
    """
    S_bar[k, g, j, a] = (2a + 1) / (dE dmu)
                        sum_{a', g', j'} M[m_k, a', g', j', a, g, j] psi~[k, g', j', a'].
    """
    K, G, J, P = psi.shape
    out = np.zeros((K, P, G, J))
    for k in range(K):
        Mk = M[cell_material[k]]
        for b in range(P):
            for gp in range(G):
                for jp in range(J):
                    if psi[k, gp, jp, b] != 0.0:
                        out[k] += Mk[b, gp, jp] * psi[k, gp, jp, b]
    result = np.zeros((K, G, J, P))
    for a in range(P):
        for g in range(G):
            for j in range(J):
                result[:, g, j, a] = (
                    (2.0 * a + 1.0)
                    * out[:, a, g, j]
                    / ((E_edges[g + 1] - E_edges[g]) * (mu_edges[j + 1] - mu_edges[j]))
                )
    return result


@njit(cache=True)
def _interpolated_cell(psi_left, psi_right, k, s):
    """psi~ of cell k at s as a one-cell array: (1 - s) left + s right."""
    out = np.empty((1,) + psi_left.shape[1:])
    out[0] = (1.0 - s) * psi_left[k] + s * psi_right[k]
    return out


@njit(cache=True)
def correction_masses(
    psi,
    S_bar,
    psi_right,
    S_right,
    linear_z,
    z_edges,
    E_edges,
    mu_edges,
    cell_material,
    rx_offsets,
    rx_ID,
    rx_nuclide,
    rx_density,
    rx_type,
    simulation,
    data,
):
    """
    Sampling masses A[k, g, j] ~ int_bin |T - S_bar| from 2 x 2 Gauss points per bin,
    plus 0.1 h dE dmu sum_a |S_bar_a| so every bin that receives in-scatter (where r
    can be nonzero) has positive probability. Only the proposal depends on these.
    With linear_z, psi / S_bar are the cells' left and psi_right / S_right their
    right nodal values, and the pilot is evaluated at the cell midpoint.
    """
    K, G, J = psi.shape[:3]
    A = np.zeros((K, G, J))
    x = np.array([-0.5773502691896258, 0.5773502691896258])
    for k in range(K):
        m = cell_material[k]
        start = rx_offsets[m]
        end = rx_offsets[m + 1]
        h = z_edges[k + 1] - z_edges[k]
        for g in range(G):
            dE = E_edges[g + 1] - E_edges[g]
            for j in range(J):
                S_kgj = S_bar[k, g, j]
                if linear_z:
                    S_kgj = 0.5 * (S_bar[k, g, j] + S_right[k, g, j])
                floor = np.sum(np.abs(S_kgj))
                if floor == 0.0:
                    continue
                dmu = mu_edges[j + 1] - mu_edges[j]
                mean = 0.0
                for a in range(2):
                    E_out = E_edges[g] + 0.5 * dE * (1.0 + x[a])
                    for b in range(2):
                        mu_out = mu_edges[j] + 0.5 * dmu * (1.0 + x[b])
                        k_eval, psi_eval = k, psi
                        if linear_z:
                            k_eval = 0
                            psi_eval = _interpolated_cell(psi, psi_right, k, 0.5)
                        T = in_scatter_density(
                            k_eval,
                            E_out,
                            mu_out,
                            psi_eval,
                            E_edges,
                            mu_edges,
                            start,
                            end,
                            rx_ID,
                            rx_nuclide,
                            rx_density,
                            rx_type,
                            simulation,
                            data,
                        )
                        S = _trial_value(S_kgj, E_out, E_edges[g], E_edges[g + 1])
                        mean += 0.25 * abs(T - S)
                A[k, g, j] = h * dE * dmu * (mean + 0.1 * floor)
    return A


@njit(cache=True)
def sample_correction_integrated(
    i_start,
    i_end,
    N_term,
    N_total,
    seed,
    z_edges,
    E_edges,
    mu_edges,
    psi,
    S_bar,
    psi_right,
    S_right,
    linear_z,
    mass,
    cell_material,
    rx_offsets,
    rx_ID,
    rx_nuclide,
    rx_density,
    rx_type,
    simulation,
    data,
):
    """
    Bank sample indices [i_start, i_end) of N_term particles for one correction term:
    bin (k, g, j) with probability P ~ mass, (z, E_out, mu_out) uniform in the bin,
    weight (N_total / N_term) (T - S_bar) / q with q = P / (h dE dmu). With linear_z,
    psi~ and S_bar at z interpolate the cell's left (psi, S_bar) and right
    (psi_right, S_right) nodal values.
    """
    K, G, J = psi.shape[:3]
    if N_term == 0:
        return
    scale = N_total / N_term
    total = np.sum(mass)
    if total == 0.0:
        for i in range(i_start, i_end):
            _bank_dead(_new_container(rng.split_seed(uint64(i), seed)), simulation)
        return
    cdf = np.zeros(K * G * J + 1)
    cdf[1:] = np.cumsum(mass.ravel()) / total

    for i in range(i_start, i_end):
        container = _new_container(rng.split_seed(uint64(i), seed))
        idx = _sample_cdf(cdf, rng.lcg(container))
        k = idx // (G * J)
        g = (idx // J) % G
        j = idx % J
        h = z_edges[k + 1] - z_edges[k]
        dE = E_edges[g + 1] - E_edges[g]
        dmu = mu_edges[j + 1] - mu_edges[j]
        s = rng.lcg(container)
        z = z_edges[k] + s * h
        E_out = E_edges[g] + rng.lcg(container) * dE
        mu_out = mu_edges[j] + rng.lcg(container) * dmu
        m = cell_material[k]
        k_eval, psi_eval = k, psi
        S_kgj = S_bar[k, g, j]
        if linear_z:
            k_eval = 0
            psi_eval = _interpolated_cell(psi, psi_right, k, s)
            S_kgj = (1.0 - s) * S_bar[k, g, j] + s * S_right[k, g, j]
        T = in_scatter_density(
            k_eval,
            E_out,
            mu_out,
            psi_eval,
            E_edges,
            mu_edges,
            rx_offsets[m],
            rx_offsets[m + 1],
            rx_ID,
            rx_nuclide,
            rx_density,
            rx_type,
            simulation,
            data,
        )
        q = mass[k, g, j] / total / (h * dE * dmu)
        S = _trial_value(S_kgj, E_out, E_edges[g], E_edges[g + 1])
        w = scale * (T - S) / q
        _bank(container, z, E_out, mu_out, w, simulation)
