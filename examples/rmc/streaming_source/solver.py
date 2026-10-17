"""One-group characteristic RMC experiment; see README.md and the math writeup.

All outer faces are vacuum. Arrays are flattened in x-y-z order. Cross sections
may differ in every cell; the external source is isotropic and constant per cell.
This research transport kernel is shared by RMC and SMC, not production MC/DC.
"""

import time

import numpy as np
from numba import njit
from scipy.stats import qmc


@njit(cache=True)
def locate(point, edges):
    return min(max(np.searchsorted(edges, point, side="right") - 1, 0), len(edges) - 2)


@njit(cache=True)
def direction(u, v):
    mu = 2.0 * u - 1.0
    phi = 2.0 * np.pi * v
    s = np.sqrt(max(0.0, 1.0 - mu * mu))
    return np.array([s * np.cos(phi), s * np.sin(phi), mu])


@njit(cache=True)
def next_face(position, omega, ijk, edges):
    distance, axis = np.inf, 0
    for a in range(3):
        if omega[a] == 0.0:
            continue
        face = ijk[a] + (1 if omega[a] > 0.0 else 0)
        d = max(0.0, (edges[a][face] - position[a]) / omega[a])
        if d < distance:
            distance, axis = d, a
    return distance, axis


@njit(cache=True)
def cell_index(ijk, shape):
    return (ijk[0] * shape[1] + ijk[1]) * shape[2] + ijk[2]


@njit(cache=True)
def characteristic(position, omega, edges, sigma, source, slopes=None):
    """Integral of source along a ray, with exact piecewise-constant attenuation."""
    shape = np.array([len(e) - 1 for e in edges])
    ijk = np.array([locate(position[a], edges[a]) for a in range(3)])
    point = position.copy()
    attenuation, value = 1.0, 0.0
    while True:
        c = cell_index(ijk, shape)
        distance, axis = next_face(point, omega, ijk, edges)
        if sigma[c] == 0.0:
            integral = distance
        else:
            integral = -np.expm1(-sigma[c] * distance) / sigma[c]
        emission = source[c]
        slope = 0.0
        if slopes is not None:
            for a in range(3):
                h = edges[a][ijk[a] + 1] - edges[a][ijk[a]]
                emission += slopes[c, a] * (2 * (point[a] - edges[a][ijk[a]]) / h - 1)
                slope += slopes[c, a] * 2 * omega[a] / h
        value += attenuation * emission * integral
        if slopes is not None:
            tau = sigma[c] * distance
            if abs(tau) < 1e-3:
                first_moment = distance**2 * (
                    0.5 - tau / 3 + tau**2 / 8 - tau**3 / 30 + tau**4 / 144
                )
            else:
                first_moment = (integral - distance * np.exp(-tau)) / sigma[c]
            value += attenuation * slope * first_moment
        attenuation *= np.exp(-sigma[c] * distance)
        point += distance * omega
        ijk[axis] += 1 if omega[axis] > 0.0 else -1
        if ijk[axis] < 0 or ijk[axis] >= shape[axis]:
            return value


@njit(cache=True)
def first_flight_matrix(points, edges, sigma, volume):
    """P[i,j]: cell-i average scalar flux from unit isotropic density in cell j."""
    shape = np.array([len(e) - 1 for e in edges])
    cells, samples = points.shape[:2]
    P = np.zeros((cells, cells))
    for j in range(cells):
        ijk0 = np.array(
            [j // (shape[1] * shape[2]), (j // shape[2]) % shape[1], j % shape[2]]
        )
        for n in range(samples):
            u = points[j, n]
            point = np.array(
                [
                    edges[a][ijk0[a]]
                    + u[a] * (edges[a][ijk0[a] + 1] - edges[a][ijk0[a]])
                    for a in range(3)
                ]
            )
            omega = direction(u[3], u[4])
            ijk = ijk0.copy()
            attenuation = 1.0
            while True:
                i = cell_index(ijk, shape)
                distance, axis = next_face(point, omega, ijk, edges)
                integral = (
                    distance
                    if sigma[i] == 0.0
                    else -np.expm1(-sigma[i] * distance) / sigma[i]
                )
                P[i, j] += volume[j] / volume[i] * attenuation * integral / samples
                attenuation *= np.exp(-sigma[i] * distance)
                point += distance * omega
                ijk[axis] += 1 if omega[axis] > 0.0 else -1
                if ijk[axis] < 0 or ijk[axis] >= shape[axis]:
                    break
    return P


@njit(cache=True)
def baseline(points, edges, sigma, source, slopes=None):
    """Independent unbiased RQMC estimates of the trial's cell averages."""
    shape = np.array([len(e) - 1 for e in edges])
    cells, samples = points.shape[:2]
    out = np.zeros(cells)
    for c in range(cells):
        ijk = np.array(
            [c // (shape[1] * shape[2]), (c // shape[2]) % shape[1], c % shape[2]]
        )
        for n in range(samples):
            u = points[c, n]
            point = np.array(
                [
                    edges[a][ijk[a]] + u[a] * (edges[a][ijk[a] + 1] - edges[a][ijk[a]])
                    for a in range(3)
                ]
            )
            out[c] += (
                characteristic(
                    point, direction(u[3], u[4]), edges, sigma, source, slopes
                )
                / samples
            )
    return out


@njit(cache=True)
def targeted_baseline(points, edges, sigma, source, volume):
    """Unbiased source-directed cone sampling, including exact angular weights."""
    shape = np.array([len(e) - 1 for e in edges])
    cells, samples = points.shape[:2]
    emission = np.abs(source) * volume
    probability = emission / emission.sum()
    cdf = np.cumsum(probability)
    centers = np.zeros((cells, 3))
    radii = np.zeros(cells)
    for j in range(cells):
        ijk = np.array(
            [j // (shape[1] * shape[2]), (j // shape[2]) % shape[1], j % shape[2]]
        )
        for a in range(3):
            centers[j, a] = 0.5 * (edges[a][ijk[a]] + edges[a][ijk[a] + 1])
            radii[j] += (0.5 * (edges[a][ijk[a] + 1] - edges[a][ijk[a]])) ** 2
        radii[j] = np.sqrt(radii[j])
    out = np.zeros(cells)
    for c in range(cells):
        ijk = np.array(
            [c // (shape[1] * shape[2]), (c // shape[2]) % shape[1], c % shape[2]]
        )
        for n in range(samples):
            u = points[c, n]
            point = np.array(
                [
                    edges[a][ijk[a]] + u[a] * (edges[a][ijk[a] + 1] - edges[a][ijk[a]])
                    for a in range(3)
                ]
            )
            j = min(np.searchsorted(cdf, u[3], side="right"), cells - 1)
            axis = centers[j] - point
            distance = np.sqrt(np.sum(axis**2))
            if distance <= radii[j]:
                omega = direction(u[4], u[5])
                angular_weight = 1.0
            else:
                axis /= distance
                cosine = np.sqrt(max(0.0, 1.0 - (radii[j] / distance) ** 2))
                angular_weight = 0.5 * (1.0 - cosine)
                mu = 1.0 - u[4] * (1.0 - cosine)
                reference = (
                    np.array([0.0, 0.0, 1.0])
                    if abs(axis[2]) < 0.9
                    else np.array([0.0, 1.0, 0.0])
                )
                e1 = np.cross(axis, reference)
                e1 /= np.sqrt(np.sum(e1**2))
                e2 = np.cross(axis, e1)
                phi = 2 * np.pi * u[5]
                omega = mu * axis + np.sqrt(max(0.0, 1 - mu**2)) * (
                    np.cos(phi) * e1 + np.sin(phi) * e2
                )
            selected = np.zeros(cells)
            selected[j] = source[j]
            out[c] += (
                angular_weight
                * characteristic(point, omega, edges, sigma, selected)
                / probability[j]
                / samples
            )
    return out


@njit(cache=True)
def transport(point, omega, weight, edges, sigma, scatter):
    """Analog transport: per-history track length in each cell, including progeny."""
    shape = np.array([len(e) - 1 for e in edges])
    ijk = np.array([locate(point[a], edges[a]) for a in range(3)])
    score = np.zeros(len(sigma))
    tau = -np.log(max(np.random.random(), 1e-300))
    while True:
        c = cell_index(ijk, shape)
        distance, axis = next_face(point, omega, ijk, edges)
        collision = tau / sigma[c] if sigma[c] > 0.0 else np.inf
        d = min(distance, collision)
        score[c] += weight * d
        point += d * omega
        if collision < distance:
            if np.random.random() * sigma[c] >= scatter[c]:
                return score
            omega = direction(np.random.random(), np.random.random())
            tau = -np.log(max(np.random.random(), 1e-300))
        else:
            tau -= distance * sigma[c]
            ijk[axis] += 1 if omega[axis] > 0.0 else -1
            if ijk[axis] < 0 or ijk[axis] >= shape[axis]:
                return score


@njit(cache=True)
def histories(
    edges,
    sigma,
    scatter,
    q,
    Q,
    volume,
    probability,
    N,
    inner,
    seed,
    slopes=None,
    antithetic=False,
):
    """SMC if inner=0; otherwise transport an unbiased estimate of the residual."""
    np.random.seed(seed)
    shape = np.array([len(e) - 1 for e in edges])
    cdf = np.cumsum(probability)
    total, squared = np.zeros(len(q)), np.zeros(len(q))
    weight_squared = 0.0
    trial_source = q + Q
    for n in range(N):
        c = min(np.searchsorted(cdf, np.random.random(), side="right"), len(q) - 1)
        ijk = np.array(
            [c // (shape[1] * shape[2]), (c // shape[2]) % shape[1], c % shape[2]]
        )
        point = np.array(
            [
                edges[a][ijk[a]]
                + np.random.random() * (edges[a][ijk[a] + 1] - edges[a][ijk[a]])
                for a in range(3)
            ]
        )
        if inner == 0:
            weight = q[c] * volume[c] / probability[c]
        else:
            shift_mu, shift_phi = np.random.random(), np.random.random()
            phi_trial = 0.0
            for k in range(inner):
                index = k // 2 if antithetic else k
                count = inner // 2 if antithetic else inner
                u = ((index + 0.5) / count + shift_mu) % 1.0
                v = (index * 0.6180339887498949 + shift_phi) % 1.0
                omega_inner = direction(u, v)
                if antithetic and k % 2:
                    omega_inner = -omega_inner
                phi_trial += (
                    characteristic(
                        point, omega_inner, edges, sigma, trial_source, slopes
                    )
                    / inner
                )
            local_Q = Q[c]
            if slopes is not None:
                for a in range(3):
                    h = edges[a][ijk[a] + 1] - edges[a][ijk[a]]
                    local_Q += slopes[c, a] * (
                        2 * (point[a] - edges[a][ijk[a]]) / h - 1
                    )
            weight = (scatter[c] * phi_trial - local_Q) * volume[c] / probability[c]
        weight_squared += weight**2
        omega = direction(np.random.random(), np.random.random())
        score = transport(point, omega, weight, edges, sigma, scatter) / volume
        total += score
        squared += score**2
    mean = total / N
    variance = np.maximum(0.0, squared / N - mean**2) / (N - 1)
    return mean, variance, weight_squared / N


def geometry(edges):
    edges = tuple(np.asarray(e, dtype=float) for e in edges)
    if any(len(e) < 2 or np.any(np.diff(e) <= 0) for e in edges):
        raise ValueError("Mesh edges must increase")
    volume = np.einsum("i,j,k->ijk", *(np.diff(e) for e in edges)).ravel()
    return edges, volume


def sobol_cells(cells, power, seed, dimensions=5):
    # Shared local coordinates correlate cells, but each cell's marginal is
    # still uniform. Independent scrambles across repeats retain valid errors.
    # One generator avoids paying its construction cost separately per cell.
    points = qmc.Sobol(dimensions, scramble=True, seed=int(seed)).random_base2(power)
    return np.broadcast_to(points, (cells,) + points.shape).copy()


def solve(
    edges,
    sigma,
    scatter,
    q,
    *,
    N=10000,
    inner=16,
    power=9,
    repeats=8,
    seed=1,
    forward=False,
    pilot=64,
    targeted=False,
    split=False,
    linear=False,
    antithetic=False,
    fit_source=True,
):
    """Frozen-source RMC with independent source, baseline and residual estimates.

    Standard errors include independently scrambled baseline noise as well as
    residual-history noise. Times exclude compilation only when warmed by caller.
    """
    start = time.perf_counter()
    edges, volume = geometry(edges)
    sigma, scatter, q = (
        np.asarray(a, dtype=float).reshape(-1) for a in (sigma, scatter, q)
    )
    if any(len(a) != len(volume) for a in (sigma, scatter, q)):
        raise ValueError("Cross sections and source must have one value per cell")
    if (
        np.any(sigma < 0)
        or np.any(scatter < 0)
        or np.any(scatter > sigma)
        or np.any((sigma > 0) & (scatter >= sigma))
    ):
        raise ValueError(
            "Require 0 <= Sigma_s < Sigma_t in material cells; voids have both zero"
        )
    if np.any(q < 0) or not np.isclose(q @ volume, 1.0):
        raise ValueError("External source must be nonnegative and integrate to one")
    if N < 2 or inner < 1 or repeats < 2:
        raise ValueError("Need N >= 2, inner >= 1 and baseline repeats >= 2")
    if antithetic and inner % 2:
        raise ValueError("Antipodal pairs need an even inner ray count")
    seeds = np.random.SeedSequence(seed).generate_state(3 + repeats)
    P = first_flight_matrix(sobol_cells(len(q), power, seeds[0]), edges, sigma, volume)
    Q = (
        np.linalg.solve(np.eye(len(q)) - scatter[:, None] * P, scatter * (P @ q))
        if fit_source
        else np.zeros_like(q)
    )
    slopes = None
    if linear:
        if forward or targeted or split:
            raise ValueError(
                "Linear source experiment currently uses the backward baseline"
            )
        shape = tuple(len(e) - 1 for e in edges)
        flux = (P @ (q + Q)).reshape(shape)
        slopes = np.zeros((len(q), 3))
        for axis in range(3):
            if shape[axis] > 1:
                center = 0.5 * (edges[axis][1:] + edges[axis][:-1])
                gradient = np.gradient(flux, center, axis=axis)
                hshape = [1, 1, 1]
                hshape[axis] = -1
                slopes[:, axis] = (
                    gradient * (0.5 * np.diff(edges[axis])).reshape(hshape)
                ).ravel() * scatter
        magnitude = np.abs(slopes).sum(axis=1)
        slopes *= np.minimum(1.0, 0.95 * Q / np.maximum(magnitude, 1e-300))[:, None]
    probability = volume / volume.sum()
    if pilot > 1:
        # Freeze an estimate of the response second moment by birthplace, using
        # the approximate source solve for response scales, never an SMC oracle.
        # The positive volume mixture guarantees support even when a pilot
        # misses rare scores or happens to estimate a zero residual.
        scale = np.maximum(P @ (q + Q), 1e-12)
        mass = np.zeros(len(q))
        for c in range(len(q)):
            local = np.zeros(len(q))
            local[c] = 1.0
            mean, variance, _ = histories(
                edges,
                sigma,
                scatter,
                q,
                Q,
                volume,
                local,
                pilot,
                inner,
                int((int(seeds[1]) + c) % 2**32),
                slopes,
                antithetic,
            )
            mass[c] = np.sqrt(np.mean(((pilot - 1) * variance + mean**2) / scale**2))
        if mass.sum() > 0:
            probability = 0.9 * mass / mass.sum() + 0.1 * probability
    setup = time.perf_counter() - start
    start = time.perf_counter()
    if split and np.count_nonzero(q) < len(q) / 2:
        base_q = np.stack(
            [
                targeted_baseline(
                    sobol_cells(len(q), power, s, 6), edges, sigma, q, volume
                )
                for s in seeds[3:]
            ]
        )
        base_Q = np.stack(
            [
                baseline(sobol_cells(len(q), power, s), edges, sigma, Q)
                for s in seeds[3:]
            ]
        )
        base = base_q + base_Q
    elif targeted:
        base = np.stack(
            [
                targeted_baseline(
                    sobol_cells(len(q), power, s, 6), edges, sigma, q + Q, volume
                )
                for s in seeds[3:]
            ]
        )
    elif forward:
        base = np.stack(
            [
                first_flight_matrix(sobol_cells(len(q), power, s), edges, sigma, volume)
                @ (q + Q)
                for s in seeds[3:]
            ]
        )
    else:
        base = np.stack(
            [
                baseline(sobol_cells(len(q), power, s), edges, sigma, q + Q, slopes)
                for s in seeds[3:]
            ]
        )
    base_mean, base_var = base.mean(axis=0), base.var(axis=0, ddof=1) / repeats
    baseline_time = time.perf_counter() - start
    start = time.perf_counter()
    # A uniform-in-volume proposal guarantees support independently of the
    # noisy residual estimate and the sampled source-construction matrix.
    # The frozen pilot may now redistribute 90% of it; 10% retains that support.
    correction, variance, weight_second = histories(
        edges,
        sigma,
        scatter,
        q,
        Q,
        volume,
        probability,
        N,
        inner,
        int(seeds[2]),
        slopes,
        antithetic,
    )
    correction_time = time.perf_counter() - start
    return dict(
        mean=base_mean + correction,
        variance=base_var + variance,
        baseline_variance=base_var,
        weight_second=weight_second,
        Q=Q,
        setup=setup,
        baseline=baseline_time,
        correction=correction_time,
        total=setup + baseline_time + correction_time,
    )


def smc(edges, sigma, scatter, q, *, N=100000, seed=1):
    start = time.perf_counter()
    edges, volume = geometry(edges)
    q = np.asarray(q, dtype=float)
    probability = q * volume / (q @ volume)
    mean, variance, weights = histories(
        edges,
        np.asarray(sigma),
        np.asarray(scatter),
        q,
        np.zeros_like(q),
        volume,
        probability,
        N,
        0,
        seed,
    )
    return dict(
        mean=mean,
        variance=variance,
        total=time.perf_counter() - start,
        weight_second=weights,
    )
