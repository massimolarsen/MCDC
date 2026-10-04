"""
Angular transfer between polar-cosine bins about the slab axis.

For incident polar cosine mu_in in bin j' and lab scattering cosine mu0 (azimuth
uniform), the outgoing-bin probability Pi_j(mu_in, mu0) is analytic
(`kinematics.azimuthal_bin_probability`). Because emission densities do not depend
on mu_in, the transfer moments only need

    A_{j j'}(mu0) = int_{bin j'} Pi_j(mu_in, mu0) dmu_in,

which satisfies sum_j A_{j j'}(mu0) = width of bin j'.

A is evaluated exactly by splitting the mu_in integral at its kinks
(mu_in = b mu0 +- sqrt(1 - b^2) sqrt(1 - mu0^2) for each edge b of bin j) and using a
cosine change of variables that removes the square-root endpoint behavior. For fast
lookup it is tabulated on a uniform grid in theta0 = arccos(mu0) and linearly
interpolated. Lookup error only enlarges the scattering residual that corrects it;
it does not bias RMC.
"""

import math

import numpy as np

from numba import njit

####

from mcdc.constant import PI
from mcdc.rmc.kinematics import azimuthal_bin_probability

N_THETA_DEFAULT = 8193

_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


@njit(cache=True)
def angular_transfer_exact(mu0, mu_edges, j, jp):
    """A_{j j'}(mu0), evaluated to round-off."""
    low = mu_edges[jp]
    high = mu_edges[jp + 1]

    # Breakpoints: bin j' edges and the kinks from bin j edges
    points = np.empty(6)
    points[0] = low
    points[1] = high
    N = 2
    s0 = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    for edge in (mu_edges[j], mu_edges[j + 1]):
        s = math.sqrt(max(0.0, 1.0 - edge * edge)) * s0
        for kink in (edge * mu0 - s, edge * mu0 + s):
            if low < kink < high:
                points[N] = kink
                N += 1
    points = np.sort(points[:N])

    total = 0.0
    for p in range(N - 1):
        c = points[p]
        d = points[p + 1]
        if d <= c:
            continue
        for q in range(len(_GL_X)):
            t = 0.5 * (_GL_X[q] + 1.0)
            x = c + 0.5 * (d - c) * (1.0 - math.cos(PI * t))
            dx = 0.5 * (d - c) * PI * math.sin(PI * t)
            total += (
                0.5
                * _GL_W[q]
                * dx
                * azimuthal_bin_probability(x, mu0, mu_edges[j], mu_edges[j + 1])
            )
    return total


@njit(cache=True)
def build_angular_transfer_table(mu_edges, N_theta):
    """Table A[j, j', k] at theta0_k = pi k / (N_theta - 1), mu0 = cos(theta0_k)."""
    J = len(mu_edges) - 1
    table = np.zeros((J, J, N_theta))
    for k in range(N_theta):
        mu0 = math.cos(PI * k / (N_theta - 1))
        for j in range(J):
            for jp in range(J):
                table[j, jp, k] = angular_transfer_exact(mu0, mu_edges, j, jp)
    return table


@njit(cache=True)
def angular_transfer(mu0, table, out):
    """Fill out[j, j'] with A_{j j'}(mu0) by linear interpolation in theta0."""
    N_theta = table.shape[2]
    mu0 = min(1.0, max(-1.0, mu0))
    x = math.acos(mu0) / PI * (N_theta - 1)
    k = min(int(x), N_theta - 2)
    f = x - k
    J = table.shape[0]
    for j in range(J):
        for jp in range(J):
            out[j, jp] = (1.0 - f) * table[j, jp, k] + f * table[j, jp, k + 1]
