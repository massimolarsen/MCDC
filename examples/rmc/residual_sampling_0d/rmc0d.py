"""
0D continuous-energy RMC prototype: compare residual samplers / trial spaces.

Infinite medium, classical elastic scattering isotropic in COM on mass A (E_out uniform
on [alpha E', E']), Sigma_s(E) with a scattering resonance, Sigma_a(E) = 1/v + an
absorption resonance. Source uniform in the top bin. Neutrons below E_min are killed.

Trial spaces (per bin g, Legendre P_b of x_g(E)):
  flat : psi~(E) = sum_b c_b P_b                      (current feature/RMC basis)
  coll : psi~(E) = sum_b c_b P_b / Sigma_t(E)         (collision-density basis)
Test functions P_a in both; update dc = B^-1 t with t_a = int_g eps P_a.

Residual r(E) = q(E) + T[psi~](E) - Sigma_t(E) psi~(E), with T the exact
continuous-energy in-scatter, evaluated through a precomputed bin-to-point table
U[g'b](E_i) (the in-scatter at fine nodes from one incident basis function).
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------- problem data
A = 12.0
ALPHA = ((A - 1) / (A + 1)) ** 2
E_MIN, E_MAX = 1.0, 1000.0
SRC_LO = 1000.0 / 10 ** 0.1  # source uniform on [SRC_LO, E_MAX] (coarse top bin)


@njit(cache=True)
def sig_s(E):
    return 1.0 + 30.0 / (1.0 + ((E - 300.0) / 1.5) ** 2)


@njit(cache=True)
def sig_a(E):
    return 0.02 * np.sqrt(100.0 / E) + 15.0 / (1.0 + ((E - 50.0) / 0.5) ** 2)


@njit(cache=True)
def sig_t(E):
    return sig_s(E) + sig_a(E)


def make_edges(G):
    return np.geomspace(E_MIN, E_MAX, G + 1)


def fine_grid(edges, n_per_bin=400):
    pts = [np.geomspace(edges[g], edges[g + 1], n_per_bin + 1) for g in range(len(edges) - 1)]
    for E0, half in ((50.0, 0.5), (300.0, 1.5)):
        pts.append(E0 + half * np.sinh(np.linspace(-9, 9, 4001)))
    Ef = np.unique(np.concatenate(pts))
    Ef = Ef[(Ef > E_MIN) & (Ef < E_MAX)]
    # each bin gets its own one-sided edge nodes, so discontinuous trial functions
    # are represented exactly on either side of an edge
    inner = edges[1:-1]
    near = np.min(np.abs(Ef[:, None] / inner[None, :] - 1.0), axis=1) < 1e-9
    Ef = Ef[~near]
    eps = 1e-12
    return np.unique(np.concatenate([Ef, inner * (1 - eps), inner * (1 + eps), [E_MIN, E_MAX]]))


# ---------------------------------------------------------------- reference
@njit(cache=True)
def reference_collision_density(Ef, q):
    """F(E) = q + int_E^{min(E/alpha,Emax)} c F / ((1-alpha) E') dE', marched downward.
    Returns F at Ef. H(E) = int_E^Emax cF/((1-alpha)E'), F = q + H(E) - H(E/alpha)."""
    n = Ef.size
    F = np.zeros(n)
    H = np.zeros(n)
    for i in range(n - 1, -1, -1):
        E = Ef[i]
        Eu = E / ALPHA
        if Eu >= E_MAX:
            Hu = 0.0
        else:
            j = np.searchsorted(Ef, Eu)  # Ef[j-1] < Eu <= Ef[j]
            t = (Eu - Ef[j - 1]) / (Ef[j] - Ef[j - 1])
            Hu = H[j - 1] * (1 - t) + H[j] * t
        if i == n - 1:
            F[i] = q[i]
            H[i] = 0.0
            continue
        k_up = sig_s(Ef[i + 1]) / sig_t(Ef[i + 1]) / ((1 - ALPHA) * Ef[i + 1])
        k_here = sig_s(E) / sig_t(E) / ((1 - ALPHA) * E)
        dE = Ef[i + 1] - E
        # H_i = H_{i+1} + dE/2 (k_up F_{i+1} + k_here F_i);  F_i = q_i + H_i - Hu
        base = H[i + 1] + 0.5 * dE * k_up * F[i + 1]
        F[i] = (q[i] + base - Hu) / (1 - 0.5 * dE * k_here)
        H[i] = base + 0.5 * dE * k_here * F[i]
    return F


# ---------------------------------------------------------------- basis helpers
def legendre(b, x):
    return np.ones_like(x) if b == 0 else (x if b == 1 else 1.5 * x * x - 0.5)


class Setup:
    def __init__(self, G, P, basis, Ef_n=400, edges=None):
        self.edges = make_edges(G) if edges is None else np.asarray(edges)
        G = self.edges.size - 1
        self.G, self.P, self.basis = G, P, basis
        self.Ef = fine_grid(self.edges, Ef_n)
        Ef = self.Ef
        self.St = sig_t(Ef)
        self.Ss = sig_s(Ef)
        self.bin = np.clip(np.searchsorted(self.edges, Ef, side="right") - 1, 0, G - 1)
        lo, hi = self.edges[self.bin], self.edges[self.bin + 1]
        self.x = (2 * Ef - lo - hi) / (hi - lo)
        # source: uniform in top bin, unit total
        self.q = np.where(Ef > SRC_LO * (1 + 1e-13), 1.0 / (E_MAX - SRC_LO), 0.0)
        # basis functions phi_{g b}(E) at fine nodes (nonzero in bin g only)
        self.w = 1.0 / self.St if basis == "coll" else np.ones_like(Ef)
        self.phi = np.zeros((G, P, Ef.size))
        for g in range(G):
            m = self.bin == g
            for b in range(P):
                self.phi[g, b, m] = legendre(b, self.x[m]) * self.w[m]
        # fine-node index range of each bin (its own one-sided edge nodes)
        self.masks = []
        for g in range(G):
            i0 = np.searchsorted(Ef, self.edges[g], side="left")
            i1 = np.searchsorted(Ef, self.edges[g + 1], side="right") - 1
            self.masks.append((i0, i1))
        # mass matrices B[g,a,b] = int_g P_a phi_b
        self.B = np.zeros((G, P, P))
        for g in range(G):
            i0, i1 = self.masks[g]
            E = Ef[i0 : i1 + 1]
            x = (2 * E - self.edges[g] - self.edges[g + 1]) / (self.edges[g + 1] - self.edges[g])
            w = 1.0 / sig_t(E) if basis == "coll" else np.ones_like(E)
            for a in range(P):
                for b in range(P):
                    self.B[g, a, b] = np.trapezoid(legendre(a, x) * legendre(b, x) * w, E)
        self.Binv = np.linalg.inv(self.B)
        self._build_U()

    def _build_U(self):
        """U[g', b, i] = int_{g'} Sigma_s(E') phi_{g'b}(E') K(E'->E_i) dE',
        K = 1/((1-alpha)E') on E' in [E_i, E_i/alpha]. Cumulative integrals per bin."""
        G, P, Ef = self.G, self.P, self.Ef
        U = np.zeros((G, P, Ef.size))
        for gp in range(G):
            i0, i1 = self.masks[gp]
            E = Ef[i0 : i1 + 1]
            x = (2 * E - self.edges[gp] - self.edges[gp + 1]) / (self.edges[gp + 1] - self.edges[gp])
            w = 1.0 / sig_t(E) if self.basis == "coll" else np.ones_like(E)
            for b in range(P):
                f = sig_s(E) * legendre(b, x) * w / ((1 - ALPHA) * E)
                C = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(E))])
                lo = np.clip(Ef, E[0], E[-1])
                hi = np.clip(Ef / ALPHA, E[0], E[-1])
                U[gp, b] = np.interp(hi, E, C) - np.interp(lo, E, C)
        self.U = U.reshape(G * P, Ef.size)

    # trial function at fine nodes
    def psi_fine(self, c):
        out = np.zeros(self.Ef.size)
        for g in range(self.G):
            i0, i1 = self.masks[g]
            for b in range(self.P):
                E = self.Ef[i0 : i1 + 1]
                x = (2 * E - self.edges[g] - self.edges[g + 1]) / (self.edges[g + 1] - self.edges[g])
                w = 1.0 / sig_t(E) if self.basis == "coll" else 1.0
                out[i0 : i1 + 1] += c[g, b] * legendre(b, x) * w
        return out

    def T_fine(self, c):
        return c.reshape(-1) @ self.U

    def moments(self, f):
        """int_g f P_a dE for f at fine nodes (trapezoid within each bin)."""
        out = np.zeros((self.G, self.P))
        for g in range(self.G):
            i0, i1 = self.masks[g]
            E = self.Ef[i0 : i1 + 1]
            x = (2 * E - self.edges[g] - self.edges[g + 1]) / (self.edges[g + 1] - self.edges[g])
            for a in range(self.P):
                out[g, a] = np.trapezoid(f[i0 : i1 + 1] * legendre(a, x), E)
        return out

    def flux_bin_avg(self, c):
        psi = self.psi_fine(c)
        return self.moments(psi)[:, 0] / np.diff(self.edges)


# ---------------------------------------------------------------- sampling
def proposal(Ef, absr, floor_frac=0.02):
    """Piecewise-constant proposal on fine intervals, proportional to the interval max
    of |r| at its nodes plus a defensive floor (support everywhere)."""
    dE = np.diff(Ef)
    h = np.maximum(absr[1:], absr[:-1])
    mass = h * dE
    tot = mass.sum()
    if tot <= 0:
        return None
    mass = mass + floor_frac * tot * dE / dE.sum()
    cdf = np.cumsum(mass)
    return cdf / cdf[-1], mass / cdf[-1]


def sample_term(rng, Ef, r_nodes, r_exact, N):
    """Sample N energies from a proposal ~|r| and return (E, weights) with
    weight = r(E) / (N q(E)), r evaluated exactly at E by r_exact."""
    pr = proposal(Ef, np.abs(r_nodes))
    if pr is None or N == 0:
        return np.zeros(0), np.zeros(0)
    cdf, p = pr
    u = rng.random(N)
    k = np.searchsorted(cdf, u)
    k = np.minimum(k, Ef.size - 2)
    E = Ef[k] + rng.random(N) * (Ef[k + 1] - Ef[k])
    qd = p[k] / (Ef[k + 1] - Ef[k])
    return E, r_exact(E) / (N * qd)


# ---------------------------------------------------------------- transport (0D)
@njit(cache=True)
def transport(E0, w0, edges, P, seed):
    """Expected-value 0D transport: each emission at E with weight w scores
    w/Sigma_t(E) * P_a(x_g(E)) into bin g; survival weight w*c(E); E -> U[alpha E, E].
    Returns t[g, a] (estimates of int_g eps P_a)."""
    np.random.seed(seed)
    G = edges.size - 1
    t = np.zeros((G, P))
    for n in range(E0.size):
        E = E0[n]
        w = w0[n]
        while True:
            if E < E_MIN:
                break
            g = np.searchsorted(edges, E, side="right") - 1
            if g >= G:
                g = G - 1
            st = sig_t(E)
            x = (2 * E - edges[g] - edges[g + 1]) / (edges[g + 1] - edges[g])
            s = w / st
            t[g, 0] += s
            if P > 1:
                t[g, 1] += s * x
            if P > 2:
                t[g, 2] += s * (1.5 * x * x - 0.5)
            w *= sig_s(E) / st
            # Russian roulette on tiny weights (unbiased)
            if abs(w) < 1e-3 * abs(w0[n]):
                if np.random.random() < 0.5:
                    break
                w *= 2.0
            E = E * (ALPHA + (1 - ALPHA) * np.random.random())
    return t
