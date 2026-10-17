"""
Angular patches of the 3D Cartesian RMC trial space (writeups/cartesian_3d/cartesian_3d.tex).

The unit sphere is cut into patches j = j_mu J_phi + j_phi by polar-cosine edges
(mu = Omega . z) and azimuthal edges (phi = atan2(Omega_y, Omega_x), as MC/DC's tally
filter measures it). A patch has solid angle |A_j| = dmu dphi, and the trial-space
angular flux is constant on it (per steradian).

Scattering from Omega' to Omega with lab cosine mu0 = Omega' . Omega is azimuthally
symmetric about Omega', so the transfer moments need only

    A_{j j'}(mu0) = int_{patch j'} dOmega' F_j(Omega', mu0),

with F_j the fraction of the circle {Omega : Omega' . Omega = mu0} that lies in patch j.
A plays the role of the slab table of `angular.py`: sum_j A_{j j'} = |A_j'| and
A_{j j'} = A_{j' j}. F_j is computed exactly from the circle's crossings of the patch
edges; the integral over patch j' uses Gauss-Legendre panels split at the kinks of
F (where the circle becomes tangent to an edge). The table is tabulated on a uniform
grid in theta0 = arccos(mu0) and interpolated linearly by `angular.angular_transfer`.

The face residual needs int_patch |Omega_a| dOmega for each axis a and sampling of
Omega in a patch with density proportional to |Omega_a|; both are separable in
(mu, phi) because every patch lies within one octant.
"""

import math

import numpy as np

from numba import njit

####

from mcdc.constant import PI

N_THETA_PATCHES = 2049
_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)
_TWO_PI = 2.0 * PI

AXIS_X, AXIS_Y, AXIS_Z = 0, 1, 2


class AngularPatches:
    """
    Patch layout from polar-cosine edges (from -1 to 1, including 0) and azimuthal
    edges (from -pi to pi, including -pi/2, 0 and pi/2), so that every patch lies
    within one octant and the signs of Omega_x, Omega_y and Omega_z are constant on it.
    """

    def __init__(self, mu_edges, phi_edges):
        mu_edges = np.asarray(mu_edges, dtype=float)
        phi_edges = np.asarray(phi_edges, dtype=float)
        if not (np.all(np.diff(mu_edges) > 0.0) and np.all(np.diff(phi_edges) > 0.0)):
            raise ValueError("RMC: patch edges must increase")
        if not (np.isclose(mu_edges[0], -1.0) and np.isclose(mu_edges[-1], 1.0)):
            raise ValueError("RMC: mu edges must span [-1, 1]")
        if not (np.isclose(phi_edges[0], -PI) and np.isclose(phi_edges[-1], PI)):
            raise ValueError("RMC: phi edges must span [-pi, pi]")
        for value, edges, name in (
            (0.0, mu_edges, "mu = 0"),
            (-0.5 * PI, phi_edges, "phi = -pi/2"),
            (0.0, phi_edges, "phi = 0"),
            (0.5 * PI, phi_edges, "phi = pi/2"),
        ):
            if not np.any(np.isclose(edges, value, atol=1e-12)):
                raise ValueError(f"RMC: {name} must be a patch edge")
        mu_edges[0], mu_edges[-1] = -1.0, 1.0
        phi_edges[0], phi_edges[-1] = -PI, PI
        self.mu_edges, self.phi_edges = mu_edges, phi_edges
        self.J_mu, self.J_phi = len(mu_edges) - 1, len(phi_edges) - 1
        self.J = self.J_mu * self.J_phi

        j_mu, j_phi = np.divmod(np.arange(self.J), self.J_phi)
        self.mu_low, self.mu_high = mu_edges[j_mu], mu_edges[j_mu + 1]
        self.phi_low, self.phi_high = phi_edges[j_phi], phi_edges[j_phi + 1]
        self.area = (self.mu_high - self.mu_low) * (self.phi_high - self.phi_low)
        # The kernel code reads patch measures as differences of these "edges"
        self.measure_edges = np.concatenate(([0.0], np.cumsum(self.area)))
        # Center direction of each patch
        mu_c = 0.5 * (self.mu_low + self.mu_high)
        phi_c = 0.5 * (self.phi_low + self.phi_high)
        s_c = np.sqrt(1.0 - mu_c**2)
        self.center = np.stack((s_c * np.cos(phi_c), s_c * np.sin(phi_c), mu_c), axis=1)
        # int_patch |Omega_a| dOmega, a = x, y, z
        G = _sqrt_integral(self.mu_high) - _sqrt_integral(self.mu_low)
        self.abs_direction_integral = np.stack(
            (
                G * np.abs(np.sin(self.phi_high) - np.sin(self.phi_low)),
                G * np.abs(np.cos(self.phi_low) - np.cos(self.phi_high)),
                (self.phi_high - self.phi_low)
                * np.abs(
                    self.mu_high * np.abs(self.mu_high)
                    - self.mu_low * np.abs(self.mu_low)
                )
                / 2.0,
            )
        )
        # Sign of Omega_a on each patch
        self.direction_sign = np.sign(self.center.T)

    @property
    def key(self):
        """Array that identifies the layout (for cache keys)."""
        return np.concatenate((self.mu_edges, [9.0], self.phi_edges))

    def index(self, mu, phi):
        j_mu = np.clip(np.searchsorted(self.mu_edges, mu, side="right") - 1, 0, None)
        j_phi = np.clip(np.searchsorted(self.phi_edges, phi, side="right") - 1, 0, None)
        return np.minimum(j_mu, self.J_mu - 1) * self.J_phi + np.minimum(
            j_phi, self.J_phi - 1
        )

    def mirror(self, axis):
        """
        Patch index of the reflection Omega_a -> -Omega_a of every patch. Requires the
        layout to be symmetric under that reflection (a reflective boundary on axis a).
        """
        if axis == AXIS_Z:
            low, high = -self.mu_high, -self.mu_low
            match = self.index(0.5 * (low + high), 0.5 * (self.phi_low + self.phi_high))
            ok = np.allclose(self.mu_low[match], low) and np.allclose(
                self.mu_high[match], high
            )
        else:
            if axis == AXIS_X:  # phi -> pi - phi (wrapped into [-pi, pi])
                low = np.where(self.phi_high > 0, PI, -PI) - self.phi_high
                high = np.where(self.phi_high > 0, PI, -PI) - self.phi_low
            else:  # phi -> -phi
                low, high = -self.phi_high, -self.phi_low
            match = self.index(0.5 * (self.mu_low + self.mu_high), 0.5 * (low + high))
            ok = np.allclose(self.phi_low[match], low) and np.allclose(
                self.phi_high[match], high
            )
        if not ok:
            raise ValueError(
                f"RMC: a reflective boundary on axis {'xyz'[axis]} needs patch edges "
                "symmetric under that reflection"
            )
        return match


def _sqrt_integral(mu):
    """Antiderivative of sqrt(1 - mu^2)."""
    mu = np.clip(mu, -1.0, 1.0)
    return 0.5 * (mu * np.sqrt(1.0 - mu * mu) + np.arcsin(mu))


# ======================================================================================
# Arc fractions and the patch transfer table
# ======================================================================================


@njit(cache=True)
def patch_index(mu, phi, mu_edges, phi_edges):
    J_mu = len(mu_edges) - 1
    J_phi = len(phi_edges) - 1
    j_mu = min(max(np.searchsorted(mu_edges, mu, side="right") - 1, 0), J_mu - 1)
    j_phi = min(max(np.searchsorted(phi_edges, phi, side="right") - 1, 0), J_phi - 1)
    return j_mu * J_phi + j_phi


@njit(cache=True)
def arc_fractions(mu_p, phi_p, mu0, mu_edges, phi_edges, weight, out):
    """
    out[j] += weight F_j(Omega', mu0): the fraction of the circle of directions with
    cosine mu0 to Omega' = (mu_p, phi_p) that lies in patch j. Exact: the circle is cut
    at its crossings of the mu-edge cones and the azimuthal half-planes, and each arc
    is assigned to the patch of its midpoint.
    """
    s_p = math.sqrt(max(0.0, 1.0 - mu_p * mu_p))
    s0 = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    cp, sp = math.cos(phi_p), math.sin(phi_p)
    if s0 < 1e-14:
        # Degenerate circle: Omega = Omega' (mu0 = 1) or -Omega' (mu0 = -1)
        if mu0 > 0.0:
            out[patch_index(mu_p, phi_p, mu_edges, phi_edges)] += weight
        else:
            phi = phi_p + PI if phi_p < 0.0 else phi_p - PI
            out[patch_index(-mu_p, phi, mu_edges, phi_edges)] += weight
        return

    # Omega(psi) = mu0 Omega' + s0 (cos psi e1 + sin psi e2)
    o = (s_p * cp, s_p * sp, mu_p)
    e1 = (mu_p * cp, mu_p * sp, -s_p)
    e2 = (-sp, cp, 0.0)

    cuts = np.empty(2 * len(mu_edges) + 2 * len(phi_edges))
    N = 0
    # mu(psi) = mu0 mu' - s0 s' cos psi = b
    if s_p > 0.0:
        for i in range(1, len(mu_edges) - 1):
            c = (mu0 * mu_p - mu_edges[i]) / (s0 * s_p)
            if -1.0 < c < 1.0:
                a = math.acos(c)
                cuts[N] = a
                cuts[N + 1] = _TWO_PI - a
                N += 2
    # Plane through the z axis at azimuth phi_e: n . Omega(psi) = 0
    for i in range(len(phi_edges) - 1):
        n_x, n_y = -math.sin(phi_edges[i]), math.cos(phi_edges[i])
        a = mu0 * (n_x * o[0] + n_y * o[1])
        b = s0 * (n_x * e1[0] + n_y * e1[1])
        c = s0 * (n_x * e2[0] + n_y * e2[1])
        R = math.sqrt(b * b + c * c)
        if R > 0.0 and abs(a) < R:
            base = math.atan2(c, b)
            d = math.acos(-a / R)
            for psi in (base + d, base - d):
                cuts[N] = psi % _TWO_PI
                N += 1

    if N == 0:
        mu = mu0 * o[2] - s0 * e1[2]
        x = mu0 * o[0] + s0 * e1[0]
        y = mu0 * o[1] + s0 * e1[1]
        out[patch_index(mu, math.atan2(y, x), mu_edges, phi_edges)] += weight
        return
    cuts = np.sort(cuts[:N])
    for i in range(N):
        a = cuts[i]
        b = cuts[i + 1] if i + 1 < N else cuts[0] + _TWO_PI
        if b <= a:
            continue
        m = 0.5 * (a + b)
        cm, sm = math.cos(m), math.sin(m)
        x = mu0 * o[0] + s0 * (cm * e1[0] + sm * e2[0])
        y = mu0 * o[1] + s0 * (cm * e1[1] + sm * e2[1])
        mu = mu0 * o[2] + s0 * (cm * e1[2] + sm * e2[2])
        out[patch_index(mu, math.atan2(y, x), mu_edges, phi_edges)] += (
            weight * (b - a) / _TWO_PI
        )


@njit(cache=True)
def _mu_breakpoints(mu0, low, high, phi_low, phi_high, mu_edges, phi_edges):
    """
    Kinks in mu' of G(mu') = int_{patch phi range} F dphi':
      - tangency of the circle to the cones mu = b (incl. b = +-1, the circle through
        a pole): mu' = b mu0 +- s_b s0;
      - an inner kink of F in phi' (`_phi_breakpoints`) reaching the patch's own
        azimuthal edge phi_l: plane tangency, s' |sin(phi_l - phi_e)| = s0, and the
        circle through a corner, b mu' + s_b cos(phi_l - phi_e) s' = mu0.
    """
    J_phi = len(phi_edges) - 1
    points = np.empty(2 + 2 * len(mu_edges) + 4 * J_phi * (1 + len(mu_edges)))
    points[0] = low
    points[1] = high
    N = 2
    s0 = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    for b in mu_edges:
        s = math.sqrt(max(0.0, 1.0 - b * b)) * s0
        for kink in (b * mu0 - s, b * mu0 + s):
            if low < kink < high:
                points[N] = kink
                N += 1
    for phi_l in (phi_low, phi_high):
        for i in range(J_phi):
            delta = phi_l - phi_edges[i]
            # Plane tangency: s' = s0 / |sin(delta)|
            sin_delta = abs(math.sin(delta))
            if sin_delta > s0:
                m = math.sqrt(max(0.0, 1.0 - (s0 / sin_delta) ** 2))
                for kink in (-m, m):
                    if low < kink < high:
                        points[N] = kink
                        N += 1
            # Corner (b, phi_e): b mu' + s_b cos(delta) s' = mu0, with mu' = cos(t)
            cos_delta = math.cos(delta)
            for b in mu_edges[1:-1]:
                u = math.sqrt(max(0.0, 1.0 - b * b)) * cos_delta
                R = math.sqrt(b * b + u * u)
                if R <= abs(mu0) or R == 0.0:
                    continue
                beta = math.atan2(u, b)
                gamma = math.acos(mu0 / R)
                for t in (beta + gamma, beta - gamma):
                    if 0.0 <= t <= PI:
                        kink = math.cos(t)
                        if low < kink < high:
                            points[N] = kink
                            N += 1
    return np.sort(points[:N])


@njit(cache=True)
def _phi_breakpoints(mu0, mu_p, low, high, mu_edges, phi_edges):
    """
    Kinks of F in phi' at fixed mu':
      - tangency of the circle to an azimuthal plane, s' |sin(phi' - phi_e)| = s0;
      - the circle through a patch corner (mu = b, phi = phi_e),
        s' s_b cos(phi' - phi_e) + mu' b = mu0.
    """
    J_phi = len(phi_edges) - 1
    points = np.empty(2 + 4 * J_phi + 2 * J_phi * len(mu_edges))
    points[0] = low
    points[1] = high
    N = 2
    s0 = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    s_p = math.sqrt(max(0.0, 1.0 - mu_p * mu_p))
    if s0 < s_p:
        alpha = math.asin(s0 / s_p)
        for i in range(J_phi):
            for kink in (alpha, -alpha, PI - alpha, PI + alpha):
                phi = (phi_edges[i] + kink + PI) % _TWO_PI - PI
                if low < phi < high:
                    points[N] = phi
                    N += 1
    for b in mu_edges[1:-1]:
        s_b = math.sqrt(max(0.0, 1.0 - b * b))
        if s_p * s_b <= 0.0:
            continue
        c = (mu0 - mu_p * b) / (s_p * s_b)
        if not (-1.0 < c < 1.0):
            continue
        delta = math.acos(c)
        for i in range(J_phi):
            for kink in (delta, -delta):
                phi = (phi_edges[i] + kink + PI) % _TWO_PI - PI
                if low < phi < high:
                    points[N] = phi
                    N += 1
    return np.sort(points[:N])


@njit(cache=True)
def patch_transfer_row(mu0, jp, mu_edges, phi_edges, out):
    """
    out[j] += A_{j j'}(mu0) = int_{patch j'} F_j(Omega', mu0) dOmega', by
    Gauss-Legendre panels (cosine-mapped to absorb square-root endpoint behavior) split
    at the kinks of F in mu' and, at each mu', in phi'.
    """
    J_phi = len(phi_edges) - 1
    j_mu = jp // J_phi
    j_phi = jp % J_phi
    mu_points = _mu_breakpoints(
        mu0,
        mu_edges[j_mu],
        mu_edges[j_mu + 1],
        phi_edges[j_phi],
        phi_edges[j_phi + 1],
        mu_edges,
        phi_edges,
    )
    n = len(_GL_X)
    for p in range(len(mu_points) - 1):
        c = mu_points[p]
        d = mu_points[p + 1]
        if d <= c:
            continue
        for q in range(n):
            t = 0.5 * (_GL_X[q] + 1.0)
            mu_p = c + 0.5 * (d - c) * (1.0 - math.cos(PI * t))
            w_mu = 0.5 * _GL_W[q] * 0.5 * (d - c) * PI * math.sin(PI * t)
            phi_points = _phi_breakpoints(
                mu0, mu_p, phi_edges[j_phi], phi_edges[j_phi + 1], mu_edges, phi_edges
            )
            for r in range(len(phi_points) - 1):
                e = phi_points[r]
                f = phi_points[r + 1]
                if f <= e:
                    continue
                for s in range(n):
                    u = 0.5 * (_GL_X[s] + 1.0)
                    phi_p = e + 0.5 * (f - e) * (1.0 - math.cos(PI * u))
                    w_phi = 0.5 * _GL_W[s] * 0.5 * (f - e) * PI * math.sin(PI * u)
                    arc_fractions(
                        mu_p, phi_p, mu0, mu_edges, phi_edges, w_mu * w_phi, out
                    )


@njit(cache=True)
def _table_columns(mu_edges, phi_edges, N_theta, columns, k_start, k_step, table):
    J = table.shape[2]
    row = np.zeros(J)
    for k in range(k_start, N_theta, k_step):
        mu0 = math.cos(PI * k / (N_theta - 1))
        for jp in columns:
            row[:] = 0.0
            patch_transfer_row(mu0, jp, mu_edges, phi_edges, row)
            for j in range(J):
                table[0, 0, j, jp, k] = row[j]


def _symmetries(patches):
    """
    Symmetries of the layout that map patches to patches (scattering depends only on
    Omega' . Omega, so A_{j j'} is invariant under them):
      rotation by the azimuthal width when the phi edges are uniform;
      the reflection mu -> -mu when the mu edges are symmetric.
    """
    rotation = np.allclose(np.diff(patches.phi_edges), 2.0 * PI / patches.J_phi)
    mirror = np.allclose(patches.mu_edges, -patches.mu_edges[::-1])
    return rotation, mirror


def build_patch_transfer_table(patches, N_theta=N_THETA_PATCHES, comm=None):
    """
    Table A[0, 0, j, j', k] at theta0_k = pi k / (N_theta - 1), mu0 = cos(theta0_k),
    in the layout of `angular.build_angular_transfer_table` (constant angular basis).
    Only the columns j' that the layout's symmetries do not determine are computed;
    with an MPI communicator, the theta0 points are dealt round-robin over its ranks.
    """
    J_mu, J_phi, J = patches.J_mu, patches.J_phi, patches.J
    rotation, mirror = _symmetries(patches)
    j_mu_max = (J_mu + 1) // 2 if mirror else J_mu
    j_phi_max = 1 if rotation else J_phi
    columns = np.array(
        [jm * J_phi + jf for jm in range(j_mu_max) for jf in range(j_phi_max)],
        dtype=np.int64,
    )
    table = np.zeros((1, 1, J, J, N_theta))
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    _table_columns(
        patches.mu_edges, patches.phi_edges, N_theta, columns, rank, size, table
    )
    if comm is not None and size > 1:
        from mpi4py import MPI

        comm.Allreduce(MPI.IN_PLACE, table, op=MPI.SUM)

    # Fill the remaining columns from the symmetries
    A = table[0, 0].reshape(J_mu, J_phi, J_mu, J_phi, N_theta)
    if rotation:
        for jf in range(1, J_phi):
            A[:, :, :j_mu_max, jf] = np.roll(A[:, :, :j_mu_max, 0], jf, axis=1)
    if mirror:
        for jm in range(j_mu_max, J_mu):
            A[:, :, jm] = A[::-1, :, J_mu - 1 - jm]
    return table


# ======================================================================================
# Directions for the face residual: density |Omega_a| on a patch
# ======================================================================================


@njit(cache=True)
def _sqrt_antiderivative(mu):
    mu = min(1.0, max(-1.0, mu))
    return 0.5 * (mu * math.sqrt(1.0 - mu * mu) + math.asin(mu))


@njit(cache=True)
def sample_mu_sqrt(low, high, xi):
    """mu in [low, high] with density proportional to sqrt(1 - mu^2) (bisection-safe Newton)."""
    F_low = _sqrt_antiderivative(low)
    target = F_low + xi * (_sqrt_antiderivative(high) - F_low)
    a, b = low, high
    mu = low + xi * (high - low)
    for _ in range(60):
        f = _sqrt_antiderivative(mu) - target
        if f > 0.0:
            b = mu
        else:
            a = mu
        derivative = math.sqrt(max(0.0, 1.0 - mu * mu))
        step = mu - f / derivative if derivative > 1e-12 else 0.5 * (a + b)
        mu = step if a < step < b else 0.5 * (a + b)
        if b - a < 1e-15:
            break
    return mu


@njit(cache=True)
def sample_direction_abs(axis, mu_low, mu_high, phi_low, phi_high, xi1, xi2):
    """
    (mu, phi) in the patch with density proportional to |Omega_axis|:
      z: mu^2 uniform (|mu| dmu), phi uniform;
      x: mu ~ sqrt(1 - mu^2), sin(phi) uniform (|cos phi| dphi);
      y: mu ~ sqrt(1 - mu^2), cos(phi) uniform (|sin phi| dphi).
    The patch lies within one octant, so each map is monotone.
    """
    if axis == AXIS_Z:
        phi = phi_low + xi2 * (phi_high - phi_low)
        if mu_low >= 0.0:
            mu = math.sqrt(
                mu_low * mu_low + xi1 * (mu_high * mu_high - mu_low * mu_low)
            )
        else:
            mu = -math.sqrt(
                mu_high * mu_high + xi1 * (mu_low * mu_low - mu_high * mu_high)
            )
        return mu, phi
    mu = sample_mu_sqrt(mu_low, mu_high, xi1)
    phi_mid = 0.5 * (phi_low + phi_high)
    if axis == AXIS_X:
        t = math.sin(phi_low) + xi2 * (math.sin(phi_high) - math.sin(phi_low))
        t = min(1.0, max(-1.0, t))
        if math.cos(phi_mid) >= 0.0:
            phi = math.asin(t)
        elif phi_mid > 0.0:
            phi = PI - math.asin(t)
        else:
            phi = -PI - math.asin(t)
    else:
        t = math.cos(phi_low) + xi2 * (math.cos(phi_high) - math.cos(phi_low))
        t = min(1.0, max(-1.0, t))
        phi = math.acos(t) if phi_mid >= 0.0 else -math.acos(t)
    return mu, phi
