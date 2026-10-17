"""
Projection of the S_N reference onto candidate RMC trial spaces, and the L1 norm of
the residual of each projection (writeups/method/basis_study.tex).

For a trial space with projection Pi, r = q - L Pi psi is an ideal approximation
residual, distinct from a noisy binned fixed point. Its norm screens source-weight
second moments; transported response variance and cost must be measured. Each trial space is a
tensor product of
  - an energy basis on the coarse bins: "E0" constant, "E1" linear in E;
  - an angular basis: patches (J = 8 or 32, mu x phi tiles) carrying complete
    polynomials of degree <= p in their local (mu, phi) coordinates (p = 0, 1, 2),
    or real spherical harmonics up to degree N ("P1", "P3", ...);
  - a spatial basis on the coarse cells (mcdc.rmc.space3d: dg0, dg1, dg2, cg1, cg2).
The projections are applied in turn (they commute). The residual is evaluated on the
fine grid of the reference: per fine cell, fine group and direction for the volume
part, and on the fine faces of the coarse faces for the jumps of Pi psi (incoming
directions on vacuum and reflective boundaries), with Omega . grad and the in-scatter
of Pi psi computed exactly for the polynomial and the fine-group data.
"""

import numpy as np
from scipy.special import eval_legendre

from mcdc.rmc.mesh3d import CartesianMesh
from mcdc.rmc.patches import AngularPatches
from mcdc.rmc.residual import BOUNDARY_REFLECTIVE
from mcdc.rmc.space3d import SpatialBasis

from sn3d import REFLECTIVE, harmonic_degree, real_harmonics

PATCH_LAYOUTS = {8: (2, 4), 32: (4, 8), 128: (8, 16)}


# ======================================================================================
# Angular bases: functions b_f(d) at the quadrature directions and projection weights
# ======================================================================================


class AngularBasis:
    """B[d, f] values and P[d, f] projection weights: coefficients = psi @ P."""

    def __init__(self, name, mu, phi, w):
        self.name = name
        if name.startswith("P"):
            N = int(name[1:])
            Y = real_harmonics(N, mu, phi)  # (F, D), orthonormal
            self.B = Y.T.copy()
            self.P = (Y * w).T.copy()
            if not np.allclose(self.P.T @ self.B, np.eye(len(Y)), atol=1e-8, rtol=0.0):
                raise ValueError("Harmonic quadrature is underresolved; increase Gauss points")
            self.per_patch = None
        else:
            J, p = (int(v) for v in name[1:].split("p"))
            n_mu, n_phi = PATCH_LAYOUTS[J]
            mu_edges = np.linspace(-1.0, 1.0, n_mu + 1)
            phi_edges = np.linspace(-np.pi, np.pi, n_phi + 1)
            jm = np.clip(np.searchsorted(mu_edges, mu, side="right") - 1, 0, n_mu - 1)
            jp = np.clip(np.searchsorted(phi_edges, phi, side="right") - 1, 0, n_phi - 1)
            u = 2.0 * (mu - mu_edges[jm]) / np.diff(mu_edges)[jm] - 1.0
            v = 2.0 * (phi - phi_edges[jp]) / np.diff(phi_edges)[jp] - 1.0
            orders = [(a, b) for a in range(p + 1) for b in range(p + 1) if a + b <= p]
            patch = jm * n_phi + jp
            area = (2.0 / n_mu) * (2.0 * np.pi / n_phi)
            F = J * len(orders)
            self.B = np.zeros((len(mu), F))
            self.P = np.zeros((len(mu), F))
            for i, (a, b) in enumerate(orders):
                value = eval_legendre(a, u) * eval_legendre(b, v)
                col = patch * len(orders) + i
                self.B[np.arange(len(mu)), col] = value
                self.P[np.arange(len(mu)), col] = (
                    w * value * (2 * a + 1) * (2 * b + 1) / area
                )
            self.per_patch = len(orders)
        self.F = self.B.shape[1]


# ======================================================================================
# Fine-cell tables of the coarse-cell Legendre polynomials
# ======================================================================================


def _legendre_averages(n):
    """avg[a, s]: mean of P_a over the s-th of n equal sub-intervals of [-1, 1]."""
    lo = -1.0 + 2.0 * np.arange(n) / n
    hi = lo + 2.0 / n
    return np.array(
        [
            np.ones(n),
            0.5 * (lo + hi),
            0.5 * (lo * lo + lo * hi + hi * hi) - 0.5,
        ]
    )


class Study:
    """
    Reference solution and geometry of one problem. coarse: (x, y, z) coarse edges;
    n: fine cells per coarse cell per axis; psi[G, cell, D] (float32);
    E_fine, n_fine (fine groups per coarse bin); Sigma_t[m, G], T[m, L+1, G, G],
    material[cell], q[cell, G], bc[3, 2], quadrature (mu, phi, w), L.
    """

    def __init__(self, coarse, n, psi, E_fine, n_fine, Sigma_t, T, material, q, bc,
                 quadrature, L):
        self.coarse = [np.asarray(e, dtype=float) for e in coarse]
        self.n = n
        self.K = tuple(len(e) - 1 for e in self.coarse)
        self.fine_shape = tuple(n * k for k in self.K)
        self.psi = psi
        self.E_fine = E_fine
        self.n_fine = n_fine
        self.G = len(E_fine) - 1
        self.Gc = self.G // n_fine
        self.Sigma_t, self.T, self.material, self.q = Sigma_t, T, material, q
        self.bc = np.asarray(bc)
        self.mu, self.phi, self.w = quadrature
        s = np.sqrt(1.0 - self.mu**2)
        self.omega = np.stack([s * np.cos(self.phi), s * np.sin(self.phi), self.mu])
        from sn3d import mirror_directions

        self.mirror = mirror_directions(self.mu, self.phi)
        self.L = L
        self.Y = real_harmonics(L, self.mu, self.phi)  # (n_lm, D)
        self.degree = harmonic_degree(L)
        self.mesh = CartesianMesh(*self.coarse)
        # A patch object only for SpatialBasis (no strong boundary conditions here)
        self._patches = AngularPatches([-1.0, 0.0, 1.0], np.linspace(-np.pi, np.pi, 5))
        self.avg = _legendre_averages(n)
        # Fine-cell geometry
        self.fine_edges = [
            np.concatenate([np.linspace(e[k], e[k + 1], n + 1)[:-1] for k in range(len(e) - 1)] + [e[-1:]])
            for e in self.coarse
        ]
        hx, hy, hz = (np.diff(e) for e in self.fine_edges)
        self.fine_volume = (hx[:, None, None] * hy[None, :, None] * hz[None, None, :]).ravel()
        self.dE = np.diff(E_fine)
        # Energy: position of each fine group in its coarse bin (mean of x over it)
        self.bin_of = np.arange(self.G) // n_fine
        lo = E_fine[self.bin_of * n_fine]
        hi = E_fine[(self.bin_of + 1) * n_fine]
        self.x_mean = (E_fine[:-1] + E_fine[1:] - lo - hi) / (hi - lo)
        self.q_norm = float(np.sum(np.abs(q) * self.fine_volume[:, None] * self.dE) * 4 * np.pi)

    # ----------------------------------------------------------------------------------

    def _fine_index(self):
        """(kx, ky, kz, sx, sy, sz) of every fine cell, in the reference's order."""
        n = self.n
        ix, iy, iz = np.meshgrid(*(np.arange(f) for f in self.fine_shape), indexing="ij")
        return [a.ravel() for a in (ix // n, iy // n, iz // n, ix % n, iy % n, iz % n)]

    def evaluate(self, basis, C):
        """
        Fine-cell averages of the modal polynomial C[kx, ky, kz, A, B, M]:
        (cell, A, B).
        """
        kx, ky, kz, sx, sy, sz = self._fine_index()
        a, b, c = basis.modes.T
        table = self.avg[a][:, sx] * self.avg[b][:, sy] * self.avg[c][:, sz]  # (M, cell)
        return np.einsum("cabm,mc->cab", C[kx, ky, kz], table, optimize=True)

    def moments(self, basis, values):
        """Coarse-cell moments int P_m of fine-cell values (cell, A, B): (Kx, Ky, Kz, A, B, M)."""
        kx, ky, kz, sx, sy, sz = self._fine_index()
        a, b, c = basis.modes.T
        table = self.avg[a][:, sx] * self.avg[b][:, sy] * self.avg[c][:, sz]
        weighted = values * self.fine_volume[:, None, None]
        out = np.zeros(self.K + values.shape[1:] + (basis.M,))
        flat = np.einsum("cab,mc->cabm", weighted, table, optimize=True)
        np.add.at(out, (kx, ky, kz), flat)
        return out

    # ----------------------------------------------------------------------------------

    def residual(self, energy, angular, spatial):
        """
        ||r||_1 / ||q||_1 of the projection onto energy x angular x spatial, split into
        (volume, faces), and the number of unknowns per coarse cell and bin.
        """
        ang = angular
        basis = SpatialBasis(spatial, self.mesh, self._patches, ((BOUNDARY_REFLECTIVE,) * 2,) * 3)
        nE = 1 if energy == "E0" else 2
        # 1. Angular coefficients of every fine group: (G, cell, F)
        # 2. Energy: coefficients per coarse bin (Gc, nE, cell, F)
        coef_E = np.zeros((self.Gc, nE, self.psi.shape[1], ang.F))
        for g in range(self.G):
            a_g = self.psi[g].astype(np.float64) @ ang.P  # (cell, F)
            b = self.bin_of[g]
            share = self.dE[g] / np.sum(self.dE[self.bin_of == b])
            coef_E[b, 0] += share * a_g
            if nE == 2:
                coef_E[b, 1] += 3.0 * share * self.x_mean[g] * a_g
        # 3. Space: modal coefficients per coarse bin, (Kx, Ky, Kz, Gc*nE, F, M)
        values = np.moveaxis(coef_E.reshape(self.Gc * nE, -1, ang.F), 1, 0)  # (cell, GcnE, F)
        C = basis.project_modal(self.moments(basis, values))
        grad = basis.gradient(C)  # (3, Kx, Ky, Kz, GcnE, F, M)
        V = self.evaluate(basis, C)  # (cell, GcnE, F)
        Gv = [self.evaluate(basis, grad[a]) for a in range(3)]

        # Scattering moments of Pi psi for every fine group: SH moments via F-space
        to_lm = ang.B.T @ (self.Y * self.w).T  # (F, n_lm)

        def fine_coefficients(arr, g):
            """Angular coefficients of Pi psi at fine group g from bin coefficients."""
            b = self.bin_of[g]
            out = arr[:, b * nE]
            if nE == 2:
                out = out + self.x_mean[g] * arr[:, b * nE + 1]
            return out  # (cell, F)

        moments = np.zeros((self.G, V.shape[0], len(self.degree)))
        for g in range(self.G):
            moments[g] = fine_coefficients(V, g) @ to_lm

        # Volume residual per fine group, directions in one product
        volume = 0.0
        D = len(self.w)
        Bt = ang.B.T  # (F, D)
        for g in range(self.G - 1, -1, -1):
            s_lm = np.zeros((V.shape[0], len(self.degree)))
            for gp in range(g, self.G):
                Tl = self.T[self.material, :, gp, g][:, self.degree]
                if np.any(Tl):
                    s_lm += Tl * moments[gp]
            s_lm /= self.dE[g]
            value = fine_coefficients(V, g)  # (cell, F)
            Sigma = self.Sigma_t[self.material, g][:, None]
            r = self.q[:, g][:, None] + s_lm @ self.Y - (Sigma * value) @ Bt
            for a in range(3):
                r -= (fine_coefficients(Gv[a], g) @ Bt) * self.omega[a][None, :]
            volume += np.sum(np.abs(r) * self.fine_volume[:, None] * self.w[None, :]) * self.dE[g]

        # Faces: jumps of Pi psi across coarse faces, on their fine faces
        faces = 0.0
        for axis in range(3):
            faces += self._face_residual(basis, C, ang, axis, nE)
        unknowns = self._unknowns(basis) * ang.F * nE
        return volume / self.q_norm, faces / self.q_norm, unknowns

    def _unknowns(self, basis):
        """Spatial unknowns per coarse cell (nodes per cell for continuous bases)."""
        if basis.continuous:
            return np.prod(basis.node_shape) / np.prod(self.K)
        return basis.M

    def _face_residual(self, basis, C, ang, axis, nE):
        n = self.n
        others = [d for d in range(3) if d != axis]
        face_modes = basis.face_modes[axis]
        # Traces on the low and high side of every coarse cell: (K..., GcnE, F, Mf)
        low = np.einsum("fm,xyzabm->xyzabf", basis.trace[axis][0], C)
        high = np.einsum("fm,xyzabm->xyzabf", basis.trace[axis][1], C)
        lo = np.moveaxis(low, axis, 0)
        hi = np.moveaxis(high, axis, 0)
        Ka = lo.shape[0]
        # Fine-face table: average of the face polynomial over each fine face
        s1, s2 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        table = (self.avg[face_modes[:, 0]][:, s1.ravel()]
                 * self.avg[face_modes[:, 1]][:, s2.ravel()])  # (Mf, n*n)
        widths = [np.diff(self.coarse[d]) for d in others]
        omega_n = self.omega[axis]
        total = 0.0
        for g in range(self.G):
            b = self.bin_of[g]

            def at_group(arr):
                out = arr[..., b * nE, :, :]
                if nE == 2:
                    out = out + self.x_mean[g] * arr[..., b * nE + 1, :, :]
                return out  # (Ka, K1, K2, F, Mf)

            L = np.einsum("akjfm,ms->akjsf", at_group(lo), table)  # fine-face values per F
            H = np.einsum("akjfm,ms->akjsf", at_group(hi), table)
            Lv = L @ ang.B.T  # (Ka, K1, K2, n*n, D)
            Hv = H @ ang.B.T
            area = (widths[0][:, None] * widths[1][None, :] / (n * n))[None, :, :, None, None]
            weight = np.abs(omega_n)[None, None, None, None, :] * self.w * area * self.dE[g]
            # Interior faces: jump downstream - upstream (both signs of Omega_n)
            D_int = Lv[1:] - Hv[:-1]
            total += np.sum(np.abs(D_int) * weight)
            # Boundaries: incoming directions only
            for side, trace, sign in ((0, Lv[0], omega_n > 0), (1, Hv[-1], omega_n < 0)):
                if self.bc[axis, side] == REFLECTIVE:
                    jump = trace - trace[..., self.mirror[axis]]
                else:
                    jump = trace
                total += np.sum(np.abs(jump[..., sign]) * weight[0][..., sign])
        return total


# ======================================================================================
# Streaming-exact trial functions: psi~ = T^-1 (q + Q~), Q~ a projected scattering source
# ======================================================================================


def streaming_exact(study, energy, N, spatial):
    """
    ||r||_1 / ||q||_1 for the trial function psi~ = T^-1 (q + Pi S psi), T = Omega.grad +
    Sigma_t (no scattering), with Pi the projection of the reference scattering source
    onto (energy basis) x (spherical harmonics of degree <= N) x (spatial basis). Since
    psi = T^-1 (q + S psi),

        r = q - L psi~ = S psi~ - Pi S psi = (I - S T^-1) e,   e = (I - Pi) S psi,

    For an exact reference the identity holds. The fixed-up reference is not exact
    for the linear inverse used here, so the implementation constructs psi~ directly
    with sn3d.transport_only and measures S psi~ - Pi S psi instead.
    Returns (||e||/||q||, ||r||/||q||, unknowns per coarse cell and bin of Q~).
    """
    from sn3d import transport_only

    s = study
    basis = SpatialBasis(spatial, s.mesh, s._patches, ((BOUNDARY_REFLECTIVE,) * 2,) * 3)
    n_lm = len(s.degree)
    Yw = (s.Y * s.w).T  # (D, n_lm)
    # Reference moments and scattering source moments per fine group: (G, cell, n_lm)
    moments = np.stack([s.psi[g].astype(np.float64) @ Yw for g in range(s.G)])
    S = np.zeros_like(moments)
    for g in range(s.G):
        for gp in range(g, s.G):
            S[g] += s.T[s.material, :, gp, g][:, s.degree] * moments[gp]
        S[g] /= s.dE[g]
    # Projection: keep degrees <= N, energy bins, spatial basis
    keep = s.degree <= N
    nE = 1 if energy == "E0" else 2
    coef_E = np.zeros((s.Gc, nE) + S.shape[1:])
    for g in range(s.G):
        b = s.bin_of[g]
        share = s.dE[g] / np.sum(s.dE[s.bin_of == b])
        coef_E[b, 0] += share * S[g]
        if nE == 2:
            coef_E[b, 1] += 3.0 * share * s.x_mean[g] * S[g]
    coef_E[..., ~keep] = 0.0
    values = np.moveaxis(coef_E.reshape(s.Gc * nE, -1, n_lm), 1, 0)  # (cell, GcnE, lm)
    C = basis.project_modal(s.moments(basis, values))
    projected = s.evaluate(basis, C)  # (cell, GcnE, lm)
    e = np.zeros_like(S)
    for g in range(s.G):
        b = s.bin_of[g]
        value = projected[:, b * nE]
        if nE == 2:
            value = value + s.x_mean[g] * projected[:, b * nE + 1]
        e[g] = S[g] - value
    # r = e - S T^-1 e, groups from the top (downscatter only)
    # That identity requires a reference solved with this same linear inverse.
    # The reference uses nonlinear negative-flux fixup, so construct the trial
    # directly instead and measure S T^-1(q + Pi S psi_ref) - Pi S psi_ref.
    problem = dict(x=s.fine_edges[0], y=s.fine_edges[1], z=s.fine_edges[2], bc=s.bc)
    t_moments = np.zeros_like(e)
    for g in range(s.G - 1, -1, -1):
        psi_e = transport_only(problem, (s.mu, s.phi, s.w),
                               s.q[:, g, None] + (S[g] - e[g]) @ s.Y,
                               s.Sigma_t[s.material, g])
        t_moments[g] = psi_e @ Yw
    norm_e = norm_r = 0.0
    for g in range(s.G):
        St = np.zeros((e.shape[1], n_lm))
        for gp in range(g, s.G):
            St += s.T[s.material, :, gp, g][:, s.degree] * t_moments[gp]
        St /= s.dE[g]
        weight = s.fine_volume[:, None] * s.w[None, :] * s.dE[g]
        norm_e += np.sum(np.abs(e[g] @ s.Y) * weight)
        norm_r += np.sum(np.abs((St - S[g] + e[g]) @ s.Y) * weight)
    unknowns = s._unknowns(basis) * int(np.sum(keep)) * nE
    return norm_e / s.q_norm, norm_r / s.q_norm, unknowns
