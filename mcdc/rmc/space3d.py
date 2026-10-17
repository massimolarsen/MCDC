"""
Polynomial spatial bases of the 3D Cartesian RMC trial space
(writeups/cartesian_3d/spatial_bases.tex).

On each structured cell, with local coordinates t = (t_x, t_y, t_z) in [-1, 1]^3,

    psi~(t) = sum_m C_m P_a(t_x) P_b(t_y) P_c(t_z),     m = (a, b, c),

per energy bin and angular patch (both still piecewise constant). Every basis is held
in this per-cell Legendre form, so the residual, its sampling and the correction
passes are written once:

  - "dg1", "dg2": discontinuous, complete polynomials of degree <= p (4 or 10 modes);
    the coefficients are free per cell and the L2 projection is local:
    C_m = (2a+1)(2b+1)(2c+1) / |V| int_V eps P_m.
  - "cg1", "cg2": continuous, the tensor-product Lagrange space Q_p (nodes at the
    cell vertices, and for p = 2 the edge, face and cell midpoints). The unknowns are
    nodal values; the per-cell Legendre coefficients (all modes with a, b, c <= p)
    follow exactly. The L2 projection solves the mass system, which is the Kronecker
    product of three 1D banded matrices. Vacuum boundaries are imposed strongly: for
    an incoming patch the boundary nodal values are zero (a patch is wholly incoming
    or outgoing on each face).

The MC/DC tally gives the cell moments int_V eps P_m (scores "flux-xyz<a><b><c>").
"""

import numpy as np

####

from mcdc.rmc.patches import AXIS_X, AXIS_Y, AXIS_Z
from mcdc.rmc.residual import BOUNDARY_VACUUM

SPATIAL_BASES = ("constant", "dg1", "dg2", "cg1", "cg2")

# 1D Lagrange basis of one cell in Legendre form: phi_l(t) = sum_a B[l, a] P_a(t),
#   nodes t = -1, 1 (p = 1) or -1, 0, 1 (p = 2)
_LAGRANGE = {
    1: np.array([[0.5, -0.5], [0.5, 0.5]]),
    2: np.array(
        [
            [1.0 / 6.0, -0.5, 1.0 / 3.0],
            [2.0 / 3.0, 0.0, -2.0 / 3.0],
            [1.0 / 6.0, 0.5, 1.0 / 3.0],
        ]
    ),
}


def legendre(n, t):
    """P_n(t) for n <= 2."""
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return t
    return 1.5 * t * t - 0.5


class SpatialBasis:
    """
    Modes, derivative and trace maps, and (continuous bases) the nodal structure and
    the projection.

    Parameters
    ----------
    name : {"dg1", "dg2", "cg1", "cg2"}
    mesh : CartesianMesh
    patches : AngularPatches
    boundary : ((x_low, x_high), (y_low, y_high), (z_low, z_high)) boundary codes
    """

    def __init__(self, name, mesh, patches, boundary):
        # "dg0" (one mode) is the constant basis in this form, for analysis tools;
        #   run3d keeps its own constant path
        if name not in ("dg0",) + SPATIAL_BASES[1:]:
            raise ValueError(f"RMC: unknown spatial basis {name}")
        self.name = name
        self.continuous = name.startswith("cg")
        self.p = p = int(name[2])
        self.mesh = mesh
        modes = [
            (a, b, c)
            for c in range(p + 1)
            for b in range(p + 1)
            for a in range(p + 1)
            if (max(a, b, c) <= p if self.continuous else a + b + c <= p)
        ]
        modes.sort(key=lambda m: (sum(m), m[::-1]))
        self.modes = np.array(modes, dtype=np.int64)
        self.M = len(modes)
        self.index = {m: i for i, m in enumerate(modes)}
        # int_{[-1,1]^3} P_m^2 = 8 / norm_m
        self.norm = np.prod(2.0 * self.modes + 1.0, axis=1)
        self.tally_scores = [f"flux-xyz{a}{b}{c}" for a, b, c in modes]

        # d/dt_axis in modal form: P_1' = P_0, P_2' = 3 P_1
        self.derivative = np.zeros((3, self.M, self.M))
        for i, m in enumerate(modes):
            for axis in range(3):
                n = m[axis]
                if n == 0:
                    continue
                lower = list(m)
                lower[axis] = n - 1
                self.derivative[axis, self.index[tuple(lower)], i] = (
                    1.0 if n == 1 else 3.0
                )

        # Traces on the faces normal to each axis: face modes (the other two orders)
        #   and trace[axis, side] (M_face, M) with P_n(+1) = 1, P_n(-1) = (-1)^n
        self.face_modes = []
        self.trace = []
        for axis in range(3):
            others = [d for d in range(3) if d != axis]
            face = sorted({(m[others[0]], m[others[1]]) for m in modes})
            lookup = {f: i for i, f in enumerate(face)}
            T = np.zeros((2, len(face), self.M))
            for i, m in enumerate(modes):
                f = lookup[(m[others[0]], m[others[1]])]
                T[0, f, i] = (-1.0) ** m[axis]  # low side, t = -1
                T[1, f, i] = 1.0  # high side, t = +1
            self.face_modes.append(np.array(face, dtype=np.int64))
            self.trace.append(T)

        if self.continuous:
            self._setup_nodes(patches, boundary)

    # ==================================================================================
    # Continuous bases: nodes, conversion and projection
    # ==================================================================================

    def _setup_nodes(self, patches, boundary):
        p = self.p
        B = _LAGRANGE[p]
        self.B = B
        self.node_shape = tuple(p * K + 1 for K in self.mesh.shape)
        # Local node indices of each cell along each axis
        self.local = [
            p * np.arange(K)[:, None] + np.arange(p + 1)[None, :]
            for K in self.mesh.shape
        ]
        # 1D mass matrices: local int phi_l phi_l' dx = h/2 sum_a B B 2/(2a+1)
        local_mass = B @ np.diag(2.0 / (2.0 * np.arange(p + 1) + 1.0)) @ B.T
        self.mass_1d = []
        for axis, K in enumerate(self.mesh.shape):
            h = self.mesh.width[axis]
            matrix = np.zeros((self.node_shape[axis],) * 2)
            for k in range(K):
                idx = self.local[axis][k]
                matrix[np.ix_(idx, idx)] += 0.5 * h[k] * local_mass
            self.mass_1d.append(matrix)
        # Strong vacuum conditions: per patch and axis, the free nodes
        J = patches.J
        self.free = []
        for axis in range(3):
            free = np.ones((J, self.node_shape[axis]), dtype=bool)
            low, high = boundary[axis]
            sign = patches.direction_sign[axis]
            if low == BOUNDARY_VACUUM:
                free[sign > 0, 0] = False
            if high == BOUNDARY_VACUUM:
                free[sign < 0, -1] = False
            self.free.append(free)

    def nodal_to_modal(self, N):
        """N[nx, ny, nz, G, J] -> C[kx, ky, kz, G, J, m]."""
        ix, iy, iz = self.local
        local = N[
            ix[:, None, None, :, None, None],
            iy[None, :, None, None, :, None],
            iz[None, None, :, None, None, :],
        ]  # (Kx, Ky, Kz, l, m, n, G, J)
        B = self.B
        full = np.einsum("xyzlmngj,la,mb,nc->xyzgjabc", local, B, B, B, optimize=True)
        a, b, c = self.modes.T
        return np.ascontiguousarray(full[..., a, b, c])

    def project(self, moments):
        """
        L2 projection from the cell moments int_V eps P_m (all tensor modes), per
        unit energy and solid angle: moments[kx, ky, kz, G, J, m]. Returns the nodal
        values N[nx, ny, nz, G, J] of the projection onto the constrained space.
        """
        p = self.p
        B = self.B
        Kx, Ky, Kz, G, J = moments.shape[:5]
        full = np.zeros((Kx, Ky, Kz, G, J, p + 1, p + 1, p + 1))
        a, b, c = self.modes.T
        full[..., a, b, c] = moments
        # Hat-function moments int eps phi_I, assembled over cells
        local = np.einsum("xyzgjabc,la,mb,nc->xyzlmngj", full, B, B, B, optimize=True)
        rhs = np.zeros(self.node_shape + (G, J))
        ix, iy, iz = self.local
        np.add.at(
            rhs,
            (
                ix[:, None, None, :, None, None],
                iy[None, :, None, None, :, None],
                iz[None, None, :, None, None, :],
            ),
            local,
        )
        # Solve (Mx (x) My (x) Mz) N = rhs on the free nodes of each patch
        if all(free.all() for free in self.free):
            # No strong conditions: one solve per axis for every column
            values = rhs
            for axis in range(3):
                moved = np.moveaxis(values, axis, 0)
                solved = np.linalg.solve(
                    self.mass_1d[axis], moved.reshape(moved.shape[0], -1)
                ).reshape(moved.shape)
                values = np.moveaxis(solved, 0, axis)
            return values
        N = np.zeros_like(rhs)
        for j in range(J):
            values = rhs[..., j]
            for axis in range(3):
                free = self.free[axis][j]
                matrix = self.mass_1d[axis][np.ix_(free, free)]
                moved = np.moveaxis(values, axis, 0)
                solved = np.zeros_like(moved)
                solved[free] = np.linalg.solve(
                    matrix, moved[free].reshape(free.sum(), -1)
                ).reshape((free.sum(),) + moved.shape[1:])
                values = np.moveaxis(solved, 0, axis)
            N[..., j] = values
        return N

    # ==================================================================================
    # Both kinds
    # ==================================================================================

    def project_modal(self, moments):
        """
        Per-cell Legendre coefficients of the L2 projection of eps (see `project`):
        local for discontinuous bases; through the nodal values for continuous ones.
        """
        if self.continuous:
            return self.nodal_to_modal(self.project(moments))
        volume = self.mesh.volume[..., None, None, None]
        return moments * self.norm / volume

    def gradient(self, C):
        """
        Modal coefficients of d psi~ / dx_axis, (3, Kx, Ky, Kz, G, J, M): the modal
        derivative times 2 / h of each cell.
        """
        out = np.einsum("dnm,xyzgjm->dxyzgjn", self.derivative, C)
        for axis in range(3):
            shape = [1, 1, 1, 1, 1, 1]
            shape[axis] = -1
            out[axis] *= (2.0 / self.mesh.width[axis]).reshape(shape)
        return out

    def face_jumps(self, C, axis, boundary, patches, mirror):
        """
        Jump D = psi~(+ side) - psi~(- side) across the faces normal to `axis`, in face
        modes: (face grid with N_axis + 1 along `axis`, G, J, M_face). At a boundary,
        for incoming patches only, psi_bc = 0 (vacuum) or the mirrored patch
        (reflective), as in `mesh3d.face_jumps_3d`. Continuous bases have no interior
        jumps, and none at a vacuum boundary (imposed strongly).
        """
        low_trace = np.einsum("fm,xyzgjm->xyzgjf", self.trace[axis][0], C)
        high_trace = np.einsum("fm,xyzgjm->xyzgjf", self.trace[axis][1], C)
        lo = np.moveaxis(low_trace, axis, 0)
        hi = np.moveaxis(high_trace, axis, 0)
        D = np.zeros((lo.shape[0] + 1,) + lo.shape[1:])
        if not self.continuous:
            D[1:-1] = lo[1:] - hi[:-1]
        sign = patches.direction_sign[axis]
        low, high = boundary
        if not (self.continuous and low == BOUNDARY_VACUUM):
            outside = lo[0][..., mirror, :] if low != BOUNDARY_VACUUM else 0.0
            D[0][..., sign > 0, :] = (lo[0] - outside)[..., sign > 0, :]
        if not (self.continuous and high == BOUNDARY_VACUUM):
            outside = hi[-1][..., mirror, :] if high != BOUNDARY_VACUUM else 0.0
            D[-1][..., sign < 0, :] = (outside - hi[-1])[..., sign < 0, :]
        return np.moveaxis(D, 0, axis)

    def mean_square(self, coefficients, axis=-1):
        """Mean over the cell of the square of a modal polynomial: sum C_m^2 / norm_m."""
        return np.sum(coefficients**2 / self.norm, axis=axis)

    def face_norm(self, axis):
        """norm of each face mode of `axis` (the product over the two other orders)."""
        return np.prod(2.0 * self.face_modes[axis] + 1.0, axis=1)
