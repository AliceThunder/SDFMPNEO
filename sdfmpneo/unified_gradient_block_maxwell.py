"""Compatible scalar-gradient block for the Cartesian E-form Maxwell solver.

The production E operator is

    A = C.T H_mu C + D,

where ``D`` contains the dielectric, conductive, and open-boundary edge mass
terms.  The Cartesian incidence operators satisfy the exact sequence identity
``C G = 0``.  Therefore the difficult gradient part of the Maxwell solve is not
an approximate near-nullspace: on ``range(G)`` it is governed exactly by

    S_g = G_g.T D G_g = G_g.T A G_g,

with one constant scalar gauge removed from ``G_g``.

This module factors that scalar block and combines it multiplicatively with the
existing bounded-fill edge ILU.  The physical Maxwell matrix and RHS are never
shifted or projected.  In particular an open two-terminal source may have a
non-zero discrete divergence; the scalar block resolves that longitudinal
response instead of deleting it from the source.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _node_id(background, i, j, k):
    return (int(i) * (background.ny + 1) + int(j)) * (background.nz + 1) + int(k)


def node_coordinates(background):
    cached = getattr(background, "_compatible_node_coordinates", None)
    if cached is not None:
        return cached
    X, Y, Z = np.meshgrid(background.x, background.y, background.z, indexing="ij")
    coordinates = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    background._compatible_node_coordinates = coordinates
    return coordinates


def gradient_operator(background, *, gauge_fixed=False):
    """Return the exact edge<-node incidence matrix for the active edge topology."""
    cached = getattr(background, "_compatible_edge_gradient", None)
    if cached is None:
        n_nodes = (background.nx + 1) * (background.ny + 1) * (background.nz + 1)
        rows, cols, data = [], [], []
        for edge, (axis, i, j, k) in enumerate(background.edge_tuples):
            start = _node_id(background, i, j, k)
            if axis == 0:
                stop = _node_id(background, i + 1, j, k)
            elif axis == 1:
                stop = _node_id(background, i, j + 1, k)
            else:
                stop = _node_id(background, i, j, k + 1)
            rows.extend((edge, edge))
            cols.extend((start, stop))
            data.extend((-1.0, 1.0))
        cached = sp.csr_matrix(
            (data, (rows, cols)), shape=(background.n_edges, n_nodes), dtype=float
        )
        background._compatible_edge_gradient = cached

    if not gauge_fixed:
        return cached
    # The rectilinear open domain is connected.  Remove one constant-potential
    # coordinate; no penalty parameter or numerical rank threshold is used.
    fixed = getattr(background, "_compatible_edge_gradient_gauge_fixed", None)
    if fixed is None:
        if cached.shape[1] < 2:
            raise ValueError("Maxwell scalar gradient space has no non-constant coordinate")
        fixed = cached[:, 1:].tocsr()
        background._compatible_edge_gradient_gauge_fixed = fixed
    return fixed


def source_terminal_divergence(background, source):
    """Discrete terminal charge/current-balance diagnostics for an open source."""
    s = np.asarray(source, float).reshape(-1)
    if s.shape != (background.n_edges,) or np.any(~np.isfinite(s)):
        raise ValueError("source must contain one finite value per Maxwell edge")
    G = gradient_operator(background, gauge_fixed=False)
    q = np.asarray(G.T @ s, float).reshape(-1)
    scale = max(float(np.linalg.norm(q, 1)), float(np.linalg.norm(s)), np.finfo(float).tiny)
    net = float(abs(np.sum(q)) / scale)
    coords = node_coordinates(background)
    moment = np.asarray(coords.T @ q, float).reshape(3)
    return q, net, moment


def _edge_mass_diagonal(background, context, *, mqs=False, mqs_admittance=None):
    sigma, eps, _mu_inv, _k, _cap, _temperature = background.cell_properties(
        context, None, em=True
    )
    sigma = np.asarray(sigma, float)
    hs = np.asarray(background.edge_cell_hodge @ sigma).reshape(-1)
    diagonal = 1j * background.omega * hs.astype(complex)
    if not mqs:
        eps = np.asarray(eps, float)
        he = np.asarray(background.edge_cell_hodge @ eps).reshape(-1)
        diagonal = diagonal - (background.omega ** 2) * he

    boundary_hodge = getattr(background, "boundary_edge_hodge", None)
    if boundary_hodge is not None:
        if mqs:
            if mqs_admittance is None:
                raise ValueError("MQS gradient block requires the MQS boundary admittance")
            admittance = complex(mqs_admittance)
        else:
            admittance = complex(background.boundary_admittance())
        diagonal = diagonal + 1j * background.omega * admittance * np.asarray(
            boundary_hodge, float
        ).reshape(-1)
    if diagonal.shape != (background.n_edges,) or np.any(~np.isfinite(diagonal)):
        raise FloatingPointError("Maxwell edge mass diagonal is invalid")
    return np.asarray(diagonal, complex)


@dataclass
class GradientBlock:
    gradient: sp.csr_matrix
    scalar_matrix: sp.csc_matrix
    factor: object
    build_seconds: float
    topology_error: float

    @property
    def scalar_dofs(self):
        return int(self.gradient.shape[1])

    def solve(self, rhs):
        value = np.asarray(rhs, complex).reshape(-1)
        if value.shape != (self.gradient.shape[0],):
            raise ValueError("gradient-block RHS dimension mismatch")
        scalar_rhs = np.asarray(self.gradient.T @ value, complex).reshape(-1)
        phi = np.asarray(self.factor.solve(scalar_rhs), complex).reshape(-1)
        return np.asarray(self.gradient @ phi, complex).reshape(-1)

    def scalar_residual(self, edge_rhs, edge_correction):
        residual = np.asarray(edge_rhs, complex).reshape(-1) - np.asarray(
            edge_correction, complex
        ).reshape(-1)
        return float(
            np.linalg.norm(self.gradient.T @ residual)
            / max(float(np.linalg.norm(self.gradient.T @ edge_rhs)), np.finfo(float).tiny)
        )


def build_gradient_block(
    background,
    context,
    *,
    mqs=False,
    mqs_admittance=None,
    check_topology=True,
):
    """Factor the exact gauge-fixed scalar block ``G.T A G``.

    The curl-curl contribution is omitted analytically, not approximately,
    because ``C G = 0``.  This avoids cancellation between a very large curl
    block and the much smaller dielectric/conductive longitudinal block.
    """
    cache = getattr(background, "_sdfmpneo_gradient_block_cache", None)
    if cache is None:
        cache = {}
        background._sdfmpneo_gradient_block_cache = cache
    key = (
        id(context),
        bool(mqs),
        None if mqs_admittance is None else complex(mqs_admittance),
        bool(check_topology),
    )
    cached_block = cache.get(key)
    if cached_block is not None:
        return cached_block

    started = time.perf_counter()
    G = gradient_operator(background, gauge_fixed=True)
    topology_error = 0.0
    if check_topology:
        CG = (background.curl @ G).tocsr()
        CG.eliminate_zeros()
        topology_error = (
            float(np.max(np.abs(CG.data))) if CG.nnz else 0.0
        )
        if topology_error > 1e-12:
            raise RuntimeError(
                "compatible Maxwell topology failed C*G=0: "
                f"maximum entry={topology_error:.3e}"
            )

    diagonal = _edge_mass_diagonal(
        background,
        context,
        mqs=mqs,
        mqs_admittance=mqs_admittance,
    )
    scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    try:
        factor = spla.splu(
            scalar,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(scalar)
    elapsed = float(time.perf_counter() - started)
    block = GradientBlock(G, scalar, factor, elapsed, topology_error)
    cache[key] = block
    print(
        f"Maxwell scalar-gradient block: scalar_dofs={G.shape[1]}, "
        f"topology={topology_error:.1e}, factor={elapsed:.1f}s",
        flush=True,
    )
    return block


def compose_block_preconditioner(A, edge_preconditioner, gradient_block, *, post_correct=True):
    """Multiplicative edge/gradient approximate inverse.

    First remove the exact gradient component of the incoming residual, apply
    the edge ILU only to the remaining component, and optionally clean the
    gradient residual introduced by that approximate edge solve.  Every scalar
    correction is derived from the original ``A`` through ``G.T A G``; no source
    projection and no gauge penalty are introduced.
    """
    Gblock = gradient_block

    def apply(vector):
        v = np.asarray(vector, complex).reshape(-1)
        zg = Gblock.solve(v)
        transverse_rhs = v - A @ zg
        zt = np.asarray(edge_preconditioner @ transverse_rhs, complex).reshape(-1)
        z = zg + zt
        if post_correct:
            remainder = v - A @ z
            z = z + Gblock.solve(remainder)
        return np.asarray(z, complex).reshape(-1)

    return spla.LinearOperator(A.shape, matvec=apply, dtype=A.dtype)


__all__ = [
    "GradientBlock",
    "build_gradient_block",
    "compose_block_preconditioner",
    "gradient_operator",
    "node_coordinates",
    "source_terminal_divergence",
]
