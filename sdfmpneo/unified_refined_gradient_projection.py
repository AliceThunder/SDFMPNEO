"""Certified scalar-gradient projection for localized self truth extraction.

The fast GradientBlock.solve() remains unchanged because it is used repeatedly
inside Maxwell preconditioners.  Localized self extraction is different: it is
performed only once after a certified full-field solve, and on the finest local
grid the transverse remainder can be only a few parts per million of the full
field.  A one-shot complex128 scalar solve is then not accurate enough to form
that small remainder reliably.

This module reuses the existing sparse LU factor as an iterative-refinement
solver, forms the scalar residual with the compensated CSR machinery, retains
small scalar corrections in a high/low expansion, and evaluates G*phi with
error-free edge differences.
"""
from __future__ import annotations

import numpy as np

from .unified_accurate_residual import accurate_residual_vector
from .unified_compensated_field import (
    CompensatedComplexField,
    as_compensated_field,
    compensated_add,
    field_norm,
    field_parts,
)


def _two_sum(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s = a + b
    bp = s - a
    e = (a - (s - bp)) + (b - bp)
    return s, e


def _edge_node_ids(background):
    cached = getattr(background, "_localized_edge_node_ids", None)
    if cached is not None:
        return cached
    starts = np.empty(background.n_edges, dtype=np.int64)
    stops = np.empty(background.n_edges, dtype=np.int64)
    ny1 = int(background.ny + 1)
    nz1 = int(background.nz + 1)

    def node(i, j, k):
        return (int(i) * ny1 + int(j)) * nz1 + int(k)

    for edge, (axis, i, j, k) in enumerate(background.edge_tuples):
        starts[edge] = node(i, j, k)
        if axis == 0:
            stops[edge] = node(i + 1, j, k)
        elif axis == 1:
            stops[edge] = node(i, j + 1, k)
        else:
            stops[edge] = node(i, j, k + 1)
    cached = (starts, stops)
    background._localized_edge_node_ids = cached
    return cached


def _compensated_gradient_field(background, scalar_field):
    """Return G*phi as a two-term edge expansion.

    GradientBlock uses the gauge-fixed matrix G[:, 1:], so scalar coordinate
    zero is the removed constant-potential degree of freedom.
    """
    phi_high, phi_low = field_parts(scalar_field)
    n_nodes = (background.nx + 1) * (background.ny + 1) * (background.nz + 1)
    if phi_high.shape != (n_nodes - 1,):
        raise ValueError("localized scalar projection has incompatible dimension")

    full_high = np.zeros(n_nodes, dtype=np.complex128)
    full_low = np.zeros(n_nodes, dtype=np.complex128)
    full_high[1:] = phi_high
    full_low[1:] = phi_low
    starts, stops = _edge_node_ids(background)

    ar = full_high[stops].real
    br = -full_high[starts].real
    ai = full_high[stops].imag
    bi = -full_high[starts].imag
    hr, lr = _two_sum(ar, br)
    hi, li = _two_sum(ai, bi)
    edge = CompensatedComplexField(
        np.asarray(hr + 1j * hi, dtype=np.complex128),
        np.asarray(lr + 1j * li, dtype=np.complex128),
    )
    edge = compensated_add(edge, full_low[stops])
    edge = compensated_add(edge, full_low[starts], scale=-1.0)
    return edge


def refined_gradient_projection(
    background,
    gradient_block,
    rhs,
    *,
    relative_tolerance=5e-13,
    maximum_refinements=5,
):
    """Solve the compatible scalar projection with certified residual refinement."""
    G = gradient_block.gradient
    scalar_matrix = gradient_block.scalar_matrix
    scalar_rhs = np.asarray(G.T @ np.asarray(rhs, complex).reshape(-1), complex).reshape(-1)
    norm_rhs = max(float(np.linalg.norm(scalar_rhs)), np.finfo(float).tiny)
    phi0 = np.asarray(gradient_block.factor.solve(scalar_rhs), complex).reshape(-1)
    phi = as_compensated_field(phi0)

    residual, diagnostics = accurate_residual_vector(
        scalar_matrix,
        phi,
        scalar_rhs,
        target_relative=float(relative_tolerance),
    )
    initial = float(np.linalg.norm(residual) / norm_rhs)
    current = initial
    refinements = 0

    for _ in range(int(maximum_refinements)):
        if current <= float(relative_tolerance):
            break
        delta = np.asarray(gradient_block.factor.solve(residual), complex).reshape(-1)
        if np.any(~np.isfinite(delta)):
            break
        candidate = compensated_add(phi, delta)
        candidate_residual, candidate_diag = accurate_residual_vector(
            scalar_matrix,
            candidate,
            scalar_rhs,
            target_relative=float(relative_tolerance),
        )
        candidate_value = float(np.linalg.norm(candidate_residual) / norm_rhs)
        if not np.isfinite(candidate_value) or candidate_value >= current:
            break
        phi = candidate
        residual = candidate_residual
        diagnostics = candidate_diag
        current = candidate_value
        refinements += 1

    longitudinal = _compensated_gradient_field(background, phi)
    low_relative = float(
        np.linalg.norm(field_parts(longitudinal)[1])
        / max(field_norm(longitudinal), np.finfo(float).tiny)
    )
    report = {
        "initial_relative_residual": initial,
        "relative_residual": current,
        "relative_tolerance": float(relative_tolerance),
        "refinements": int(refinements),
        "accumulation_mode": str(diagnostics.get("accumulation_mode", "unknown")),
        "edge_low_relative_norm": low_relative,
        "converged": bool(current <= float(relative_tolerance)),
    }
    print(
        "local Maxwell localized gradient projection: "
        f"initial={initial:.3e}, final={current:.3e}, "
        f"refinements={refinements}, low={low_relative:.3e}, "
        f"mode={report['accumulation_mode']}",
        flush=True,
    )
    if not report["converged"]:
        raise RuntimeError(
            "localized scalar-gradient projection did not reach its certified residual; "
            f"residual={current:.3e}, tolerance={float(relative_tolerance):.3e}"
        )
    return longitudinal, report


__all__ = ["refined_gradient_projection"]
