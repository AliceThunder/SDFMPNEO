"""Equilibrated linear algebra for the final Maxwell defect equation.

This module never changes the physical equation.  Given ``A delta = r`` it
builds diagonal row/column scalings and solves the exactly equivalent system

    (R A C) y = R r,    delta = C y.

The production two-level preconditioner ``M ~= A^-1`` is mapped into the same
coordinates as ``C^-1 M R^-1``.  Scaling is used only for the final iterative
refinement where the raw Maxwell system has a very large dynamic range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _positive_span(values):
    values = np.asarray(values, float).reshape(-1)
    good = values[np.isfinite(values) & (values > np.finfo(float).tiny)]
    if good.size == 0:
        return float("inf")
    return float(np.max(good) / np.min(good))


def _segment_max(values, indptr, size):
    values = np.asarray(values, float).reshape(-1)
    indptr = np.asarray(indptr, dtype=np.intp)
    counts = np.diff(indptr)
    out = np.zeros(int(size), float)
    nonempty = counts > 0
    if np.any(nonempty):
        starts = indptr[:-1][nonempty]
        out[nonempty] = np.maximum.reduceat(values, starts)
    return out


def _row_max(matrix, column_scale):
    values = np.abs(matrix.data) * np.asarray(column_scale, float)[matrix.indices]
    return _segment_max(values, matrix.indptr, matrix.shape[0])


def _column_max(csc, row_scale):
    values = np.abs(csc.data) * np.asarray(row_scale, float)[csc.indices]
    return _segment_max(values, csc.indptr, csc.shape[1])


@dataclass(frozen=True)
class EquilibratedDefectSystem:
    physical_A: object
    operator: object
    preconditioner: object | None
    rhs: np.ndarray
    row_scale: np.ndarray
    column_scale: np.ndarray
    row_span_before: float
    column_span_before: float
    row_span_after: float
    column_span_after: float
    row_scale_span: float
    column_scale_span: float

    def physical_correction(self, scaled_solution):
        value = np.asarray(scaled_solution, complex).reshape(-1)
        if value.shape != self.column_scale.shape:
            raise ValueError("scaled Maxwell correction has wrong dimension")
        return self.column_scale * value


def build_equilibrated_defect_system(A, defect, M=None, *, iterations=4, scale_limit=1e12):
    """Build an exactly equivalent Ruiz-equilibrated defect system."""
    if not sp.issparse(A):
        raise TypeError("equilibrated Maxwell defect requires a sparse physical matrix")
    matrix = A.tocsr()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("equilibrated Maxwell defect requires a square matrix")
    rhs = np.asarray(defect, complex).reshape(-1)
    n = matrix.shape[0]
    if rhs.shape != (n,) or np.any(~np.isfinite(rhs)):
        raise ValueError("equilibrated Maxwell defect rhs is invalid")
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("equilibration iterations must be positive")
    limit = float(scale_limit)
    if not np.isfinite(limit) or limit <= 1.0:
        raise ValueError("equilibration scale_limit must exceed one")

    csc = matrix.tocsc()
    ones = np.ones(n, float)
    row_before = _row_max(matrix, ones)
    col_before = _column_max(csc, ones)
    row_scale = np.ones(n, float)
    column_scale = np.ones(n, float)
    tiny = np.finfo(float).tiny
    lower = 1.0 / limit

    # Alternating Ruiz infinity-norm equilibration.  Each update uses the
    # currently scaled operator, but the sparse physical matrix is never copied
    # or modified.
    for _ in range(iterations):
        row_norm = row_scale * _row_max(matrix, column_scale)
        good = np.isfinite(row_norm) & (row_norm > tiny)
        update = np.ones(n, float)
        update[good] = 1.0 / np.sqrt(row_norm[good])
        row_scale = np.clip(row_scale * update, lower, limit)

        col_norm = column_scale * _column_max(csc, row_scale)
        good = np.isfinite(col_norm) & (col_norm > tiny)
        update = np.ones(n, float)
        update[good] = 1.0 / np.sqrt(col_norm[good])
        column_scale = np.clip(column_scale * update, lower, limit)

    row_after = row_scale * _row_max(matrix, column_scale)
    col_after = column_scale * _column_max(csc, row_scale)

    def matvec(value):
        value = np.asarray(value, complex).reshape(-1)
        return row_scale * np.asarray(matrix @ (column_scale * value), complex).reshape(-1)

    operator = spla.LinearOperator(matrix.shape, matvec=matvec, dtype=matrix.dtype)
    preconditioner = None
    if M is not None:
        def psolve(value):
            value = np.asarray(value, complex).reshape(-1)
            physical_rhs = value / row_scale
            physical_solution = np.asarray(M @ physical_rhs, complex).reshape(-1)
            return physical_solution / column_scale

        preconditioner = spla.LinearOperator(matrix.shape, matvec=psolve, dtype=matrix.dtype)

    scaled_rhs = row_scale * rhs
    return EquilibratedDefectSystem(
        physical_A=matrix,
        operator=operator,
        preconditioner=preconditioner,
        rhs=np.asarray(scaled_rhs, complex),
        row_scale=row_scale,
        column_scale=column_scale,
        row_span_before=_positive_span(row_before),
        column_span_before=_positive_span(col_before),
        row_span_after=_positive_span(row_after),
        column_span_after=_positive_span(col_after),
        row_scale_span=_positive_span(row_scale),
        column_scale_span=_positive_span(column_scale),
    )


__all__ = ["EquilibratedDefectSystem", "build_equilibrated_defect_system"]
