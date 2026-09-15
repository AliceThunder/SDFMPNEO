"""Accurate residual formation for ill-scaled sparse Maxwell systems.

The finest local Maxwell solve can reach a point where a correction equation is
solved successfully in double precision but the conventional recomputation
``b - A @ x`` stagnates several orders of magnitude above the requested
certificate.  For a curl-curl system this can be a residual-evaluation problem:
individual row products are much larger than the final residual and cancel.

This module keeps the stored physical matrix and field unchanged.  It provides
an independently accumulated residual of that same matrix/vector pair and a
roundoff-floor diagnostic.  On platforms with genuinely extended ``longdouble``
precision a blocked extended-precision CSR matvec is used.  Standard Windows
NumPy builds normally expose no extra long-double mantissa, so a compensated
(double-double product + double-double row sum) CSR path is provided as the
portable fallback.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp


_SPLITTER = 134217729.0  # 2**27 + 1, Dekker splitter for IEEE binary64.


def _as_csr(A):
    matrix = A.tocsr() if sp.issparse(A) else None
    if matrix is None:
        raise TypeError("accurate Maxwell residual requires a SciPy sparse matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("accurate Maxwell residual requires a square matrix")
    return matrix


def residual_roundoff_diagnostics(A, field, rhs, *, standard_residual=None):
    """Estimate whether ordinary CSR residual formation is roundoff limited."""
    matrix = _as_csr(A)
    x = np.asarray(field, complex).reshape(-1)
    b = np.asarray(rhs, complex).reshape(-1)
    if x.shape != (matrix.shape[1],) or b.shape != (matrix.shape[0],):
        raise ValueError("Maxwell residual dimensions do not match")
    if standard_residual is None:
        standard_residual = np.asarray(b - matrix @ x, complex).reshape(-1)
    else:
        standard_residual = np.asarray(standard_residual, complex).reshape(-1)

    norm_b = max(float(np.linalg.norm(b)), np.finfo(float).tiny)
    absolute_action = np.asarray(abs(matrix) @ np.abs(x), float).reshape(-1)
    row_nnz = np.diff(matrix.indptr).astype(float)
    eps = np.finfo(float).eps
    # A complex multiply/add consumes several real operations.  The factor 8 is
    # intentionally conservative; this is a diagnostic upper scale, not an
    # acceptance tolerance.
    operations = np.maximum(1.0, 8.0 * row_nnz + 2.0)
    gamma = operations * eps
    gamma = gamma / np.maximum(1.0 - gamma, 0.5)
    row_bound = gamma * absolute_action + eps * np.abs(b)
    relative_bound = float(np.linalg.norm(row_bound) / norm_b)
    dynamic_range = float(np.linalg.norm(absolute_action + np.abs(b)) / norm_b)
    standard_relative = float(np.linalg.norm(standard_residual) / norm_b)
    cancellation = (absolute_action + np.abs(b)) / np.maximum(
        np.abs(standard_residual), np.finfo(float).tiny
    )
    finite_cancellation = cancellation[np.isfinite(cancellation)]
    maximum_cancellation = (
        float(np.max(finite_cancellation)) if finite_cancellation.size else float("inf")
    )
    return {
        "standard_relative_residual": standard_relative,
        "roundoff_bound_relative": relative_bound,
        "action_rhs_dynamic_range": dynamic_range,
        "maximum_row_cancellation_ratio": maximum_cancellation,
        "maximum_row_nnz": int(np.max(row_nnz)) if row_nnz.size else 0,
    }


def _extended_longdouble_available():
    try:
        return bool(np.finfo(np.longdouble).eps < 0.5 * np.finfo(np.float64).eps)
    except (TypeError, ValueError):
        return False


def _extended_csr_residual(matrix, x, b, *, rows_per_chunk=8192):
    """Form b-Ax with true platform long-double precision when available."""
    n = matrix.shape[0]
    out = np.empty(n, dtype=np.complex128)
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    x_ext = np.asarray(x, dtype=np.clongdouble)
    b_ext = np.asarray(b, dtype=np.clongdouble)

    for r0 in range(0, n, int(rows_per_chunk)):
        r1 = min(n, r0 + int(rows_per_chunk))
        p0 = int(indptr[r0])
        p1 = int(indptr[r1])
        starts = np.asarray(indptr[r0 : r1 + 1] - p0, dtype=np.intp)
        values = (
            np.asarray(data[p0:p1], dtype=np.clongdouble)
            * x_ext[np.asarray(indices[p0:p1], dtype=np.intp)]
        )
        counts = np.diff(starts)
        if values.size and np.all(counts > 0):
            sums = np.add.reduceat(values, starts[:-1])
        else:
            sums = np.zeros(r1 - r0, dtype=np.clongdouble)
            for local in range(r1 - r0):
                a = int(starts[local])
                z = int(starts[local + 1])
                if z > a:
                    sums[local] = np.sum(values[a:z], dtype=np.clongdouble)
        residual = b_ext[r0:r1] - sums
        out[r0:r1] = np.asarray(residual, dtype=np.complex128)
    return out


def _two_product_array(a, b):
    """Error-free binary64 product expansion p+e ~= exact(a*b), vectorized."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    p = a * b
    ca = _SPLITTER * a
    cb = _SPLITTER * b
    ah = ca - (ca - a)
    bh = cb - (cb - b)
    al = a - ah
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _two_sum_array(a, b):
    s = a + b
    bp = s - a
    e = (a - (s - bp)) + (b - bp)
    return s, e


def _two_sum_scalar(a, b):
    s = a + b
    bp = s - a
    e = (a - (s - bp)) + (b - bp)
    return s, e


def _dd_add_scalar(hi, lo, value_hi, value_lo=0.0):
    s, e = _two_sum_scalar(float(hi), float(value_hi))
    lo2 = float(lo) + float(e) + float(value_lo)
    s2, e2 = _two_sum_scalar(s, lo2)
    return s2, e2


def _compensated_csr_residual(matrix, x, b, *, rows_per_chunk=4096):
    """Portable compensated complex CSR residual using binary64 expansions."""
    n = matrix.shape[0]
    out = np.empty(n, dtype=np.complex128)
    indptr = matrix.indptr
    indices = matrix.indices
    data = np.asarray(matrix.data, dtype=np.complex128)
    x = np.asarray(x, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    for r0 in range(0, n, int(rows_per_chunk)):
        r1 = min(n, r0 + int(rows_per_chunk))
        p0 = int(indptr[r0])
        p1 = int(indptr[r1])
        local_data = data[p0:p1]
        local_x = x[np.asarray(indices[p0:p1], dtype=np.intp)]

        ar = local_data.real
        ai = local_data.imag
        xr = local_x.real
        xi = local_x.imag
        p_rr, e_rr = _two_product_array(ar, xr)
        p_ii, e_ii = _two_product_array(ai, xi)
        p_ri, e_ri = _two_product_array(ar, xi)
        p_ir, e_ir = _two_product_array(ai, xr)

        real_hi, real_sum_error = _two_sum_array(p_rr, -p_ii)
        real_lo = real_sum_error + e_rr - e_ii
        imag_hi, imag_sum_error = _two_sum_array(p_ri, p_ir)
        imag_lo = imag_sum_error + e_ri + e_ir

        starts = np.asarray(indptr[r0 : r1 + 1] - p0, dtype=np.intp)
        for local_row in range(r1 - r0):
            row = r0 + local_row
            a = int(starts[local_row])
            z = int(starts[local_row + 1])
            rh = float(b[row].real)
            rl = 0.0
            ih = float(b[row].imag)
            il = 0.0
            for q in range(a, z):
                rh, rl = _dd_add_scalar(rh, rl, -real_hi[q], -real_lo[q])
                ih, il = _dd_add_scalar(ih, il, -imag_hi[q], -imag_lo[q])
            # math.fsum preserves the final low component instead of discarding
            # it in an ordinary binary64 addition.
            out[row] = complex(math.fsum((rh, rl)), math.fsum((ih, il)))
    return out


def accurate_residual_vector(A, field, rhs, *, target_relative=1e-9):
    """Return an accurately accumulated residual and numerical diagnostics."""
    matrix = _as_csr(A)
    x = np.asarray(field, complex).reshape(-1)
    b = np.asarray(rhs, complex).reshape(-1)
    standard = np.asarray(b - matrix @ x, complex).reshape(-1)
    diagnostics = residual_roundoff_diagnostics(
        matrix, x, b, standard_residual=standard
    )

    # If the conservative rounding scale is far below the requested certificate,
    # the ordinary sparse matvec is already sufficient and substantially faster.
    target = max(float(target_relative), np.finfo(float).tiny)
    if diagnostics["roundoff_bound_relative"] <= 0.05 * target:
        diagnostics["accumulation_mode"] = "standard-csr"
        diagnostics["accurate_relative_residual"] = diagnostics[
            "standard_relative_residual"
        ]
        return standard, diagnostics

    if _extended_longdouble_available():
        residual = _extended_csr_residual(matrix, x, b)
        mode = "extended-longdouble-csr"
    else:
        residual = _compensated_csr_residual(matrix, x, b)
        mode = "compensated-double-double-csr"

    norm_b = max(float(np.linalg.norm(b)), np.finfo(float).tiny)
    diagnostics["accumulation_mode"] = mode
    diagnostics["accurate_relative_residual"] = float(
        np.linalg.norm(residual) / norm_b
    )
    diagnostics["standard_accurate_residual_ratio"] = float(
        diagnostics["standard_relative_residual"]
        / max(diagnostics["accurate_relative_residual"], np.finfo(float).tiny)
    )
    return np.asarray(residual, complex).reshape(-1), diagnostics


__all__ = ["accurate_residual_vector", "residual_roundoff_diagnostics"]
