"""Scale-robust ILU stabilization for compatible Maxwell solves.

After the exact scalar-gradient block has removed the longitudinal response, the
remaining correction ``e_t`` satisfies

    G.T A e_t = G.T D e_t = 0,

because ``G.T C.T = 0`` for the compatible Cartesian complex and
``A = C.T H_mu C + D``.  A full diagonal shift was useful before the gradient
block existed, but it also perturbs the physical transverse block and becomes a
poor preconditioner on the 2.25-mm validation grid.

This module builds an ILU only for preconditioning from

    P = A + alpha D G W G.T D,

with diagonal ``W`` approximating ``(G.T D G)^-1``.  The added term vanishes on
the exact transverse correction because ``G.T D e_t = 0``.  Therefore it can
regularize the weak gradient directions seen by ILU without changing the
transverse equation that Krylov is trying to solve.  The production Maxwell
matrix/RHS are never modified and certification still uses their true residual.

SuperLU ILU may encounter a zero pivot on very large incomplete factorizations
even when the physical matrix is nonsingular.  The 118k production solve also
shows that ``MMD_AT_PLUS_A`` gives a much better Maxwell approximate inverse than
``COLAMD``.  Large-grid recovery therefore keeps the successful MMD ordering and
adds only a tiny row-scaled diagonal inside the preconditioner before considering
COLAMD.  None of these fallbacks changes the physical Maxwell matrix seen by
Krylov or its certificate.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_gradient_block_maxwell import _edge_mass_diagonal


def _positive_median(values):
    values = np.asarray(values, float).reshape(-1)
    values = values[np.isfinite(values) & (values > np.finfo(float).tiny)]
    if values.size == 0:
        raise ValueError("compatible transverse ILU scaling has no positive reference values")
    return float(np.median(values))


def _row_scaled_regularization(matrix, relative_scale):
    """Add a tiny positive pivot floor to a *preconditioner matrix only*."""
    beta = float(relative_scale)
    if beta <= 0.0:
        return matrix
    row_scale = np.asarray(abs(matrix).sum(axis=1)).reshape(-1)
    reference = _positive_median(row_scale)
    scale = np.maximum(row_scale, reference * 1e-12)
    return (matrix + sp.diags(beta * scale, format="csr")).tocsr()


def _factor_with_pivot_recovery(matrix, *, drop_tol, fill_factor):
    """Build bounded-fill ILU with ordering/pivot recovery for numerical zero pivots.

    The 254k validation grid can make SuperLU's incomplete factorization report
    ``Factor is exactly singular``.  We first preserve the MMD ordering that is
    empirically effective on the 118k Maxwell system and regularize only its
    numerical pivots.  COLAMD is a last-resort factorization path because a
    factorization that exists is not necessarily a useful Maxwell preconditioner.
    """
    n = int(matrix.shape[0])
    if n >= 200000:
        attempts = (
            ("MMD_AT_PLUS_A", 0.01, 0.0),
            ("MMD_AT_PLUS_A", 0.01, 1e-10),
            ("MMD_AT_PLUS_A", 0.01, 1e-8),
            ("MMD_AT_PLUS_A", 0.01, 1e-6),
            ("MMD_AT_PLUS_A", 0.01, 1e-4),
            ("COLAMD", 0.0, 0.0),
            ("COLAMD", 0.0, 1e-8),
        )
    else:
        attempts = (
            ("MMD_AT_PLUS_A", 0.01, 0.0),
            ("MMD_AT_PLUS_A", 0.01, 1e-10),
            ("MMD_AT_PLUS_A", 0.01, 1e-8),
            ("COLAMD", 0.0, 0.0),
        )

    failures = []
    for index, (ordering, pivot_threshold, regularization) in enumerate(attempts, start=1):
        candidate = _row_scaled_regularization(matrix, regularization)
        attempt_started = time.perf_counter()
        try:
            ilu = spla.spilu(
                candidate.tocsc(),
                drop_tol=float(drop_tol),
                fill_factor=float(fill_factor),
                permc_spec=ordering,
                diag_pivot_thresh=float(pivot_threshold),
                drop_rule="basic,area",
                options={"Equil": True},
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            elapsed = time.perf_counter() - attempt_started
            message = str(exc).replace("\n", " ")
            if len(message) > 180:
                message = message[:177] + "..."
            failures.append((ordering, pivot_threshold, regularization, type(exc).__name__, message))
            print(
                "Maxwell compatible transverse ILU factor attempt failed: "
                f"attempt={index}/{len(attempts)}, ordering={ordering}, "
                f"pivot={pivot_threshold:.2g}, regularization={regularization:.1e}, "
                f"error={type(exc).__name__}: {message}, time={elapsed:.1f}s",
                flush=True,
            )
            continue

        elapsed = time.perf_counter() - attempt_started
        if index > 1 or regularization > 0.0:
            print(
                "Maxwell compatible transverse ILU factor recovered: "
                f"attempt={index}/{len(attempts)}, ordering={ordering}, "
                f"pivot={pivot_threshold:.2g}, regularization={regularization:.1e}, "
                f"time={elapsed:.1f}s",
                flush=True,
            )
        return ilu, {
            "factor_ordering": ordering,
            "factor_pivot_threshold": float(pivot_threshold),
            "factor_pivot_regularization": float(regularization),
            "factor_recovery_attempt": int(index),
        }

    details = "; ".join(
        f"{ordering}/pivot={pivot:.2g}/reg={regularization:.1e}:{error}"
        for ordering, pivot, regularization, error, _message in failures
    )
    raise RuntimeError(
        "all compatible transverse ILU factorization strategies failed"
        + (f" ({details})" if details else "")
    )


def build_transverse_stabilized_matrix(
    A,
    background,
    context,
    gradient_block,
    *,
    stabilization_factor=3e-2,
    mqs=False,
    mqs_admittance=None,
):
    """Return a preconditioner matrix whose augmentation vanishes on e_t."""
    factor = float(stabilization_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("transverse ILU stabilization_factor must be positive")

    G = gradient_block.gradient
    if G.shape[0] != A.shape[0]:
        raise ValueError("gradient block and Maxwell matrix dimensions disagree")

    d = _edge_mass_diagonal(
        background,
        context,
        mqs=bool(mqs),
        mqs_admittance=mqs_admittance,
    )
    scalar_diag = np.asarray(gradient_block.scalar_matrix.diagonal(), complex).reshape(-1)
    magnitude = np.abs(scalar_diag)
    reference = _positive_median(magnitude)
    floor = max(reference * 1e-12, np.finfo(float).tiny)
    inverse_diag = np.conj(scalar_diag) / (magnitude * magnitude + floor * floor)

    DG = (sp.diags(d, format="csr") @ G).tocsr()
    local_inverse = sp.diags(inverse_diag, format="csr")
    lift = (DG @ local_inverse @ DG.T).tocsr()
    lift.sum_duplicates()
    lift.eliminate_zeros()

    row_scale = np.asarray(abs(A).sum(axis=1)).reshape(-1)
    row_reference = _positive_median(row_scale)
    mass_reference = _positive_median(np.abs(d))
    gain = factor * row_reference / mass_reference
    gain = float(np.clip(gain, 1.0, 1e12))

    augmented = (A + gain * lift).tocsr()
    augmented.sum_duplicates()
    augmented.eliminate_zeros()
    diagonal = np.abs(np.asarray(augmented.diagonal(), complex).reshape(-1))
    diagonal_reference = _positive_median(diagonal)
    near_zero_diagonal = int(np.count_nonzero(diagonal <= diagonal_reference * 1e-14))
    stats = {
        "stabilization_factor": factor,
        "augmentation_gain": gain,
        "row_reference": row_reference,
        "mass_reference": mass_reference,
        "augmentation_nnz": int(lift.nnz),
        "augmented_nnz": int(augmented.nnz),
        "near_zero_diagonal_count": near_zero_diagonal,
        "diagonal_reference": diagonal_reference,
    }
    return augmented, stats


def build_transverse_ilu(
    A,
    background,
    context,
    gradient_block,
    *,
    drop_tol,
    fill_factor,
    stabilization_factor,
    mqs=False,
    mqs_admittance=None,
):
    """Factor the compatible transverse-stabilized matrix and return an inverse."""
    started = time.perf_counter()
    print(
        "Maxwell compatible transverse ILU build: "
        f"edges={A.shape[0]}, fill={float(fill_factor):g}, "
        f"drop={float(drop_tol):.1e}, stabilization={float(stabilization_factor):.2e}",
        flush=True,
    )
    stage = "augmentation"
    try:
        augmented, stats = build_transverse_stabilized_matrix(
            A,
            background,
            context,
            gradient_block,
            stabilization_factor=stabilization_factor,
            mqs=bool(mqs),
            mqs_admittance=mqs_admittance,
        )
        print(
            "Maxwell compatible transverse ILU matrix: "
            f"edges={A.shape[0]}, nnz={stats['augmented_nnz']}, "
            f"augmentation_nnz={stats['augmentation_nnz']}, "
            f"near_zero_diag={stats['near_zero_diagonal_count']}, "
            f"build={time.perf_counter()-started:.1f}s",
            flush=True,
        )
        stage = "spilu"
        ilu, factor_stats = _factor_with_pivot_recovery(
            augmented,
            drop_tol=float(drop_tol),
            fill_factor=float(fill_factor),
        )
        stats.update(factor_stats)
    except (RuntimeError, ValueError, MemoryError) as exc:
        elapsed = time.perf_counter() - started
        message = str(exc).replace("\n", " ")
        if len(message) > 320:
            message = message[:317] + "..."
        print(
            "Maxwell compatible transverse ILU FAILED: "
            f"stage={stage}, edges={A.shape[0]}, error={type(exc).__name__}: {message}, "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        raise

    elapsed = float(time.perf_counter() - started)
    stats = dict(stats)
    stats["seconds"] = elapsed
    stats["drop_tolerance"] = float(drop_tol)
    stats["fill_factor"] = float(fill_factor)
    print(
        "Maxwell compatible transverse ILU: "
        f"gain={stats['augmentation_gain']:.3e}, "
        f"fill={float(fill_factor):g}, drop={float(drop_tol):.1e}, "
        f"ordering={stats['factor_ordering']}, "
        f"pivot_reg={stats['factor_pivot_regularization']:.1e}, "
        f"factor={elapsed:.1f}s",
        flush=True,
    )
    return spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype), stats


__all__ = [
    "build_transverse_ilu",
    "build_transverse_stabilized_matrix",
]
