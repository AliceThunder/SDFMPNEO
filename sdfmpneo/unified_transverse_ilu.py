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
    """Return a preconditioner matrix whose augmentation vanishes on e_t.

    ``stabilization_factor`` has the same intuitive scale as the old row-diagonal
    shift, but the added magnitude is confined to the compatible gradient
    directions instead of being applied to every Maxwell edge DOF.
    """
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
    # Stable complex reciprocal.  This is used only in the preconditioner.
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
    # Keep pathological material data from overflowing a preconditioner while
    # retaining enough gain to lift the weak gradient block to the curl scale.
    gain = float(np.clip(gain, 1.0, 1e12))

    augmented = (A + gain * lift).tocsr()
    augmented.sum_duplicates()
    augmented.eliminate_zeros()
    stats = {
        "stabilization_factor": factor,
        "augmentation_gain": gain,
        "row_reference": row_reference,
        "mass_reference": mass_reference,
        "augmentation_nnz": int(lift.nnz),
        "augmented_nnz": int(augmented.nnz),
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
    augmented, stats = build_transverse_stabilized_matrix(
        A,
        background,
        context,
        gradient_block,
        stabilization_factor=stabilization_factor,
        mqs=bool(mqs),
        mqs_admittance=mqs_admittance,
    )
    ilu = spla.spilu(
        augmented.tocsc(),
        drop_tol=float(drop_tol),
        fill_factor=float(fill_factor),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.01,
    )
    elapsed = float(time.perf_counter() - started)
    stats = dict(stats)
    stats["seconds"] = elapsed
    stats["drop_tolerance"] = float(drop_tol)
    stats["fill_factor"] = float(fill_factor)
    print(
        "Maxwell compatible transverse ILU: "
        f"gain={stats['augmentation_gain']:.3e}, "
        f"fill={float(fill_factor):g}, drop={float(drop_tol):.1e}, "
        f"factor={elapsed:.1f}s",
        flush=True,
    )
    return spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype), stats


__all__ = ["build_transverse_ilu", "build_transverse_stabilized_matrix"]
