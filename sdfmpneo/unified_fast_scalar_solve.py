"""Certified iterative linear solver for nested Cartesian scalar patches.

This module contains linear algebra only.  It does not define source, material,
patch, correction or Gate semantics.  Large 3-D scalar Dirichlet systems first
use a parent-grid Galerkin coarse space plus damped fine Jacobi inside LGMRES.
If that route does not reach the unchanged 1e-9 true-residual certificate, a
bounded-fill fine-grid ILU + LGMRES fallback is tried before the historical
fill-heavy sparse direct solve.  Direct solve remains the final correctness
fallback; no physics or convergence tolerance is relaxed.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import unified_certified_local_solve as _local_solve


_DIRECT_THRESHOLD = 60000
_RESIDUAL_TOLERANCE = 1e-9


def _parent_subset_axis(parent_axis, patch_axis):
    parent = np.asarray(parent_axis, float)
    patch = np.asarray(patch_axis, float)
    scale = max(float(np.max(np.abs(parent))), float(np.max(np.abs(patch))), 1.0)
    tol = 256.0 * np.finfo(float).eps * scale
    mask = (parent >= patch[0] - tol) & (parent <= patch[-1] + tol)
    coarse = parent[mask].copy()
    if coarse.size < 3:
        raise RuntimeError("scalar patch has fewer than two parent coarse cells")
    if abs(float(coarse[0] - patch[0])) > tol or abs(float(coarse[-1] - patch[-1])) > tol:
        raise RuntimeError("scalar patch boundary is not aligned to parent nodes")
    coarse[0] = patch[0]
    coarse[-1] = patch[-1]
    return coarse


def _coarse_axes(parent, patch):
    return tuple(
        _parent_subset_axis(parent_axis, patch_axis)
        for parent_axis, patch_axis in zip(
            (parent.x, parent.y, parent.z), (patch.x, patch.y, patch.z)
        )
    )


def _interpolation_1d(coarse_axis, fine_axis):
    coarse = np.asarray(coarse_axis, float)
    fine = np.asarray(fine_axis, float)
    nf = max(0, fine.size - 2)
    nc = max(0, coarse.size - 2)
    rows, cols, data = [], [], []
    scale = max(float(np.max(np.abs(coarse))), float(np.max(np.abs(fine))), 1.0)
    tol = 256.0 * np.finfo(float).eps * scale
    for row, value in enumerate(fine[1:-1]):
        index = int(np.searchsorted(coarse, value, side="left"))
        if index < coarse.size and abs(float(coarse[index] - value)) <= tol:
            if 0 < index < coarse.size - 1:
                rows.append(row)
                cols.append(index - 1)
                data.append(1.0)
            continue
        right = min(max(index, 1), coarse.size - 1)
        left = right - 1
        width = float(coarse[right] - coarse[left])
        if width <= 0.0:
            raise FloatingPointError("coarse interpolation axis is not monotone")
        wr = float((value - coarse[left]) / width)
        wl = 1.0 - wr
        if 0 < left < coarse.size - 1 and abs(wl) > np.finfo(float).tiny:
            rows.append(row)
            cols.append(left - 1)
            data.append(wl)
        if 0 < right < coarse.size - 1 and abs(wr) > np.finfo(float).tiny:
            rows.append(row)
            cols.append(right - 1)
            data.append(wr)
    return sp.csr_matrix((data, (rows, cols)), shape=(nf, nc))


def _nodal_prolongation(coarse_axes, fine_axes):
    pieces = [_interpolation_1d(c, f) for c, f in zip(coarse_axes, fine_axes)]
    P = sp.kron(sp.kron(pieces[0], pieces[1], format="csr"), pieces[2], format="csr")
    expected_fine = int(np.prod([len(axis) - 2 for axis in fine_axes], dtype=np.int64))
    expected_coarse = int(np.prod([len(axis) - 2 for axis in coarse_axes], dtype=np.int64))
    if P.shape != (expected_fine, expected_coarse):
        raise AssertionError("scalar two-level prolongation shape mismatch")
    return P


def _direct_solve(A, rhs):
    try:
        factor = spla.splu(
            A.tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(A.tocsc())
    return np.asarray(factor.solve(rhs), complex).reshape(-1)


def _inverse_diagonal(A):
    diagonal = np.asarray(A.diagonal(), complex).reshape(-1)
    magnitude = np.abs(diagonal)
    good = magnitude[np.isfinite(magnitude) & (magnitude > np.finfo(float).tiny)]
    if good.size == 0:
        raise RuntimeError("scalar patch diagonal has no usable entries")
    floor = max(float(np.median(good)) * 1e-12, np.finfo(float).tiny)
    return np.conj(diagonal) / (magnitude * magnitude + floor * floor)


def _two_level_operator(A, P, coarse_factor, inverse_diagonal, *, sweeps, weight=0.65):
    sweeps = max(1, int(sweeps))

    def smooth(residual):
        r = np.asarray(residual, complex).reshape(-1).copy()
        z = np.zeros_like(r)
        for _ in range(sweeps):
            delta = float(weight) * inverse_diagonal * r
            z += delta
            r -= A @ delta
        return z, r

    def apply(vector):
        rhs = np.asarray(vector, complex).reshape(-1)
        zpre, r1 = smooth(rhs)
        coarse_rhs = np.asarray(P.T @ r1, complex).reshape(-1)
        coarse = np.asarray(coarse_factor.solve(coarse_rhs), complex).reshape(-1)
        zc = np.asarray(P @ coarse, complex).reshape(-1)
        r2 = r1 - A @ zc
        zpost, _r3 = smooth(r2)
        return np.asarray(zpre + zc + zpost, complex).reshape(-1)

    return spla.LinearOperator(A.shape, matvec=apply, dtype=A.dtype)


def _ilu_fallback(A, rhs, best, best_residual):
    attempts = (
        (2e-3, 4.0, 0.0, 28, 24, "ilu-fast"),
        (5e-4, 7.0, 0.0, 40, 30, "ilu-strong"),
        (5e-4, 7.0, 1e-10, 40, 30, "ilu-shifted"),
    )
    for drop_tol, fill_factor, shift_factor, maxiter, inner_m, label in attempts:
        started = time.perf_counter()
        try:
            M = _local_solve._ilu_preconditioner(
                A,
                drop_tol=float(drop_tol),
                fill_factor=float(fill_factor),
                shift_factor=float(shift_factor),
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            print(
                "longitudinal scalar "
                f"{label} preconditioner unavailable ({exc}); trying next solver",
                flush=True,
            )
            continue
        build_seconds = float(time.perf_counter() - started)
        t0 = time.perf_counter()
        candidate, info = _local_solve._lgmres(
            A,
            rhs,
            x0=best,
            M=M,
            rtol=max(0.2 * _RESIDUAL_TOLERANCE, 1e-12),
            maxiter=int(maxiter),
            inner_m=int(inner_m),
        )
        candidate = np.asarray(candidate, complex).reshape(-1)
        residual = float(_local_solve._relative_residual(A, candidate, rhs))
        solve_seconds = float(time.perf_counter() - t0)
        print(
            "longitudinal scalar fine-ILU: "
            f"solver={label}, dofs={A.shape[0]}, residual={residual:.3e}, "
            f"info={int(info)}, build={build_seconds:.1f}s, solve={solve_seconds:.1f}s",
            flush=True,
        )
        if np.isfinite(residual) and residual < best_residual:
            best = candidate
            best_residual = residual
        if best_residual <= _RESIDUAL_TOLERANCE:
            return best, best_residual, f"{label}-lgmres"
    return best, best_residual, None


def solve_refined(A, rhs, *, parent, patch, x0):
    n = int(A.shape[0])
    rhs = np.asarray(rhs, complex).reshape(-1)
    if n < _DIRECT_THRESHOLD:
        started = time.perf_counter()
        field = _direct_solve(A, rhs)
        residual = float(_local_solve._relative_residual(A, field, rhs))
        print(
            f"longitudinal scalar direct: dofs={n}, residual={residual:.3e}, "
            f"time={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return field, residual, "direct"

    coarse_axes = _coarse_axes(parent, patch)
    fine_axes = tuple(np.asarray(axis, float) for axis in (patch.x, patch.y, patch.z))
    best = None if x0 is None else np.asarray(x0, complex).reshape(-1).copy()
    best_residual = (
        float("inf")
        if best is None
        else float(_local_solve._relative_residual(A, best, rhs))
    )
    started = time.perf_counter()
    try:
        P = _nodal_prolongation(coarse_axes, fine_axes)
        Ac = (P.T @ (A @ P)).tocsc()
        Ac.sum_duplicates()
        Ac.eliminate_zeros()
        if Ac.shape[0] == 0:
            raise RuntimeError("scalar two-level coarse space is empty")
        try:
            coarse_factor = spla.splu(
                Ac,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.01,
                options={"Equil": True},
            )
        except (RuntimeError, ValueError):
            coarse_factor = spla.splu(Ac)
        inverse_diagonal = _inverse_diagonal(A)
        build_seconds = float(time.perf_counter() - started)
        for sweeps, maxiter, inner_m in ((1, 24, 24), (2, 40, 30)):
            M = _two_level_operator(A, P, coarse_factor, inverse_diagonal, sweeps=sweeps)
            t0 = time.perf_counter()
            candidate, info = _local_solve._lgmres(
                A,
                rhs,
                x0=best,
                M=M,
                rtol=max(0.2 * _RESIDUAL_TOLERANCE, 1e-12),
                maxiter=maxiter,
                inner_m=inner_m,
            )
            candidate = np.asarray(candidate, complex).reshape(-1)
            residual = float(_local_solve._relative_residual(A, candidate, rhs))
            elapsed = float(time.perf_counter() - t0)
            print(
                "longitudinal scalar two-level: "
                f"dofs={n}, coarse={Ac.shape[0]}, P_nnz={P.nnz}, sweeps={sweeps}, "
                f"residual={residual:.3e}, info={int(info)}, build={build_seconds:.1f}s, "
                f"solve={elapsed:.1f}s",
                flush=True,
            )
            if np.isfinite(residual) and residual < best_residual:
                best = candidate
                best_residual = residual
            if best_residual <= _RESIDUAL_TOLERANCE:
                return best, best_residual, "two-level-lgmres"
        print(
            "longitudinal scalar two-level did not reach residual Gate; "
            "trying bounded-fill fine-grid ILU",
            flush=True,
        )
    except (RuntimeError, ValueError, MemoryError, FloatingPointError) as exc:
        print(
            f"longitudinal scalar two-level unavailable ({exc}); trying fine-grid ILU",
            flush=True,
        )

    best, best_residual, label = _ilu_fallback(A, rhs, best, best_residual)
    if label is not None:
        return best, best_residual, label

    print(
        "longitudinal scalar iterative solvers did not reach residual Gate; "
        "falling back to sparse direct solve",
        flush=True,
    )
    t0 = time.perf_counter()
    field = _direct_solve(A, rhs)
    residual = float(_local_solve._relative_residual(A, field, rhs))
    print(
        f"longitudinal scalar fallback direct: dofs={n}, residual={residual:.3e}, "
        f"time={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return field, residual, "fallback-direct"


__all__ = ["_nodal_prolongation", "solve_refined"]
