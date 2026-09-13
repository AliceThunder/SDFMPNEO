"""Residual-driven common Maxwell space for the unified background.

The basis rank is not a user-chosen hyperparameter.  It is the smallest rank
reached by residual-greedy enrichment that satisfies the requested physical
anchor residual, subject only to the ambient Maxwell space and numerical
linear-independence of new residual directions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def safe_diag(A):
    d = np.asarray(A.diagonal(), complex)
    scale = max(float(np.max(np.abs(d))), np.finfo(float).tiny)
    return np.where(np.abs(d) > 1e-12 * scale, d, scale + 0j)


def orthonormal_append(V, v):
    q = np.asarray(v, complex).copy()
    if V.size:
        for _ in range(2):
            q -= V @ (V.conj().T @ q)
    n = float(np.linalg.norm(q))
    if n <= 1e-12 * max(float(np.linalg.norm(v)), 1.0):
        return V, False
    q /= n
    return (q[:, None] if V.size == 0 else np.column_stack([V, q])), True


def _solve_reduced(Ar, br):
    if Ar.shape[0] == 0:
        return np.empty((0, br.shape[1]), complex)
    try:
        return scipy.linalg.solve(Ar, br, assume_a="gen", check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return np.linalg.lstsq(Ar, br, rcond=None)[0]


def reduced_solution(A, B, V):
    if V.shape[1] == 0:
        return np.zeros((A.shape[0], B.shape[1]), complex)
    AV = A @ V
    Ar = V.conj().T @ AV
    br = V.conj().T @ B
    return V @ _solve_reduced(Ar, br)


@dataclass(frozen=True)
class BasisReport:
    basis_dimension: int
    maximum_anchor_relative_residual: float
    target_relative_residual: float
    sample_count: int
    enrichment_steps: int
    converged: bool
    stop_reason: str


def _expand_anchor(anchor, old_basis, new_vector):
    """Update one anchor's reduced matrices after appending one basis vector."""
    aq = anchor["A"] @ new_vector
    old_rank = old_basis.shape[1]
    if old_rank == 0:
        Ar = np.asarray([[np.vdot(new_vector, aq)]], complex)
        br = np.asarray(new_vector.conj() @ anchor["B"], complex).reshape(1, -1)
        AV = aq[:, None]
    else:
        top_right = old_basis.conj().T @ aq
        bottom_left = new_vector.conj() @ anchor["AV"]
        bottom = np.vdot(new_vector, aq)
        Ar = np.empty((old_rank + 1, old_rank + 1), complex)
        Ar[:-1, :-1] = anchor["Ar"]
        Ar[:-1, -1] = top_right
        Ar[-1, :-1] = bottom_left
        Ar[-1, -1] = bottom
        br = np.vstack([anchor["br"], new_vector.conj() @ anchor["B"]])
        AV = np.column_stack([anchor["AV"], aq])
    anchor["Ar"] = Ar
    anchor["br"] = br
    anchor["AV"] = AV


def _anchor_residual(anchor):
    if anchor["Ar"].shape[0] == 0:
        residual = anchor["B"].copy()
    else:
        coeff = _solve_reduced(anchor["Ar"], anchor["br"])
        residual = anchor["B"] - anchor["AV"] @ coeff
    relative = np.linalg.norm(residual, axis=0) / anchor["denominator"]
    return residual, relative


def build_residual_basis(
    background,
    geometry_samples,
    state_samples,
    *,
    target_relative_residual=2e-1,
    monitor=None,
):
    """Automatically determine the common Maxwell rank from physical residual.

    Starting from an empty basis, repeatedly find the globally worst
    ``(geometry, thermal-state, port)`` anchor and append an independent lift of
    its true Maxwell residual.  Enrichment stops only when every anchor reaches
    ``target_relative_residual`` or when the physical residuals contain no new
    numerically independent direction.  No Maxwell solution labels and no fixed
    rank target are used.
    """
    target = float(target_relative_residual)
    if not 0 < target < 1:
        raise ValueError("target_relative_residual must lie in (0, 1)")
    pairs = list(zip(geometry_samples, state_samples))
    if not pairs:
        raise ValueError("basis samples cannot be empty")

    anchors = []
    for si, (geometry, state) in enumerate(pairs):
        if monitor is not None:
            monitor.checkpoint()
        context = background.geometry_context(geometry, assemble_thermal=False)
        A = background.em_operator(context, state)
        B = background.rhs_matrix(context)
        anchors.append(
            {
                "A": A,
                "B": B,
                "diag": safe_diag(A),
                "denominator": np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny),
                "AV": np.empty((A.shape[0], 0), complex),
                "Ar": np.empty((0, 0), complex),
                "br": np.empty((0, B.shape[1]), complex),
            }
        )
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(phase="maxwell_basis", training_points=si + 1, basis_rank=0)
        print(
            f"准备 Maxwell anchor……{100 * (si + 1) / len(pairs):5.1f}%  ({si + 1}/{len(pairs)})",
            flush=True,
        )

    V = np.empty((background.n_edges, 0), complex)
    enrichment_steps = 0
    stop_reason = "target_reached"

    while True:
        if monitor is not None:
            monitor.checkpoint()

        candidates = []
        for ai, anchor in enumerate(anchors):
            residual, relative = _anchor_residual(anchor)
            for port, value in enumerate(relative):
                candidates.append((float(value), ai, port, residual[:, port]))
        candidates.sort(key=lambda item: item[0], reverse=True)
        worst = candidates[0][0]
        if worst <= target:
            stop_reason = "target_reached"
            break
        if V.shape[1] >= background.n_edges:
            stop_reason = "ambient_space_exhausted"
            break

        old_basis = V
        chosen = None
        for relative, ai, port, residual in candidates:
            anchor = anchors[ai]
            for lift in (residual / anchor["diag"], residual):
                trial, added = orthonormal_append(V, lift)
                if added:
                    V = trial
                    chosen = (relative, ai, port)
                    break
            if chosen is not None:
                break
        if chosen is None:
            stop_reason = "no_independent_residual_direction"
            break

        enrichment_steps += 1
        new_vector = V[:, -1]
        for anchor in anchors:
            _expand_anchor(anchor, old_basis, new_vector)

        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="maxwell_basis",
                    training_points=len(anchors),
                    basis_rank=V.shape[1],
                    basis_residual=float(worst),
                )
        if V.shape[1] == 1 or V.shape[1] % 4 == 0:
            rel, ai, port = chosen
            print(
                f"构建统一 Maxwell 空间……rank={V.shape[1]}  "
                f"worst residual={worst:.3e}  enriched=anchor[{ai}]/port[{port}] ({rel:.3e})",
                flush=True,
            )

    final_worst = 0.0
    for anchor in anchors:
        _, relative = _anchor_residual(anchor)
        final_worst = max(final_worst, float(np.max(relative)))
    converged = final_worst <= target
    print(
        f"Maxwell 公共空间完成：自动 rank={V.shape[1]}，"
        f"maximum anchor residual={final_worst:.3e}，target={target:.3e}，"
        f"stop={stop_reason}",
        flush=True,
    )
    return V, BasisReport(
        V.shape[1], final_worst, target, len(pairs), enrichment_steps, converged, stop_reason
    )


__all__ = ["BasisReport", "build_residual_basis", "reduced_solution", "safe_diag"]
