"""Residual-driven common Maxwell space for the unified background.

The basis rank is not a user-chosen hyperparameter. It is determined by the
requested physical Maxwell residual on all training anchors.
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


def reduced_solution(A, B, V):
    """Minimum-residual solution in ``span(V)``.

    This deliberately does *not* use ``V^H A V c = V^H B``. The Maxwell
    operator here is complex and generally non-Hermitian/indefinite, so a
    Galerkin solve does not guarantee that the Euclidean residual decreases as
    the trial space grows. Solving ``min ||B-A V c||_2`` does.
    """
    if V.shape[1] == 0:
        return np.zeros((A.shape[0], B.shape[1]), complex)
    AV = np.asarray(A @ V, complex)
    coeff = scipy.linalg.lstsq(
        AV, np.asarray(B, complex), cond=1e-12, lapack_driver="gelsy",
        check_finite=False,
    )[0]
    return V @ coeff


@dataclass(frozen=True)
class BasisReport:
    basis_dimension: int
    maximum_anchor_relative_residual: float
    target_relative_residual: float
    sample_count: int
    enrichment_steps: int
    converged: bool
    stop_reason: str


def _append_image_direction(anchor, new_vector):
    """Increment the orthonormal basis of ``range(A V)`` for one anchor."""
    image = np.asarray(anchor["A"] @ new_vector, complex)
    q = image.copy()
    W = anchor["image_basis"]
    if W.size:
        for _ in range(2):
            q -= W @ (W.conj().T @ q)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12 * max(float(np.linalg.norm(image)), 1.0):
        return False
    q /= norm
    anchor["image_basis"] = q[:, None] if W.size == 0 else np.column_stack([W, q])
    row = np.asarray(q.conj() @ anchor["B"], complex).reshape(1, -1)
    projection = anchor["image_rhs"]
    anchor["image_rhs"] = row if projection.size == 0 else np.vstack([projection, row])
    return True


def _anchor_residual(anchor):
    """True minimum residual over the current common trial space.

    If ``W`` is an orthonormal basis for ``range(A V)``, the least-squares
    projection is ``W W^H B`` and the residual is ``B-W W^H B``. Because the
    image space is nested during enrichment, its residual norm is monotone
    non-increasing up to roundoff.
    """
    W = anchor["image_basis"]
    if W.shape[1] == 0:
        residual = anchor["B"].copy()
    else:
        residual = anchor["B"] - W @ anchor["image_rhs"]
    relative = np.linalg.norm(residual, axis=0) / anchor["denominator"]
    return residual, relative


def _worst_candidates(anchors):
    candidates = []
    for ai, anchor in enumerate(anchors):
        residual, relative = _anchor_residual(anchor)
        for port, value in enumerate(relative):
            candidates.append((float(value), ai, port, residual[:, port]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def build_residual_basis(
    background,
    geometry_samples,
    state_samples,
    *,
    target_relative_residual=2e-1,
    monitor=None,
):
    """Automatically determine the common Maxwell rank from physical residual.

    The enrichment is global residual-greedy across every
    ``(geometry, thermal-state, port)`` anchor. For each anchor the quality
    measure is the *minimum* full-background residual over the current common
    trial space. No Maxwell solution labels and no fixed rank target are used.
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
        B = np.asarray(background.rhs_matrix(context), complex)
        anchors.append(
            {
                "A": A,
                "B": B,
                "diag": safe_diag(A),
                "denominator": np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny),
                "image_basis": np.empty((A.shape[0], 0), complex),
                "image_rhs": np.empty((0, B.shape[1]), complex),
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
    previous_worst = float("inf")
    stagnant_steps = 0

    while True:
        if monitor is not None:
            monitor.checkpoint()

        candidates = _worst_candidates(anchors)
        worst_before = candidates[0][0]
        if worst_before <= target:
            stop_reason = "target_reached"
            break
        if V.shape[1] >= background.n_edges:
            stop_reason = "ambient_space_exhausted"
            break

        chosen = None
        old_rank = V.shape[1]
        for relative, ai, port, residual in candidates:
            anchor = anchors[ai]
            for lift in (residual / anchor["diag"], residual):
                trial, added = orthonormal_append(V, lift)
                if not added:
                    continue
                # The new trial direction must also add a new image direction
                # for the anchor we are trying to improve.
                image = np.asarray(anchor["A"] @ trial[:, -1], complex)
                W = anchor["image_basis"]
                orth = image.copy()
                if W.size:
                    for _ in range(2):
                        orth -= W @ (W.conj().T @ orth)
                if np.linalg.norm(orth) <= 1e-12 * max(float(np.linalg.norm(image)), 1.0):
                    continue
                V = trial
                chosen = (relative, ai, port)
                break
            if chosen is not None:
                break
        if chosen is None:
            stop_reason = "no_independent_residual_direction"
            break
        if V.shape[1] != old_rank + 1:
            raise RuntimeError("Maxwell basis enrichment lost nested-space structure")

        enrichment_steps += 1
        new_vector = V[:, -1]
        for anchor in anchors:
            _append_image_direction(anchor, new_vector)

        post_candidates = _worst_candidates(anchors)
        worst_after = post_candidates[0][0]
        # Nested minimum-residual spaces cannot become worse. A larger increase
        # means numerical orthogonality has broken and must not be hidden.
        monotone_tol = 5e-11 * max(1.0, worst_before)
        if worst_after > worst_before + monotone_tol:
            raise FloatingPointError(
                "minimum-residual Maxwell basis lost monotonicity: "
                f"before={worst_before:.6e}, after={worst_after:.6e}"
            )

        improvement = previous_worst - worst_after
        if np.isfinite(previous_worst) and improvement <= 1e-10 * max(1.0, previous_worst):
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        previous_worst = worst_after
        if stagnant_steps >= 16:
            stop_reason = "residual_stagnation"
            break

        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="maxwell_basis",
                    training_points=len(anchors),
                    basis_rank=V.shape[1],
                    basis_residual=float(worst_after),
                )
        if V.shape[1] == 1 or V.shape[1] % 4 == 0 or worst_after <= target:
            rel, ai, port = chosen
            print(
                f"构建统一 Maxwell 空间……rank={V.shape[1]}  "
                f"worst residual={worst_after:.3e}  enriched=anchor[{ai}]/port[{port}] ({rel:.3e})",
                flush=True,
            )

    final_candidates = _worst_candidates(anchors)
    final_worst = final_candidates[0][0]
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
