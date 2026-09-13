"""Automatic residual-driven thermal basis for the unified solver.

The thermal rank is not configured. It is the rank reached by a global
residual-greedy algorithm over physically generated Joule-heating anchors. For
each sampled geometry the basis must represent both the steady conduction
problem ``K T = q`` and the initial dynamic response ``M dT/dt = q`` to the
requested relative residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla


def _weighted_append(Phi, vector, weights):
    q = np.asarray(vector, float).reshape(-1).copy()
    if np.any(~np.isfinite(q)):
        return Phi, False
    if Phi.size:
        for _ in range(2):
            q -= Phi @ (Phi.T @ (weights * q))
    norm = float(np.sqrt(max(np.dot(q, weights * q), 0.0)))
    reference = max(float(np.sqrt(max(np.dot(vector, weights * vector), 0.0))), 1.0)
    if norm <= 1e-11 * reference:
        return Phi, False
    q /= norm
    return (q[:, None] if Phi.size == 0 else np.column_stack([Phi, q])), True


def _image_append(W, vector):
    q = np.asarray(vector, float).reshape(-1).copy()
    if W.size:
        for _ in range(2):
            q -= W @ (W.T @ q)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12 * max(float(np.linalg.norm(vector)), 1.0):
        return W
    q /= norm
    return q[:, None] if W.size == 0 else np.column_stack([W, q])


def _port_current_vectors(n_ports):
    vectors = []
    eye = np.eye(n_ports, dtype=complex)
    vectors.extend(eye[:, p] for p in range(n_ports))
    for i in range(n_ports):
        for j in range(i + 1, n_ports):
            for phase in (1.0, -1.0, 1.0j, -1.0j):
                vectors.append(eye[:, i] + phase * eye[:, j])
    return vectors


def _joule_heat_anchors(background, context):
    """Return full-cell physical heat patterns for representative port combinations."""
    A = background.em_operator(context, None)
    B = background.rhs_matrix(context)
    try:
        lu = spla.splu(A.tocsc())
        X = np.column_stack([lu.solve(B[:, p]) for p in range(B.shape[1])])
    except RuntimeError:
        X = np.column_stack([spla.spsolve(A, B[:, p]) for p in range(B.shape[1])])
    ex, ey, ez, weight = background.material_joule_cells(context, None, X)
    resistances = background.wire_resistances(context, None)
    patterns = []
    for current in _port_current_vectors(B.shape[1]):
        Ex = ex @ current
        Ey = ey @ current
        Ez = ez @ current
        q = weight * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        for p, line_weights in enumerate(context.line_heat_weights):
            q = q + 0.5 * resistances[p] * abs(current[p]) ** 2 * line_weights
        q = np.asarray(q.real, float)
        if np.any(~np.isfinite(q)):
            raise FloatingPointError("thermal Joule anchor contains non-finite values")
        if np.linalg.norm(q) > np.finfo(float).tiny:
            patterns.append(q)
    if not patterns:
        raise RuntimeError("physical geometry produced no nonzero thermal heat anchors")
    return np.column_stack(patterns)


@dataclass(frozen=True)
class ThermalBasisReport:
    basis_dimension: int
    maximum_anchor_relative_residual: float
    target_relative_residual: float
    geometry_sample_count: int
    heat_pattern_count: int
    equation_anchor_count: int
    enrichment_steps: int
    converged: bool
    stop_reason: str


def _expand_anchor(anchor, new_vector):
    anchor["W"] = _image_append(anchor["W"], anchor["L"] @ new_vector)


def _anchor_residual(anchor):
    B = anchor["B"]
    W = anchor["W"]
    residual = B.copy() if W.shape[1] == 0 else B - W @ (W.T @ B)
    relative = np.linalg.norm(residual, axis=0) / anchor["denominator"]
    return residual, relative


def _solve_lift(anchor, residual):
    if anchor["kind"] == "mass":
        return residual / anchor["mass_diag"]
    solver = anchor.get("solver")
    if solver is None:
        solver = spla.splu(anchor["L"].tocsc())
        anchor["solver"] = solver
    return solver.solve(residual)


def build_thermal_basis(background, geometry_samples, *, target_relative_residual=5e-2, monitor=None):
    """Build and install one automatic-rank thermal basis.

    Anchors come from full Maxwell Joule heating at ambient temperature. For
    every geometry, both ``K T = q`` and ``M v = q`` are controlled. Since each
    anchor uses orthogonal projection onto ``range(L Phi)``, its best residual
    cannot increase as ``Phi`` grows.
    """
    target = float(target_relative_residual)
    if not 0.0 < target < 1.0:
        raise ValueError("thermal target_relative_residual must lie in (0, 1)")
    geometries = list(geometry_samples)
    if not geometries:
        raise ValueError("thermal geometry samples cannot be empty")

    anchors = []
    heat_pattern_count = 0
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        context = background.geometry_context(geometry, assemble_thermal=False)
        M, K = background.thermal_operator_full(context.fractions)
        Q = _joule_heat_anchors(background, context)
        heat_pattern_count += Q.shape[1]
        denominator = np.maximum(np.linalg.norm(Q, axis=0), np.finfo(float).tiny)
        anchors.append({
            "kind": "stiffness", "L": K, "B": Q, "denominator": denominator,
            "W": np.empty((background.n_cells, 0), float),
            "label": f"geometry[{index}]/steady",
        })
        anchors.append({
            "kind": "mass", "L": M, "B": Q, "denominator": denominator,
            "W": np.empty((background.n_cells, 0), float),
            "mass_diag": np.asarray(M.diagonal(), float),
            "label": f"geometry[{index}]/dynamic",
        })
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(phase="thermal_basis", training_points=index + 1,
                                    thermal_basis_rank=0)
        print(f"准备 thermal anchor……{100 * (index + 1) / len(geometries):5.1f}%  "
              f"({index + 1}/{len(geometries)})", flush=True)

    Phi = np.empty((background.n_cells, 0), float)
    previous_worst = float("inf")
    steps = 0
    stop_reason = "target_reached"
    while True:
        if monitor is not None:
            monitor.checkpoint()
        candidates = []
        for ai, anchor in enumerate(anchors):
            residual, relative = _anchor_residual(anchor)
            for column, value in enumerate(relative):
                candidates.append((float(value), ai, column, residual[:, column]))
        candidates.sort(key=lambda item: item[0], reverse=True)
        worst_before = candidates[0][0]
        if worst_before <= target:
            stop_reason = "target_reached"
            break
        if Phi.shape[1] >= background.n_cells:
            stop_reason = "ambient_space_exhausted"
            break

        chosen = None
        for relative, ai, column, residual in candidates:
            try:
                lift = _solve_lift(anchors[ai], residual)
            except RuntimeError:
                continue
            trial, added = _weighted_append(Phi, lift, background.cell_volumes)
            if added:
                Phi = trial
                chosen = (relative, ai, column)
                break
        if chosen is None:
            stop_reason = "no_independent_thermal_direction"
            break

        new_vector = Phi[:, -1]
        for anchor in anchors:
            _expand_anchor(anchor, new_vector)
        steps += 1

        current_worst = 0.0
        for anchor in anchors:
            _, relative = _anchor_residual(anchor)
            current_worst = max(current_worst, float(np.max(relative)))
        if current_worst > previous_worst * (1.0 + 5e-10) + 5e-12:
            raise FloatingPointError(
                "minimum-residual thermal basis lost monotonicity: "
                f"{previous_worst:.6e} -> {current_worst:.6e}"
            )
        previous_worst = min(previous_worst, current_worst)

        if monitor is not None:
            with monitor._lock:
                monitor.data.update(phase="thermal_basis", training_points=len(geometries),
                                    thermal_basis_rank=Phi.shape[1],
                                    thermal_basis_residual=current_worst)
        if Phi.shape[1] == 1 or Phi.shape[1] % 4 == 0 or current_worst <= target:
            rel, ai, column = chosen
            print(f"构建 thermal 公共空间……rank={Phi.shape[1]}  "
                  f"worst residual={current_worst:.3e}  "
                  f"enriched={anchors[ai]['label']}/heat[{column}] ({rel:.3e})",
                  flush=True)

    final_worst = 0.0
    for anchor in anchors:
        _, relative = _anchor_residual(anchor)
        final_worst = max(final_worst, float(np.max(relative)))
    converged = final_worst <= target
    if Phi.shape[1] == 0:
        raise RuntimeError("thermal residual-greedy produced an empty basis")
    background.set_thermal_basis(Phi)
    print(f"thermal 公共空间完成：自动 rank={background.thermal_rank}，"
          f"maximum anchor residual={final_worst:.3e}，target={target:.3e}，"
          f"stop={stop_reason}", flush=True)
    return background.thermal_basis.copy(), ThermalBasisReport(
        background.thermal_rank, final_worst, target, len(geometries),
        heat_pattern_count, len(anchors), steps, converged, stop_reason,
    )


__all__ = ["ThermalBasisReport", "build_thermal_basis"]
