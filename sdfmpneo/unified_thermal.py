"""Transient-aware shared thermal basis for the unified tensor surrogate.

The basis is built from the complete finite-dimensional current-induced heat
span, explicit wire-heat directions, optional initial-condition directions and
multiple resolvent shifts. Rank selection uses the actual Galerkin solution
error in the A_s=K+sM energy norm rather than an unrelated Euclidean image
residual.

Basis construction uses three geometry roles:

* training geometries seed the greedy basis;
* enrichment geometries may add directions when the seeded basis does not
  generalize across the production geometry family;
* validation geometries stay strictly held out and only certify the final basis.

This avoids the previous failure mode where validation could report a very large
error after basis construction had already stopped, while also avoiding the
invalid shortcut of training on the same geometries that are called validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla


def _weighted_append(phi, vector, weights):
    q = np.asarray(vector, float).reshape(-1).copy()
    reference = float(np.sqrt(max(np.dot(q, weights * q), 0.0)))
    if not np.isfinite(reference) or reference <= np.finfo(float).tiny:
        return phi, False
    if phi.size:
        for _ in range(2):
            q -= phi @ (phi.T @ (weights * q))
    norm = float(np.sqrt(max(np.dot(q, weights * q), 0.0)))
    if norm <= 1e-11 * max(reference, 1.0):
        return phi, False
    q /= norm
    return (q[:, None] if phi.size == 0 else np.column_stack([phi, q])), True


def _port_current_vectors(n_ports):
    """A deterministic real basis for the Hermitian port quadratic space."""
    n = int(n_ports)
    eye = np.eye(n, dtype=complex)
    vectors = [eye[:, p] for p in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            vectors.append(eye[:, i] + eye[:, j])
            vectors.append(eye[:, i] + 1j * eye[:, j])
    if len(vectors) != n * n:
        raise AssertionError("Hermitian current basis dimension mismatch")
    return vectors


def _maxwell_port_fields(background, context):
    A = background.em_operator(context, None)
    B = background.rhs_matrix(context)
    try:
        lu = spla.splu(A.tocsc())
        X = np.column_stack([lu.solve(B[:, p]) for p in range(B.shape[1])])
    except RuntimeError:
        X = np.column_stack([spla.spsolve(A, B[:, p]) for p in range(B.shape[1])])
    if np.any(~np.isfinite(X)):
        raise FloatingPointError("Maxwell truth solve produced non-finite fields")
    return X


def _volume_heat(background, context, field):
    sigma, _, _, _, _, _ = background.cell_properties(context, None, em=True)
    edge_energy = np.abs(np.asarray(field, complex).reshape(-1)) ** 2
    # Exact cell partition of the same mass-lumped loss Hodge used by Maxwell:
    # sum(Q) = 0.5 * E^H diag(edge_cell_hodge @ sigma) E.
    cell_edge_energy = np.asarray(background.edge_cell_hodge.T @ edge_energy).reshape(-1)
    q = 0.5 * np.asarray(sigma, float) * cell_edge_energy
    return np.asarray(q.real, float)


def _source_patterns(background, context):
    X = _maxwell_port_fields(background, context)
    patterns = []
    for current in _port_current_vectors(X.shape[1]):
        q = _volume_heat(background, context, X @ current)
        if np.linalg.norm(q) > np.finfo(float).tiny:
            patterns.append(q)
    # Wire heat has fixed spatial directions; temperature/current only changes amplitudes.
    for weights in context.line_heat_weights:
        q = np.asarray(weights, float).reshape(-1)
        if np.linalg.norm(q) > np.finfo(float).tiny:
            patterns.append(q)
    if not patterns:
        raise RuntimeError("geometry produced no nonzero thermal source directions")
    return patterns


def _resolvent_shifts(time_scales):
    scales = np.asarray(list(time_scales), float).reshape(-1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("thermal time_scales must be finite and positive")
    shifts = [0.0]
    shifts.extend(float(1.0 / t) for t in scales)
    return np.asarray(sorted(set(shifts)), float)


def _solve(A, b):
    try:
        return np.asarray(spla.spsolve(A.tocsc(), b), float).reshape(-1)
    except RuntimeError:
        return np.asarray(spla.lsmr(A, b, atol=1e-12, btol=1e-12)[0], float).reshape(-1)


def _make_anchors(
    background,
    geometries,
    shifts,
    include_uniform_initial_condition,
    monitor=None,
    *,
    role="geometry",
):
    anchors = []
    source_count = 0
    geometries = list(geometries)
    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        context = background.geometry_context(geometry, assemble_thermal=False)
        M, K = background.thermal_operator_full(context.fractions)
        sources = _source_patterns(background, context)
        source_count += len(sources)
        rhs_items = [(f"source[{j}]", q) for j, q in enumerate(sources)]
        if include_uniform_initial_condition:
            theta0 = np.ones(background.n_cells, float)
            rhs_items.append(("initial[uniform]", np.asarray(M @ theta0).reshape(-1)))
        for shift in shifts:
            A = (K + float(shift) * M).tocsr()
            for label, b in rhs_items:
                u = _solve(A, b)
                denom2 = float(np.real(u @ (A @ u)))
                if not np.isfinite(denom2) or denom2 <= np.finfo(float).tiny:
                    continue
                anchors.append({
                    "A": A,
                    "b": np.asarray(b, float),
                    "u": u,
                    "denom2": denom2,
                    "label": f"{role}[{gi}]/{label}/s={shift:.6g}",
                })
        if geometries:
            print(
                f"准备 {role} transient thermal anchors……"
                f"{100.0 * (gi + 1) / len(geometries):5.1f}%  ({gi + 1}/{len(geometries)})",
                flush=True,
            )
    if geometries and not anchors:
        raise RuntimeError(f"thermal {role} anchor set is empty")
    return anchors, source_count


def _anchor_error(anchor, phi):
    A = anchor["A"]
    b = anchor["b"]
    u = anchor["u"]
    if phi.shape[1] == 0:
        approx = np.zeros_like(u)
    else:
        Ar = phi.T @ (A @ phi)
        br = phi.T @ b
        try:
            a = np.linalg.solve(Ar, br)
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(Ar, br, rcond=None)[0]
        approx = phi @ a
    error = u - approx
    numerator2 = max(float(np.real(error @ (A @ error))), 0.0)
    relative = float(np.sqrt(numerator2 / anchor["denom2"]))
    return relative, error


def _worst_anchor(anchors, phi):
    if not anchors:
        return 0.0, None, None
    worst = (-1.0, None, None)
    for anchor in anchors:
        relative, error = _anchor_error(anchor, phi)
        if relative > worst[0]:
            worst = (relative, anchor, error)
    return worst


def _greedy_enrich(background, phi, anchors, target, rank_limit, monitor, *, phase):
    """Enrich ``phi`` until every supplied anchor meets the energy-error target."""
    steps = 0
    stop_reason = "target_reached"
    while True:
        if monitor is not None:
            monitor.checkpoint()
        worst, anchor, error = _worst_anchor(anchors, phi)
        if worst <= target:
            stop_reason = "target_reached"
            break
        if phi.shape[1] >= rank_limit:
            stop_reason = "maximum_rank_reached"
            break
        trial, added = _weighted_append(phi, error, background.cell_volumes)
        if not added:
            # The exact solution itself is a robust fallback enrichment direction.
            trial, added = _weighted_append(phi, anchor["u"], background.cell_volumes)
        if not added:
            stop_reason = "no_independent_thermal_direction"
            break
        phi = trial
        steps += 1
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="thermal_basis",
                    thermal_basis_stage=str(phase),
                    thermal_basis_rank=phi.shape[1],
                    thermal_basis_energy_error=worst,
                )
        if phi.shape[1] == 1 or phi.shape[1] % 4 == 0:
            new_worst = _worst_anchor(anchors, phi)[0]
            print(
                f"构建 transient thermal 公共空间[{phase}]……rank={phi.shape[1]}  "
                f"worst energy error={new_worst:.3e}  enriched={anchor['label']}",
                flush=True,
            )
    return phi, steps, stop_reason


@dataclass(frozen=True)
class ThermalBasisReport:
    basis_dimension: int
    maximum_anchor_relative_energy_error: float
    maximum_enrichment_relative_energy_error: float
    maximum_validation_relative_energy_error: float
    target_relative_error: float
    geometry_sample_count: int
    enrichment_geometry_count: int
    validation_geometry_count: int
    source_direction_count: int
    equation_anchor_count: int
    enrichment_steps: int
    shifts: tuple
    converged: bool
    stop_reason: str

    # Backwards-readable aliases; the quantity is now an energy-norm solution error.
    @property
    def maximum_anchor_relative_residual(self):
        return self.maximum_anchor_relative_energy_error

    @property
    def target_relative_residual(self):
        return self.target_relative_error


def build_thermal_basis(
    background,
    geometry_samples,
    *,
    enrichment_geometries=None,
    validation_geometries=None,
    target_relative_error=5e-2,
    target_relative_residual=None,
    time_scales=(1e-3, 1.0, 1000.0),
    include_uniform_initial_condition=True,
    maximum_rank=None,
    monitor=None,
):
    """Build, install and independently validate one shared automatic-rank basis.

    ``geometry_samples`` seed the greedy space. ``enrichment_geometries`` are a
    separate candidate pool that is allowed to add directions if the seeded
    basis does not generalize. ``validation_geometries`` remain strictly held
    out: they never add basis vectors and therefore retain their meaning as an
    independent thermal-ROM audit.
    """
    if target_relative_residual is not None:
        target_relative_error = target_relative_residual
    target = float(target_relative_error)
    if not 0.0 < target < 1.0:
        raise ValueError("thermal target_relative_error must lie in (0, 1)")
    geometries = list(geometry_samples)
    enrichment = [] if enrichment_geometries is None else list(enrichment_geometries)
    validation = [] if validation_geometries is None else list(validation_geometries)
    if not geometries:
        raise ValueError("thermal geometry samples cannot be empty")

    shifts = _resolvent_shifts(time_scales)
    anchors, source_count = _make_anchors(
        background,
        geometries,
        shifts,
        bool(include_uniform_initial_condition),
        monitor,
        role="train",
    )
    enrichment_anchors, enrichment_source_count = _make_anchors(
        background,
        enrichment,
        shifts,
        bool(include_uniform_initial_condition),
        monitor,
        role="enrichment",
    )
    validation_anchors, _ = _make_anchors(
        background,
        validation,
        shifts,
        bool(include_uniform_initial_condition),
        monitor,
        role="validation",
    )

    phi = np.empty((background.n_cells, 0), float)
    rank_limit = (
        background.n_cells
        if maximum_rank is None
        else min(int(maximum_rank), background.n_cells)
    )

    # Stage 1: fit the seed geometry bank.
    phi, seed_steps, seed_stop = _greedy_enrich(
        background,
        phi,
        anchors,
        target,
        rank_limit,
        monitor,
        phase="seed",
    )

    # Stage 2: only the designated enrichment pool may influence the basis.
    # Rechecking the seed anchors together prevents an enrichment direction from
    # trading accuracy away from the original geometry bank.
    combined_anchors = anchors + enrichment_anchors
    if enrichment_anchors and seed_stop == "target_reached":
        phi, extra_steps, enrich_stop = _greedy_enrich(
            background,
            phi,
            combined_anchors,
            target,
            rank_limit,
            monitor,
            phase="generalization",
        )
    else:
        extra_steps = 0
        enrich_stop = seed_stop

    training_worst = _worst_anchor(anchors, phi)[0]
    enrichment_worst = _worst_anchor(enrichment_anchors, phi)[0]
    validation_worst = _worst_anchor(validation_anchors, phi)[0]

    if seed_stop != "target_reached":
        stop_reason = seed_stop
    elif enrich_stop != "target_reached":
        stop_reason = enrich_stop
    elif validation and validation_worst > target:
        stop_reason = "validation_target_not_met"
    else:
        stop_reason = "target_reached"

    background.set_thermal_basis(phi)
    converged = (
        training_worst <= target
        and (not enrichment or enrichment_worst <= target)
        and (not validation or validation_worst <= target)
    )
    print(
        f"thermal 公共空间完成：自动 rank={background.thermal_rank}，"
        f"train={training_worst:.3e}，enrichment={enrichment_worst:.3e}，"
        f"held-out validation={validation_worst:.3e}，target={target:.3e}，"
        f"stop={stop_reason}",
        flush=True,
    )
    return background.thermal_basis.copy(), ThermalBasisReport(
        basis_dimension=background.thermal_rank,
        maximum_anchor_relative_energy_error=float(training_worst),
        maximum_enrichment_relative_energy_error=float(enrichment_worst),
        maximum_validation_relative_energy_error=float(validation_worst),
        target_relative_error=target,
        geometry_sample_count=len(geometries),
        enrichment_geometry_count=len(enrichment),
        validation_geometry_count=len(validation),
        source_direction_count=source_count + enrichment_source_count,
        equation_anchor_count=len(combined_anchors),
        enrichment_steps=seed_steps + extra_steps,
        shifts=tuple(float(v) for v in shifts),
        converged=converged,
        stop_reason=stop_reason,
    )


__all__ = ["ThermalBasisReport", "build_thermal_basis", "_port_current_vectors", "_volume_heat"]
