"""Geometry-local thermal ROM built online from predicted spatial Joule tensors.

This module deliberately has no cross-geometry thermal state basis.  For one
query geometry, the true thermal M/K operators are assembled on the fixed
background and a small block rational Krylov space is generated from the full
Hermitian current-quadratic volume source span, exact wire source shapes and the
uniform initial condition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_geometry import UnifiedUWPTGeometry
from .unified_thermal import (
    _group_anchors,
    _port_current_vectors,
    _resolvent_shifts,
    _solve_block,
    _weighted_append,
    _worst_anchor_grouped,
)


@dataclass(frozen=True)
class OnlineThermalReport:
    rank: int
    source_count: int
    anchor_count: int
    maximum_anchor_relative_energy_error: float
    conditioning: float
    time_scales: tuple
    shifts: tuple
    representation: str = "geometry_local_rational_krylov_v1"


def _basis_condition(phi, weights):
    value = np.asarray(phi, float)
    if value.ndim != 2 or value.shape[1] < 1:
        return float("inf")
    weighted = np.sqrt(np.asarray(weights, float))[:, None] * value
    try:
        singular = np.linalg.svd(weighted, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("inf")
    if singular.size == 0 or singular[-1] <= 0.0:
        return float("inf")
    return float(singular[0] / singular[-1])


def _cell_heat(cell_h, current):
    c = np.asarray(current, complex).reshape(-1)
    tensors = np.asarray(cell_h, complex)
    if tensors.ndim != 3 or tensors.shape[1:] != (c.size, c.size):
        raise ValueError("cell Joule tensor field and current dimensions differ")
    q = 0.5 * np.real(
        np.einsum("p,kpq,q->k", c.conj(), tensors, c, optimize=True)
    )
    scale = max(float(np.max(np.abs(q))), 1.0)
    if float(np.min(q)) < -1e-10 * scale:
        raise ValueError("cell Joule tensor field is not PSD on current span")
    return np.maximum(np.asarray(q, float), 0.0)


def _source_block(background, context, cell_h):
    n_ports = len(context.geometry.coils)
    currents = _port_current_vectors(n_ports)
    rhs = []
    labels = []
    for index, current in enumerate(currents):
        q = _cell_heat(cell_h, current)
        if np.linalg.norm(q) <= np.finfo(float).tiny:
            continue
        rhs.append(q)
        labels.append(f"volume[{index}]")

    for p, weights in enumerate(context.line_heat_weights):
        q = np.asarray(weights, float).reshape(-1)
        if q.shape != (background.n_cells,):
            raise ValueError("wire heat shape has incompatible cell dimension")
        total = float(np.sum(q))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("wire heat shape is empty")
        rhs.append(q / total)
        labels.append(f"wire[{p}]")

    if not rhs:
        raise RuntimeError("online thermal ROM has no physical heat sources")
    return np.column_stack(rhs), tuple(labels)


def build_online_thermal_context(
    background,
    geometry,
    cell_h,
    *,
    time_scales=(0.1, 1.0, 10.0),
    conditioning_limit=1e10,
    target_relative_error=5e-2,
):
    """Assemble one small, certified thermal ROM for a query geometry."""
    g = (
        geometry
        if isinstance(geometry, UnifiedUWPTGeometry)
        else UnifiedUWPTGeometry.from_mapping(geometry)
    )
    context = background.geometry_context(g, assemble_thermal=False)
    M, K = background.thermal_operator_full(context.fractions)
    B_heat, labels = _source_block(background, context, cell_h)
    initial = np.ones(background.n_cells, float)
    b_initial = np.asarray(M @ initial, float).reshape(-1)
    B = np.column_stack((B_heat, b_initial))
    all_labels = labels + ("initial[uniform]",)
    shifts = _resolvent_shifts(time_scales)

    weights = np.asarray(background.cell_volumes, float)
    phi = np.empty((background.n_cells, 0), float)
    # Preserve a spatially uniform supplied initial temperature exactly at t=0.
    # The shifted resolvent snapshots alone approximate its decay but do not
    # generally contain the initial vector itself.
    phi, initial_added = _weighted_append(phi, initial, weights)
    if not initial_added:
        raise RuntimeError("online thermal ROM rejected uniform initial direction")
    anchors = []
    geometry_index = 0

    for shift in shifts:
        A = (
            K
            if abs(float(shift)) <= 1e-15
            else (K + float(shift) * M).tocsr()
        )
        U = _solve_block(A, B)
        for j, label in enumerate(all_labels):
            u = np.asarray(U[:, j], float).reshape(-1)
            b = np.asarray(B[:, j], float).reshape(-1)
            denom2 = float(np.real(np.dot(u, b)))
            if not np.isfinite(denom2) or denom2 <= np.finfo(float).tiny:
                continue
            anchors.append(
                {
                    "A": A,
                    "b": b,
                    "u": u,
                    "denom2": denom2,
                    "label": f"online/{label}/s={float(shift):.6g}",
                    "case_label": label,
                    "source_kind": (
                        "initial" if label == "initial[uniform]"
                        else ("wire" if label.startswith("wire[") else "volume")
                    ),
                    "shift": float(shift),
                    "geometry_index": geometry_index,
                    "rhs_norm": float(np.linalg.norm(b)),
                }
            )
            phi2, added = _weighted_append(phi, u, weights)
            if added:
                phi = phi2

    if phi.shape[1] < 1:
        raise RuntimeError("online thermal ROM produced an empty basis")

    condition = _basis_condition(phi, weights)
    if not np.isfinite(condition) or condition > float(conditioning_limit):
        raise RuntimeError(
            "online thermal basis conditioning failed: "
            f"cond={condition:.3e}, limit={float(conditioning_limit):.3e}"
        )

    groups = _group_anchors(anchors)
    error, worst, _ = _worst_anchor_grouped(groups, phi)
    if error > float(target_relative_error):
        raise RuntimeError(
            "online thermal resolvent certificate failed: "
            f"error={float(error):.3e}, target={float(target_relative_error):.3e}, "
            f"worst={None if worst is None else worst.get('label')}"
        )

    Mr = np.asarray(phi.T @ (M @ phi), float)
    Kr = np.asarray(phi.T @ (K @ phi), float)
    Mr = 0.5 * (Mr + Mr.T)
    Kr = 0.5 * (Kr + Kr.T)
    if np.min(np.linalg.eigvalsh(Mr)) <= 0.0:
        raise RuntimeError("online reduced thermal mass is not SPD")
    if np.min(np.linalg.eigvalsh(Kr)) <= 0.0:
        raise RuntimeError("online reduced thermal stiffness is not SPD")

    # BackgroundContext is intentionally a mutable runtime container.
    context.thermal_basis = np.asarray(phi, float)
    context.thermal_mass_full = M
    context.thermal_stiffness_full = K
    context.thermal_mass_reduced = Mr
    context.thermal_stiffness_reduced = Kr
    context.online_thermal_report = OnlineThermalReport(
        rank=int(phi.shape[1]),
        source_count=int(B_heat.shape[1]),
        anchor_count=int(len(anchors)),
        maximum_anchor_relative_energy_error=float(error),
        conditioning=float(condition),
        time_scales=tuple(float(v) for v in time_scales),
        shifts=tuple(float(v) for v in shifts),
    )
    return context


def audit_online_thermal_trajectories(
    background,
    geometry,
    cell_h,
    *,
    times=(0.1, 1.0, 10.0, 100.0),
    time_scales=(0.1, 1.0, 10.0),
    conditioning_limit=1e10,
    target_relative_error=5e-2,
):
    """Full-cell vs geometry-local ROM transient audit for the physical source span."""
    audit_times = np.asarray(times, float).reshape(-1)
    if (
        audit_times.size == 0
        or np.any(~np.isfinite(audit_times))
        or np.any(audit_times <= 0.0)
    ):
        raise ValueError("online thermal audit times must be positive")
    audit_times = np.unique(np.sort(audit_times))

    context = build_online_thermal_context(
        background,
        geometry,
        cell_h,
        time_scales=time_scales,
        conditioning_limit=conditioning_limit,
        target_relative_error=target_relative_error,
    )
    M = context.thermal_mass_full
    K = context.thermal_stiffness_full
    phi = np.asarray(context.thermal_basis, float)
    Mr = np.asarray(context.thermal_mass_reduced, float)
    Kr = np.asarray(context.thermal_stiffness_reduced, float)
    B_heat, labels = _source_block(background, context, cell_h)

    steady_full = _solve_block(K, B_heat)
    reduced_rhs = phi.T @ B_heat
    try:
        steady_reduced = np.linalg.solve(Kr, reduced_rhs)
    except np.linalg.LinAlgError:
        steady_reduced = np.linalg.lstsq(Kr, reduced_rhs, rcond=None)[0]

    initial_full = np.ones(background.n_cells, float)
    try:
        initial_reduced = np.linalg.solve(
            Mr,
            phi.T @ (M @ initial_full),
        )
    except np.linalg.LinAlgError:
        initial_reduced = np.linalg.lstsq(
            Mr,
            phi.T @ (M @ initial_full),
            rcond=None,
        )[0]

    mass_diag = np.asarray(M.diagonal(), float)
    if np.any(~np.isfinite(mass_diag)) or np.any(mass_diag <= 0.0):
        raise RuntimeError("full thermal mass diagonal is invalid")
    A_full = (-sp.diags(1.0 / mass_diag) @ K).tocsr()
    A_red = -np.linalg.solve(Mr, Kr)

    full_block = np.column_stack((steady_full, initial_full))
    reduced_block = np.column_stack(
        (steady_reduced, initial_reduced)
    )
    initial_index = full_block.shape[1] - 1
    worst = 0.0
    worst_row = {}

    def mass_error(truth, approx):
        diff = np.asarray(truth - approx, float)
        ref = np.asarray(truth, float)
        num = float(np.dot(diff, mass_diag * diff))
        den = float(np.dot(ref, mass_diag * ref))
        return float(
            np.sqrt(
                max(num, 0.0)
                / max(den, np.finfo(float).tiny)
            )
        )

    for time in audit_times:
        full_decay = np.asarray(
            spla.expm_multiply(A_full * float(time), full_block),
            float,
        )
        reduced_decay = np.asarray(
            spla.expm_multiply(A_red * float(time), reduced_block),
            float,
        )
        for j, label in enumerate(labels):
            truth = steady_full[:, j] - full_decay[:, j]
            approx = phi @ (
                steady_reduced[:, j] - reduced_decay[:, j]
            )
            error = mass_error(truth, approx)
            if error > worst:
                worst = error
                worst_row = {
                    "case": str(label),
                    "time": float(time),
                    "mass_relative_error": float(error),
                }

        truth_initial = full_decay[:, initial_index]
        approx_initial = phi @ reduced_decay[:, initial_index]
        error = mass_error(truth_initial, approx_initial)
        if error > worst:
            worst = error
            worst_row = {
                "case": "initial[uniform]",
                "time": float(time),
                "mass_relative_error": float(error),
            }

    return {
        "maximum_mass_relative_error": float(worst),
        "worst": worst_row,
        "rank": int(phi.shape[1]),
        "resolvent_error": float(
            context.online_thermal_report.maximum_anchor_relative_energy_error
        ),
        "conditioning": float(
            context.online_thermal_report.conditioning
        ),
        "source_count": int(
            context.online_thermal_report.source_count
        ),
        "times": audit_times.tolist(),
        "converged": bool(worst <= float(target_relative_error)),
    }


__all__ = [
    "OnlineThermalReport",
    "build_online_thermal_context",
    "audit_online_thermal_trajectories",
]
