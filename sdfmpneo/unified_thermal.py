"""Geometry-aware deterministic thermal ROM basis construction.

Production thermal reduction follows the frozen theory:

    Phi(g) = [Phi_bg, T_tx(g) Psi_tx, T_rx(g) Psi_rx, ...]

The canonical mode count and ordering are fixed. Geometry changes transport the
local blocks deterministically; the neural network never predicts thermal modes
or thermal operators. Reduced M/K are always projections of the true thermal
operators for the queried geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from .unified_geometry import UnifiedUWPTGeometry


def _weighted_append(phi, vector, weights):
    q = np.asarray(vector, float).reshape(-1).copy()
    reference = float(np.sqrt(max(np.dot(q, weights * q), 0.0)))
    if not np.isfinite(reference) or reference <= np.finfo(float).tiny:
        return phi, False
    if phi.size:
        for _ in range(2):
            q -= phi @ (phi.T @ (weights * q))
    norm = float(np.sqrt(max(np.dot(q, weights * q), 0.0)))
    # Linear independence must be scale invariant.  Using max(reference,1)
    # imposed an absolute 1e-11 field-amplitude floor and could reject valid
    # thermal directions solely because the physical temperature response was
    # numerically small.
    if norm <= 1e-11 * reference:
        return phi, False
    q /= norm
    return (q[:, None] if phi.size == 0 else np.column_stack([phi, q])), True


def _port_current_vectors(n_ports):
    """Deterministic current vectors spanning the Hermitian current quadratic space."""
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
    """Use the production-certified shared multi-port Maxwell solver.

    Thermal anchor generation previously bypassed the installed global solver and
    performed a fresh fill-heavy sparse LU for every geometry.  Calling the
    tensor-truth solve surface keeps exactly the same A/B physics while reusing
    the compatible-gradient/transverse Krylov path and its 1e-9 true-residual
    certificate.
    """
    from . import unified_tensor_surrogate as _tensor_truth

    X, residual = _tensor_truth._solve_port_fields(background, context)
    X = np.asarray(X, complex)
    tolerance = float(
        dict(getattr(background, "background_config", {}) or {})
        .get("linear_solver", {})
        .get("relative_residual_tolerance", 1e-9)
    )
    if np.any(~np.isfinite(X)) or not np.isfinite(residual):
        raise FloatingPointError("Maxwell truth solve produced non-finite fields")
    if float(residual) > tolerance:
        raise RuntimeError(
            "thermal anchor Maxwell solve failed residual certificate: "
            f"{float(residual):.3e} > {tolerance:.3e}"
        )
    return X


def _volume_heat(background, context, field):
    sigma, _, _, _, _, _ = background.cell_properties(context, None, em=True)
    edge_energy = np.abs(np.asarray(field, complex).reshape(-1)) ** 2
    cell_edge_energy = np.asarray(background.edge_cell_hodge.T @ edge_energy).reshape(-1)
    return np.asarray((0.5 * np.asarray(sigma, float) * cell_edge_energy).real, float)


def _resolvent_shifts(time_scales):
    scales = np.asarray(list(time_scales), float).reshape(-1)
    if scales.size == 0 or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("thermal time_scales must be finite and positive")
    shifts = [0.0]
    shifts.extend(float(1.0 / t) for t in scales)
    return np.asarray(sorted(set(shifts)), float)


def _trajectory_times(time_scales, requested=None):
    if requested is None:
        base = np.asarray(list(time_scales), float).reshape(-1)
        values = np.r_[base, 10.0 * np.max(base)]
    else:
        values = np.asarray(list(requested), float).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("thermal trajectory times must be finite and positive")
    return np.asarray(sorted(set(float(v) for v in values)), float)


def _solve(A, b):
    try:
        return np.asarray(spla.spsolve(A.tocsc(), b), float).reshape(-1)
    except RuntimeError:
        return np.asarray(spla.lsmr(A, b, atol=1e-12, btol=1e-12)[0], float).reshape(-1)


def _solve_block(A, B):
    """Factor one thermal matrix once and solve all source directions."""
    rhs = np.asarray(B, float)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != A.shape[0]:
        raise ValueError("thermal block RHS has incompatible shape")
    try:
        lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A")
        out = np.asarray(lu.solve(rhs), float)
    except RuntimeError:
        out = np.column_stack([_solve(A, rhs[:, j]) for j in range(rhs.shape[1])])
    if out.shape != rhs.shape or np.any(~np.isfinite(out)):
        raise FloatingPointError("thermal block solve produced invalid values")
    return out


def _geometry_anchors(
    background,
    geometry,
    shifts,
    geometry_index,
    *,
    volume=True,
    wire=True,
    uniform_initial=True,
    wire_port=None,
):
    context = background.geometry_context(geometry, assemble_thermal=False)
    M, K = background.thermal_operator_full(context.fractions)
    rhs_items = []
    if volume:
        X = _maxwell_port_fields(background, context)
        for j, current in enumerate(_port_current_vectors(X.shape[1])):
            q = _volume_heat(background, context, X @ current)
            if np.linalg.norm(q) > np.finfo(float).tiny:
                rhs_items.append((f"volume[{j}]", "volume", q))
    if wire:
        for p, weights in enumerate(context.line_heat_weights):
            if wire_port is not None and p != int(wire_port):
                continue
            q = np.asarray(weights, float).reshape(-1)
            if np.linalg.norm(q) > np.finfo(float).tiny:
                rhs_items.append((f"wire[{p}]", f"wire[{p}]", q))
    if uniform_initial:
        rhs_items.append(
            ("initial[uniform]", "initial", np.asarray(M @ np.ones(background.n_cells)).reshape(-1))
        )
    if not rhs_items:
        return []

    B = np.column_stack([np.asarray(item[2], float).reshape(-1) for item in rhs_items])
    anchors = []
    for shift in shifts:
        A = (K + float(shift) * M).tocsr()
        U = _solve_block(A, B)
        for j, (label, kind, b) in enumerate(rhs_items):
            b = np.asarray(b, float).reshape(-1)
            u = np.asarray(U[:, j], float).reshape(-1)
            # A u = b, so u^T A u = u^T b without another sparse matvec.
            denom2 = float(np.real(np.dot(u, b)))
            if not np.isfinite(denom2) or denom2 <= np.finfo(float).tiny:
                continue
            anchors.append(
                {
                    "A": A,
                    "b": b,
                    "u": u,
                    "denom2": denom2,
                    "label": f"geometry[{geometry_index}]/{label}/s={shift:.6g}",
                    "case_label": str(label),
                    "source_kind": str(kind),
                    "shift": float(shift),
                    "geometry_index": int(geometry_index),
                    "rhs_norm": float(np.linalg.norm(b)),
                }
            )
    return anchors


def _self_volume_local_window(background, geometry, port):
    """Smooth pose-following window for the genuinely local part of self-volume heat.

    Pure-port Maxwell Joule heating contains both a near-coil hotspot and a broad
    environmental tail.  Only the near field is rigidly transportable with the
    coil.  The complementary source remains in the fixed background block.
    """
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    p = int(port)
    if p < 0 or p >= g.n_ports:
        raise ValueError("thermal self-volume port index is out of range")
    coil = g.coils[p]
    package = g.packages[p]

    minimum_step = float(
        min(
            np.min(np.asarray(background.dx, float)),
            np.min(np.asarray(background.dy, float)),
            np.min(np.asarray(background.dz, float)),
        )
    )
    if not np.isfinite(minimum_step) or minimum_step <= 0.0:
        raise ValueError("thermal background has invalid cell spacing")

    spacing = max(minimum_step, min(float(coil.pitch), float(coil.outer_half_size)) / 4.0)
    centerline = np.asarray(coil.centerline(spacing), float)
    coil_local = coil.pose.inverse(centerline)
    conductor_extent = np.array(
        [
            0.5 * float(coil.conductor_width),
            0.5 * float(coil.conductor_width),
            0.5 * float(coil.conductor_thickness),
        ],
        float,
    )
    coil_extent = np.max(np.abs(coil_local), axis=0) + conductor_extent

    signs = np.asarray(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        float,
    )
    package_corners = package.pose.apply(signs * np.asarray(package.half_extent, float))
    package_local = coil.pose.inverse(package_corners)
    package_extent = np.max(np.abs(package_local), axis=0)

    # Keep at least ~1.5 fine cells of guaranteed local support and another
    # ~2 cells of cosine taper.  The exact complement is retained globally.
    core = np.maximum(coil_extent, package_extent) + 1.5 * minimum_step
    outer = core + 2.0 * minimum_step
    local = np.abs(coil.pose.inverse(background.cell_centers))

    axis_windows = []
    for axis in range(3):
        d = local[:, axis]
        t = np.clip(
            (d - core[axis]) / max(float(outer[axis] - core[axis]), np.finfo(float).tiny),
            0.0,
            1.0,
        )
        taper = 0.5 * (1.0 + np.cos(np.pi * t))
        taper[d <= core[axis]] = 1.0
        taper[d >= outer[axis]] = 0.0
        axis_windows.append(taper)
    window = np.prod(np.column_stack(axis_windows), axis=1)
    window = np.clip(np.asarray(window, float), 0.0, 1.0)
    if window.shape != (background.n_cells,) or np.any(~np.isfinite(window)):
        raise FloatingPointError("thermal self-volume local window is invalid")
    if float(np.max(window)) <= 0.0:
        raise RuntimeError("thermal self-volume local window lost all support")
    return window


def _pure_volume_port_index(anchor, n_ports):
    if anchor.get("source_kind") != "volume":
        return None
    label = str(anchor.get("case_label", ""))
    if not (label.startswith("volume[") and label.endswith("]")):
        return None
    try:
        index = int(label[7:-1])
    except ValueError:
        return None
    return index if 0 <= index < int(n_ports) else None


def _partitioned_anchor(template, b, u, *, component, port):
    b = np.asarray(b, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    if np.linalg.norm(b) <= np.finfo(float).tiny:
        return None
    denom2 = float(np.real(np.dot(u, b)))
    if not np.isfinite(denom2) or denom2 <= np.finfo(float).tiny:
        return None
    row = dict(template)
    row.update(
        b=b,
        u=u,
        denom2=denom2,
        label=f"{template['label']}::{component}",
        case_label=f"volume-{component}[{int(port)}]",
        source_kind=f"volume-{component}",
        rhs_norm=float(np.linalg.norm(b)),
        port_index=int(port),
    )
    return row


def _partition_self_volume_anchors(background, geometry, anchors):
    """Split pure-port volume anchors into transportable near and fixed far parts.

    For every pure-port source b, b_local + b_far == b exactly.  Thermal
    linearity gives the same exact decomposition for the resolvent solution.
    Cross-port volume response and uniform initial response remain global.
    Wire anchors are intentionally handled by the existing local-block path.
    """
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    windows = tuple(
        _self_volume_local_window(background, g, p)
        for p in range(g.n_ports)
    )
    background_anchors = []
    local_anchors = [[] for _ in range(g.n_ports)]
    grouped = {}

    for anchor in anchors:
        port = _pure_volume_port_index(anchor, g.n_ports)
        if port is not None:
            key = (float(anchor["shift"]), id(anchor["A"]))
            grouped.setdefault(key, []).append((port, anchor))
            continue
        if anchor.get("source_kind") in {"volume", "initial"}:
            background_anchors.append(anchor)

    for group in grouped.values():
        A = group[0][1]["A"]
        B_local = np.column_stack(
            [windows[port] * np.asarray(anchor["b"], float) for port, anchor in group]
        )
        U_local = _solve_block(A, B_local)
        for column, (port, anchor) in enumerate(group):
            b_local = np.asarray(B_local[:, column], float)
            u_local = np.asarray(U_local[:, column], float)
            local_row = _partitioned_anchor(
                anchor,
                b_local,
                u_local,
                component="local",
                port=port,
            )
            if local_row is not None:
                local_anchors[port].append(local_row)

            b_far = np.asarray(anchor["b"], float) - b_local
            u_far = np.asarray(anchor["u"], float) - u_local
            far_row = _partitioned_anchor(
                anchor,
                b_far,
                u_far,
                component="far",
                port=port,
            )
            if far_row is not None:
                background_anchors.append(far_row)

    return background_anchors, tuple(tuple(rows) for rows in local_anchors)


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
    return float(np.sqrt(numerator2 / anchor["denom2"])), error


def _worst_anchor(anchors, phi):
    if not anchors:
        return 0.0, None, None
    worst = (-1.0, None, None)
    for anchor in anchors:
        relative, error = _anchor_error(anchor, phi)
        if relative > worst[0]:
            worst = (relative, anchor, error)
    return worst


def _group_anchors(anchors):
    """Group anchors that share one full operator so reduced RHS solves can be batched."""
    groups = {}
    for anchor in anchors:
        key = (int(anchor["geometry_index"]), float(anchor["shift"]), id(anchor["A"]))
        groups.setdefault(key, []).append(anchor)
    return tuple(groups.values())


def _anchor_relative_errors_grouped(groups, phi):
    """Evaluate all RHS for each shared resolvent with one reduced solve."""
    rows = []
    for group in groups:
        if not group:
            continue
        if phi.shape[1] == 0:
            rows.extend((anchor, 1.0) for anchor in group)
            continue
        A = group[0]["A"]
        APhi = A @ phi
        Ar = phi.T @ APhi
        B = np.column_stack([anchor["b"] for anchor in group])
        Br = phi.T @ B
        try:
            coeff = np.linalg.solve(Ar, Br)
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(Ar, Br, rcond=None)[0]
        U = np.column_stack([anchor["u"] for anchor in group])
        E = U - phi @ coeff
        AE = A @ E
        numerators = np.maximum(np.real(np.sum(E * AE, axis=0)), 0.0)
        denominators = np.asarray([anchor["denom2"] for anchor in group], float)
        relatives = np.sqrt(numerators / denominators)
        rows.extend((anchor, float(relative)) for anchor, relative in zip(group, relatives))
    return rows


def _worst_anchor_grouped(groups, phi):
    """Return the worst anchor while sharing one reduced solve per operator group."""
    if not groups:
        return 0.0, None, None
    worst = (-1.0, None, None)
    for group in groups:
        if not group:
            continue
        if phi.shape[1] == 0:
            anchor = group[0]
            if 1.0 > worst[0]:
                worst = (1.0, anchor, np.asarray(anchor["u"], float).copy())
            continue
        A = group[0]["A"]
        APhi = A @ phi
        Ar = phi.T @ APhi
        B = np.column_stack([anchor["b"] for anchor in group])
        Br = phi.T @ B
        try:
            coeff = np.linalg.solve(Ar, Br)
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(Ar, Br, rcond=None)[0]
        U = np.column_stack([anchor["u"] for anchor in group])
        E = U - phi @ coeff
        AE = A @ E
        numerators = np.maximum(np.real(np.sum(E * AE, axis=0)), 0.0)
        denominators = np.asarray([anchor["denom2"] for anchor in group], float)
        relatives = np.sqrt(numerators / denominators)
        index = int(np.argmax(relatives))
        relative = float(relatives[index])
        if relative > worst[0]:
            worst = (relative, group[index], E[:, index].copy())
    return worst


def _joint_geometry_worst(background, reference, bg_modes, local_modes, geometries, anchor_sets, time_scales, conditioning_limit):
    library = GeometryAwareThermalLibrary(
        reference,
        np.asarray(bg_modes, float),
        tuple(np.asarray(m, float) for m in local_modes),
        tuple(float(v) for v in time_scales),
        float(conditioning_limit),
    )
    worst = (-1.0, None, None, None)
    for geometry, anchors in zip(geometries, anchor_sets):
        phi = library.basis_for_geometry(background, geometry)
        relative, anchor, error = _worst_anchor_grouped(_group_anchors(anchors), phi)
        if anchor is not None and relative > worst[0]:
            worst = (float(relative), anchor, error, geometry)
    return worst


def _transported_local_columns(background, reference, local_modes, geometry):
    """Transport and normalize the local block once for one training geometry."""
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    columns = []
    for p, modes in enumerate(local_modes):
        if modes.shape[1] == 0:
            continue
        source_pose = reference.coils[p].pose
        target_pose = g.coils[p].pose
        for j in range(modes.shape[1]):
            q = _transport_field(background, modes[:, j], source_pose, target_pose)
            norm = float(np.sqrt(max(np.dot(q, background.cell_volumes * q), 0.0)))
            if not np.isfinite(norm) or norm <= 1e-14:
                raise RuntimeError("transported thermal mode lost support inside the physical domain")
            columns.append(q / norm)
    if not columns:
        return np.empty((background.n_cells, 0), float)
    return np.column_stack(columns)


def _check_cached_basis_conditioning(
    background,
    background_modes,
    transported_local,
    conditioning_limit,
):
    """Apply the same raw-basis conditioning gate as basis_for_geometry."""
    blocks = []
    for block in (background_modes, transported_local):
        block = np.asarray(block, float)
        if block.size:
            blocks.append(block)
    if not blocks:
        raise RuntimeError("geometry-aware thermal library is empty")
    raw = np.column_stack(blocks)
    norms = np.sqrt(
        np.maximum(
            np.sum(background.cell_volumes[:, None] * raw * raw, axis=0),
            0.0,
        )
    )
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-14):
        raise RuntimeError("transported thermal mode lost support inside the physical domain")
    normalized = raw / norms
    gram = normalized.T @ (background.cell_volumes[:, None] * normalized)
    gram = 0.5 * (gram + gram.T)
    eig = np.linalg.eigvalsh(gram)
    if eig[0] <= 0.0:
        raise RuntimeError("geometry-aware thermal basis became rank deficient")
    condition = float(eig[-1] / eig[0])
    if not np.isfinite(condition) or condition > float(conditioning_limit):
        raise RuntimeError(
            f"geometry-aware thermal basis conditioning failed: cond={condition:.3e}"
        )
    return normalized


def _combined_cached_basis(
    background,
    background_modes,
    transported_local,
    conditioning_limit,
):
    """Build the exact production span once, then update it incrementally."""
    normalized = _check_cached_basis_conditioning(
        background,
        background_modes,
        transported_local,
        conditioning_limit,
    )
    phi = np.empty((background.n_cells, 0), float)
    for j in range(normalized.shape[1]):
        phi2, added = _weighted_append(phi, normalized[:, j], background.cell_volumes)
        if not added:
            raise RuntimeError(f"geometry-aware thermal mode {j} became linearly dependent")
        phi = phi2
    return phi


def _enrich_background_against_full_library(
    background,
    reference,
    bg_modes,
    local_modes,
    geometries,
    anchor_sets,
    target,
    maximum_rank,
    time_scales,
    conditioning_limit,
    monitor,
):
    """Add global residual modes without rebuilding transported local spans each step."""
    del time_scales  # The cached span already contains the complete production block.
    geometries = list(geometries)
    phi_bg = np.asarray(bg_modes, float)

    local_columns = [
        _transported_local_columns(background, reference, local_modes, geometry)
        for geometry in geometries
    ]
    anchors_by_geometry = [[] for _ in geometries]
    for anchors in anchor_sets:
        for anchor in anchors:
            gi = int(anchor["geometry_index"])
            if 0 <= gi < len(anchors_by_geometry):
                anchors_by_geometry[gi].append(anchor)
    groups_by_geometry = [_group_anchors(items) for items in anchors_by_geometry]

    spans = [
        _combined_cached_basis(
            background,
            phi_bg,
            local_columns[gi],
            conditioning_limit,
        )
        for gi in range(len(geometries))
    ]

    steps = 0
    stop = "target_reached"

    def worst_state():
        worst = (-1.0, None, None)
        for gi, groups in enumerate(groups_by_geometry):
            relative, anchor, error = _worst_anchor_grouped(groups, spans[gi])
            if relative > worst[0]:
                worst = (relative, anchor, error)
        return worst

    while True:
        if monitor is not None:
            monitor.checkpoint()
        worst, anchor, error = worst_state()
        if worst <= target:
            break
        total_rank = int(phi_bg.shape[1] + sum(m.shape[1] for m in local_modes))
        if maximum_rank is not None and total_rank >= int(maximum_rank):
            stop = "maximum_rank_reached"
            break

        phi2, added = _weighted_append(phi_bg, error, background.cell_volumes)
        if not added:
            phi2, added = _weighted_append(phi_bg, anchor["u"], background.cell_volumes)
        if not added:
            stop = "no_independent_background_residual"
            break

        new_direction = phi2[:, -1]
        for gi, local in enumerate(local_columns):
            _check_cached_basis_conditioning(
                background,
                phi2,
                local,
                conditioning_limit,
            )
            span2, span_added = _weighted_append(
                spans[gi],
                new_direction,
                background.cell_volumes,
            )
            if not span_added:
                raise RuntimeError(
                    "geometry-aware thermal background residual became linearly dependent"
                )
            spans[gi] = span2

        phi_bg = phi2
        steps += 1
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="geometry_aware_thermal_basis",
                    thermal_basis_stage="background-residual",
                    thermal_basis_rank=phi_bg.shape[1],
                    thermal_basis_energy_error=worst,
                )
        if steps == 1 or steps % 4 == 0:
            print(
                "构建 geometry-aware thermal background residual……"
                f"rank={phi_bg.shape[1]}  worst full-library energy error={worst:.3e}",
                flush=True,
            )

    return phi_bg, steps, stop, float(worst_state()[0])


def _greedy_basis(background, anchors, target, maximum_rank, monitor, label):
    phi = np.empty((background.n_cells, 0), float)
    rank_limit = background.n_cells if maximum_rank is None else min(int(maximum_rank), background.n_cells)
    steps = 0
    stop = "target_reached"
    groups = _group_anchors(anchors)
    while True:
        if monitor is not None:
            monitor.checkpoint()
        worst, anchor, error = _worst_anchor_grouped(groups, phi)
        if worst <= target:
            break
        if phi.shape[1] >= rank_limit:
            stop = "maximum_rank_reached"
            break
        phi2, added = _weighted_append(phi, error, background.cell_volumes)
        if not added:
            phi2, added = _weighted_append(phi, anchor["u"], background.cell_volumes)
        if not added:
            stop = "no_independent_thermal_direction"
            break
        phi = phi2
        steps += 1
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="geometry_aware_thermal_basis",
                    thermal_basis_stage=str(label),
                    thermal_basis_rank=phi.shape[1],
                    thermal_basis_energy_error=worst,
                )
        if phi.shape[1] == 1 or phi.shape[1] % 4 == 0:
            after = _worst_anchor_grouped(groups, phi)[0]
            print(
                f"构建 geometry-aware thermal canonical block[{label}]……"
                f"rank={phi.shape[1]}  worst energy error={after:.3e}",
                flush=True,
            )
    return phi, steps, stop, float(_worst_anchor_grouped(groups, phi)[0])


def _canonicalize_poses(geometry, reference):
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    ref = reference if isinstance(reference, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(reference)
    mapping = g.to_mapping()
    for i in range(g.n_ports):
        mapping["coils"][i]["translation"] = ref.coils[i].pose.translation.tolist()
        mapping["coils"][i]["angles"] = ref.coils[i].pose.angles.tolist()
        mapping["packages"][i]["translation"] = ref.packages[i].pose.translation.tolist()
        mapping["packages"][i]["angles"] = ref.packages[i].pose.angles.tolist()
    return UnifiedUWPTGeometry.from_mapping(mapping)


def _transport_field(background, field, source_pose, target_pose):
    """Rigidly pull a scalar cell field from source pose to target pose."""
    values = np.asarray(field, float).reshape(background.nx, background.ny, background.nz)
    interp = RegularGridInterpolator(
        background.cell_axes,
        values,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    local = target_pose.inverse(background.cell_centers)
    sample_points = source_pose.apply(local)
    out = np.asarray(interp(sample_points), float).reshape(-1)
    if np.any(~np.isfinite(out)):
        raise FloatingPointError("thermal rigid transport produced non-finite values")
    return out


@dataclass(frozen=True)
class GeometryAwareThermalLibrary:
    reference_geometry: UnifiedUWPTGeometry
    background_modes: np.ndarray
    local_modes: tuple
    time_scales: tuple
    conditioning_limit: float = 1e10

    @property
    def rank(self):
        return int(self.background_modes.shape[1] + sum(m.shape[1] for m in self.local_modes))

    @property
    def block_ranks(self):
        return (int(self.background_modes.shape[1]),) + tuple(int(m.shape[1]) for m in self.local_modes)

    def basis_for_geometry(self, background, geometry):
        g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
        if g.n_ports != len(self.local_modes):
            raise ValueError("thermal library port count differs from geometry")
        columns = []
        if self.background_modes.size:
            columns.extend(self.background_modes[:, j] for j in range(self.background_modes.shape[1]))
        for p, modes in enumerate(self.local_modes):
            source_pose = self.reference_geometry.coils[p].pose
            target_pose = g.coils[p].pose
            for j in range(modes.shape[1]):
                columns.append(_transport_field(background, modes[:, j], source_pose, target_pose))
        if not columns:
            raise RuntimeError("geometry-aware thermal library is empty")

        raw = np.column_stack(columns)
        norms = np.sqrt(np.maximum(np.sum(background.cell_volumes[:, None] * raw * raw, axis=0), 0.0))
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-14):
            raise RuntimeError("transported thermal mode lost support inside the physical domain")
        normalized = raw / norms
        gram = normalized.T @ (background.cell_volumes[:, None] * normalized)
        gram = 0.5 * (gram + gram.T)
        eig = np.linalg.eigvalsh(gram)
        if eig[0] <= 0.0:
            raise RuntimeError("geometry-aware thermal basis became rank deficient")
        condition = float(eig[-1] / eig[0])
        if not np.isfinite(condition) or condition > float(self.conditioning_limit):
            raise RuntimeError(f"geometry-aware thermal basis conditioning failed: cond={condition:.3e}")

        phi = np.empty((background.n_cells, 0), float)
        for j in range(normalized.shape[1]):
            phi2, added = _weighted_append(phi, normalized[:, j], background.cell_volumes)
            if not added:
                raise RuntimeError(f"geometry-aware thermal mode {j} became linearly dependent")
            phi = phi2
        if phi.shape[1] != self.rank:
            raise RuntimeError("geometry-aware thermal rank changed with geometry")
        return phi

    def mode_bounds(self, background, geometry):
        phi = self.basis_for_geometry(background, geometry)
        return np.min(phi, axis=0), np.max(phi, axis=0)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": 1,
            "reference_geometry": self.reference_geometry.to_mapping(),
            "time_scales": list(self.time_scales),
            "conditioning_limit": float(self.conditioning_limit),
            "local_count": len(self.local_modes),
        }
        arrays = {
            "metadata_json": np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)),
            "background_modes": np.asarray(self.background_modes, float),
        }
        for p, modes in enumerate(self.local_modes):
            arrays[f"local_modes_{p}"] = np.asarray(modes, float)
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            if int(meta.get("schema_version", -1)) != 1:
                raise ValueError("unsupported geometry-aware thermal library version")
            local = tuple(
                np.asarray(data[f"local_modes_{p}"], float)
                for p in range(int(meta["local_count"]))
            )
            return cls(
                UnifiedUWPTGeometry.from_mapping(meta["reference_geometry"]),
                np.asarray(data["background_modes"], float),
                local,
                tuple(float(v) for v in meta["time_scales"]),
                float(meta.get("conditioning_limit", 1e10)),
            )


@dataclass(frozen=True)
class ThermalBasisReport:
    basis_dimension: int
    background_rank: int
    local_ranks: tuple
    maximum_anchor_relative_energy_error: float
    maximum_validation_relative_energy_error: float
    maximum_validation_trajectory_relative_error: float
    target_relative_error: float
    geometry_sample_count: int
    validation_geometry_count: int
    source_direction_count: int
    equation_anchor_count: int
    enrichment_steps: int
    shifts: tuple
    trajectory_times: tuple
    converged: bool
    stop_reason: str
    worst_training_anchor: dict
    worst_validation_anchor: dict
    worst_validation_trajectory: dict
    training_diagnostics: dict
    validation_diagnostics: dict
    trajectory_diagnostics: dict

    @property
    def maximum_anchor_relative_residual(self):
        return self.maximum_anchor_relative_energy_error

    @property
    def target_relative_residual(self):
        return self.target_relative_error


def _anchor_summary(anchor, error):
    if anchor is None:
        return {}
    shift = float(anchor["shift"])
    return {
        "label": anchor["label"],
        "geometry_index": int(anchor["geometry_index"]),
        "source_kind": anchor["source_kind"],
        "shift": shift,
        "time_scale": None if shift == 0.0 else float(1.0 / shift),
        "relative_energy_error": float(error),
        "rhs_norm": float(anchor["rhs_norm"]),
        "solution_energy_norm": float(np.sqrt(anchor["denom2"])),
    }


def _audit_geometries(
    background,
    library,
    geometries,
    shifts,
    monitor,
    role,
    *,
    anchor_sets=None,
):
    worst = (-1.0, None)
    diagnostics = {}
    count = 0
    geometries = list(geometries)
    if anchor_sets is not None and len(anchor_sets) != len(geometries):
        raise ValueError("precomputed thermal anchor set count does not match geometries")
    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        phi = library.basis_for_geometry(background, geometry)
        anchors = (
            list(anchor_sets[gi])
            if anchor_sets is not None
            else _geometry_anchors(
                background,
                geometry,
                shifts,
                gi,
                volume=True,
                wire=True,
                uniform_initial=True,
            )
        )
        count += len(anchors)
        local_worst = 0.0
        for anchor, error in _anchor_relative_errors_grouped(_group_anchors(anchors), phi):
            shift = float(anchor["shift"])
            tau = "steady" if shift == 0.0 else f"{1.0 / shift:g}s"
            key = f"{anchor['source_kind']}@{tau}"
            diagnostics[key] = max(float(diagnostics.get(key, 0.0)), float(error))
            if error > worst[0]:
                worst = (float(error), anchor)
            local_worst = max(local_worst, float(error))
        print(
            f"{role} geometry-aware thermal audit……{gi + 1}/{len(geometries)}  "
            f"worst={local_worst:.3e}",
            flush=True,
        )
    if worst[0] < 0.0:
        return 0.0, {}, diagnostics, count
    return worst[0], _anchor_summary(worst[1], worst[0]), diagnostics, count


def _thermal_trajectory_cases(background, geometry, prepared_anchors=None):
    context = background.geometry_context(geometry, assemble_thermal=False)
    M, K = background.thermal_operator_full(context.fractions)
    labels = []
    rhs = []
    steady = None
    if prepared_anchors is not None:
        rows = [
            anchor
            for anchor in prepared_anchors
            if abs(float(anchor["shift"])) <= 1e-15
            and anchor["source_kind"] != "initial"
        ]
        labels = [str(anchor.get("case_label", anchor["source_kind"])) for anchor in rows]
        rhs = [np.asarray(anchor["b"], float).reshape(-1) for anchor in rows]
        steady = (
            np.column_stack([np.asarray(anchor["u"], float).reshape(-1) for anchor in rows])
            if rows
            else np.empty((background.n_cells, 0), float)
        )
    else:
        X = _maxwell_port_fields(background, context)
        for j, current in enumerate(_port_current_vectors(X.shape[1])):
            q = _volume_heat(background, context, X @ current)
            if np.linalg.norm(q) > np.finfo(float).tiny:
                labels.append(f"volume[{j}]")
                rhs.append(q)
        for p, weights in enumerate(context.line_heat_weights):
            q = np.asarray(weights, float).reshape(-1)
            if np.linalg.norm(q) > np.finfo(float).tiny:
                labels.append(f"wire[{p}]")
                rhs.append(q)
    return context, M.tocsr(), K.tocsr(), labels, rhs, steady


def _trajectory_metric(background, context, mass_diag, truth, approx):
    truth = np.asarray(truth, float).reshape(-1)
    approx = np.asarray(approx, float).reshape(-1)
    error = approx - truth
    denom2 = float(np.dot(mass_diag * truth, truth))
    field = float(np.sqrt(max(float(np.dot(mass_diag * error, error)), 0.0) / max(denom2, np.finfo(float).tiny)))
    scale = max(float(np.max(np.abs(truth))), np.finfo(float).tiny)
    minimum = float(abs(np.min(approx) - np.min(truth)) / scale)
    maximum = float(abs(np.max(approx) - np.max(truth)) / scale)
    wire_truth = []
    wire_approx = []
    for weights in context.line_heat_weights:
        weights = np.asarray(weights, float).reshape(-1)
        wire_truth.append(float(np.dot(weights, truth)))
        wire_approx.append(float(np.dot(weights, approx)))
    if wire_truth:
        wire_truth = np.asarray(wire_truth, float)
        wire_approx = np.asarray(wire_approx, float)
        # Treat wire temperatures as one observable vector. Component-wise
        # relative error is ill-posed when an unexcited remote wire is
        # physically near zero and previously produced 1e9-scale false errors.
        denominator = max(
            float(np.max(np.abs(wire_truth))),
            scale * 1e-12,
            np.finfo(float).tiny,
        )
        wire = float(np.max(np.abs(wire_approx - wire_truth)) / denominator)
    else:
        wire = 0.0
    return {
        "field_mass_relative_error": field,
        "minimum_temperature_relative_error": minimum,
        "maximum_temperature_relative_error": maximum,
        "maximum_wire_average_relative_error": float(wire),
        "composite_relative_error": float(max(field, minimum, maximum, wire)),
    }


def audit_geometry_aware_thermal_trajectories(
    background,
    library,
    geometries,
    *,
    times=None,
    monitor=None,
    anchor_sets=None,
):
    """Compare full and geometry-aware ROM thermal trajectories on held-out geometries.

    The audit is independent of the resolvent construction metric. Constant physical
    volume/wire source directions are propagated from zero initial condition, and the
    declared uniform initial-condition family is propagated homogeneously. Full and
    reduced linear systems use matrix exponential actions, so the reported discrepancy
    is ROM error rather than a time-stepping tolerance artifact.
    """
    geometries = list(geometries)
    audit_times = _trajectory_times(library.time_scales, times)
    if not geometries:
        return 0.0, {}, {}, tuple(float(v) for v in audit_times)
    if anchor_sets is not None and len(anchor_sets) != len(geometries):
        raise ValueError("trajectory anchor set count does not match geometries")

    worst = (-1.0, {})
    diagnostics = {}
    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        g = background.validate_geometry(geometry)
        prepared = None if anchor_sets is None else anchor_sets[gi]
        context, M, K, labels, rhs, prepared_steady = _thermal_trajectory_cases(
            background, g, prepared
        )
        phi = library.basis_for_geometry(background, g)
        mass_diag = np.asarray(M.diagonal(), float)
        if np.any(~np.isfinite(mass_diag)) or np.any(mass_diag <= 0.0):
            raise RuntimeError("full thermal mass must be positive for trajectory audit")
        Mr = phi.T @ (M @ phi)
        Kr = phi.T @ (K @ phi)
        if np.min(np.linalg.eigvalsh(0.5 * (Mr + Mr.T))) <= 0.0:
            raise RuntimeError("reduced thermal mass is not positive definite")
        if np.min(np.linalg.eigvalsh(0.5 * (Kr + Kr.T))) <= 0.0:
            raise RuntimeError("reduced thermal stiffness is not positive definite")

        A_full = (-sp.diags(1.0 / mass_diag) @ K).tocsr()
        A_red = -np.linalg.solve(Mr, Kr)

        if rhs:
            B = np.column_stack(rhs)
            steady_full = (
                np.asarray(prepared_steady, float)
                if prepared_steady is not None
                else _solve_block(K, B)
            )
            reduced_rhs = phi.T @ B
            steady_reduced = np.linalg.solve(Kr, reduced_rhs)
        else:
            steady_full = np.empty((background.n_cells, 0), float)
            steady_reduced = np.empty((phi.shape[1], 0), float)

        initial_full = np.ones(background.n_cells, float)
        initial_reduced = np.linalg.solve(Mr, phi.T @ (M @ initial_full))
        full_block = np.column_stack([steady_full, initial_full])
        reduced_block = np.column_stack([steady_reduced, initial_reduced])
        initial_index = full_block.shape[1] - 1

        local_worst = 0.0

        # Steady forced solutions, including the a_* coordinate comparison requested
        # by the frozen theory.
        for j, label in enumerate(labels):
            truth = steady_full[:, j]
            approx = phi @ steady_reduced[:, j]
            metrics = _trajectory_metric(background, context, mass_diag, truth, approx)
            projected = np.linalg.solve(Mr, phi.T @ (M @ truth))
            coordinate_error = float(
                np.linalg.norm(steady_reduced[:, j] - projected)
                / max(np.linalg.norm(projected), np.finfo(float).tiny)
            )
            metrics["steady_coordinate_relative_error"] = coordinate_error
            metrics["composite_relative_error"] = max(metrics["composite_relative_error"], coordinate_error)
            key = f"{label}@steady"
            diagnostics[key] = max(float(diagnostics.get(key, 0.0)), metrics["composite_relative_error"])
            row = {"geometry_index": gi, "case": label, "time": "steady", **metrics}
            if metrics["composite_relative_error"] > worst[0]:
                worst = (metrics["composite_relative_error"], row)
            local_worst = max(local_worst, metrics["composite_relative_error"])

        # Initial projection is itself part of the declared initial-condition audit.
        initial_metrics = _trajectory_metric(
            background, context, mass_diag, initial_full, phi @ initial_reduced
        )
        diagnostics["initial[uniform]@0s"] = max(
            float(diagnostics.get("initial[uniform]@0s", 0.0)),
            initial_metrics["composite_relative_error"],
        )
        row = {"geometry_index": gi, "case": "initial[uniform]", "time": 0.0, **initial_metrics}
        if initial_metrics["composite_relative_error"] > worst[0]:
            worst = (initial_metrics["composite_relative_error"], row)
        local_worst = max(local_worst, initial_metrics["composite_relative_error"])

        for time in audit_times:
            full_decay = np.asarray(spla.expm_multiply(A_full * float(time), full_block), float)
            reduced_decay = np.asarray(spla.expm_multiply(A_red * float(time), reduced_block), float)
            for j, label in enumerate(labels):
                truth = steady_full[:, j] - full_decay[:, j]
                approx = phi @ (steady_reduced[:, j] - reduced_decay[:, j])
                metrics = _trajectory_metric(background, context, mass_diag, truth, approx)
                key = f"{label}@{float(time):g}s"
                diagnostics[key] = max(float(diagnostics.get(key, 0.0)), metrics["composite_relative_error"])
                row = {"geometry_index": gi, "case": label, "time": float(time), **metrics}
                if metrics["composite_relative_error"] > worst[0]:
                    worst = (metrics["composite_relative_error"], row)
                local_worst = max(local_worst, metrics["composite_relative_error"])

            truth = full_decay[:, initial_index]
            approx = phi @ reduced_decay[:, initial_index]
            metrics = _trajectory_metric(background, context, mass_diag, truth, approx)
            key = f"initial[uniform]@{float(time):g}s"
            diagnostics[key] = max(float(diagnostics.get(key, 0.0)), metrics["composite_relative_error"])
            row = {"geometry_index": gi, "case": "initial[uniform]", "time": float(time), **metrics}
            if metrics["composite_relative_error"] > worst[0]:
                worst = (metrics["composite_relative_error"], row)
            local_worst = max(local_worst, metrics["composite_relative_error"])

        print(
            f"held-out full-vs-ROM thermal trajectory audit……{gi + 1}/{len(geometries)}  "
            f"worst={local_worst:.3e}",
            flush=True,
        )

    if worst[0] < 0.0:
        return 0.0, {}, diagnostics, tuple(float(v) for v in audit_times)
    return float(worst[0]), dict(worst[1]), diagnostics, tuple(float(v) for v in audit_times)


def build_geometry_aware_thermal_library(
    background,
    reference_geometry,
    geometry_samples,
    *,
    validation_geometries=None,
    target_relative_error=5e-2,
    time_scales=(0.1, 1.0, 10.0),
    trajectory_times=None,
    maximum_rank=None,
    conditioning_limit=1e10,
    monitor=None,
):
    """Build canonical BG/local blocks and run independent held-out ROM audits."""
    target = float(target_relative_error)
    if not 0.0 < target < 1.0:
        raise ValueError("thermal target_relative_error must lie in (0, 1)")
    reference = background.validate_geometry(reference_geometry)
    training = [background.validate_geometry(g) for g in list(geometry_samples)]
    if not training:
        training = [reference]
    else:
        training = [reference] + training
    validation = [] if validation_geometries is None else [background.validate_geometry(g) for g in validation_geometries]
    shifts = _resolvent_shifts(time_scales)

    # Preserve the theoretical block split:
    #   Phi(g) = [Phi_bg, T_tx Psi_tx, T_rx Psi_rx].
    # Phi_bg is trained on global volume/initial response only; wire-local
    # response belongs to the transported canonical local blocks.  The v35
    # experiment that also inserted wire anchors into Phi_bg duplicated the
    # local blocks and made the assembled basis rank deficient.
    training_anchor_sets = []
    bg_anchors = []
    for gi, geometry in enumerate(training):
        anchors = _geometry_anchors(
            background,
            geometry,
            shifts,
            gi,
            volume=True,
            wire=True,
            uniform_initial=True,
        )
        training_anchor_sets.append(anchors)
        partitioned_background, _local_unused = _partition_self_volume_anchors(
            background,
            geometry,
            anchors,
        )
        bg_anchors.extend(partitioned_background)
        print(
            f"准备 background/local thermal anchors……{100.0 * (gi + 1) / len(training):5.1f}%",
            flush=True,
        )
    bg_modes, bg_steps, bg_stop, bg_error = _greedy_basis(
        background, bg_anchors, target, maximum_rank, monitor, "background"
    )

    canonical = []
    for geometry in training:
        try:
            candidate = _canonicalize_poses(geometry, reference)
            canonical.append(background.validate_geometry(candidate))
        except ValueError:
            continue
    if not canonical:
        canonical = [reference]

    # Canonical local anchors are prepared once for both ports.  Each local
    # block receives its own self-volume hotspot plus wire source; cross-port
    # volume directions remain global/background.
    canonical_anchor_sets = []
    canonical_local_volume = [[] for _ in range(reference.n_ports)]
    canonical_wire = [[] for _ in range(reference.n_ports)]
    for gi, geometry in enumerate(canonical):
        anchors = _geometry_anchors(
            background,
            geometry,
            shifts,
            gi,
            volume=True,
            wire=True,
            uniform_initial=False,
            wire_port=None,
        )
        canonical_anchor_sets.append(anchors)
        _background_unused, local_volume = _partition_self_volume_anchors(
            background,
            geometry,
            anchors,
        )
        for p in range(reference.n_ports):
            canonical_local_volume[p].extend(local_volume[p])
            wire_kind = f"wire[{p}]"
            canonical_wire[p].extend(
                anchor for anchor in anchors
                if anchor.get("source_kind") == wire_kind
            )

    local_modes = []
    local_steps = 0
    local_stop = "target_reached"
    local_errors = []
    for p in range(reference.n_ports):
        anchors = list(canonical_wire[p]) + list(canonical_local_volume[p])
        modes, steps, stop, error = _greedy_basis(
            background, anchors, target, maximum_rank, monitor, f"local-port-{p}"
        )
        local_modes.append(modes)
        local_steps += steps
        local_errors.append(error)
        if stop != "target_reached":
            local_stop = stop

    # The local blocks carry the moving self-hotspots.  Enrich the fixed
    # background only with whatever global residual remains when the complete
    # transported library is used on the training geometries.
    bg_modes, residual_steps, residual_stop, residual_error = (
        _enrich_background_against_full_library(
            background,
            reference,
            bg_modes,
            tuple(local_modes),
            training,
            training_anchor_sets,
            target,
            maximum_rank,
            time_scales,
            conditioning_limit,
            monitor,
        )
    )
    bg_steps += residual_steps
    bg_error = max(float(bg_error), float(residual_error))
    if residual_stop != "target_reached":
        bg_stop = residual_stop

    library = GeometryAwareThermalLibrary(
        reference,
        bg_modes,
        tuple(local_modes),
        tuple(float(v) for v in time_scales),
        float(conditioning_limit),
    )
    if maximum_rank is not None and library.rank > int(maximum_rank):
        stop_reason = "maximum_rank_reached"
    elif bg_stop != "target_reached":
        stop_reason = bg_stop
    elif local_stop != "target_reached":
        stop_reason = local_stop
    else:
        stop_reason = "target_reached"

    training_error, worst_train, train_diag, training_anchor_count = _audit_geometries(
        background,
        library,
        training,
        shifts,
        monitor,
        "train",
        anchor_sets=training_anchor_sets,
    )
    if validation:
        validation_anchor_sets = []
        for gi, geometry in enumerate(validation):
            if monitor is not None:
                monitor.checkpoint()
                with monitor._lock:
                    monitor.data.update(
                        phase="geometry_aware_thermal_basis",
                        thermal_basis_stage="validation-anchor-prep",
                        thermal_basis_rank=library.rank,
                        thermal_basis_energy_error=None,
                    )
            validation_anchor_sets.append(
                _geometry_anchors(
                    background,
                    geometry,
                    shifts,
                    gi,
                    volume=True,
                    wire=True,
                    uniform_initial=True,
                )
            )
            print(
                f"准备 held-out thermal anchors……{100.0 * (gi + 1) / len(validation):5.1f}%",
                flush=True,
            )
        validation_error, worst_val, val_diag, validation_anchor_count = _audit_geometries(
            background,
            library,
            validation,
            shifts,
            monitor,
            "held-out validation",
            anchor_sets=validation_anchor_sets,
        )
        if validation_error > target:
            trajectory_error = 0.0
            worst_trajectory = {}
            trajectory_diag = {
                "audit_skipped_validation_energy_error": float(validation_error),
            }
            audited_times = tuple(float(v) for v in _trajectory_times(time_scales, trajectory_times))
            print(
                "held-out resolvent audit already exceeds target; "
                "skipping expensive full-vs-ROM trajectory audit.",
                flush=True,
            )
        else:
            trajectory_error, worst_trajectory, trajectory_diag, audited_times = (
                audit_geometry_aware_thermal_trajectories(
                    background,
                    library,
                    validation,
                    times=trajectory_times,
                    monitor=monitor,
                    anchor_sets=validation_anchor_sets,
                )
            )
    else:
        validation_error, worst_val, val_diag, validation_anchor_count = 0.0, {}, {}, 0
        trajectory_error, worst_trajectory, trajectory_diag = 0.0, {}, {}
        audited_times = tuple(float(v) for v in _trajectory_times(time_scales, trajectory_times))

    if training_error > target:
        stop_reason = "training_target_not_met"
    elif validation and validation_error > target:
        stop_reason = "validation_target_not_met"
    elif validation and trajectory_error > target:
        stop_reason = "validation_trajectory_target_not_met"

    converged = (
        stop_reason == "target_reached"
        and bg_error <= target
        and max(local_errors or [0.0]) <= target
        and training_error <= target
        and (not validation or validation_error <= target)
        and (not validation or trajectory_error <= target)
    )
    report = ThermalBasisReport(
        basis_dimension=library.rank,
        background_rank=bg_modes.shape[1],
        local_ranks=tuple(m.shape[1] for m in local_modes),
        maximum_anchor_relative_energy_error=float(training_error),
        maximum_validation_relative_energy_error=float(validation_error),
        maximum_validation_trajectory_relative_error=float(trajectory_error),
        target_relative_error=target,
        geometry_sample_count=len(training),
        validation_geometry_count=len(validation),
        source_direction_count=reference.n_ports * reference.n_ports + reference.n_ports,
        equation_anchor_count=len(bg_anchors) + training_anchor_count + validation_anchor_count,
        enrichment_steps=bg_steps + local_steps,
        shifts=tuple(float(v) for v in shifts),
        trajectory_times=tuple(float(v) for v in audited_times),
        converged=bool(converged),
        stop_reason=stop_reason,
        worst_training_anchor=worst_train,
        worst_validation_anchor=worst_val,
        worst_validation_trajectory=worst_trajectory,
        training_diagnostics=train_diag,
        validation_diagnostics=val_diag,
        trajectory_diagnostics=trajectory_diag,
    )
    print(
        f"geometry-aware thermal ROM 完成：rank={library.rank} "
        f"(bg={bg_modes.shape[1]}, local={tuple(m.shape[1] for m in local_modes)})，"
        f"train={training_error:.3e}，validation={validation_error:.3e}，"
        f"trajectory={trajectory_error:.3e}，target={target:.3e}，stop={stop_reason}",
        flush=True,
    )
    if worst_val:
        print("held-out worst anchor: " + json.dumps(worst_val, ensure_ascii=False, sort_keys=True), flush=True)
    if worst_trajectory:
        print(
            "held-out worst trajectory: "
            + json.dumps(worst_trajectory, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    return library, report


__all__ = [
    "GeometryAwareThermalLibrary",
    "ThermalBasisReport",
    "audit_geometry_aware_thermal_trajectories",
    "build_geometry_aware_thermal_library",
    "_port_current_vectors",
    "_volume_heat",
]
