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
    if norm <= 1e-11 * max(reference, 1.0):
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


def _anchor(A, b, label, *, source_kind, shift, geometry_index, port_index=None):
    b = np.asarray(b, float).reshape(-1)
    u = _solve(A, b)
    denom2 = float(np.real(u @ (A @ u)))
    if not np.isfinite(denom2) or denom2 <= np.finfo(float).tiny:
        return None
    return {
        "A": A,
        "b": b,
        "u": u,
        "denom2": denom2,
        "label": str(label),
        "source_kind": str(source_kind),
        "shift": float(shift),
        "geometry_index": int(geometry_index),
        "port_index": None if port_index is None else int(port_index),
        "rhs_norm": float(np.linalg.norm(b)),
    }


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
    volume_port=None,
):
    context = background.geometry_context(geometry, assemble_thermal=False)
    M, K = background.thermal_operator_full(context.fractions)
    rhs_items = []
    if volume:
        X = _maxwell_port_fields(background, context)
        currents = _port_current_vectors(X.shape[1])
        if volume_port is None:
            volume_items = list(enumerate(currents))
        else:
            p = int(volume_port)
            if p < 0 or p >= X.shape[1]:
                raise ValueError("volume_port is out of range")
            volume_items = [(p, currents[p])]
        for j, current in volume_items:
            q = _volume_heat(background, context, X @ current)
            if np.linalg.norm(q) > np.finfo(float).tiny:
                pure_port = j if j < X.shape[1] else None
                rhs_items.append((f"volume[{j}]", "volume", q, pure_port))
    if wire:
        for p, weights in enumerate(context.line_heat_weights):
            if wire_port is not None and p != int(wire_port):
                continue
            q = np.asarray(weights, float).reshape(-1)
            if np.linalg.norm(q) > np.finfo(float).tiny:
                rhs_items.append((f"wire[{p}]", f"wire[{p}]", q, p))
    if uniform_initial:
        rhs_items.append(
            ("initial[uniform]", "initial", np.asarray(M @ np.ones(background.n_cells)).reshape(-1), None)
        )

    anchors = []
    for shift in shifts:
        A = (K + float(shift) * M).tocsr()
        for label, kind, b, port_index in rhs_items:
            item = _anchor(
                A,
                b,
                f"geometry[{geometry_index}]/{label}/s={shift:.6g}",
                source_kind=kind,
                shift=shift,
                geometry_index=geometry_index,
                port_index=port_index,
            )
            if item is not None:
                anchors.append(item)
    return anchors


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
    """Group resolvent anchors that share one geometry and one operator shift."""
    groups = {}
    for anchor in anchors:
        key = (int(anchor["geometry_index"]), float(anchor["shift"]), id(anchor["A"]))
        groups.setdefault(key, []).append(anchor)
    return tuple(groups.values())


def _anchor_relative_errors_grouped(groups, phi):
    """Return (anchor, relative_error) while sharing reduced solves per A/shift."""
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
    """Evaluate many RHS against one reduced resolvent factorization per group."""
    if not groups:
        return 0.0, None, None
    worst = (-1.0, None, None)
    for group in groups:
        if not group:
            continue
        if phi.shape[1] == 0:
            for anchor in group:
                relative = 1.0
                if relative > worst[0]:
                    worst = (relative, anchor, np.asarray(anchor["u"], float).copy())
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


def _snapshot_relative_error(vector, phi, weights):
    vector = np.asarray(vector, float).reshape(-1)
    denom2 = float(np.dot(vector, weights * vector))
    if not np.isfinite(denom2) or denom2 <= np.finfo(float).tiny:
        return 0.0, np.zeros_like(vector)
    if phi.size:
        approximation = phi @ (phi.T @ (weights * vector))
    else:
        approximation = np.zeros_like(vector)
    error = vector - approximation
    num2 = max(float(np.dot(error, weights * error)), 0.0)
    return float(np.sqrt(num2 / denom2)), error


def _augment_snapshot_basis(background, phi, vectors, target, maximum_rank, monitor, label):
    """Greedily compress transported physical snapshots in the volume metric."""
    phi = np.asarray(phi, float)
    vectors = [np.asarray(v, float).reshape(-1) for v in vectors]
    vectors = [v for v in vectors if np.linalg.norm(v) > np.finfo(float).tiny]
    if not vectors:
        return phi, 0, "target_reached", 0.0
    rank_limit = background.n_cells if maximum_rank is None else min(int(maximum_rank), background.n_cells)
    steps = 0
    stop = "target_reached"
    while True:
        if monitor is not None:
            monitor.checkpoint()
        worst = (-1.0, None)
        for vector in vectors:
            relative, error = _snapshot_relative_error(vector, phi, background.cell_volumes)
            if relative > worst[0]:
                worst = (relative, error)
        if worst[0] <= target:
            break
        if phi.shape[1] >= rank_limit:
            stop = "maximum_rank_reached"
            break
        phi2, added = _weighted_append(phi, worst[1], background.cell_volumes)
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
                    thermal_basis_energy_error=worst[0],
                )
    final = max((_snapshot_relative_error(v, phi, background.cell_volumes)[0] for v in vectors), default=0.0)
    return phi, steps, stop, float(final)


def _transported_local_columns(background, reference, local_modes, geometry):
    """Return normalized transported local columns without a background block."""
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    columns = []
    for p, modes in enumerate(local_modes):
        source_pose = reference.coils[p].pose
        target_pose = g.coils[p].pose
        for j in range(modes.shape[1]):
            q = _transport_field(background, modes[:, j], source_pose, target_pose)
            norm = float(np.sqrt(max(np.dot(q, background.cell_volumes * q), 0.0)))
            if not np.isfinite(norm) or norm <= 1e-14:
                raise RuntimeError("transported thermal mode lost support inside the physical domain")
            columns.append(q / norm)
    return np.empty((background.n_cells, 0), float) if not columns else np.column_stack(columns)


def _combined_basis(background, background_modes, transported_local):
    """Volume-orthonormalize the exact span used by the production library."""
    phi = np.empty((background.n_cells, 0), float)
    for block in (background_modes, transported_local):
        block = np.asarray(block, float)
        for j in range(block.shape[1]):
            phi2, added = _weighted_append(phi, block[:, j], background.cell_volumes)
            if added:
                phi = phi2
    return phi


def _greedy_background_residual(
    background,
    anchors,
    geometries,
    reference,
    local_modes,
    target,
    maximum_total_rank,
    monitor,
):
    """Build only the non-transportable residual after local physical modes.

    The expensive part is repeated reduced resolvent evaluation at increasing
    rank.  Keep each geometry's transported local span resident and batch every
    RHS that shares the same thermal operator/shift, so each greedy step needs
    one reduced factorization per geometry/shift rather than one per source.
    """
    geometries = list(geometries)
    local_columns = [
        _transported_local_columns(background, reference, local_modes, geometry)
        for geometry in geometries
    ]
    anchors_by_geometry = [[] for _ in geometries]
    for anchor in anchors:
        gi = int(anchor["geometry_index"])
        if 0 <= gi < len(anchors_by_geometry):
            anchors_by_geometry[gi].append(anchor)
    groups_by_geometry = [_group_anchors(items) for items in anchors_by_geometry]

    # The span is what matters for Galerkin error.  Start from the local span,
    # then append every accepted fixed-background direction incrementally instead
    # of rebuilding/orthogonalizing the whole [BG, local] matrix on every trial.
    spans = [
        _combined_basis(background, np.empty((background.n_cells, 0), float), local)
        for local in local_columns
    ]
    phi_bg = np.empty((background.n_cells, 0), float)
    local_rank = int(sum(m.shape[1] for m in local_modes))
    rank_limit = background.n_cells
    if maximum_total_rank is not None:
        rank_limit = max(0, min(background.n_cells, int(maximum_total_rank) - local_rank))
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
        if phi_bg.shape[1] >= rank_limit:
            stop = "maximum_rank_reached"
            break
        phi2, added = _weighted_append(phi_bg, error, background.cell_volumes)
        if not added:
            phi2, added = _weighted_append(phi_bg, anchor["u"], background.cell_volumes)
        if not added:
            stop = "no_independent_thermal_direction"
            break
        new_direction = phi2[:, -1]
        phi_bg = phi2
        for gi, span in enumerate(spans):
            span2, span_added = _weighted_append(span, new_direction, background.cell_volumes)
            if span_added:
                spans[gi] = span2
        steps += 1
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="geometry_aware_thermal_basis",
                    thermal_basis_stage="background-residual",
                    thermal_basis_rank=phi_bg.shape[1],
                    thermal_basis_energy_error=worst,
                )
        if phi_bg.shape[1] == 1 or phi_bg.shape[1] % 4 == 0:
            after = worst_state()[0]
            print(
                "构建 geometry-aware thermal canonical block[background-residual]……"
                f"rank={phi_bg.shape[1]}  worst energy error={after:.3e}",
                flush=True,
            )
    return phi_bg, steps, stop, float(worst_state()[0])


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
        "port_index": anchor.get("port_index"),
        "shift": shift,
        "time_scale": None if shift == 0.0 else float(1.0 / shift),
        "relative_energy_error": float(error),
        "rhs_norm": float(anchor["rhs_norm"]),
        "solution_energy_norm": float(np.sqrt(anchor["denom2"])),
    }


def _audit_geometries(background, library, geometries, shifts, monitor, role):
    worst = (-1.0, None)
    diagnostics = {}
    count = 0
    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        phi = library.basis_for_geometry(background, geometry)
        anchors = _geometry_anchors(
            background,
            geometry,
            shifts,
            gi,
            volume=True,
            wire=True,
            uniform_initial=True,
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


def _thermal_trajectory_cases(background, geometry):
    context = background.geometry_context(geometry, assemble_thermal=False)
    M, K = background.thermal_operator_full(context.fractions)
    labels = []
    rhs = []
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
    return context, M.tocsr(), K.tocsr(), labels, rhs


def _trajectory_metric(background, context, mass_diag, truth, approx):
    truth = np.asarray(truth, float).reshape(-1)
    approx = np.asarray(approx, float).reshape(-1)
    error = approx - truth
    denom2 = float(np.dot(mass_diag * truth, truth))
    field = float(np.sqrt(max(float(np.dot(mass_diag * error, error)), 0.0) / max(denom2, np.finfo(float).tiny)))
    scale = max(float(np.max(np.abs(truth))), np.finfo(float).tiny)
    minimum = float(abs(np.min(approx) - np.min(truth)) / scale)
    maximum = float(abs(np.max(approx) - np.max(truth)) / scale)
    wire = 0.0
    for weights in context.line_heat_weights:
        weights = np.asarray(weights, float).reshape(-1)
        full_value = float(np.dot(weights, truth))
        rom_value = float(np.dot(weights, approx))
        denominator = max(abs(full_value), scale * 1e-12, np.finfo(float).tiny)
        wire = max(wire, abs(rom_value - full_value) / denominator)
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

    worst = (-1.0, {})
    diagnostics = {}
    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        g = background.validate_geometry(geometry)
        context, M, K, labels, rhs = _thermal_trajectory_cases(background, g)
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
            try:
                lu = spla.splu(K.tocsc())
                steady_full = np.column_stack([lu.solve(B[:, j]) for j in range(B.shape[1])])
            except RuntimeError:
                steady_full = np.column_stack([_solve(K, B[:, j]) for j in range(B.shape[1])])
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

    # Full physical anchors are solved once and reused.  In particular, the
    # pure-port volume responses become geometry-following local snapshots instead
    # of forcing the fixed background block to memorize every coil pose.
    bg_anchors = []
    for gi, geometry in enumerate(training):
        bg_anchors.extend(
            _geometry_anchors(
                background,
                geometry,
                shifts,
                gi,
                volume=True,
                wire=False,
                uniform_initial=True,
            )
        )
        print(
            f"准备 thermal anchors……{100.0 * (gi + 1) / len(training):5.1f}%",
            flush=True,
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

    local_modes = []
    local_steps = 0
    local_stop = "target_reached"
    local_errors = []
    for p in range(reference.n_ports):
        wire_anchors = []
        for gi, geometry in enumerate(canonical):
            wire_anchors.extend(
                _geometry_anchors(
                    background,
                    geometry,
                    shifts,
                    gi,
                    volume=False,
                    wire=True,
                    uniform_initial=False,
                    wire_port=p,
                )
            )
        modes, steps, stop, error = _greedy_basis(
            background, wire_anchors, target, maximum_rank, monitor, f"local-port-{p}"
        )

        # Reuse already-solved pure-port volume thermal responses and pull them
        # back to the reference pose.  This is the critical geometry-generalization
        # step: self-volume Joule heat follows the coil instead of being encoded as
        # dozens of fixed global modes for each sampled translation/rotation.
        self_volume_snapshots = []
        for anchor in bg_anchors:
            if anchor.get("source_kind") != "volume" or anchor.get("port_index") != p:
                continue
            gi = int(anchor["geometry_index"])
            source_pose = training[gi].coils[p].pose
            target_pose = reference.coils[p].pose
            self_volume_snapshots.append(
                _transport_field(background, anchor["u"], source_pose, target_pose)
            )
        modes, extra_steps, snapshot_stop, snapshot_error = _augment_snapshot_basis(
            background,
            modes,
            self_volume_snapshots,
            target,
            maximum_rank,
            monitor,
            f"local-port-{p}-self-volume",
        )
        local_modes.append(modes)
        local_steps += steps + extra_steps
        local_errors.append(max(error, snapshot_error))
        if stop != "target_reached":
            local_stop = stop
        if snapshot_stop != "target_reached":
            local_stop = snapshot_stop

    # Only the part that cannot follow either coil is represented by a fixed
    # background block.  This includes uniform/global diffusion and mutual/nonlocal
    # volume-heating residuals.
    bg_modes, bg_steps, bg_stop, bg_error = _greedy_background_residual(
        background,
        bg_anchors,
        training,
        reference,
        tuple(local_modes),
        target,
        maximum_rank,
        monitor,
    )

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
        background, library, training, shifts, monitor, "train"
    )
    if validation:
        validation_error, worst_val, val_diag, validation_anchor_count = _audit_geometries(
            background, library, validation, shifts, monitor, "held-out validation"
        )
        # A badly missed resolvent space already proves this basis cannot pass.
        # Do not spend another several minutes on expm-based trajectory audits
        # merely to rediscover the same failure; trajectory validation resumes as
        # soon as the cheaper independent energy gate passes.
        if validation_error > target:
            trajectory_error, worst_trajectory = 0.0, {}
            trajectory_diag = {"audit_skipped_validation_energy_error": float(validation_error)}
            audited_times = tuple(float(v) for v in _trajectory_times(time_scales, trajectory_times))
            print(
                "held-out thermal trajectory audit skipped: "
                f"resolvent validation error={validation_error:.3e} > target={target:.3e}",
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