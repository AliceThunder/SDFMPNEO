"""Scalable training primitives for funnel-shaped analytic response networks."""
from __future__ import annotations

from dataclasses import dataclass
from threading import local
from types import SimpleNamespace
import numpy as np
import scipy.linalg

from .research_helpers import (
    _combined_metrics,
    _evaluate_semigroup,
    _gn_apply_field_jacobian,
    _hard_weights,
    _physics_vector_field,
    _work,
)
from .research_linearization import _balanced_physics_subset
from .source_prefit_factorization import fit_source_factors


_SOLVE_STATE = local()


@dataclass(frozen=True)
class _PreparedLayerLeastSquares:
    ids: np.ndarray
    rw: np.ndarray
    normalized_jacobian: np.ndarray
    scales: np.ndarray
    gram: np.ndarray
    rhs: np.ndarray
    row_space: bool


def source_prefit(network, field, config, monitor=None):
    """Learn the first-layer source subspace and amplitudes from t=0 physics."""
    samples = np.asarray(config.source_points(), dtype=float)
    n = network.n_modes
    desired = []
    total = len(samples)
    for i, row in enumerate(samples):
        _work(monitor, "source_prefit", i, total)
        a0 = row[:n]
        operating = row[n:]
        physical = _physics_vector_field(field, a0, operating)
        desired.append(
            np.asarray(physical, float) + np.asarray(network.lambdas, float) * a0
        )
    _work(monitor, "source_prefit", total, total)
    fitted, residual = fit_source_factors(network, samples, np.vstack(desired))
    norms = np.linalg.norm(residual, axis=1)
    rms = float(np.sqrt(np.mean(norms * norms))) if len(norms) else 0.0
    maximum = float(np.max(norms, initial=0.0))
    return fitted, SimpleNamespace(
        rms=rms,
        maximum=maximum,
        sample_count=int(len(samples)),
    )


def residual_target_modes(records, width):
    """Pick unique thermal targets explaining the largest modal residual energy."""
    width = int(width)
    if width < 1:
        raise ValueError("response layer width must be positive")
    if not records:
        raise ValueError("target selection requires physics residual records")
    residual = np.vstack([np.asarray(record.residual, float) for record in records])
    energy = np.mean(residual * residual, axis=0)
    width = min(width, residual.shape[1])
    ids = np.argsort(energy)[::-1][:width]
    return np.asarray(ids, dtype=int), energy


def evaluate_layer_physics(
    network,
    field,
    points,
    layer,
    *,
    monitor=None,
    work_label="physics_layer_jacobian",
):
    """Exact residual plus exact current-layer amplitude Jacobian."""
    points = np.asarray(points, dtype=float)
    n = network.n_modes
    records = []
    ids = None
    total = len(points)
    for i, point in enumerate(points):
        _work(monitor, work_label, i, total)
        initial, operating, time = point[:n], point[n:-1], float(point[-1])
        a, da, ja, jda, local_ids = network.evaluate_layer_amplitude_jacobian(
            time, a0=initial, operating=operating, layer=layer
        )
        if ids is None:
            ids = np.asarray(local_ids, dtype=int)
        elif not np.array_equal(ids, local_ids):
            raise RuntimeError("layer amplitude parameter block changed across collocation points")
        F = _physics_vector_field(field, a, operating)
        JFJa = _gn_apply_field_jacobian(field, a, operating, ja)
        records.append(SimpleNamespace(
            residual=np.asarray(da - F, dtype=float),
            parameter_jacobian=np.asarray(jda - JFJa, dtype=float),
            parameter_indices=ids,
        ))
    _work(monitor, work_label, total, total)
    return records, np.asarray(
        network.layer_amplitude_parameter_indices(layer) if ids is None else ids,
        dtype=int,
    )


def layer_linearization(network, field, points, evaluated, layer, config, monitor=None):
    budget = min(int(config.jacobian_point_budget), len(points))
    subset = _balanced_physics_subset(
        points, evaluated.physics_norms, budget, config
    )
    records, ids = evaluate_layer_physics(
        network,
        field,
        subset,
        layer,
        monitor=monitor,
        work_label=f"physics_layer_{layer + 1}_jacobian",
    )
    result = _combined_metrics(records, [])
    result.physics_subset = subset
    result.parameter_indices = ids
    return result


def evaluate_layer_semigroup(
    network,
    rows,
    layer,
    *,
    monitor=None,
    work_label="restart_layer_jacobian",
):
    """Inexact but bounded layer-amplitude semigroup Jacobian.

    The direct and restarted explicit parameter dependences are exact. The
    indirect derivative through the first segment's restart state is omitted and
    handled by physical trust-region acceptance on the full semigroup defect.
    """
    rows = np.asarray(rows, dtype=float)
    n = network.n_modes
    horizon = float(network.max_response_time)
    records = []
    ids = None
    total_rows = len(rows)
    for i, row in enumerate(rows):
        _work(monitor, work_label, i, total_rows)
        initial = row[:n]
        operating = row[n:-2]
        t1, t2 = float(row[-2]), float(row[-1])
        total = min(t1 + t2, horizon)
        direct, _, Jdirect, _, local_ids = network.evaluate_layer_amplitude_jacobian(
            total, a0=initial, operating=operating, layer=layer
        )
        first, _ = network.evaluate(t1, a0=initial, operating=operating)
        restarted, _, Jrestart, _, restart_ids = network.evaluate_layer_amplitude_jacobian(
            t2, a0=first, operating=operating, layer=layer
        )
        if not np.array_equal(local_ids, restart_ids):
            raise RuntimeError("restart layer parameter block changed")
        if ids is None:
            ids = np.asarray(local_ids, dtype=int)
        defect = np.asarray(direct - restarted, dtype=float)
        records.append(SimpleNamespace(
            residual=defect / horizon,
            raw_defect=defect,
            parameter_jacobian=np.asarray(Jdirect - Jrestart, dtype=float) / horizon,
            parameter_indices=ids,
        ))
    _work(monitor, work_label, total_rows, total_rows)
    return records, np.asarray(
        network.layer_amplitude_parameter_indices(layer) if ids is None else ids,
        dtype=int,
    )


def _prepare_layer_least_squares(records, weights, parameter_indices):
    ids = np.asarray(parameter_indices, dtype=int).reshape(-1)
    residual = np.vstack([np.asarray(record.residual, float) for record in records])
    jacobian = np.stack([
        np.asarray(record.parameter_jacobian, float) for record in records
    ])
    if jacobian.shape[2] != ids.size:
        raise ValueError("local layer Jacobian does not match amplitude block")
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(records),):
        raise ValueError("least-squares weights do not match residual records")
    root = np.sqrt(w)[:, None]
    rw = np.ascontiguousarray((root * residual).reshape(-1))
    Jw = np.ascontiguousarray(
        (root[:, :, None] * jacobian).reshape(-1, ids.size)
    )
    scales = np.linalg.norm(Jw, axis=0)
    scales[scales < 1e-14] = 1.0
    Jn = np.ascontiguousarray(Jw / scales)
    m, p = Jn.shape
    row_space = m <= p
    if row_space:
        gram = np.ascontiguousarray(Jn @ Jn.T)
        rhs = -rw
    else:
        gram = np.ascontiguousarray(Jn.T @ Jn)
        rhs = np.ascontiguousarray(-(Jn.T @ rw))
    return _PreparedLayerLeastSquares(
        ids=ids.copy(),
        rw=rw,
        normalized_jacobian=Jn,
        scales=np.ascontiguousarray(scales),
        gram=gram,
        rhs=np.ascontiguousarray(rhs),
        row_space=bool(row_space),
    )


def _prepared_layer_least_squares(records, weights, parameter_indices):
    """Reuse one normalized normal system across all damping retries."""
    ids = np.asarray(parameter_indices, dtype=int).reshape(-1)
    cached = getattr(_SOLVE_STATE, "last", None)
    if (
        cached is not None
        and cached.records is records
        and cached.weights is weights
        and np.array_equal(cached.prepared.ids, ids)
    ):
        return cached.prepared
    prepared = _prepare_layer_least_squares(records, weights, ids)
    _SOLVE_STATE.last = SimpleNamespace(
        records=records,
        weights=weights,
        prepared=prepared,
    )
    return prepared


def _solve_prepared_layer_direction(prepared, damping, network):
    ids = prepared.ids
    if ids.size == 0:
        return np.zeros(network.parameter_count)
    mu = max(float(damping), 1e-12)
    system = prepared.gram.copy()
    size = system.shape[0]
    system.flat[::size + 1] += mu
    solved = scipy.linalg.solve(
        system,
        prepared.rhs,
        assume_a="pos",
        check_finite=False,
    )
    if prepared.row_space:
        q = prepared.normalized_jacobian.T @ solved
    else:
        q = solved
    local = q / prepared.scales
    trust = 2.0 * max(1.0, float(np.linalg.norm(network.parameters[ids])))
    norm = float(np.linalg.norm(local))
    if norm > trust:
        local *= trust / norm
    delta = np.zeros(network.parameter_count)
    delta[ids] = local
    return delta


def solve_layer_direction(records, weights, damping, network, parameter_indices):
    """LM direction with one normal-system build per linearization.

    Damping retries change only ``mu I``.  The weighted residual, column scaling,
    normalized Jacobian, Gram matrix and right-hand side are invariant throughout
    that retry loop and are therefore prepared once and reused.
    """
    ids = np.asarray(parameter_indices, dtype=int).reshape(-1)
    if ids.size == 0 or not records:
        return np.zeros(network.parameter_count)
    prepared = _prepared_layer_least_squares(records, weights, ids)
    return _solve_prepared_layer_direction(prepared, damping, network)


def predicted_layer_metrics(records, weights, delta, parameter_indices, factor):
    ids = np.asarray(parameter_indices, dtype=int)
    local = np.asarray(delta, dtype=float)[ids]
    residual = np.stack([np.asarray(record.residual, float) for record in records])
    jacobian = np.stack([
        np.asarray(record.parameter_jacobian, float) for record in records
    ])
    predicted = residual + float(factor) * np.einsum(
        "nrp,p->nr", jacobian, local, optimize=True
    )
    norms = np.linalg.norm(predicted, axis=1)
    w = np.asarray(weights, float)
    merit = float(np.dot(w, norms * norms))
    wrms = float(np.sqrt(merit / max(float(np.sum(w)), np.finfo(float).tiny)))
    return float(np.max(norms, initial=0.0)), wrms, merit


def combined_layer_linearization(
    network,
    field,
    physics_points,
    semigroup_points,
    evaluated,
    layer,
    config,
    monitor=None,
):
    physics = layer_linearization(
        network, field, physics_points, evaluated, layer, config, monitor
    )
    sg_records = []
    if len(semigroup_points):
        sg_budget = min(int(config.semigroup_jacobian_point_budget), len(semigroup_points))
        rows = np.asarray(semigroup_points, float)[:sg_budget]
        sg_records, sg_ids = evaluate_layer_semigroup(
            network, rows, layer, monitor=monitor
        )
        if not np.array_equal(physics.parameter_indices, sg_ids):
            raise RuntimeError("physics and restart parameter blocks differ")
    result = _combined_metrics(physics.records, sg_records)
    result.parameter_indices = physics.parameter_indices
    result.physics_subset = physics.physics_subset
    return result


__all__ = [
    "source_prefit", "residual_target_modes", "evaluate_layer_physics",
    "layer_linearization", "evaluate_layer_semigroup", "solve_layer_direction",
    "predicted_layer_metrics", "combined_layer_linearization",
]
