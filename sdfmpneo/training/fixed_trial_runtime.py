from __future__ import annotations

from threading import local

import numpy as np

from .parallel_runtime import _ordered_map, training_point_workers
from .fixed_physics_runtime import (
    _prepare_operating_contexts,
    _prepare_working_set,
    evaluate_physics_batch,
    evaluate_semigroup_batch,
)

_STATE = local()
_TRIAL_PARENTS: dict[int, int] = {}
_EVALUATED: dict[int, object] = {}
_ORIGINAL_WITH_PARAMETERS = None
_INSTALLED = False


class CandidateEarlyRejected(ValueError):
    pass


def _candidate_thresholds(parent, tolerance):
    from . import research_helpers as rh

    old = np.asarray(parent.norms, dtype=float)
    weights = rh._hard_weights(old, tolerance)
    old_max = float(np.max(old, initial=0.0))
    old_merit = float(np.dot(weights, old * old))
    numerical = 64.0 * np.finfo(float).eps * max(old_merit, 1.0)
    margin = 64.0 * np.finfo(float).eps * max(float(tolerance), old_max, 1e-30)
    if old_max <= 20.0 * tolerance:
        return weights, old_max + margin, 1.02 * old_merit + numerical
    return weights, 1.10 * old_max + margin, 1.05 * old_merit + numerical


def _candidate_exact_result(
    network, field, points, semigroup_points, include_semigroup,
    monitor, parent, tolerance,
):
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    semigroup_points = np.asarray(semigroup_points, dtype=float)
    _prepare_working_set(field, points)
    if len(points):
        _prepare_operating_contexts(field, points[:, network.n_modes:-1])
    weights, max_limit, merit_limit = _candidate_thresholds(parent, tolerance)
    n_physics = len(points)
    n_semigroup = len(semigroup_points) if include_semigroup else 0
    expected = n_physics + n_semigroup
    if weights.shape != (expected,):
        if include_semigroup:
            return rh._evaluate_all(network, field, points, semigroup_points, monitor=monitor)
        return rh._combined_metrics(
            evaluate_physics_batch(network, field, points, monitor=monitor), []
        )

    items = [(float(parent.norms[i]), 0, i, row) for i, row in enumerate(points)]
    if include_semigroup:
        items.extend(
            (float(parent.norms[n_physics + i]), 1, n_physics + i, row)
            for i, row in enumerate(semigroup_points)
        )
    items.sort(key=lambda item: item[0], reverse=True)

    records = [None] * expected
    partial_merit = 0.0
    partial_max = 0.0
    partial_physics_max = 0.0
    physics_guard = None
    if include_semigroup:
        physics_guard = max(1.05 * tolerance, 1.02 * float(parent.physics_max))
    batch_size = min(training_point_workers(), max(1, expected))

    def evaluate_item(item):
        _, kind, index, values = item
        if kind == 0:
            record = evaluate_physics_batch(
                network, field, np.asarray(values, float)[None, :], monitor=None
            )[0]
        else:
            record = evaluate_semigroup_batch(
                network, np.asarray(values, float)[None, :], monitor=None
            )[0]
        return index, kind, record

    completed = 0
    for start in range(0, len(items), batch_size):
        values = _ordered_map(evaluate_item, items[start:start + batch_size], monitor=monitor)
        for index, kind, record in values:
            records[index] = record
            norm = float(np.linalg.norm(record.residual))
            partial_merit += float(weights[index]) * norm * norm
            partial_max = max(partial_max, norm)
            if kind == 0:
                partial_physics_max = max(partial_physics_max, norm)
        completed += len(values)
        rh._work(monitor, "candidate_exact_residual", completed, expected)
        if partial_max > max_limit or partial_merit > merit_limit:
            raise CandidateEarlyRejected(
                "candidate cannot satisfy exact trust acceptance"
            )
        if physics_guard is not None and partial_physics_max > physics_guard:
            raise CandidateEarlyRejected("candidate exceeds exact physics guard")

    return rh._combined_metrics(
        records[:n_physics], records[n_physics:] if include_semigroup else []
    )


def _install_trial_tracking(network_class):
    global _ORIGINAL_WITH_PARAMETERS
    if getattr(network_class, "_sdfmpneo_trial_tracking_installed", False):
        return
    _ORIGINAL_WITH_PARAMETERS = network_class.with_parameters

    def with_parameters(self, parameters):
        result = _ORIGINAL_WITH_PARAMETERS(self, parameters)
        _TRIAL_PARENTS[id(result)] = id(self)
        return result

    network_class.with_parameters = with_parameters
    network_class._sdfmpneo_trial_tracking_installed = True


def install_trial_acceleration() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
    from . import research_trainer as rt

    _install_trial_tracking(FixedAnalyticResponseNetwork)
    original_exact_result = rt._exact_result

    def exact_result(network, field, points, semigroup_points, include_semigroup, monitor):
        tolerance = getattr(_STATE, "tolerance", None)
        parent_id = _TRIAL_PARENTS.get(id(network))
        parent = _EVALUATED.get(parent_id) if parent_id is not None else None
        if tolerance is not None and parent is not None:
            result = _candidate_exact_result(
                network, field, points, semigroup_points, include_semigroup,
                monitor, parent, float(tolerance),
            )
        else:
            result = original_exact_result(
                network, field, points, semigroup_points, include_semigroup, monitor
            )
        _EVALUATED[id(network)] = result
        return result

    rt._exact_result = exact_result
    _INSTALLED = True


def accelerated_train_research_network(original, field, config, **kwargs):
    install_trial_acceleration()
    previous = getattr(_STATE, "tolerance", None)
    _STATE.tolerance = float(config.residual_tolerance)
    _TRIAL_PARENTS.clear()
    _EVALUATED.clear()
    try:
        return original(field, config, **kwargs)
    finally:
        _TRIAL_PARENTS.clear()
        _EVALUATED.clear()
        if previous is None:
            try:
                delattr(_STATE, "tolerance")
            except AttributeError:
                pass
        else:
            _STATE.tolerance = previous
