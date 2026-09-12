from __future__ import annotations

from threading import local
from types import SimpleNamespace

import numpy as np

from .parallel_runtime import _ordered_map, training_point_workers
from .fixed_physics_runtime import (
    _prepare_operating_contexts,
    _prepare_working_set,
    evaluate_physics_batch,
    evaluate_semigroup_batch,
    physics_vector_field,
)
from .fixed_layer_cache_runtime import clear_layer_program_cache, layer_program

_STATE = local()
_TRIAL_PARENTS: dict[int, int] = {}
_NETWORKS: dict[int, object] = {}
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


def _candidate_amplitude_layer(parent_network, network):
    """Return the sole changed amplitude layer, or ``None`` for a general trial."""
    old = np.asarray(parent_network.parameters, dtype=float)
    new = np.asarray(network.parameters, dtype=float)
    if old.shape != new.shape:
        return None
    changed = np.flatnonzero(old != new)
    if changed.size == 0:
        return None
    for layer in range(network.depth):
        ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
        if ids.size and np.all(np.isin(changed, ids, assume_unique=False)):
            if any(not network._layer_is_zero(k) for k in range(layer + 1, network.depth)):
                return None
            return int(layer), ids
    return None


def _compiled_physics_records(network, field, rows, program):
    """Evaluate exact physics on a candidate without rebuilding analytic signals."""
    values = np.asarray(rows, dtype=float)
    if len(values) == 0:
        return []
    n = network.n_modes
    states, derivatives = program.reconstruct(network, values)

    def one(index):
        operating = values[index, n:-1]
        physical = physics_vector_field(field, states[index], operating)
        return SimpleNamespace(
            residual=np.asarray(derivatives[index] - physical, dtype=float)
        )

    return _ordered_map(one, range(len(values)), monitor=None)


def _candidate_exact_result(
    network,
    field,
    points,
    semigroup_points,
    include_semigroup,
    monitor,
    parent,
    tolerance,
    parent_network=None,
):
    """Exact trust-region result with batched early rejection.

    Expensive analytic structure is compiled once by ``layer_program``.  A trial
    candidate then performs only dense state reconstruction plus the real physical
    vector field.  Physics points are processed from worst parent residual to best
    in small batches so hopeless candidates still exit early.
    """
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

    program = None
    if parent_network is not None:
        affine = _candidate_amplitude_layer(parent_network, network)
        if affine is not None:
            layer, ids = affine
            if layer > 0 and len(ids) <= 4096:
                # The preceding Jacobian step normally prewarms this exact program.
                # If a caller reaches candidate evaluation directly, compile once
                # here and reuse it for every subsequent backtrack/iteration.
                program = layer_program(parent_network, points, layer, monitor=monitor)

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

    # Small batches preserve early-reject power; each batch is internally point
    # parallel.  Using at most one worker wave avoids the old nested thread/batch
    # overhead where evaluate_physics_batch was invoked once per point.
    batch_size = min(training_point_workers(), max(1, expected))
    completed = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        physics_items = [item for item in batch if item[1] == 0]
        restart_items = [item for item in batch if item[1] == 1]

        if physics_items:
            rows = np.asarray([item[3] for item in physics_items], dtype=float)
            if program is not None:
                batch_records = _compiled_physics_records(network, field, rows, program)
            else:
                batch_records = evaluate_physics_batch(
                    network, field, rows, monitor=None
                )
            for item, record in zip(physics_items, batch_records):
                index = item[2]
                records[index] = record
                norm = float(np.linalg.norm(record.residual))
                partial_merit += float(weights[index]) * norm * norm
                partial_max = max(partial_max, norm)
                partial_physics_max = max(partial_physics_max, norm)

        if restart_items:
            rows = np.asarray([item[3] for item in restart_items], dtype=float)
            batch_records = evaluate_semigroup_batch(network, rows, monitor=None)
            for item, record in zip(restart_items, batch_records):
                index = item[2]
                records[index] = record
                norm = float(np.linalg.norm(record.residual))
                partial_merit += float(weights[index]) * norm * norm
                partial_max = max(partial_max, norm)

        completed += len(batch)
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
        _NETWORKS[id(self)] = self
        _NETWORKS[id(result)] = result
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
        network_id = id(network)
        _NETWORKS[network_id] = network
        parent_id = _TRIAL_PARENTS.get(network_id)
        parent = _EVALUATED.get(parent_id) if parent_id is not None else None
        parent_network = _NETWORKS.get(parent_id) if parent_id is not None else None
        if tolerance is not None and parent is not None:
            result = _candidate_exact_result(
                network,
                field,
                points,
                semigroup_points,
                include_semigroup,
                monitor,
                parent,
                float(tolerance),
                parent_network=parent_network,
            )
        else:
            result = original_exact_result(
                network, field, points, semigroup_points, include_semigroup, monitor
            )
        _EVALUATED[network_id] = result
        return result

    rt._exact_result = exact_result
    _INSTALLED = True


def accelerated_train_research_network(original, field, config, **kwargs):
    install_trial_acceleration()
    previous = getattr(_STATE, "tolerance", None)
    _STATE.tolerance = float(config.residual_tolerance)
    _TRIAL_PARENTS.clear()
    _NETWORKS.clear()
    _EVALUATED.clear()
    clear_layer_program_cache()
    try:
        return original(field, config, **kwargs)
    finally:
        _TRIAL_PARENTS.clear()
        _NETWORKS.clear()
        _EVALUATED.clear()
        clear_layer_program_cache()
        if previous is None:
            try:
                delattr(_STATE, "tolerance")
            except AttributeError:
                pass
        else:
            _STATE.tolerance = previous
