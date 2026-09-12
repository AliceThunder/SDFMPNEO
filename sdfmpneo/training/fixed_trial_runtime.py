from __future__ import annotations

import hashlib
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

_STATE = local()
_TRIAL_PARENTS: dict[int, int] = {}
_NETWORKS: dict[int, object] = {}
_EVALUATED: dict[int, object] = {}
_AFFINE_POINT_CACHE: dict[tuple[bytes, bytes], tuple] = {}
_ACTIVE_AFFINE_SIGNATURE: bytes | None = None
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
            # The affine reconstruction is exact only when no later active layer
            # consumes the current layer as an input feature.
            if any(not network._layer_is_zero(k) for k in range(layer + 1, network.depth)):
                return None
            return int(layer), ids
    return None


def _affine_signature(network, layer, ids):
    """Signature of every quantity that fixes a layer's analytic feature basis."""
    theta = np.asarray(network.parameters, dtype=np.float64).copy()
    theta[np.asarray(ids, dtype=int)] = 0.0
    digest = hashlib.blake2b(digest_size=20)
    digest.update(theta.tobytes())
    digest.update(np.asarray(network.lambdas, dtype=np.float64).tobytes())
    digest.update(np.asarray(network.input_center, dtype=np.float64).tobytes())
    digest.update(np.asarray(network.input_scale, dtype=np.float64).tobytes())
    digest.update(np.asarray(network.layer_widths[: layer + 1], dtype=np.int64).tobytes())
    for targets in network.layer_targets[: layer + 1]:
        digest.update(np.asarray(targets, dtype=np.int64).tobytes())
    return digest.digest()


def _activate_affine_signature(signature):
    global _ACTIVE_AFFINE_SIGNATURE
    if _ACTIVE_AFFINE_SIGNATURE != signature:
        _AFFINE_POINT_CACHE.clear()
        _ACTIVE_AFFINE_SIGNATURE = signature


def _affine_basis_for_point(network, point, layer, ids, signature):
    """Cache the exact response basis for one collocation point.

    During layerwise training only output amplitudes of the active layer change.
    The source features and response poles therefore stay fixed, making ``a`` and
    ``da/dt`` exactly affine in that amplitude block.  We store only target rows
    of the Jacobians to keep the cache compact for high thermal rank.
    """
    point = np.ascontiguousarray(np.asarray(point, dtype=np.float64))
    key = (signature, point.tobytes())
    cached = _AFFINE_POINT_CACHE.get(key)
    if cached is not None:
        return cached

    n = network.n_modes
    initial, operating, time = point[:n], point[n:-1], float(point[-1])
    a, da, ja, jda, actual_ids = network.evaluate_layer_amplitude_jacobian(
        time, a0=initial, operating=operating, layer=layer
    )
    actual_ids = np.asarray(actual_ids, dtype=int)
    if not np.array_equal(actual_ids, ids):
        raise RuntimeError("layer amplitude block changed while building exact candidate cache")

    targets = np.unique(np.asarray(network.layer_targets[layer], dtype=int))
    theta = np.asarray(network.parameters, dtype=float)[ids]
    ja_rows = np.ascontiguousarray(np.asarray(ja, dtype=float)[targets])
    jda_rows = np.ascontiguousarray(np.asarray(jda, dtype=float)[targets])
    base_a = np.asarray(a, dtype=float).copy()
    base_da = np.asarray(da, dtype=float).copy()
    base_a[targets] -= ja_rows @ theta
    base_da[targets] -= jda_rows @ theta
    cached = (targets, base_a, base_da, ja_rows, jda_rows)
    _AFFINE_POINT_CACHE[key] = cached
    return cached


def _affine_physics_record(network, field, point, layer, ids, signature):
    targets, base_a, base_da, ja_rows, jda_rows = _affine_basis_for_point(
        network, point, layer, ids, signature
    )
    theta = np.asarray(network.parameters, dtype=float)[ids]
    a = base_a.copy()
    da = base_da.copy()
    a[targets] += ja_rows @ theta
    da[targets] += jda_rows @ theta
    n = network.n_modes
    operating = np.asarray(point, dtype=float)[n:-1]
    F = physics_vector_field(field, a, operating)
    return SimpleNamespace(residual=np.asarray(da - F, dtype=float))


def _candidate_exact_result(
    network, field, points, semigroup_points, include_semigroup,
    monitor, parent, tolerance, parent_network=None,
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

    affine = None
    if parent_network is not None:
        affine = _candidate_amplitude_layer(parent_network, network)
    if affine is not None:
        layer, ids = affine
        # Layer 1 is already fast and its amplitude block can be very wide; the
        # compressed cache is aimed at the expensive deeper funnel layers.
        if layer <= 0 or len(ids) > 2048:
            affine = None
        else:
            signature = _affine_signature(network, layer, ids)
            _activate_affine_signature(signature)

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
            if affine is not None:
                layer, ids = affine
                record = _affine_physics_record(
                    network, field, values, layer, ids, signature
                )
            else:
                record = evaluate_physics_batch(
                    network, field, np.asarray(values, float)[None, :], monitor=None
                )[0]
        else:
            # Restart composition is not affine in the layer amplitudes because
            # the first segment becomes the second segment's initial condition.
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
                network, field, points, semigroup_points, include_semigroup,
                monitor, parent, float(tolerance), parent_network=parent_network,
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
    global _ACTIVE_AFFINE_SIGNATURE
    install_trial_acceleration()
    previous = getattr(_STATE, "tolerance", None)
    _STATE.tolerance = float(config.residual_tolerance)
    _TRIAL_PARENTS.clear()
    _NETWORKS.clear()
    _EVALUATED.clear()
    _AFFINE_POINT_CACHE.clear()
    _ACTIVE_AFFINE_SIGNATURE = None
    try:
        return original(field, config, **kwargs)
    finally:
        _TRIAL_PARENTS.clear()
        _NETWORKS.clear()
        _EVALUATED.clear()
        _AFFINE_POINT_CACHE.clear()
        _ACTIVE_AFFINE_SIGNATURE = None
        if previous is None:
            try:
                delattr(_STATE, "tolerance")
            except AttributeError:
                pass
        else:
            _STATE.tolerance = previous
