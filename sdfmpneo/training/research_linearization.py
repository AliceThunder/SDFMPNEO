"""Bounded-memory collocation selection for fixed-network Gauss-Newton."""
from __future__ import annotations

import numpy as np

from .research_helpers import (
    _combined_metrics,
    _evaluate_network,
    _evaluate_semigroup,
    _hard_subset,
)

# At r=198 this allows twelve physics points (2376 residual rows): half are
# current worst-residual points and half cover the full state/static/time box.
# This remains tiny compared with a full 68-point Jacobian while avoiding a
# search direction fitted only to a local cluster of failures.
_MAX_PHYSICS_JACOBIAN_ROWS = 2560
_MAX_SEMIGROUP_JACOBIAN_ROWS = 640


def _normalized_collocation(points, config):
    """Map collocation rows to an O(1) box for geometric representative sampling."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        return np.asarray(values, dtype=float)
    lo = np.min(values, axis=0)
    hi = np.max(values, axis=0)
    span = hi - lo
    span[span <= 0.0] = 1.0
    result = (values - lo) / span

    # The training design deliberately mixes very early logarithmic times with
    # late linear times.  A linear Euclidean metric would collapse all early
    # transient points near t=0, so preserve their separation in the sampling
    # metric while leaving the actual collocation values unchanged.
    if getattr(config, "time_sampling", None) == "mixed_log" and values.shape[1]:
        t = np.maximum(values[:, -1], 0.0)
        tmin = max(float(getattr(config, "time_min", 1e-6)), np.finfo(float).tiny)
        horizon = max(float(getattr(config, "max_response_time", 1.0)), tmin)
        denom = np.log1p(horizon / tmin)
        result[:, -1] = np.log1p(t / tmin) / max(denom, np.finfo(float).tiny)
    return result


def _representative_subset(points, excluded, count, config):
    """Farthest-point sample the parameter/time domain outside ``excluded``."""
    values = np.asarray(points, dtype=float)
    count = min(max(0, int(count)), len(values))
    if count == 0 or len(values) == 0:
        return values[:0]

    excluded_rows = np.asarray(excluded, dtype=float)
    blocked = set(tuple(float(v) for v in row) for row in excluded_rows)
    candidates = np.asarray(
        [i for i, row in enumerate(values) if tuple(float(v) for v in row) not in blocked],
        dtype=int,
    )
    if candidates.size == 0:
        return values[:0]
    count = min(count, candidates.size)
    metric = _normalized_collocation(values, config)

    if len(excluded_rows):
        excluded_metric = _normalized_collocation(
            np.vstack([values, excluded_rows]), config
        )[len(values):]
        distance = np.min(
            np.sum((metric[candidates, None, :] - excluded_metric[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    else:
        center = np.full(metric.shape[1], 0.5)
        distance = np.sum((metric[candidates] - center) ** 2, axis=1)

    chosen = []
    available = candidates.copy()
    current_distance = distance.copy()
    for _ in range(count):
        local = int(np.argmax(current_distance))
        idx = int(available[local])
        chosen.append(idx)
        available = np.delete(available, local)
        current_distance = np.delete(current_distance, local)
        if available.size == 0:
            break
        new_distance = np.sum((metric[available] - metric[idx]) ** 2, axis=1)
        current_distance = np.minimum(current_distance, new_distance)
    return values[np.asarray(chosen, dtype=int)]


def _balanced_physics_subset(points, norms, budget, config):
    """Use half worst points and half domain-covering representatives."""
    values = np.asarray(points, dtype=float)
    budget = min(len(values), max(1, int(budget)))
    hard_count = min(budget, max(1, (budget + 1) // 2))
    hard = _hard_subset(values, norms, hard_count)
    representatives = _representative_subset(values, hard, budget - len(hard), config)
    if len(representatives) == 0:
        return hard
    return np.vstack([hard, representatives])


def linearization(network, field, points, semigroup_points, evaluated, config, monitor):
    n_modes = max(1, int(network.n_modes))
    physics_budget = min(
        int(config.jacobian_point_budget),
        max(1, _MAX_PHYSICS_JACOBIAN_ROWS // n_modes),
    )
    physics_subset = _balanced_physics_subset(
        points, evaluated.physics_norms, physics_budget, config
    )
    physics = _evaluate_network(
        network,
        field,
        physics_subset,
        jacobian=True,
        monitor=monitor,
        work_label="physics_jacobian",
    )

    semigroup = []
    if len(semigroup_points):
        semigroup_budget = min(
            int(config.semigroup_jacobian_point_budget),
            max(1, _MAX_SEMIGROUP_JACOBIAN_ROWS // n_modes),
        )
        semigroup_subset = _hard_subset(
            semigroup_points,
            evaluated.semigroup_norms,
            semigroup_budget,
        )
        semigroup = _evaluate_semigroup(
            network,
            semigroup_subset,
            jacobian=True,
            monitor=monitor,
            work_label="restart_jacobian",
        )
    result = _combined_metrics(physics, semigroup)
    result.physics_subset = physics_subset
    result.physics_hard_count = min(len(physics_subset), max(1, (len(physics_subset) + 1) // 2))
    return result


__all__ = ["linearization", "_balanced_physics_subset", "_representative_subset"]
