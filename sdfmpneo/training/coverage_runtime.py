from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np


_HIGH_DIMENSIONAL_PARAMETER_THRESHOLD = 8
# Validation is now a numerical residual-search pool rather than a small random
# batch whose every failed point is promoted.  Search much more broadly, then
# exchange only the strongest few counterexamples into the persistent set.
_SEARCH_QMC_MULTIPLIER = 8
_EXCHANGE_BATCH_SIZE = 4
_COVERAGE_POLICY_VERSION = 2


def _stable_unique_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("collocation points must be a matrix")
    rows = []
    seen = set()
    for row in values:
        # np.inf is intentional for the stationary limit and is hashable.
        key = tuple(float(value) for value in row)
        if key not in seen:
            seen.add(key)
            rows.append(row.copy())
    return np.vstack(rows) if rows else np.empty((0, values.shape[1]), dtype=float)


def _axis_boundary_anchors(config, *, search: bool = False) -> np.ndarray:
    """Deterministic face-centre probes for every non-time parameter bound."""
    lo = np.asarray(config.initial_lower + config.operating_lower, dtype=float)
    hi = np.asarray(config.initial_upper + config.operating_upper, dtype=float)
    centre = 0.5 * (lo + hi)

    if search:
        if config.time_sampling == "mixed_log":
            middle = float(np.sqrt(config.time_min * config.time_horizon))
            times = [0.0, float(config.time_min), middle, float(config.time_horizon)]
        else:
            times = [0.0, 0.5 * float(config.time_horizon), float(config.time_horizon)]
        if config.include_steady_state:
            times.append(np.inf)
    else:
        # Keep the persistent base set compact.  QMC covers the transient
        # interior; anchors guarantee every parameter face is represented at
        # the exact initial and stationary endpoints.
        times = [0.0]
        if config.include_steady_state:
            times.append(np.inf)
        elif config.time_sampling == "mixed_log":
            times.append(float(np.sqrt(config.time_min * config.time_horizon)))
        else:
            times.append(0.5 * float(config.time_horizon))

    rows = []
    for time in times:
        for index in range(lo.size):
            lower = centre.copy()
            upper = centre.copy()
            lower[index] = lo[index]
            upper[index] = hi[index]
            rows.append(np.concatenate([lower, [time]]))
            rows.append(np.concatenate([upper, [time]]))
    return np.vstack(rows) if rows else np.empty((0, lo.size + 1), dtype=float)


def _select_worst_exchange_points(records, points, *, tolerance: float):
    """Exchange only the strongest currently discovered residual counterexamples.

    The full search pool is evaluated, so its maximum remains the numerical
    convergence check.  Passed search points are deliberately not retained as
    guards: a new low-discrepancy search pool is generated at the next exchange.
    This keeps the public state machine simply

        train -> residual search -> add worst hotspots -> train.
    """
    values = np.asarray(points, dtype=float)
    norms = np.array([np.linalg.norm(record.residual) for record in records], dtype=float)
    if values.ndim != 2 or values.shape[0] != norms.size:
        raise ValueError("search records and points do not match")

    failing = np.flatnonzero(norms > float(tolerance))
    if failing.size:
        # Stable sort preserves deterministic QMC order for exact score ties.
        order = failing[np.argsort(-norms[failing], kind="stable")]
        selected = order[:_EXCHANGE_BATCH_SIZE]
        worst = values[selected].copy()
    else:
        worst = np.empty((0, values.shape[1]), dtype=float)
    passed_guards = np.empty((0, values.shape[1]), dtype=float)
    return worst, passed_guards, norms


def install_high_dimensional_collocation() -> None:
    """Install high-dimensional worst-residual exchange semantics.

    Low-dimensional research/demo sampling is unchanged.  For parameter boxes
    with at least eight non-time coordinates, the persistent training set keeps
    the configured QMC count plus cheap axis-boundary anchors.  Each numerical
    validation becomes a much broader residual-search pool; only its strongest
    few failed points are exchanged into training.  No labels, tolerances,
    candidate dictionaries or capacity limits change.
    """
    from . import research as r
    from . import adaptive_runtime as adaptive
    from . import late_stage_runtime as late_stage

    if getattr(r.ResearchTrainingConfig, "_high_dimensional_coverage_installed", False):
        return

    original_points = r.ResearchTrainingConfig.points
    original_signature = adaptive._config_signature
    original_selector = adaptive._select_adaptive_validation_points

    def points(self, validation=False, seed=None):
        parameter_dimension = len(self.initial_lower) + len(self.operating_lower)
        if parameter_dimension < _HIGH_DIMENSIONAL_PARAMETER_THRESHOLD:
            return original_points(self, validation=validation, seed=seed)

        if validation:
            expanded = replace(
                self,
                validation_count=self.validation_count * _SEARCH_QMC_MULTIPLIER,
            )
            values = original_points(expanded, validation=True, seed=seed)
            values = np.vstack([values, _axis_boundary_anchors(self, search=True)])
        else:
            values = original_points(self, validation=False, seed=seed)
            values = np.vstack([values, _axis_boundary_anchors(self, search=False)])
        return _stable_unique_rows(values)

    def selector(records, points, *, tolerance: float):
        values = np.asarray(points, dtype=float)
        if values.ndim != 2:
            raise ValueError("search points must be a matrix")
        # The selector itself cannot see config.  High-dimensional calls are
        # recognizable by point width: >=8 non-time coordinates plus time.
        if values.shape[1] - 1 < _HIGH_DIMENSIONAL_PARAMETER_THRESHOLD:
            return original_selector(records, points, tolerance=tolerance)
        return _select_worst_exchange_points(records, points, tolerance=tolerance)

    def signature(config):
        base = original_signature(config)
        parameter_dimension = len(config.initial_lower) + len(config.operating_lower)
        if parameter_dimension < _HIGH_DIMENSIONAL_PARAMETER_THRESHOLD:
            return base
        payload = {
            "base": base,
            "coverage_policy_version": _COVERAGE_POLICY_VERSION,
            "search_qmc_multiplier": _SEARCH_QMC_MULTIPLIER,
            "exchange_batch_size": _EXCHANGE_BATCH_SIZE,
            "axis_boundary_anchors": True,
            "passed_search_points_are_guards": False,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    r.ResearchTrainingConfig.points = points
    r.ResearchTrainingConfig._high_dimensional_coverage_installed = True
    adaptive._config_signature = signature
    adaptive._select_adaptive_validation_points = selector
    # late_stage_runtime imported the selector by value, so patch its binding too.
    late_stage._select_adaptive_validation_points = selector
