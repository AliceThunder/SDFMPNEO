from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np


_HIGH_DIMENSIONAL_PARAMETER_THRESHOLD = 8
_HIGH_DIMENSIONAL_QMC_MULTIPLIER = 2
_COVERAGE_POLICY_VERSION = 1


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


def _axis_boundary_anchors(config) -> np.ndarray:
    """Cheap deterministic coverage of every parameter face through the box centre."""
    lo = np.asarray(config.initial_lower + config.operating_lower, dtype=float)
    hi = np.asarray(config.initial_upper + config.operating_upper, dtype=float)
    centre = 0.5 * (lo + hi)

    # QMC points cover the interior/time continuum.  These anchors deliberately
    # cover the two dynamical endpoints where extrapolation between sparse high-
    # dimensional samples is most dangerous: t=0 and, when requested, t=+inf.
    times = [0.0]
    if config.include_steady_state:
        times.append(np.inf)
    else:
        if config.time_sampling == "mixed_log":
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


def install_high_dimensional_collocation() -> None:
    """Strengthen high-dimensional coverage without changing training stages.

    For parameter boxes with at least eight non-time coordinates, use twice the
    configured low-discrepancy count for both training and fresh validation.
    Training additionally receives axis-aligned face-centre anchors.  No labels,
    residual tolerances, candidate dictionaries, or model-capacity limits change.
    """
    from . import research as r
    from . import adaptive_runtime as adaptive

    if getattr(r.ResearchTrainingConfig, "_high_dimensional_coverage_installed", False):
        return

    original_points = r.ResearchTrainingConfig.points
    original_signature = adaptive._config_signature

    def points(self, validation=False, seed=None):
        parameter_dimension = len(self.initial_lower) + len(self.operating_lower)
        if parameter_dimension < _HIGH_DIMENSIONAL_PARAMETER_THRESHOLD:
            return original_points(self, validation=validation, seed=seed)

        expanded = replace(
            self,
            sample_count=self.sample_count * _HIGH_DIMENSIONAL_QMC_MULTIPLIER,
            validation_count=self.validation_count * _HIGH_DIMENSIONAL_QMC_MULTIPLIER,
        )
        values = original_points(expanded, validation=validation, seed=seed)
        if not validation:
            values = np.vstack([values, _axis_boundary_anchors(self)])
        return _stable_unique_rows(values)

    def signature(config):
        base = original_signature(config)
        parameter_dimension = len(config.initial_lower) + len(config.operating_lower)
        if parameter_dimension < _HIGH_DIMENSIONAL_PARAMETER_THRESHOLD:
            return base
        payload = {
            "base": base,
            "coverage_policy_version": _COVERAGE_POLICY_VERSION,
            "qmc_multiplier": _HIGH_DIMENSIONAL_QMC_MULTIPLIER,
            "axis_boundary_anchors": True,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    r.ResearchTrainingConfig.points = points
    r.ResearchTrainingConfig._high_dimensional_coverage_installed = True
    adaptive._config_signature = signature
