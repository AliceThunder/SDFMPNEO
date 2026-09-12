"""Reachable thermal-state domain probing with the real reduced physics.

The neural state box is a scientific input, not an optimizer hyperparameter.
This module derives a reproducible suggested box from sampled physical
trajectories and, optionally, nearby physical steady states. No neural model or
Joule-tensor label is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .dataset import latin_hypercube_box


@dataclass(frozen=True)
class ReachableStateDomainReport:
    physical_signature: str | None
    thermal_rank: int
    geometry_dimension: int
    operating_dimension: int
    trajectory_count: int
    samples_per_trajectory: int
    time_horizon: float
    time_min: float
    successful_steady_states: int
    failed_steady_states: int
    observed_state_lower: np.ndarray
    observed_state_upper: np.ndarray
    suggested_state_lower: np.ndarray
    suggested_state_upper: np.ndarray
    margin: np.ndarray
    initial_lower: np.ndarray
    initial_upper: np.ndarray
    geometry_lower: np.ndarray
    geometry_upper: np.ndarray
    operating_lower: np.ndarray
    operating_upper: np.ndarray
    seed: int
    rtol: float
    atol: float
    include_steady_state: bool


def _finite_bounds(lower, upper, *, name: str, allow_equal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.shape != hi.shape or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError(f"{name} bounds must be finite and dimensionally aligned")
    if allow_equal:
        invalid = hi < lo
    else:
        invalid = hi <= lo
    if np.any(invalid):
        relation = "ordered" if allow_equal else "strictly ordered"
        raise ValueError(f"{name} bounds must be {relation}")
    return lo, hi


def _sample_box_allow_fixed(lower: np.ndarray, upper: np.ndarray, n: int, *, seed: int) -> np.ndarray:
    """Latin-hypercube sample varying dimensions while preserving fixed ones."""
    n = int(n)
    if n < 1:
        raise ValueError("sample count must be positive")
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError("sampling bounds do not match")
    if lo.size == 0:
        return np.empty((n, 0), dtype=float)
    varying = hi > lo
    result = np.repeat(lo[None, :], n, axis=0)
    if np.any(varying):
        result[:, varying] = latin_hypercube_box(lo[varying], hi[varying], n, seed=seed)
    return result


def _probe_times(time_horizon: float, samples_per_trajectory: int, time_min: float) -> np.ndarray:
    horizon = float(time_horizon)
    count = int(samples_per_trajectory)
    minimum = float(time_min)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("time_horizon must be finite and positive")
    if count < 2:
        raise ValueError("samples_per_trajectory must be at least two")
    if not np.isfinite(minimum) or minimum <= 0.0 or minimum > horizon:
        raise ValueError("time_min must satisfy 0 < time_min <= time_horizon")
    if count == 2:
        return np.array([0.0, horizon], dtype=float)
    return np.concatenate([[0.0], np.geomspace(minimum, horizon, count - 1)])


def probe_reachable_state_domain(
    physical_vector_field,
    *,
    initial_lower,
    initial_upper,
    geometry_lower,
    geometry_upper,
    operating_lower,
    operating_upper,
    trajectory_count: int = 64,
    samples_per_trajectory: int = 24,
    time_horizon: float,
    time_min: float = 1e-6,
    seed: int = 0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    include_steady_state: bool = True,
    physical_jacobian_factory=None,
    steady_residual_tolerance: float = 1e-9,
    steady_maxfev: int = 400,
    margin_fraction: float = 0.10,
    absolute_margin: float | np.ndarray = 1e-8,
    physical_signature: str | None = None,
) -> ReachableStateDomainReport:
    """Probe a conservative neural state box from real reduced dynamics.

    Trajectories are sampled jointly over the declared initial, geometry and
    operating boxes. When requested, a physical steady-state root is attempted
    from every trajectory endpoint. Failed steady solves are recorded and never
    silently substituted by neural or clipped values.

    The returned suggested box is the observed/initial envelope plus a per-mode
    margin. It is intentionally a suggestion: final production coverage still
    has to be verified by independent trajectories in Gate 6.
    """
    init_lo, init_hi = _finite_bounds(initial_lower, initial_upper, name="initial")
    geom_lo, geom_hi = _finite_bounds(geometry_lower, geometry_upper, name="geometry")
    op_lo, op_hi = _finite_bounds(operating_lower, operating_upper, name="operating")
    if init_lo.size < 1:
        raise ValueError("thermal state dimension must be positive")
    count = int(trajectory_count)
    if count < 1:
        raise ValueError("trajectory_count must be positive")
    margin_fraction = float(margin_fraction)
    if not np.isfinite(margin_fraction) or margin_fraction < 0.0:
        raise ValueError("margin_fraction must be finite and non-negative")
    rtol = float(rtol)
    atol = float(atol)
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol/atol must be finite and positive")
    steady_residual_tolerance = float(steady_residual_tolerance)
    if not np.isfinite(steady_residual_tolerance) or steady_residual_tolerance <= 0.0:
        raise ValueError("steady_residual_tolerance must be finite and positive")
    if int(steady_maxfev) < 1:
        raise ValueError("steady_maxfev must be positive")

    absolute = np.asarray(absolute_margin, dtype=float)
    if absolute.ndim == 0:
        absolute = np.full(init_lo.size, float(absolute), dtype=float)
    else:
        absolute = absolute.reshape(-1)
    if absolute.shape != init_lo.shape or np.any(~np.isfinite(absolute)) or np.any(absolute < 0.0):
        raise ValueError("absolute_margin must be a non-negative scalar or one value per thermal mode")

    times = _probe_times(time_horizon, samples_per_trajectory, time_min)
    initial = _sample_box_allow_fixed(init_lo, init_hi, count, seed=seed)
    geometry = _sample_box_allow_fixed(geom_lo, geom_hi, count, seed=seed + 1)
    operating = _sample_box_allow_fixed(op_lo, op_hi, count, seed=seed + 2)

    observed_lower = init_lo.copy()
    observed_upper = init_hi.copy()
    successful_steady = 0
    failed_steady = 0

    for a0, g, u in zip(initial, geometry, operating):
        def rhs(_time, state):
            value = np.asarray(physical_vector_field(state, g, u), dtype=float).reshape(-1)
            if value.shape != init_lo.shape or np.any(~np.isfinite(value)):
                raise FloatingPointError("physical vector field returned an invalid value during domain probing")
            return value

        solution = solve_ivp(
            rhs,
            (0.0, float(times[-1])),
            a0,
            method="Radau",
            dense_output=True,
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"physical domain-probe trajectory failed: {solution.message}")
        sampled = np.asarray(solution.sol(times).T, dtype=float)
        if sampled.shape != (len(times), init_lo.size) or np.any(~np.isfinite(sampled)):
            raise FloatingPointError("physical domain-probe trajectory produced invalid states")
        observed_lower = np.minimum(observed_lower, np.min(sampled, axis=0))
        observed_upper = np.maximum(observed_upper, np.max(sampled, axis=0))

        if include_steady_state:
            endpoint = sampled[-1]

            def steady_fun(state):
                return rhs(0.0, state)

            if physical_jacobian_factory is None:
                jac = None
            else:
                def jac(state):
                    value = np.asarray(physical_jacobian_factory(state, g, u), dtype=float)
                    if value.shape != (init_lo.size, init_lo.size) or np.any(~np.isfinite(value)):
                        raise FloatingPointError("physical Jacobian returned an invalid value during domain probing")
                    return value

            try:
                solved = root(
                    steady_fun,
                    endpoint,
                    jac=jac,
                    method="hybr",
                    options={"maxfev": int(steady_maxfev)},
                )
                candidate = np.asarray(solved.x, dtype=float).reshape(-1)
                residual = steady_fun(candidate)
                valid = (
                    bool(solved.success)
                    and candidate.shape == init_lo.shape
                    and np.all(np.isfinite(candidate))
                    and float(np.linalg.norm(residual)) <= steady_residual_tolerance
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                valid = False
                candidate = None
            if valid:
                observed_lower = np.minimum(observed_lower, candidate)
                observed_upper = np.maximum(observed_upper, candidate)
                successful_steady += 1
            else:
                failed_steady += 1

    observed_span = observed_upper - observed_lower
    initial_span = init_hi - init_lo
    scale = np.maximum(observed_span, initial_span)
    margin = margin_fraction * scale + absolute
    suggested_lower = np.minimum(observed_lower, init_lo) - margin
    suggested_upper = np.maximum(observed_upper, init_hi) + margin
    if np.any(~np.isfinite(suggested_lower + suggested_upper)) or np.any(suggested_upper <= suggested_lower):
        raise FloatingPointError("domain probing produced an invalid suggested state box")

    return ReachableStateDomainReport(
        physical_signature=None if physical_signature is None else str(physical_signature),
        thermal_rank=int(init_lo.size),
        geometry_dimension=int(geom_lo.size),
        operating_dimension=int(op_lo.size),
        trajectory_count=count,
        samples_per_trajectory=int(samples_per_trajectory),
        time_horizon=float(time_horizon),
        time_min=float(time_min),
        successful_steady_states=int(successful_steady),
        failed_steady_states=int(failed_steady),
        observed_state_lower=observed_lower,
        observed_state_upper=observed_upper,
        suggested_state_lower=suggested_lower,
        suggested_state_upper=suggested_upper,
        margin=margin,
        initial_lower=init_lo,
        initial_upper=init_hi,
        geometry_lower=geom_lo,
        geometry_upper=geom_hi,
        operating_lower=op_lo,
        operating_upper=op_hi,
        seed=int(seed),
        rtol=rtol,
        atol=atol,
        include_steady_state=bool(include_steady_state),
    )


def _report_from_mapping(value: dict) -> ReachableStateDomainReport:
    payload = dict(value)
    array_names = (
        "observed_state_lower",
        "observed_state_upper",
        "suggested_state_lower",
        "suggested_state_upper",
        "margin",
        "initial_lower",
        "initial_upper",
        "geometry_lower",
        "geometry_upper",
        "operating_lower",
        "operating_upper",
    )
    for name in array_names:
        if name not in payload:
            raise ValueError(f"state-domain report is missing {name}")
        payload[name] = np.asarray(payload[name], dtype=float)
    return ReachableStateDomainReport(**payload)


def load_reachable_state_domain_report(path: str | Path) -> ReachableStateDomainReport:
    """Load either a direct report mapping or the CLI wrapper JSON."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("kind") == "reachable_state_domain":
        payload = payload.get("report")
    if not isinstance(payload, dict):
        raise ValueError("invalid reachable-state domain report")
    report = _report_from_mapping(payload)
    if report.thermal_rank != report.suggested_state_lower.size:
        raise ValueError("state-domain report thermal rank is inconsistent")
    if report.geometry_dimension != report.geometry_lower.size:
        raise ValueError("state-domain report geometry dimension is inconsistent")
    if report.operating_dimension != report.operating_lower.size:
        raise ValueError("state-domain report operating dimension is inconsistent")
    return report


def _same_domain(saved: np.ndarray, expected: np.ndarray) -> bool:
    saved = np.asarray(saved, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float).reshape(-1)
    if saved.shape != expected.shape:
        return False
    scale = max(1.0, float(np.max(np.abs(expected))) if expected.size else 1.0)
    return bool(np.allclose(saved, expected, rtol=0.0, atol=64.0 * np.finfo(float).eps * scale))


def validated_state_bounds(
    report: ReachableStateDomainReport,
    *,
    expected_physical_signature: str,
    geometry_lower,
    geometry_upper,
    operating_lower,
    operating_upper,
    thermal_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return suggested bounds only after fail-closed physical-domain checks."""
    if report.physical_signature is None or str(report.physical_signature) != str(expected_physical_signature):
        raise ValueError("state-domain report physical signature does not match the training physics")
    if thermal_rank is not None and int(report.thermal_rank) != int(thermal_rank):
        raise ValueError("state-domain report thermal rank does not match the training physics")
    if not _same_domain(report.geometry_lower, geometry_lower) or not _same_domain(report.geometry_upper, geometry_upper):
        raise ValueError("state-domain report geometry domain does not match training")
    if not _same_domain(report.operating_lower, operating_lower) or not _same_domain(report.operating_upper, operating_upper):
        raise ValueError("state-domain report operating domain does not match training")
    lower = np.asarray(report.suggested_state_lower, dtype=float).copy()
    upper = np.asarray(report.suggested_state_upper, dtype=float).copy()
    if lower.shape != (report.thermal_rank,) or upper.shape != lower.shape or np.any(upper <= lower):
        raise ValueError("state-domain report suggested bounds are invalid")
    return lower, upper


__all__ = [
    "ReachableStateDomainReport",
    "load_reachable_state_domain_report",
    "probe_reachable_state_domain",
    "validated_state_bounds",
]
