from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentedRolloutResult:
    state: np.ndarray
    derivative: np.ndarray
    segment_count: int
    segment_durations: tuple[float, ...]


@dataclass(frozen=True)
class SteadyStateSolveResult:
    state: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool


def _check_box(state, lower, upper):
    if lower is None or upper is None:
        return
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if state.shape != lo.shape or hi.shape != lo.shape:
        raise ValueError("restart-state bounds do not match network thermal rank")
    if np.any(state < lo) or np.any(state > hi):
        raise ValueError(
            "segmented rollout left the trained restart-state box; "
            "retrain with wider initial/restart bounds or explicitly allow extrapolation"
        )


def rollout_fixed_network(
    network,
    time,
    *,
    a0,
    operating,
    state_lower=None,
    state_upper=None,
    allow_extrapolation=False,
):
    """Compose the finite-horizon analytic flow until the requested finite time."""
    t = float(time)
    if not np.isfinite(t) or t < 0:
        raise ValueError("rollout time must be finite and non-negative")
    state = np.asarray(a0, dtype=float).copy()
    operating = np.asarray(operating, dtype=float)
    if state.shape != (network.n_modes,):
        raise ValueError("initial state dimension does not match network thermal rank")
    if operating.shape != (len(network.operating_names),):
        raise ValueError("operating dimension does not match network")
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(operating)):
        raise ValueError("initial state and operating input must be finite")
    if not allow_extrapolation:
        _check_box(state, state_lower, state_upper)

    horizon = float(network.max_response_time)
    if t == 0.0:
        state, derivative = network.evaluate(0.0, a0=state, operating=operating)
        return SegmentedRolloutResult(state, derivative, 0, ())

    durations = []
    remaining = t
    derivative = np.zeros_like(state)
    while remaining > 0.0:
        dt = min(horizon, remaining)
        state, derivative = network.evaluate(dt, a0=state, operating=operating)
        durations.append(float(dt))
        remaining = max(0.0, remaining - dt)
        if remaining > 0.0 and not allow_extrapolation:
            _check_box(state, state_lower, state_upper)
    return SegmentedRolloutResult(
        np.asarray(state, dtype=float),
        np.asarray(derivative, dtype=float),
        len(durations),
        tuple(durations),
    )


def solve_physical_steady_state(
    field,
    operating,
    *,
    initial_guess,
    tolerance=1e-10,
    max_iterations=40,
):
    """Damped Newton solve of F(a, operating)=0 using the physical reduced field."""
    tolerance = float(tolerance)
    max_iterations = int(max_iterations)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("steady-state tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("steady-state max_iterations must be positive")
    state = np.asarray(initial_guess, dtype=float).copy()
    operating = np.asarray(operating, dtype=float)
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(operating)):
        raise ValueError("steady-state inputs must be finite")

    for iteration in range(max_iterations + 1):
        evaluation = field.evaluate(state, operating)
        residual = np.asarray(evaluation.vector_field, dtype=float)
        norm = float(np.linalg.norm(residual))
        if norm <= tolerance:
            return SteadyStateSolveResult(state, norm, iteration, True)
        if iteration == max_iterations:
            break
        jacobian = np.asarray(evaluation.vector_field_jacobian, dtype=float)
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        accepted = False
        factor = 1.0
        for _ in range(12):
            trial = state + factor * step
            try:
                trial_residual = np.asarray(field.evaluate(trial, operating).vector_field, dtype=float)
                trial_norm = float(np.linalg.norm(trial_residual))
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                trial_norm = float("inf")
            if np.isfinite(trial_norm) and trial_norm < norm:
                state = trial
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            break

    final = np.asarray(field.evaluate(state, operating).vector_field, dtype=float)
    return SteadyStateSolveResult(
        state,
        float(np.linalg.norm(final)),
        max_iterations,
        False,
    )
