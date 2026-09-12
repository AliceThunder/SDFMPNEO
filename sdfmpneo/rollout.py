from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .em.modal_heat import heat_source_for_reduced_model


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


def _steady_residual_and_direction_jacobian(field, state, operating):
    """Exact physical residual plus a cheap dissipative Newton Jacobian."""
    a = np.asarray(state, dtype=float)
    u = np.asarray(operating, dtype=float)

    if hasattr(field, "split") and hasattr(field, "thermal_model"):
        context, current = field.split(u)
        rhs = context.rhs.evaluate(current)
        heat = heat_source_for_reduced_model(context.em, a, rhs)
        residual = np.linalg.solve(context.M, -context.K @ a + heat)
        jacobian = np.linalg.solve(context.M, -context.K)
        return residual, jacobian

    if hasattr(field, "em_model") and hasattr(field, "thermal_model") and hasattr(field, "rhs"):
        rhs = field.rhs(u if getattr(field, "rhs_map", None) is not None else None)
        heat = heat_source_for_reduced_model(field.em_model, a, rhs)
        lambdas = np.asarray(field.thermal_model.lambdas, dtype=float)
        forcing = np.asarray(getattr(field, "thermal_forcing", np.zeros_like(a)), dtype=float)
        residual = -lambdas * a + heat + forcing
        return residual, -np.diag(lambdas)

    evaluation = field.evaluate(a, u)
    return (
        np.asarray(evaluation.vector_field, dtype=float),
        np.asarray(evaluation.vector_field_jacobian, dtype=float),
    )


def solve_physical_steady_state(
    field,
    operating,
    *,
    initial_guess,
    tolerance=1e-10,
    max_iterations=40,
):
    """Solve F(a,operating)=0 with exact residual and damped inexact Newton steps.

    The line search always evaluates the complete electromagnetic Joule-heating
    residual. For nonlinear tetrahedral UWPT models the search direction freezes
    d(q_EM)/da and uses only the exact dissipative thermal Jacobian. This avoids
    O(r^2) loss-operator derivative assembly at high thermal rank without
    changing the steady-state equation being solved.
    """
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
        residual, jacobian = _steady_residual_and_direction_jacobian(
            field, state, operating
        )
        norm = float(np.linalg.norm(residual))
        if norm <= tolerance:
            return SteadyStateSolveResult(state, norm, iteration, True)
        if iteration == max_iterations:
            break
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
                trial_residual, _ = _steady_residual_and_direction_jacobian(
                    field, trial, operating
                )
                trial_norm = float(np.linalg.norm(trial_residual))
            except (
                ValueError,
                FloatingPointError,
                OverflowError,
                np.linalg.LinAlgError,
            ):
                trial_norm = float("inf")
            if np.isfinite(trial_norm) and trial_norm < norm:
                state = trial
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            break

    final, _ = _steady_residual_and_direction_jacobian(field, state, operating)
    return SteadyStateSolveResult(
        state,
        float(np.linalg.norm(final)),
        max_iterations,
        False,
    )
