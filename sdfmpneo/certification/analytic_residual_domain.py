from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrothermal_domain import certify_electrothermal_domain_bounds
from .em_domain import ContinuousEMResidualCertificate, ParameterBox


def _time_factor_sup(power: int, rho: float, lower: float, upper: float) -> float:
    if lower < 0.0 or upper < lower:
        raise ValueError("time interval must satisfy 0 <= lower <= upper")
    candidates = [lower, upper]
    if power > 0 and rho > 0.0:
        stationary = power / rho
        if lower <= stationary <= upper:
            candidates.append(stationary)
    return float(max((t**power) * np.exp(-rho * t) for t in candidates))


def _series_sup_abs(series, lambdas, parameter_lower, parameter_upper, t_lower, t_upper) -> float:
    lo = np.asarray(parameter_lower, dtype=float)
    hi = np.asarray(parameter_upper, dtype=float)
    pmax = np.maximum(np.abs(lo), np.abs(hi))
    total = 0.0
    for (power, decay, params), coefficient in series.terms.items():
        monomial = 1.0
        for bound, exponent in zip(pmax, params):
            if exponent:
                monomial *= bound**exponent
        rho = float(np.dot(np.asarray(decay, dtype=float), lambdas))
        total += abs(complex(coefficient)) * monomial * _time_factor_sup(
            int(power), rho, t_lower, t_upper
        )
    return float(np.nextafter(total, np.inf))


@dataclass(frozen=True)
class AnalyticResidualBoxBound:
    box: ParameterBox
    center_residual_norm: float
    residual_upper_bound: float
    state_variation_bounds: np.ndarray
    derivative_variation_bounds: np.ndarray
    directional_contributions: np.ndarray
    contraction_margin_lower_bound: float


def bound_analytic_residual_on_box(operator, box: ParameterBox) -> AnalyticResidualBoxBound:
    graph = operator.graph
    field = operator.vector_field
    n = graph.n_modes
    nu = len(graph.operating_names)
    if box.lower.shape != (n + nu + 1,):
        raise ValueError("analytic box must contain [a0, operating, time]")
    if box.lower[-1] < 0.0:
        raise ValueError("analytic time domain must be non-negative")

    center = box.midpoint
    a0c = center[:n]
    uc = center[n : n + nu]
    tc = float(center[-1])
    prediction = operator.evaluate(tc, a0=a0c, operating=uc, stable=True)
    compiled = graph.compile()
    p_lo = box.lower[: n + nu]
    p_hi = box.upper[: n + nu]
    half = box.halfwidth

    state_var = np.zeros(n, dtype=float)
    derivative_var = np.zeros(n, dtype=float)
    state_direction = np.zeros((n, n + nu + 1), dtype=float)
    derivative_direction = np.zeros_like(state_direction)

    for i, series in enumerate(compiled.mode_series):
        dtime = series.derivative(compiled.lambdas)
        ddtime = dtime.derivative(compiled.lambdas)
        time_state = _series_sup_abs(
            dtime, compiled.lambdas, p_lo, p_hi, box.lower[-1], box.upper[-1]
        )
        time_derivative = _series_sup_abs(
            ddtime, compiled.lambdas, p_lo, p_hi, box.lower[-1], box.upper[-1]
        )
        state_direction[i, -1] = time_state
        derivative_direction[i, -1] = time_derivative
        state_var[i] += half[-1] * time_state
        derivative_var[i] += half[-1] * time_derivative
        for j in range(n + nu):
            dp = series.parameter_derivative(j)
            ddp = dp.derivative(compiled.lambdas)
            s = _series_sup_abs(dp, compiled.lambdas, p_lo, p_hi, box.lower[-1], box.upper[-1])
            ds = _series_sup_abs(ddp, compiled.lambdas, p_lo, p_hi, box.lower[-1], box.upper[-1])
            state_direction[i, j] = s
            derivative_direction[i, j] = ds
            state_var[i] += half[j] * s
            derivative_var[i] += half[j] * ds

    thermal_lower = prediction.state - state_var
    thermal_upper = prediction.state + state_var
    operating_lower = box.lower[n : n + nu]
    operating_upper = box.upper[n : n + nu]
    domain = certify_electrothermal_domain_bounds(
        field,
        thermal_lower=thermal_lower,
        thermal_upper=thermal_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
    )
    lambda_max = float(np.max(np.asarray(field.thermal_model.lambdas, dtype=float)))
    L_state = lambda_max + domain.heat_source_jacobian_norm_bound
    L_operating = domain.explicit_operating_jacobian_norm_bound

    residual_upper = (
        prediction.residual_norm
        + float(np.linalg.norm(derivative_var))
        + L_state * float(np.linalg.norm(state_var))
        + L_operating * float(np.linalg.norm(box.halfwidth[n : n + nu]))
    )

    contributions = np.zeros(n + nu + 1, dtype=float)
    for j in range(n + nu + 1):
        contribution = half[j] * (
            float(np.linalg.norm(derivative_direction[:, j]))
            + L_state * float(np.linalg.norm(state_direction[:, j]))
        )
        if n <= j < n + nu:
            column = domain.explicit_operating_jacobian_entry_bounds[:, j - n]
            contribution += half[j] * float(np.linalg.norm(column))
        contributions[j] = contribution

    return AnalyticResidualBoxBound(
        box=box,
        center_residual_norm=prediction.residual_norm,
        residual_upper_bound=float(np.nextafter(residual_upper, np.inf)),
        state_variation_bounds=state_var,
        derivative_variation_bounds=derivative_var,
        directional_contributions=contributions,
        contraction_margin_lower_bound=domain.contraction_margin_lower_bound,
    )


def certify_analytic_residual_domain(
    operator,
    *,
    initial_lower: np.ndarray,
    initial_upper: np.ndarray,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
    time_lower: float,
    time_upper: float,
    tolerance: float,
    work_budget: int,
) -> ContinuousEMResidualCertificate:
    if tolerance <= 0.0 or work_budget <= 0:
        raise ValueError("tolerance and work_budget must be positive")
    lower = np.concatenate([initial_lower, operating_lower, [float(time_lower)]])
    upper = np.concatenate([initial_upper, operating_upper, [float(time_upper)]])
    pending = [ParameterBox(lower, upper)]
    processed = 0
    observed = 0.0
    resolved: list[float] = []

    while pending and processed < work_budget:
        box = pending.pop()
        try:
            bound = bound_analytic_residual_on_box(operator, box)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            bound = None
        processed += 1
        if bound is not None:
            observed = max(observed, bound.center_residual_norm)
            if bound.center_residual_norm > tolerance:
                return ContinuousEMResidualCertificate(
                    "violated", tolerance, observed, bound.residual_upper_bound,
                    processed, box.midpoint.copy(), len(pending),
                )
            if bound.residual_upper_bound <= tolerance:
                resolved.append(bound.residual_upper_bound)
                continue
        widths = box.halfwidth
        if not np.any(widths > 0.0):
            # A point that cannot be physically certified remains unresolved.
            pending.append(box)
            break
        if bound is None:
            dimension = int(np.argmax(widths))
        else:
            dimension = int(np.argmax(bound.directional_contributions))
            if bound.directional_contributions[dimension] <= 0.0:
                dimension = int(np.argmax(widths))
        left, right = box.split(dimension)
        pending.extend([right, left])

    if not pending:
        return ContinuousEMResidualCertificate(
            "certified", tolerance, observed, float(max(resolved, default=observed)),
            processed, None, 0,
        )

    unresolved_upper = []
    for box in pending:
        try:
            unresolved_upper.append(bound_analytic_residual_on_box(operator, box).residual_upper_bound)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            unresolved_upper.append(float("inf"))
    return ContinuousEMResidualCertificate(
        "indeterminate", tolerance, observed,
        float(max(resolved + unresolved_upper, default=float("inf"))),
        processed, None, len(pending),
    )
