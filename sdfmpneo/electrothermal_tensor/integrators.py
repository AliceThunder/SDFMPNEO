"""Time integrators that keep the reduced thermal operator outside the network."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp

from .vector_field import ReducedThermalOperator


@dataclass(frozen=True)
class IntegrationResult:
    state: np.ndarray
    time: float
    steps: int
    step_sizes: tuple[float, ...]
    rejected_steps: int = 0


def _etd_coefficients(lambdas: np.ndarray, step: float):
    lam = np.asarray(lambdas, dtype=float)
    h = float(step)
    if h <= 0 or not np.isfinite(h):
        raise ValueError("ETD step must be finite and positive")
    x = h * lam
    E = np.exp(-x)
    b1 = np.empty_like(lam)
    b2 = np.empty_like(lam)
    small = np.abs(x) < 1e-5
    xs = x[small]
    b1[small] = h * (1.0 - xs / 2.0 + xs**2 / 6.0 - xs**3 / 24.0)
    b2[small] = h * (0.5 - xs / 6.0 + xs**2 / 24.0 - xs**3 / 120.0)
    large = ~small
    xl = x[large]
    ll = lam[large]
    b1[large] = -np.expm1(-xl) / ll
    b2[large] = (np.exp(-xl) - 1.0 + xl) / (h * ll**2)
    return E, b1, b2


def _validate_state(validator, state: np.ndarray) -> None:
    if validator is not None:
        validator(np.asarray(state, dtype=float))


def _resolved_spectrum(
    operator: ReducedThermalOperator,
    spectrum: "GeneralizedThermalSpectrum | None",
) -> "GeneralizedThermalSpectrum":
    if spectrum is None:
        return GeneralizedThermalSpectrum(operator)
    if spectrum.operator.mass.shape != operator.mass.shape:
        raise ValueError("ETD spectrum/operator dimensions differ")
    return spectrum


class GeneralizedThermalSpectrum:
    """One generalized eigendecomposition reused by all ETD step sizes."""

    def __init__(self, operator: ReducedThermalOperator):
        self.operator = operator
        eigenvalues, vectors = scipy.linalg.eigh(
            operator.stiffness,
            operator.mass,
            check_finite=True,
        )
        if np.any(eigenvalues <= 0.0):
            raise ValueError("thermal generalized eigenvalues must be positive")
        self.lambdas = np.asarray(eigenvalues, dtype=float)
        self.vectors = np.asarray(vectors, dtype=float)

    def to_modal_state(self, state: np.ndarray) -> np.ndarray:
        a = np.asarray(state, dtype=float)
        return self.vectors.T @ (self.operator.mass @ a)

    def from_modal_state(self, modal: np.ndarray) -> np.ndarray:
        return self.vectors @ np.asarray(modal, dtype=float)

    def to_modal_source(self, source: np.ndarray) -> np.ndarray:
        return self.vectors.T @ np.asarray(source, dtype=float)

    def etd2_trial(self, state: np.ndarray, source, step: float):
        """Return ETD2 state, embedded ETD1 state and nonlinear stage."""
        current = np.asarray(state, dtype=float)
        E, b1, b2 = _etd_coefficients(self.lambdas, step)
        c0 = self.to_modal_state(current)
        q0 = np.asarray(source(current), dtype=float)
        s0 = self.to_modal_source(q0)
        c_stage = E * c0 + b1 * s0
        stage = self.from_modal_state(c_stage)
        q1 = np.asarray(source(stage), dtype=float)
        s1 = self.to_modal_source(q1)
        c2 = c_stage + b2 * (s1 - s0)
        result = self.from_modal_state(c2)
        return result, stage, stage


class GeneralizedETD2Stepper:
    """Fixed-step second-order ETD for ``M a' = -K a + q(a)``."""

    def __init__(
        self,
        operator: ReducedThermalOperator,
        step: float,
        *,
        spectrum: GeneralizedThermalSpectrum | None = None,
    ):
        self.operator = operator
        self.step = float(step)
        self.spectrum = _resolved_spectrum(operator, spectrum)
        self.lambdas = self.spectrum.lambdas
        self.vectors = self.spectrum.vectors
        self.E, self.b1, self.b2 = _etd_coefficients(self.lambdas, self.step)

    def to_modal_state(self, state: np.ndarray) -> np.ndarray:
        return self.spectrum.to_modal_state(state)

    def from_modal_state(self, modal: np.ndarray) -> np.ndarray:
        return self.spectrum.from_modal_state(modal)

    def to_modal_source(self, source: np.ndarray) -> np.ndarray:
        return self.spectrum.to_modal_source(source)

    def step_once(self, state: np.ndarray, source, *, state_validator=None) -> np.ndarray:
        current = np.asarray(state, dtype=float)
        _validate_state(state_validator, current)
        c0 = self.to_modal_state(current)
        q0 = np.asarray(source(current), dtype=float)
        s0 = self.to_modal_source(q0)
        c_stage = self.E * c0 + self.b1 * s0
        stage = self.from_modal_state(c_stage)
        _validate_state(state_validator, stage)
        q1 = np.asarray(source(stage), dtype=float)
        s1 = self.to_modal_source(q1)
        c1 = c_stage + self.b2 * (s1 - s0)
        result = self.from_modal_state(c1)
        _validate_state(state_validator, result)
        return result


def integrate_etd2(
    field,
    time: float,
    *,
    initial_state: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    max_step: float,
    state_validator=None,
    spectrum: GeneralizedThermalSpectrum | None = None,
) -> IntegrationResult:
    t = float(time)
    if not np.isfinite(t) or t < 0:
        raise ValueError("integration time must be finite and non-negative")
    a = np.asarray(initial_state, dtype=float).copy()
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)
    _validate_state(state_validator, a)
    hmax = float(max_step)
    if not np.isfinite(hmax) or hmax <= 0:
        raise ValueError("max_step must be finite and positive")
    if t == 0.0:
        return IntegrationResult(a, 0.0, 0, ())
    operator = field.thermal_operators.operator(g)
    spectrum = _resolved_spectrum(operator, spectrum)
    elapsed = 0.0
    sizes: list[float] = []
    full_h = min(hmax, t)
    full_stepper = GeneralizedETD2Stepper(operator, full_h, spectrum=spectrum)

    def source(x):
        return field.heat_source(x, g, u)

    while elapsed < t:
        h = min(hmax, t - elapsed)
        if abs(h - full_stepper.step) <= 8.0 * np.finfo(float).eps * max(1.0, h):
            stepper = full_stepper
        else:
            stepper = GeneralizedETD2Stepper(operator, h, spectrum=spectrum)
        a = stepper.step_once(a, source, state_validator=state_validator)
        if np.any(~np.isfinite(a)):
            raise FloatingPointError("ETD2 produced a non-finite thermal state")
        elapsed = min(t, elapsed + h)
        sizes.append(float(h))
    return IntegrationResult(a, t, len(sizes), tuple(sizes))


def integrate_etd2_adaptive(
    field,
    time: float,
    *,
    initial_state: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    max_step: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    initial_step: float | None = None,
    max_attempts: int = 100000,
    state_validator=None,
    spectrum: GeneralizedThermalSpectrum | None = None,
) -> IntegrationResult:
    """Adaptive ETD2 with an embedded exponential-Euler error indicator.

    A supplied generalized eigendecomposition is reused verbatim.  Otherwise it
    is built once for this integration.  Each trial only recomputes scalar
    exponential coefficients and two neural heat-source evaluations.
    """
    t = float(time)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("integration time must be finite and non-negative")
    hmax = float(max_step)
    rtol = float(rtol)
    atol = float(atol)
    attempts_limit = int(max_attempts)
    if not np.isfinite(hmax) or hmax <= 0.0:
        raise ValueError("max_step must be finite and positive")
    if not np.isfinite(rtol) or rtol <= 0.0 or not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("adaptive ETD tolerances must be finite and positive")
    if attempts_limit < 1:
        raise ValueError("max_attempts must be positive")
    a = np.asarray(initial_state, dtype=float).copy()
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)
    _validate_state(state_validator, a)
    if t == 0.0:
        return IntegrationResult(a, 0.0, 0, ())
    if initial_step is None:
        h = min(hmax, t)
    else:
        h = float(initial_step)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("initial_step must be finite and positive")
        h = min(h, hmax, t)

    operator = field.thermal_operators.operator(g)
    spectrum = _resolved_spectrum(operator, spectrum)

    def source(x):
        return field.heat_source(x, g, u)

    elapsed = 0.0
    accepted_sizes: list[float] = []
    rejected = 0
    attempts = 0
    minimum_step = max(
        np.finfo(float).eps * max(1.0, t) * 32.0,
        np.nextafter(0.0, 1.0),
    )

    while elapsed < t:
        attempts += 1
        if attempts > attempts_limit:
            raise RuntimeError("adaptive ETD2 exceeded max_attempts")
        h = min(h, hmax, t - elapsed)
        if h < minimum_step:
            raise RuntimeError("adaptive ETD2 step underflow before reaching requested tolerance")

        E, b1, b2 = _etd_coefficients(spectrum.lambdas, h)
        c0 = spectrum.to_modal_state(a)
        q0 = np.asarray(source(a), dtype=float)
        s0 = spectrum.to_modal_source(q0)
        c_euler = E * c0 + b1 * s0
        stage = spectrum.from_modal_state(c_euler)

        domain_rejected = False
        try:
            _validate_state(state_validator, stage)
        except ValueError:
            domain_rejected = True

        if domain_rejected:
            rejected += 1
            h *= 0.5
            if h < minimum_step:
                _validate_state(state_validator, stage)
            continue

        q1 = np.asarray(source(stage), dtype=float)
        s1 = spectrum.to_modal_source(q1)
        c2 = c_euler + b2 * (s1 - s0)
        trial = spectrum.from_modal_state(c2)
        try:
            _validate_state(state_validator, trial)
        except ValueError:
            rejected += 1
            h *= 0.5
            if h < minimum_step:
                _validate_state(state_validator, trial)
            continue
        if np.any(~np.isfinite(trial)):
            raise FloatingPointError("adaptive ETD2 produced a non-finite thermal state")

        error = trial - stage
        scale = atol + rtol * np.maximum(np.abs(a), np.abs(trial))
        error_ratio = float(np.sqrt(np.mean((error / scale) ** 2)))
        if not np.isfinite(error_ratio):
            raise FloatingPointError("adaptive ETD2 error indicator became non-finite")

        if error_ratio <= 1.0:
            a = trial
            elapsed = min(t, elapsed + h)
            accepted_sizes.append(float(h))
            if error_ratio == 0.0:
                factor = 5.0
            else:
                factor = float(np.clip(0.9 * error_ratio ** -0.5, 0.5, 5.0))
            h = min(hmax, h * factor)
        else:
            rejected += 1
            factor = float(np.clip(0.9 * error_ratio ** -0.5, 0.1, 0.5))
            h *= factor

    return IntegrationResult(
        a,
        t,
        len(accepted_sizes),
        tuple(accepted_sizes),
        rejected_steps=rejected,
    )


def integrate_imex_euler(
    field,
    time: float,
    *,
    initial_state: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    max_step: float,
    state_validator=None,
) -> IntegrationResult:
    """First-order robust reference: implicit thermal diffusion, explicit NN source."""
    t = float(time)
    if not np.isfinite(t) or t < 0:
        raise ValueError("integration time must be finite and non-negative")
    hmax = float(max_step)
    if not np.isfinite(hmax) or hmax <= 0.0:
        raise ValueError("max_step must be finite and positive")
    a = np.asarray(initial_state, dtype=float).copy()
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)
    _validate_state(state_validator, a)
    if t == 0.0:
        return IntegrationResult(a, 0.0, 0, ())
    operator = field.thermal_operators.operator(g)
    elapsed = 0.0
    sizes: list[float] = []
    while elapsed < t:
        h = min(hmax, t - elapsed)
        q = field.heat_source(a, g, u)
        a = scipy.linalg.solve(
            operator.mass + h * operator.stiffness,
            operator.mass @ a + h * q,
            assume_a="pos",
            check_finite=True,
        )
        _validate_state(state_validator, a)
        if np.any(~np.isfinite(a)):
            raise FloatingPointError("IMEX produced a non-finite thermal state")
        elapsed = min(t, elapsed + h)
        sizes.append(float(h))
    return IntegrationResult(a, t, len(sizes), tuple(sizes))


def integrate_reference(
    field,
    time: float,
    *,
    initial_state: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    state_validator=None,
) -> IntegrationResult:
    """High-accuracy solve_ivp oracle for validating production integrators."""
    t = float(time)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("integration time must be finite and non-negative")
    a0 = np.asarray(initial_state, dtype=float)
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)
    _validate_state(state_validator, a0)
    if t == 0.0:
        return IntegrationResult(a0.copy(), 0.0, 0, ())

    def rhs(_t, state):
        _validate_state(state_validator, state)
        return field.vector_field(state, g, u)

    solution = solve_ivp(rhs, (0.0, t), a0, method="Radau", rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    final = np.asarray(solution.y[:, -1], dtype=float)
    _validate_state(state_validator, final)
    return IntegrationResult(
        final,
        t,
        max(0, len(solution.t) - 1),
        tuple(float(v) for v in np.diff(solution.t)),
    )


__all__ = [
    "GeneralizedETD2Stepper",
    "GeneralizedThermalSpectrum",
    "IntegrationResult",
    "integrate_etd2",
    "integrate_etd2_adaptive",
    "integrate_imex_euler",
    "integrate_reference",
]
