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


class GeneralizedETD2Stepper:
    """Second-order ETD for ``M a' = -K a + q(a)``.

    A generalized symmetric eigendecomposition is built once per thermal
    operator.  Thus no assumption ``M=I`` or diagonal ``K`` is made for geometry
    families.
    """

    def __init__(self, operator: ReducedThermalOperator, step: float):
        self.operator = operator
        self.step = float(step)
        eigenvalues, vectors = scipy.linalg.eigh(
            operator.stiffness,
            operator.mass,
            check_finite=True,
        )
        if np.any(eigenvalues <= 0.0):
            raise ValueError("thermal generalized eigenvalues must be positive")
        self.lambdas = np.asarray(eigenvalues, dtype=float)
        self.vectors = np.asarray(vectors, dtype=float)
        self.E, self.b1, self.b2 = _etd_coefficients(self.lambdas, self.step)

    def to_modal_state(self, state: np.ndarray) -> np.ndarray:
        a = np.asarray(state, dtype=float)
        return self.vectors.T @ (self.operator.mass @ a)

    def from_modal_state(self, modal: np.ndarray) -> np.ndarray:
        return self.vectors @ np.asarray(modal, dtype=float)

    def to_modal_source(self, source: np.ndarray) -> np.ndarray:
        return self.vectors.T @ np.asarray(source, dtype=float)

    def step_once(self, state: np.ndarray, source) -> np.ndarray:
        c0 = self.to_modal_state(state)
        q0 = np.asarray(source(np.asarray(state, dtype=float)), dtype=float)
        s0 = self.to_modal_source(q0)
        c_stage = self.E * c0 + self.b1 * s0
        stage = self.from_modal_state(c_stage)
        q1 = np.asarray(source(stage), dtype=float)
        s1 = self.to_modal_source(q1)
        c1 = c_stage + self.b2 * (s1 - s0)
        return self.from_modal_state(c1)


def integrate_etd2(
    field,
    time: float,
    *,
    initial_state: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    max_step: float,
) -> IntegrationResult:
    t = float(time)
    if not np.isfinite(t) or t < 0:
        raise ValueError("integration time must be finite and non-negative")
    a = np.asarray(initial_state, dtype=float).copy()
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)
    if t == 0.0:
        return IntegrationResult(a, 0.0, 0, ())
    hmax = float(max_step)
    if not np.isfinite(hmax) or hmax <= 0:
        raise ValueError("max_step must be finite and positive")
    operator = field.thermal_operators.operator(g)
    elapsed = 0.0
    sizes = []
    # One full-size stepper is reused; only the final fractional step gets a new
    # propagator. Geometry is static along the autonomous trajectory.
    full_stepper = GeneralizedETD2Stepper(operator, min(hmax, t))

    def source(x):
        return field.heat_source(x, g, u)

    while elapsed < t:
        h = min(hmax, t - elapsed)
        if abs(h - full_stepper.step) <= 8.0 * np.finfo(float).eps * max(1.0, h):
            stepper = full_stepper
        else:
            stepper = GeneralizedETD2Stepper(operator, h)
        a = stepper.step_once(a, source)
        if np.any(~np.isfinite(a)):
            raise FloatingPointError("ETD2 produced a non-finite thermal state")
        elapsed = min(t, elapsed + h)
        sizes.append(float(h))
    return IntegrationResult(a, t, len(sizes), tuple(sizes))


def integrate_imex_euler(
    field,
    time: float,
    *,
    initial_state: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    max_step: float,
) -> IntegrationResult:
    """First-order robust reference: implicit thermal diffusion, explicit NN source."""
    t = float(time)
    if not np.isfinite(t) or t < 0:
        raise ValueError("integration time must be finite and non-negative")
    a = np.asarray(initial_state, dtype=float).copy()
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)
    operator = field.thermal_operators.operator(g)
    elapsed, sizes = 0.0, []
    while elapsed < t:
        h = min(float(max_step), t - elapsed)
        q = field.heat_source(a, g, u)
        a = scipy.linalg.solve(
            operator.mass + h * operator.stiffness,
            operator.mass @ a + h * q,
            assume_a="pos",
            check_finite=True,
        )
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
) -> IntegrationResult:
    """High-accuracy solve_ivp oracle for validating production integrators."""
    t = float(time)
    a0 = np.asarray(initial_state, dtype=float)
    if t == 0.0:
        return IntegrationResult(a0.copy(), 0.0, 0, ())

    def rhs(_t, state):
        return field.vector_field(state, geometry, operating)

    solution = solve_ivp(rhs, (0.0, t), a0, method="Radau", rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    return IntegrationResult(
        np.asarray(solution.y[:, -1], dtype=float),
        t,
        max(0, len(solution.t) - 1),
        tuple(float(v) for v in np.diff(solution.t)),
    )


__all__ = [
    "GeneralizedETD2Stepper",
    "IntegrationResult",
    "integrate_etd2",
    "integrate_imex_euler",
    "integrate_reference",
]
