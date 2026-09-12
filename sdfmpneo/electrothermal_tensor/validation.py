"""Frozen-test validation, stability analysis and error-budget utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp

from .quadratic_joule import quadratic_heat_source
from .symmetric import tensor_svec, quadratic_feature


@dataclass(frozen=True)
class QuadraticIdentityReport:
    sample_count: int
    relative_rms: float
    percentile_99_relative_error: float
    maximum_relative_error: float
    maximum_absolute_error: float


@dataclass(frozen=True)
class SurrogateValidationReport:
    sample_count: int
    packed_relative_rms: float
    packed_percentile_95: float
    packed_percentile_99: float
    maximum_relative_packed_error: float
    heat_relative_rms: float
    heat_percentile_95: float
    heat_percentile_99: float
    maximum_relative_heat_error: float


@dataclass(frozen=True)
class VectorFieldValidationReport:
    sample_count: int
    relative_rms: float
    percentile_95_relative_error: float
    percentile_99_relative_error: float
    maximum_relative_error: float


@dataclass(frozen=True)
class StabilityReport:
    sample_count: int
    maximum_logarithmic_norm: float
    percentile_99_logarithmic_norm: float
    minimum_contraction_margin: float
    uniformly_contractive_on_samples: bool


@dataclass(frozen=True)
class ActiveSubspaceReport:
    eigenvalues: np.ndarray
    cumulative_energy: np.ndarray


@dataclass(frozen=True)
class TrajectoryValidationReport:
    sample_count: int
    maximum_coordinate_error: float
    final_coordinate_error: float
    maximum_relative_coordinate_error: float
    maximum_temperature_error: float | None
    maximum_temperature_scalar_error: float | None


def _percentile(values, q: float) -> float:
    value = np.asarray(values, dtype=float)
    return float(np.percentile(value, q)) if value.size else 0.0


def validate_quadratic_identity(
    tensor_factory: Callable[[np.ndarray, np.ndarray], np.ndarray],
    direct_heat_factory: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    states: np.ndarray,
    geometries: np.ndarray,
    operating: np.ndarray,
) -> QuadraticIdentityReport:
    """Gate 1: verify the current-quadratic identity against direct EM heat."""
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    u = np.asarray(operating, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or u.ndim != 2 or not (len(a) == len(g) == len(u)):
        raise ValueError("quadratic identity samples must be aligned matrices")
    relative, absolute = [], []
    sq_error = sq_scale = 0.0
    for ai, gi, ui in zip(a, g, u):
        tensor = np.asarray(tensor_factory(ai, gi), dtype=float)
        predicted = quadratic_heat_source(tensor, ui)
        reference = np.asarray(direct_heat_factory(ai, gi, ui), dtype=float)
        error = float(np.linalg.norm(predicted - reference))
        scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        relative.append(error / scale)
        absolute.append(error)
        sq_error += error**2
        sq_scale += scale**2
    return QuadraticIdentityReport(
        sample_count=len(a),
        relative_rms=float(np.sqrt(sq_error / max(sq_scale, np.finfo(float).tiny))),
        percentile_99_relative_error=_percentile(relative, 99),
        maximum_relative_error=float(max(relative, default=0.0)),
        maximum_absolute_error=float(max(absolute, default=0.0)),
    )


def validate_surrogate_on_dataset(
    surrogate,
    dataset,
    *,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
    split: str = "test",
    operating_samples_per_state: int = 4,
    seed: int = 0,
) -> SurrogateValidationReport:
    """Gate 5: audit a frozen split without feeding it back into training."""
    ids = dataset.indices(split)
    lo = np.asarray(operating_lower, dtype=float).reshape(-1)
    hi = np.asarray(operating_upper, dtype=float).reshape(-1)
    if lo.shape != (dataset.current_dimension,) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("operating bounds mismatch")
    rng = np.random.default_rng(int(seed))
    packed_sq_error = packed_sq_scale = 0.0
    packed_relative = []
    heat_sq_error = heat_sq_scale = 0.0
    heat_relative = []

    for index in ids:
        state = dataset.states[index]
        geometry = dataset.geometries[index]
        true_packed = dataset.outputs[index]
        predicted_packed = surrogate.predict_packed_numpy(state, geometry)
        packed_error = float(np.linalg.norm(predicted_packed - true_packed))
        packed_scale = max(float(np.linalg.norm(true_packed)), np.finfo(float).tiny)
        packed_sq_error += packed_error**2
        packed_sq_scale += packed_scale**2
        packed_relative.append(packed_error / packed_scale)

        true_modes = true_packed.reshape(dataset.thermal_rank, dataset.packed_symmetric_size)
        predicted_modes = predicted_packed.reshape(dataset.thermal_rank, dataset.packed_symmetric_size)
        for _ in range(int(operating_samples_per_state)):
            u = rng.uniform(lo, hi)
            feature = quadratic_feature(u)
            q_true = true_modes @ feature
            q_pred = predicted_modes @ feature
            err = float(np.linalg.norm(q_pred - q_true))
            scale = max(float(np.linalg.norm(q_true)), np.finfo(float).tiny)
            heat_sq_error += err**2
            heat_sq_scale += scale**2
            heat_relative.append(err / scale)

    return SurrogateValidationReport(
        sample_count=len(ids),
        packed_relative_rms=float(np.sqrt(packed_sq_error / max(packed_sq_scale, np.finfo(float).tiny))),
        packed_percentile_95=_percentile(packed_relative, 95),
        packed_percentile_99=_percentile(packed_relative, 99),
        maximum_relative_packed_error=float(max(packed_relative, default=0.0)),
        heat_relative_rms=float(np.sqrt(heat_sq_error / max(heat_sq_scale, np.finfo(float).tiny))),
        heat_percentile_95=_percentile(heat_relative, 95),
        heat_percentile_99=_percentile(heat_relative, 99),
        maximum_relative_heat_error=float(max(heat_relative, default=0.0)),
    )


def validate_vector_field(
    neural_field,
    states: np.ndarray,
    geometries: np.ndarray,
    operating: np.ndarray,
    physical_vector_field: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> VectorFieldValidationReport:
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    u = np.asarray(operating, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or u.ndim != 2 or not (len(a) == len(g) == len(u)):
        raise ValueError("validation samples must be aligned matrices")
    sq_error = sq_scale = 0.0
    relative = []
    for ai, gi, ui in zip(a, g, u):
        reference = np.asarray(physical_vector_field(ai, gi, ui), dtype=float)
        predicted = np.asarray(neural_field.vector_field(ai, gi, ui), dtype=float)
        error = float(np.linalg.norm(predicted - reference))
        scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        sq_error += error**2
        sq_scale += scale**2
        relative.append(error / scale)
    return VectorFieldValidationReport(
        sample_count=len(a),
        relative_rms=float(np.sqrt(sq_error / max(sq_scale, np.finfo(float).tiny))),
        percentile_95_relative_error=_percentile(relative, 95),
        percentile_99_relative_error=_percentile(relative, 99),
        maximum_relative_error=float(max(relative, default=0.0)),
    )


def finite_difference_tensor_jacobian(
    tensor_factory: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state: np.ndarray,
    geometry: np.ndarray,
    *,
    relative_step: float = 1e-5,
) -> np.ndarray:
    """Gate 3 helper: central-difference Jacobian of packed ``G`` with respect to ``a``."""
    a = np.asarray(state, dtype=float).reshape(-1)
    g = np.asarray(geometry, dtype=float).reshape(-1)
    base = tensor_svec(np.asarray(tensor_factory(a, g), dtype=float)).reshape(-1)
    jacobian = np.empty((base.size, a.size), dtype=float)
    for k in range(a.size):
        h = float(relative_step) * max(1.0, abs(float(a[k])))
        plus, minus = a.copy(), a.copy()
        plus[k] += h
        minus[k] -= h
        yp = tensor_svec(np.asarray(tensor_factory(plus, g), dtype=float)).reshape(-1)
        ym = tensor_svec(np.asarray(tensor_factory(minus, g), dtype=float)).reshape(-1)
        jacobian[:, k] = (yp - ym) / (2.0 * h)
    return jacobian


def active_subspace_spectrum(jacobians: np.ndarray) -> ActiveSubspaceReport:
    """Gate 3: eigen-spectrum of mean ``J^T J`` over tensor/heat sensitivities."""
    J = np.asarray(jacobians, dtype=float)
    if J.ndim != 3 or J.shape[0] < 1 or np.any(~np.isfinite(J)):
        raise ValueError("jacobians must have shape (n_samples,n_outputs,n_states)")
    gram = np.einsum("noi,noj->ij", J, J, optimize=True) / J.shape[0]
    values = np.linalg.eigvalsh(0.5 * (gram + gram.T))[::-1]
    values = np.maximum(values, 0.0)
    total = max(float(np.sum(values)), np.finfo(float).tiny)
    return ActiveSubspaceReport(values, np.cumsum(values) / total)


def energy_logarithmic_norm(jacobian: np.ndarray, mass: np.ndarray) -> float:
    """Logarithmic norm induced by ``||x||_M`` for a real Jacobian."""
    J = np.asarray(jacobian, dtype=float)
    M = np.asarray(mass, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or M.shape != J.shape:
        raise ValueError("jacobian and mass dimensions do not match")
    symmetric = 0.5 * (M @ J + J.T @ M)
    eigenvalues = scipy.linalg.eigh(symmetric, M, eigvals_only=True, check_finite=True)
    return float(eigenvalues[-1])


def analyze_stability(field, states: np.ndarray, geometries: np.ndarray, operating: np.ndarray) -> StabilityReport:
    """Gate 4: sampled energy-norm one-sided Lipschitz analysis."""
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    u = np.asarray(operating, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or u.ndim != 2 or not (len(a) == len(g) == len(u)):
        raise ValueError("stability samples must be aligned matrices")
    values = []
    for ai, gi, ui in zip(a, g, u):
        J = field.state_jacobian(ai, gi, ui)
        M = field.thermal_operators.operator(gi).mass
        values.append(energy_logarithmic_norm(J, M))
    maximum = float(max(values, default=float("-inf")))
    return StabilityReport(
        sample_count=len(values),
        maximum_logarithmic_norm=maximum,
        percentile_99_logarithmic_norm=_percentile(values, 99),
        minimum_contraction_margin=float(-maximum),
        uniformly_contractive_on_samples=bool(maximum < 0.0),
    )


def validate_trajectory(
    neural_model,
    physical_vector_field: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    times: np.ndarray,
    *,
    initial_state: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    neural_max_step: float,
    temperature_reconstructor: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> TrajectoryValidationReport:
    """Gate 6: neural ETD trajectory versus a Radau physical-vector-field oracle."""
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size < 1 or np.any(~np.isfinite(t)) or np.any(t < 0.0) or np.any(np.diff(t) < 0.0):
        raise ValueError("times must be sorted, finite and non-negative")
    a0 = np.asarray(initial_state, dtype=float)
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)

    def rhs(_time, state):
        return np.asarray(physical_vector_field(state, g, u), dtype=float)

    if t[-1] == 0.0:
        reference = np.tile(a0, (len(t), 1))
    else:
        solution = solve_ivp(rhs, (0.0, float(t[-1])), a0, method="Radau", dense_output=True, rtol=rtol, atol=atol)
        if not solution.success:
            raise RuntimeError(solution.message)
        reference = solution.sol(t).T
    predicted = np.asarray([
        neural_model.predict(
            float(ti),
            initial_state=a0,
            geometry=g,
            operating=u,
            max_step=neural_max_step,
            method="etd2",
        ).state
        for ti in t
    ])
    errors = np.linalg.norm(predicted - reference, axis=1)
    scales = np.maximum(np.linalg.norm(reference, axis=1), np.finfo(float).tiny)
    max_temp_error = max_temp_scalar_error = None
    if temperature_reconstructor is not None:
        pred_temp = np.asarray([temperature_reconstructor(ai, g) for ai in predicted])
        ref_temp = np.asarray([temperature_reconstructor(ai, g) for ai in reference])
        max_temp_error = float(np.max(np.abs(pred_temp - ref_temp)))
        pred_max = np.max(pred_temp, axis=tuple(range(1, pred_temp.ndim)))
        ref_max = np.max(ref_temp, axis=tuple(range(1, ref_temp.ndim)))
        max_temp_scalar_error = float(np.max(np.abs(pred_max - ref_max)))
    return TrajectoryValidationReport(
        sample_count=len(t),
        maximum_coordinate_error=float(np.max(errors)),
        final_coordinate_error=float(errors[-1]),
        maximum_relative_coordinate_error=float(np.max(errors / scales)),
        maximum_temperature_error=max_temp_error,
        maximum_temperature_scalar_error=max_temp_scalar_error,
    )


def tensor_to_heat_error_bound(packed_tensor_error_norm: float, operating: np.ndarray) -> float:
    """Bound ``||Delta q||_2 <= ||Delta G||_F ||zeta||_2^2``."""
    u = np.asarray(operating, dtype=float).reshape(-1)
    zeta_norm_squared = 1.0 + float(np.dot(u, u))
    return float(abs(float(packed_tensor_error_norm)) * zeta_norm_squared)


def trajectory_error_bound(
    vector_field_error_bound: float,
    time: float,
    *,
    contraction_margin: float | None = None,
    lipschitz_constant: float | None = None,
) -> float:
    """Propagate a uniform vector-field error with contraction or Gronwall."""
    epsilon = abs(float(vector_field_error_bound))
    t = float(time)
    if t < 0 or not np.isfinite(t):
        raise ValueError("time must be finite and non-negative")
    if contraction_margin is not None and float(contraction_margin) > 0.0:
        gamma = float(contraction_margin)
        return float(epsilon * (-np.expm1(-gamma * t)) / gamma)
    if lipschitz_constant is None:
        raise ValueError("a positive contraction margin or a Lipschitz constant is required")
    L = float(lipschitz_constant)
    if L < 0 or not np.isfinite(L):
        raise ValueError("lipschitz_constant must be finite and non-negative")
    if L == 0.0:
        return epsilon * t
    return float(epsilon * np.expm1(L * t) / L)


__all__ = [
    "ActiveSubspaceReport",
    "QuadraticIdentityReport",
    "StabilityReport",
    "SurrogateValidationReport",
    "TrajectoryValidationReport",
    "VectorFieldValidationReport",
    "active_subspace_spectrum",
    "analyze_stability",
    "energy_logarithmic_norm",
    "finite_difference_tensor_jacobian",
    "tensor_to_heat_error_bound",
    "trajectory_error_bound",
    "validate_quadratic_identity",
    "validate_surrogate_on_dataset",
    "validate_trajectory",
    "validate_vector_field",
]
