"""Frozen-test validation, stability analysis and error-budget utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg

from .symmetric import quadratic_feature


@dataclass(frozen=True)
class SurrogateValidationReport:
    sample_count: int
    packed_relative_rms: float
    maximum_relative_packed_error: float
    heat_relative_rms: float
    maximum_relative_heat_error: float


@dataclass(frozen=True)
class VectorFieldValidationReport:
    sample_count: int
    relative_rms: float
    maximum_relative_error: float


@dataclass(frozen=True)
class StabilityReport:
    sample_count: int
    maximum_logarithmic_norm: float
    minimum_contraction_margin: float
    uniformly_contractive_on_samples: bool


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
    """Audit the frozen split without modifying or feeding it back into training."""
    ids = dataset.indices(split)
    lo = np.asarray(operating_lower, dtype=float).reshape(-1)
    hi = np.asarray(operating_upper, dtype=float).reshape(-1)
    if lo.shape != (dataset.current_dimension,) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("operating bounds mismatch")
    rng = np.random.default_rng(int(seed))
    packed_sq_error = 0.0
    packed_sq_scale = 0.0
    packed_relative = []
    heat_sq_error = 0.0
    heat_sq_scale = 0.0
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
        maximum_relative_packed_error=float(max(packed_relative, default=0.0)),
        heat_relative_rms=float(np.sqrt(heat_sq_error / max(heat_sq_scale, np.finfo(float).tiny))),
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
        maximum_relative_error=float(max(relative, default=0.0)),
    )


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
        minimum_contraction_margin=float(-maximum),
        uniformly_contractive_on_samples=bool(maximum < 0.0),
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
    "StabilityReport",
    "SurrogateValidationReport",
    "VectorFieldValidationReport",
    "analyze_stability",
    "energy_logarithmic_norm",
    "tensor_to_heat_error_bound",
    "trajectory_error_bound",
    "validate_surrogate_on_dataset",
    "validate_vector_field",
]
