from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from sdfmpneo.em.energy_solver import apsi_physical_energy_metric
from sdfmpneo.em.sparse_solver import (
    SparseEnergyLinearSolveCertificate,
    SparseLinearSolveCertificate,
)


@dataclass(frozen=True)
class AlgebraicHeatSourceErrorCertificate:
    """Euclidean algebraic heat-source certificate retained for compatibility."""

    electromagnetic_state_error_bound: float
    approximate_state_norm: float
    loss_operator_frobenius_bounds: np.ndarray
    component_error_bounds: np.ndarray
    heat_source_vector_error_bound: float
    certified: bool


@dataclass(frozen=True)
class PhysicalEnergyHeatSourceErrorCertificate:
    """Contrast-independent heat-source certificate in the physical EM energy norm."""

    electromagnetic_energy_state_error_bound: float
    approximate_energy_state_norm: float
    thermal_test_supremum_bounds: np.ndarray
    component_error_bounds: np.ndarray
    heat_source_vector_error_bound: float
    certified: bool


def _frobenius_norm(matrix) -> float:
    if sp.issparse(matrix):
        data = np.asarray(matrix.data)
        return float(np.sqrt(np.sum(np.abs(data) ** 2)))
    return float(np.linalg.norm(np.asarray(matrix), ord="fro"))


def certify_algebraic_heat_source_error(
    problem,
    thermal_state: np.ndarray,
    approximate_state: np.ndarray,
    linear_solve_certificate: SparseLinearSolveCertificate,
) -> AlgebraicHeatSourceErrorCertificate:
    """Bound heat-source error from a Euclidean state-error certificate."""

    a = np.asarray(thermal_state, dtype=float)
    xh = np.asarray(approximate_state, dtype=complex)
    if xh.shape != (problem.n_em,):
        raise ValueError("approximate_state dimension mismatch")
    eps_x = float(linear_solve_certificate.state_error_bound)
    if eps_x < 0.0:
        raise ValueError("state error bound must be non-negative")

    xnorm = float(np.linalg.norm(xh))
    common = 2.0 * xnorm * eps_x + eps_x * eps_x
    operator_bounds = []
    component_bounds = []
    for j in range(problem.n_thermal):
        if hasattr(problem, "loss_operator_sparse"):
            H = problem.loss_operator_sparse(j, a)
        else:
            H = problem.loss_operator(j, a)
        H_bound = _frobenius_norm(H)
        operator_bounds.append(H_bound)
        component_bounds.append(H_bound * common)

    operator_bounds_array = np.asarray(operator_bounds, dtype=float)
    component_bounds_array = np.asarray(component_bounds, dtype=float)
    vector_bound = float(np.linalg.norm(component_bounds_array))
    return AlgebraicHeatSourceErrorCertificate(
        electromagnetic_state_error_bound=eps_x,
        approximate_state_norm=xnorm,
        loss_operator_frobenius_bounds=operator_bounds_array,
        component_error_bounds=component_bounds_array,
        heat_source_vector_error_bound=vector_bound,
        certified=bool(linear_solve_certificate.certified),
    )


def certify_physical_energy_heat_source_error(
    problem,
    thermal_state: np.ndarray,
    approximate_state: np.ndarray,
    linear_solve_certificate: SparseEnergyLinearSolveCertificate,
) -> PhysicalEnergyHeatSourceErrorCertificate:
    """Propagate physical-energy field error to projected Joule heat sources.

    Let H=K+D for A=K+iD and let phi_j be the P1 thermal test mode used in
    q_j. Since the Joule operator is 0.5*omega times the conductive part D and
    |phi_j| <= m_j pointwise,

        |q_j(x)-q_j(x_h)|
        <= 0.5*omega*m_j*(2||x_h||_H eps_H + eps_H^2).

    This bound is independent of the copper/seawater conductivity contrast and
    requires neither dense loss operators nor a singular-value estimate.
    """

    if not hasattr(problem, "operator_sparse"):
        raise TypeError("problem must provide the physical sparse A-psi operator")
    if not hasattr(problem, "thermal_test_local"):
        raise TypeError("problem must expose thermal_test_local for projected heat sources")

    a = np.asarray(thermal_state, dtype=float)
    xh = np.asarray(approximate_state, dtype=complex)
    if xh.shape != (problem.n_em,):
        raise ValueError("approximate_state dimension mismatch")
    eps_H = float(linear_solve_certificate.energy_state_error_bound)
    if eps_H < 0.0:
        raise ValueError("energy state error bound must be non-negative")

    H = apsi_physical_energy_metric(problem.operator_sparse(a))
    energy_value = float(np.real(np.vdot(xh, H @ xh)))
    if energy_value < 0.0:
        backward = np.finfo(float).eps * max(1, problem.n_em) * max(1.0, float(np.linalg.norm(xh)) ** 2)
        if energy_value < -backward:
            raise np.linalg.LinAlgError("physical electromagnetic energy became negative")
        energy_value = 0.0
    xnorm_H = float(np.sqrt(energy_value))

    tests = np.asarray(problem.thermal_test_local, dtype=float)
    if tests.shape[0] != problem.n_thermal:
        raise ValueError("thermal test mode count mismatch")
    sup = np.max(np.abs(tests), axis=(1, 2))
    common = 2.0 * xnorm_H * eps_H + eps_H * eps_H
    component = 0.5 * float(problem.omega) * sup * common
    vector_bound = float(np.linalg.norm(component))
    return PhysicalEnergyHeatSourceErrorCertificate(
        electromagnetic_energy_state_error_bound=eps_H,
        approximate_energy_state_norm=xnorm_H,
        thermal_test_supremum_bounds=sup,
        component_error_bounds=np.asarray(component, dtype=float),
        heat_source_vector_error_bound=vector_bound,
        certified=bool(linear_solve_certificate.certified),
    )
