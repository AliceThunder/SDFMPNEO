from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from sdfmpneo.em.sparse_solver import SparseLinearSolveCertificate


@dataclass(frozen=True)
class AlgebraicHeatSourceErrorCertificate:
    """Propagate a certified field-solve error to reduced Joule heat sources."""

    electromagnetic_state_error_bound: float
    approximate_state_norm: float
    loss_operator_frobenius_bounds: np.ndarray
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
    """Bound heat-source error caused only by the algebraic field solve.

    For q_j=x^H H_j x and ||x-x_h||_2 <= eps_x,

        |q_j-q_j,h|
        <= ||H_j||_2 (2 ||x_h||_2 eps_x + eps_x^2)
        <= ||H_j||_F (2 ||x_h||_2 eps_x + eps_x^2).

    The Frobenius norm is evaluated directly from sparse matrix entries. The
    certificate therefore remains sparse and introduces no spectral-norm
    iteration or fitted tolerance.
    """

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
