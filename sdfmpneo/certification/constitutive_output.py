from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _ConstitutivePerturbationData:
    eps: float
    relative_factor: float
    delta_A: float
    sigma_min: float
    inverse_norm: float
    product: float
    D_tilde: np.ndarray
    L: np.ndarray
    A_tilde: np.ndarray


def _constitutive_perturbation_data(problem, thermal_state: np.ndarray) -> _ConstitutivePerturbationData:
    a = np.asarray(thermal_state, dtype=float)
    material = problem.constitutive_certificate(a)
    eps = float(material.maximum_inverse_relative_bound)
    if not 0.0 <= eps < 1.0:
        raise ValueError("constitutive relative bound must satisfy 0 <= eps < 1")

    A_tilde = np.asarray(problem.operator(a), dtype=complex)
    S_tilde = problem.conductivity_matrix(a).toarray().astype(complex)
    L = np.asarray(problem.electric_extraction(), dtype=complex)
    omega = float(problem.omega)
    if omega <= 0.0:
        raise ValueError("problem omega must be positive")

    Q = L / (-1j * omega)
    D_tilde = Q.conj().T @ S_tilde @ Q
    D_tilde = 0.5 * (D_tilde + D_tilde.conj().T)
    relative_factor = 0.0 if eps == 0.0 else eps / (1.0 - eps)
    delta_A = float(omega * relative_factor * np.linalg.norm(D_tilde, ord=2))

    singular_values = np.linalg.svd(A_tilde, compute_uv=False)
    sigma_min = float(np.min(singular_values))
    inverse_norm = float("inf") if sigma_min <= 0.0 else 1.0 / sigma_min
    product = inverse_norm * delta_A
    return _ConstitutivePerturbationData(
        eps=eps,
        relative_factor=float(relative_factor),
        delta_A=delta_A,
        sigma_min=sigma_min,
        inverse_norm=float(inverse_norm),
        product=float(product),
        D_tilde=D_tilde,
        L=L,
        A_tilde=A_tilde,
    )


@dataclass(frozen=True)
class ConstitutivePortErrorCertificate:
    """Perturbation certificate from conductivity-series error to multiport Z."""

    constitutive_relative_bound: float
    approximate_relative_factor: float
    operator_perturbation_bound: float
    approximate_minimum_singular_value: float
    approximate_inverse_norm: float
    inverse_perturbation_product: float
    inverse_difference_bound: float
    impedance_spectral_norm_bound: float
    resistance_element_bound: float
    inductance_element_bound: float
    certified: bool


def certify_constitutive_multiport_error(
    problem,
    ports,
    thermal_state: np.ndarray,
) -> ConstitutivePortErrorCertificate:
    """Propagate the problem's certified conductivity-series remainder to Z/R/L/M."""

    data = _constitutive_perturbation_data(problem, thermal_state)
    omega = float(problem.omega)
    if data.product < 1.0:
        inverse_difference = float(
            data.inverse_norm * data.inverse_norm * data.delta_A / (1.0 - data.product)
        )
        B = np.asarray(ports.coordinate_rhs, dtype=complex)
        impedance_bound = float(
            omega * (np.linalg.norm(B, ord=2) ** 2) * inverse_difference
        )
        certified = True
    else:
        inverse_difference = float("inf")
        impedance_bound = float("inf")
        certified = False

    return ConstitutivePortErrorCertificate(
        constitutive_relative_bound=data.eps,
        approximate_relative_factor=data.relative_factor,
        operator_perturbation_bound=data.delta_A,
        approximate_minimum_singular_value=data.sigma_min,
        approximate_inverse_norm=data.inverse_norm,
        inverse_perturbation_product=data.product,
        inverse_difference_bound=inverse_difference,
        impedance_spectral_norm_bound=impedance_bound,
        resistance_element_bound=impedance_bound,
        inductance_element_bound=impedance_bound / omega,
        certified=certified,
    )


@dataclass(frozen=True)
class ConstitutiveHeatSourceErrorCertificate:
    """Certified error of the projected Joule heat source caused by material series truncation."""

    constitutive_relative_bound: float
    operator_perturbation_bound: float
    inverse_perturbation_product: float
    approximate_state_norm: float
    electromagnetic_state_error_bound: float
    component_error_bounds: np.ndarray
    heat_source_vector_error_bound: float
    certified: bool


def certify_constitutive_heat_source_error(
    problem,
    thermal_state: np.ndarray,
    *,
    rhs: np.ndarray | None = None,
) -> ConstitutiveHeatSourceErrorCertificate:
    """Propagate conductivity-series error to q_em,r component by component.

    Let x be the exact electromagnetic state and x_tilde the state obtained from
    the certified constitutive series. With delta_A bounded and

        ||A_tilde^{-1}|| delta_A < 1,

    the Banach perturbation theorem gives a state error bound. For each thermal
    test function phi_j, the conductivity-operator remainder obeys

        ||delta H_j|| <= 0.5 * eps/(1-eps) * ||phi_j||_inf
                         * ||L^H M_sigma_tilde L||_2.

    Combining the state and loss-operator perturbations yields a deterministic
    bound on |q_j-q_tilde_j|. The infinity norm of phi_j is bounded by its P1
    vertex values, exactly on each tetrahedron.
    """

    if not hasattr(problem, "thermal_test_local"):
        raise TypeError("heat-source constitutive certification requires tetrahedral P1 thermal tests")

    a = np.asarray(thermal_state, dtype=float)
    data = _constitutive_perturbation_data(problem, a)
    drive = np.asarray(problem.b if rhs is None else rhs, dtype=complex)
    if drive.shape != (problem.n_em,):
        raise ValueError("rhs dimension mismatch")

    if data.product >= 1.0:
        component = np.full(problem.n_thermal, np.inf, dtype=float)
        return ConstitutiveHeatSourceErrorCertificate(
            constitutive_relative_bound=data.eps,
            operator_perturbation_bound=data.delta_A,
            inverse_perturbation_product=data.product,
            approximate_state_norm=float("inf"),
            electromagnetic_state_error_bound=float("inf"),
            component_error_bounds=component,
            heat_source_vector_error_bound=float("inf"),
            certified=False,
        )

    x_tilde = np.linalg.solve(data.A_tilde, drive)
    x_norm = float(np.linalg.norm(x_tilde))
    exact_inverse_bound = data.inverse_norm / (1.0 - data.product)
    state_error = float(exact_inverse_bound * data.delta_A * x_norm)

    # L^H S_tilde L = omega^2 D_tilde.
    omega = float(problem.omega)
    total_loss_coordinate_norm = float(
        omega * omega * np.linalg.norm(data.D_tilde, ord=2)
    )
    component_bounds = np.zeros(problem.n_thermal, dtype=float)

    for j in range(problem.n_thermal):
        H_tilde = np.asarray(problem.loss_operator(j, a), dtype=complex)
        test_inf = float(np.max(np.abs(problem.thermal_test_local[j])))
        delta_H = float(
            0.5
            * data.relative_factor
            * test_inf
            * total_loss_coordinate_norm
        )
        H_exact_norm_bound = float(np.linalg.norm(H_tilde, ord=2) + delta_H)
        state_part = H_exact_norm_bound * state_error * (2.0 * x_norm + state_error)
        operator_part = delta_H * x_norm * x_norm
        component_bounds[j] = state_part + operator_part

    return ConstitutiveHeatSourceErrorCertificate(
        constitutive_relative_bound=data.eps,
        operator_perturbation_bound=data.delta_A,
        inverse_perturbation_product=data.product,
        approximate_state_norm=x_norm,
        electromagnetic_state_error_bound=state_error,
        component_error_bounds=component_bounds,
        heat_source_vector_error_bound=float(np.linalg.norm(component_bounds)),
        certified=True,
    )
