from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstitutivePortErrorCertificate:
    """Perturbation certificate from conductivity-series error to multiport Z.

    Suppose the certified constitutive approximation obeys pointwise

        |sigma_tilde - sigma| <= eps_sigma * sigma,   eps_sigma < 1.

    Then

        |sigma_tilde - sigma| <= eps_sigma/(1-eps_sigma) * sigma_tilde.

    For the reciprocal A-psi system A = K + j*omega*Q^T M_sigma Q this yields a
    computable operator perturbation bound. A standard inverse perturbation
    argument then gives a bound for

        Z = j*omega*B^T A^{-1} B.

    No statistical or empirical calibration enters the certificate.
    """

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
    """Propagate the problem's constitutive series remainder to Z/R/L/M.

    This function currently targets `NonlinearTetrahedralApsiProblem`, whose
    reciprocal copper expansion provides a certified relative error. Exact
    constant/affine regions contribute zero approximation error, so using the
    maximum reciprocal-region bound over the total conductivity is conservative.
    """

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

    # E = -j*omega Q x, hence Q = E_extraction/(-j*omega).
    Q = L / (-1j * omega)
    D_tilde = Q.conj().T @ S_tilde @ Q
    D_tilde = 0.5 * (D_tilde + D_tilde.conj().T)

    relative_factor = 0.0 if eps == 0.0 else eps / (1.0 - eps)
    delta_A = float(omega * relative_factor * np.linalg.norm(D_tilde, ord=2))

    singular_values = np.linalg.svd(A_tilde, compute_uv=False)
    sigma_min = float(np.min(singular_values))
    if sigma_min <= 0.0:
        inverse_norm = float("inf")
    else:
        inverse_norm = 1.0 / sigma_min
    product = inverse_norm * delta_A

    if product < 1.0:
        inverse_difference = float(
            inverse_norm * inverse_norm * delta_A / (1.0 - product)
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

    # Every matrix element is bounded by the spectral norm. Re/Im cannot exceed
    # the complex element modulus. L = Im(Z)/omega.
    resistance_bound = impedance_bound
    inductance_bound = impedance_bound / omega

    return ConstitutivePortErrorCertificate(
        constitutive_relative_bound=eps,
        approximate_relative_factor=float(relative_factor),
        operator_perturbation_bound=delta_A,
        approximate_minimum_singular_value=sigma_min,
        approximate_inverse_norm=float(inverse_norm),
        inverse_perturbation_product=float(product),
        inverse_difference_bound=inverse_difference,
        impedance_spectral_norm_bound=impedance_bound,
        resistance_element_bound=resistance_bound,
        inductance_element_bound=inductance_bound,
        certified=certified,
    )
