from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ..em.reduced import RieszFactor


@dataclass(frozen=True)
class MultiPortOutputCertificate:
    electromagnetic_stability: float
    rhs_dual_norms: np.ndarray
    residual_dual_norms: np.ndarray
    impedance_abs_error_bound: np.ndarray
    resistance_abs_error_bound: np.ndarray
    inductance_abs_error_bound: np.ndarray

    @property
    def maximum_impedance_error_bound(self) -> float:
        return float(np.max(self.impedance_abs_error_bound))


def electromagnetic_stability_constant(problem, thermal_state: np.ndarray) -> float:
    """Compute beta = sigma_min(H^-1/2 A H^-1/2) at one state.

    With H=L L^H, the transformed operator is L^-1 A L^-H and

        ||e||_H <= ||r||_(H^-1) / beta.
    """

    a = np.asarray(thermal_state, dtype=float)
    A = np.asarray(problem.operator(a), dtype=complex)
    riesz = RieszFactor.build(problem.H_metric)
    n = A.shape[0]
    L_inv_H = scipy.linalg.solve_triangular(
        riesz.L.conj().T,
        np.eye(n, dtype=complex),
        lower=False,
        check_finite=True,
    )
    transformed = scipy.linalg.solve_triangular(
        riesz.L,
        A @ L_inv_H,
        lower=True,
        check_finite=True,
    )
    beta = float(np.min(scipy.linalg.svdvals(transformed)))
    if beta <= 0.0:
        raise np.linalg.LinAlgError("electromagnetic operator has no positive stability constant")
    return beta


def certify_multiport_impedance(
    problem,
    ports,
    thermal_state: np.ndarray,
    reduced_basis: np.ndarray,
) -> MultiPortOutputCertificate:
    """Propagate reduced field residuals to Z/R/L entrywise error bounds.

    For unit port RHS b_i and approximate field state x_j^r,

        |Delta Z_ij|
        = omega |b_i^T (x_j-x_j^r)|
        <= omega ||b_i||_(H^-1) ||r_j||_(H^-1) / beta.

    Port source cochains are real, so the reciprocal bilinear observation
    b_i^T x is bounded by the same H/H^-1 duality pairing.
    """

    a = np.asarray(thermal_state, dtype=float)
    V = np.asarray(reduced_basis, dtype=complex)
    A = np.asarray(problem.operator(a), dtype=complex)
    B = np.asarray(ports.coordinate_rhs, dtype=complex)
    if V.ndim != 2 or V.shape[0] != problem.n_em:
        raise ValueError("reduced_basis shape mismatch")
    if B.shape[0] != problem.n_em:
        raise ValueError("port/problem coordinate mismatch")

    beta = electromagnetic_stability_constant(problem, a)
    riesz = RieszFactor.build(problem.H_metric)

    Ar = V.conj().T @ A @ V
    Br = V.conj().T @ B
    Xr = V @ scipy.linalg.solve(Ar, Br, assume_a="gen")
    residual = B - A @ Xr

    rhs_norms = np.array([riesz.dual_norm(B[:, i]) for i in range(B.shape[1])])
    residual_norms = np.array([riesz.dual_norm(residual[:, j]) for j in range(B.shape[1])])

    z_bound = ports.omega * np.outer(rhs_norms, residual_norms) / beta
    return MultiPortOutputCertificate(
        electromagnetic_stability=beta,
        rhs_dual_norms=rhs_norms,
        residual_dual_norms=residual_norms,
        impedance_abs_error_bound=z_bound,
        resistance_abs_error_bound=z_bound.copy(),
        inductance_abs_error_bound=z_bound / ports.omega,
    )
