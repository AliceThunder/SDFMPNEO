from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from sdfmpneo.em.energy_solver import apsi_physical_energy_metric
from sdfmpneo.em.riesz_action import instantiate_riesz_action
from sdfmpneo.em.sparse_solver import CertifiedEnergySparseApsiSolver

from .em_domain import ParameterBox
from .nonlinear_em_domain import _material_energy_ratios


_BETA = CertifiedEnergySparseApsiSolver.COERCIVITY_LOWER_BOUND


@dataclass(frozen=True)
class ElectroThermalDomainBounds:
    thermal_box: ParameterBox
    operating_box: ParameterBox
    source_dual_energy_bound: float
    heat_source_jacobian_entry_bounds: np.ndarray
    heat_source_jacobian_norm_bound: float
    explicit_operating_jacobian_entry_bounds: np.ndarray
    explicit_operating_jacobian_norm_bound: float
    contraction_margin_lower_bound: float

    @property
    def contractive(self) -> bool:
        return self.contraction_margin_lower_bound > 0.0


def certify_electrothermal_domain_bounds(
    vector_field,
    *,
    thermal_lower: np.ndarray,
    thermal_upper: np.ndarray,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
) -> ElectroThermalDomainBounds:
    """Continuous-domain energy bounds for the closed reduced dynamics.

    The proof uses only the physical coercivity ``beta_H>=1/sqrt(2)``, certified
    nonlinear material ratios, source dual-energy norms and the exact quadratic
    structure of projected Joule heat.  It does not sample a Hessian.
    """

    if vector_field.rhs_map is None:
        raise ValueError("continuous operating-domain bound requires rhs_map")
    problem = vector_field.em_model.problem
    model = vector_field.em_model
    tbox = ParameterBox(thermal_lower, thermal_upper)
    ubox = ParameterBox(operating_lower, operating_upper)
    if tbox.lower.shape != (problem.n_thermal,):
        raise ValueError("thermal box dimension mismatch")
    if ubox.lower.shape != (vector_field.n_operating,):
        raise ValueError("operating box dimension mismatch")

    center = tbox.midpoint
    mu, _, theta = _material_energy_ratios(problem, tbox)
    if mu <= 0.0:
        raise ValueError("material law cannot certify positive energy over the thermal box")

    A = sp.csr_matrix(problem.operator_sparse(center), dtype=complex)
    H = apsi_physical_energy_metric(A)
    action = instantiate_riesz_action(model.riesz_action_factory, H, state=center)
    uc = ubox.midpoint
    rhs_center = vector_field.rhs_map.evaluate(uc)
    B = vector_field.rhs_map.matrix
    center_dual = action.decide_dual_norm(rhs_center, threshold=0.0).result.dual_norm_upper_bound
    direction_dual = np.array(
        [
            action.decide_dual_norm(B[:, k], threshold=0.0).result.dual_norm_upper_bound
            for k in range(B.shape[1])
        ],
        dtype=float,
    )
    source_center_metric = float(center_dual + np.dot(ubox.halfwidth, direction_dual))
    source_bound = float(source_center_metric / np.sqrt(mu))
    x_bound = source_bound / _BETA

    mode_inf = np.max(np.abs(np.asarray(problem.thermal_test_local, dtype=float)), axis=(1, 2))
    omega = float(problem.omega)
    J = np.empty((problem.n_thermal, problem.n_thermal), dtype=float)
    for j in range(problem.n_thermal):
        for k in range(problem.n_thermal):
            # q_j=x^H H_j x; H_j <= (omega/2)||phi_j||_inf H.
            J[j, k] = (
                omega
                * mode_inf[j]
                * theta[k]
                * (1.0 / _BETA + 0.5)
                * x_bound**2
            )
    Jnorm = float(np.linalg.norm(J, ord="fro"))

    Ju = np.empty((problem.n_thermal, vector_field.n_operating), dtype=float)
    for j in range(problem.n_thermal):
        for k in range(vector_field.n_operating):
            # Explicit source variation: dx=A^-1 db and d q=2 Re(dx^H H_j x).
            Ju[j, k] = omega * mode_inf[j] * source_bound * (direction_dual[k] / np.sqrt(mu)) / (_BETA**2)
    Junorm = float(np.linalg.norm(Ju, ord="fro"))

    lambda_min = float(np.min(np.asarray(vector_field.thermal_model.lambdas, dtype=float)))
    kappa = float(lambda_min - Jnorm)
    return ElectroThermalDomainBounds(
        thermal_box=tbox,
        operating_box=ubox,
        source_dual_energy_bound=source_bound,
        heat_source_jacobian_entry_bounds=J,
        heat_source_jacobian_norm_bound=Jnorm,
        explicit_operating_jacobian_entry_bounds=Ju,
        explicit_operating_jacobian_norm_bound=Junorm,
        contraction_margin_lower_bound=kappa,
    )
