from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from sdfmpneo.em.constitutive import (
    AffineConductivity,
    ConstantConductivity,
    ReciprocalLinearResistivity,
)
from sdfmpneo.em.energy_solver import apsi_physical_energy_metric
from sdfmpneo.em.riesz_action import instantiate_riesz_action
from sdfmpneo.em.sparse_solver import CertifiedEnergySparseApsiSolver

from .em_domain import BoxResidualBound, ContinuousEMResidualCertificate, ParameterBox


_BETA = CertifiedEnergySparseApsiSolver.COERCIVITY_LOWER_BOUND


def _temperature_box(problem, box: ParameterBox) -> tuple[np.ndarray, np.ndarray]:
    if box.lower.shape != (problem.n_thermal,):
        raise ValueError("thermal parameter box dimension mismatch")
    lo = np.asarray(problem.temperature_reference_local, dtype=float).copy()
    hi = lo.copy()
    for k in range(problem.n_thermal):
        mode = np.asarray(problem.thermal_modes_local[k], dtype=float)
        a = mode * box.lower[k]
        b = mode * box.upper[k]
        lo += np.minimum(a, b)
        hi += np.maximum(a, b)
    return lo, hi


def _material_energy_ratios(
    problem,
    box: ParameterBox,
) -> tuple[float, float, np.ndarray]:
    """Return certified H-box ratios and directional conductivity bounds.

    For every thermal state in the box this routine proves

        mu H(center) <= H(a) <= nu H(center)

    and

        |d D / d a_k| <= theta_k H(center)

    in quadratic-form sense.  Reciprocal copper uses the declared constitutive
    relative remainder budget, so no temperature linearization is introduced.
    """

    center = box.midpoint
    Tcenter = np.asarray(problem.temperature_local(center), dtype=float)
    Tlo, Thi = _temperature_box(problem, box)
    theta = np.zeros(problem.n_thermal, dtype=float)
    lower_ratios: list[float] = []
    upper_ratios: list[float] = []
    eps = float(getattr(problem, "constitutive_relative_error_budget", 0.0))

    for region in problem.conductivity_regions:
        mask = np.asarray(region.mask, dtype=bool)
        for q in np.flatnonzero(mask):
            law = region.law
            center_values = np.asarray(law.evaluate(Tcenter[q]), dtype=float)
            endpoint_values = np.concatenate(
                [
                    np.asarray(law.evaluate(Tlo[q]), dtype=float).ravel(),
                    np.asarray(law.evaluate(Thi[q]), dtype=float).ravel(),
                ]
            )
            if np.max(endpoint_values) == 0.0:
                continue

            rel = eps if isinstance(law, ReciprocalLinearResistivity) else 0.0
            center_lower = (1.0 - rel) * float(np.min(center_values))
            center_upper = (1.0 + rel) * float(np.max(center_values))
            box_lower = (1.0 - rel) * float(np.min(endpoint_values))
            box_upper = (1.0 + rel) * float(np.max(endpoint_values))
            if center_lower <= 0.0 or box_lower <= 0.0:
                return 0.0, float("inf"), np.full(problem.n_thermal, float("inf"))

            lower_ratios.append(box_lower / center_upper)
            upper_ratios.append(box_upper / center_lower)

            if isinstance(law, ConstantConductivity):
                continue
            if isinstance(law, AffineConductivity):
                derivative_temperature = abs(float(law.beta))
            elif isinstance(law, ReciprocalLinearResistivity):
                derivative_temperature = max(
                    float(np.max(np.abs(law.derivative(Tlo[q])))),
                    float(np.max(np.abs(law.derivative(Thi[q])))),
                ) * (1.0 + rel)
            else:
                raise TypeError(
                    f"unsupported certified nonlinear conductivity law: {type(law).__name__}"
                )

            for k in range(problem.n_thermal):
                mode_max = float(np.max(np.abs(problem.thermal_modes_local[k, q])))
                theta[k] = max(
                    theta[k],
                    derivative_temperature * mode_max / center_lower,
                )

    if not lower_ratios:
        return 1.0, 1.0, theta
    mu = min(1.0, min(lower_ratios))
    nu = max(1.0, max(upper_ratios))
    return float(mu), float(nu), theta


def bound_nonlinear_reduced_residual_on_box(model, box: ParameterBox) -> BoxResidualBound:
    """Certify a nonlinear thermal-state box without solution snapshots.

    The estimate combines the exact physical coercivity beta_H>=1/sqrt(2),
    certified material ratios over the box, and analytic conductivity derivative
    bounds.  A finite center evaluation is used only as an a-posteriori anchor;
    the rest of the box is covered by the proved Lipschitz radius.
    """

    problem = model.problem
    if not hasattr(problem, "thermal_modes_local") or not hasattr(problem, "conductivity_regions"):
        raise TypeError("nonlinear thermal-domain certificate requires the tetrahedral material problem")

    center = box.midpoint
    mu, nu, theta = _material_energy_ratios(problem, box)
    center_residual = float(
        model.residual_certificate(center).residual_dual_energy_norm_upper_bound
    )
    if mu <= 0.0 or not np.isfinite(nu):
        return BoxResidualBound(
            box=box,
            center_residual_dual_norm=center_residual,
            residual_upper_bound=float("inf"),
            reduced_stability_lower_bound=0.0,
            directional_lipschitz_bounds=np.full(problem.n_thermal, float("inf")),
        )

    A = sp.csr_matrix(problem.operator_sparse(center), dtype=complex)
    H = apsi_physical_energy_metric(A)
    action = instantiate_riesz_action(model.riesz_action_factory, H, state=center)
    source_dual = float(
        action.decide_dual_norm(problem.b, threshold=0.0).result.dual_norm_upper_bound
    )

    directional = theta * source_dual * (
        1.0 / (_BETA * mu) + nu / (_BETA * _BETA * mu * mu)
    )
    radius = float(np.dot(box.halfwidth, directional))
    upper = (center_residual + radius) / np.sqrt(mu)
    return BoxResidualBound(
        box=box,
        center_residual_dual_norm=center_residual,
        residual_upper_bound=float(np.nextafter(upper, np.inf)),
        reduced_stability_lower_bound=float(_BETA * mu),
        directional_lipschitz_bounds=np.asarray(directional, dtype=float),
    )


def certify_nonlinear_reduced_residual_domain(
    model,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
    work_budget: int,
) -> ContinuousEMResidualCertificate:
    """Branch-and-bound proof over a continuous nonlinear thermal-state box."""

    if tolerance <= 0.0 or work_budget <= 0:
        raise ValueError("tolerance and work_budget must be positive")
    pending = [ParameterBox(lower, upper)]
    processed = 0
    observed = 0.0
    resolved: list[float] = []

    while pending and processed < work_budget:
        box = pending.pop()
        bound = bound_nonlinear_reduced_residual_on_box(model, box)
        processed += 1
        observed = max(observed, bound.center_residual_dual_norm)
        if bound.center_residual_dual_norm > tolerance:
            return ContinuousEMResidualCertificate(
                "violated", float(tolerance), observed,
                max(observed, bound.residual_upper_bound), processed,
                box.midpoint.copy(), len(pending),
            )
        if bound.residual_upper_bound <= tolerance:
            resolved.append(bound.residual_upper_bound)
            continue
        widths = box.halfwidth
        if not np.any(widths > 0.0):
            resolved.append(bound.center_residual_dual_norm)
            continue
        scores = widths * bound.directional_lipschitz_bounds
        dimension = int(np.argmax(scores)) if np.any(np.isfinite(scores)) else int(np.argmax(widths))
        if not np.isfinite(scores[dimension]) or scores[dimension] == 0.0:
            dimension = int(np.argmax(widths))
        left, right = box.split(dimension)
        pending.extend([right, left])

    if not pending:
        return ContinuousEMResidualCertificate(
            "certified", float(tolerance), observed,
            float(max(resolved, default=observed)), processed, None, 0,
        )

    unresolved = [bound_nonlinear_reduced_residual_on_box(model, b).residual_upper_bound for b in pending]
    return ContinuousEMResidualCertificate(
        "indeterminate", float(tolerance), observed,
        float(max(resolved + unresolved, default=float("inf"))),
        processed, None, len(pending),
    )
