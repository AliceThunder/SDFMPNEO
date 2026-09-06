from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from sdfmpneo.em.energy_solver import apsi_physical_energy_metric
from sdfmpneo.em.riesz_action import instantiate_riesz_action
from sdfmpneo.em.sparse_solver import CertifiedEnergySparseApsiSolver
from sdfmpneo.spatial import AffineTetrahedralGeometryChart

from .em_domain import ParameterBox


_BETA = CertifiedEnergySparseApsiSolver.COERCIVITY_LOWER_BOUND


@dataclass(frozen=True)
class GeometryEMBoxBound:
    box: ParameterBox
    maximum_relative_jacobian_perturbation: float
    center_state_error_bound: float
    geometry_drift_bound: float
    family_state_error_bound: float
    directional_geometry_scores: np.ndarray


@dataclass(frozen=True)
class ContinuousGeometryEMCertificate:
    status: Literal["certified", "violated", "indeterminate"]
    tolerance: float
    global_state_error_bound: float
    boxes_processed: int
    unresolved_boxes: int
    violating_geometry: np.ndarray | None

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def bound_geometry_em_family_on_box(
    model_factory: Callable[[np.ndarray], object],
    chart: AffineTetrahedralGeometryChart,
    box: ParameterBox,
    *,
    thermal_state: np.ndarray,
    rhs: np.ndarray | None = None,
) -> GeometryEMBoxBound:
    """Bound a canonical EM-ROM family over one affine geometry chart box.

    ``model_factory(mu)`` must return the same canonical reduced space/operator
    family assembled at ``mu``. The impressed/terminal coordinate source is
    assumed work-conjugate and topologically transported by the chart.

    Nedelec curl and conductive mass forms obey the chart's exact quadratic-form
    ratios. If ``delta_A`` is the resulting normalized operator perturbation,
    resolvent perturbation plus Cea's lemma gives a full family state bound from
    the center ROM residual without geometry samples inside the box.
    """

    if box.lower.shape != (chart.n_parameters,):
        raise ValueError("geometry box/chart dimension mismatch")
    geometry = chart.certify_box(box.lower, box.upper)
    if not geometry.certified_nondegenerate:
        return GeometryEMBoxBound(
            box, geometry.maximum_relative_jacobian_perturbation,
            float("inf"), float("inf"), float("inf"),
            box.halfwidth * geometry.directional_distortion_bounds,
        )

    model = model_factory(box.midpoint)
    a = np.asarray(thermal_state, dtype=float)
    source = np.asarray(model.problem.b if rhs is None else rhs, dtype=complex)
    if source.shape != (model.problem.n_em,):
        raise ValueError("rhs dimension mismatch")

    A = model.problem.operator_sparse(a) if hasattr(model.problem, "operator_sparse") else model.problem.operator(a)
    H = apsi_physical_energy_metric(A)
    action = instantiate_riesz_action(model.riesz_action_factory, H, state=a)
    source_dual = float(action.decide_dual_norm(source, threshold=0.0).result.dual_norm_upper_bound)
    center_residual = model.residual_certificate_for_rhs(a, source)
    center_error = float(center_residual.energy_state_error_bound)

    k_lo, k_hi = geometry.hcurl_curl_ratio
    d_lo, d_hi = geometry.hcurl_mass_ratio
    mu = min(k_lo, d_lo)
    nu = max(k_hi, d_hi)
    if mu <= 0.0 or not np.isfinite(nu):
        drift = family = float("inf")
    else:
        delta_k = max(1.0 - k_lo, k_hi - 1.0)
        delta_d = max(1.0 - d_lo, d_hi - 1.0)
        delta_A = delta_k + delta_d
        drift = delta_A * source_dual / (_BETA * _BETA * mu)
        # Continuity of K+iD in the H=K+D norm is <=1.
        family = np.sqrt(nu) * (center_error + drift) / _BETA

    return GeometryEMBoxBound(
        box=box,
        maximum_relative_jacobian_perturbation=geometry.maximum_relative_jacobian_perturbation,
        center_state_error_bound=center_error,
        geometry_drift_bound=float(drift),
        family_state_error_bound=float(np.nextafter(family, np.inf)),
        directional_geometry_scores=box.halfwidth * geometry.directional_distortion_bounds,
    )


def certify_geometry_em_family_domain(
    model_factory: Callable[[np.ndarray], object],
    chart: AffineTetrahedralGeometryChart,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    thermal_state: np.ndarray,
    tolerance: float,
    work_budget: int,
    rhs: np.ndarray | None = None,
) -> ContinuousGeometryEMCertificate:
    if tolerance <= 0.0 or work_budget <= 0:
        raise ValueError("tolerance and work_budget must be positive")
    pending = [ParameterBox(lower, upper)]
    processed = 0
    resolved: list[float] = []

    while pending and processed < work_budget:
        box = pending.pop()
        try:
            bound = bound_geometry_em_family_on_box(
                model_factory, chart, box, thermal_state=thermal_state, rhs=rhs
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            bound = None
        processed += 1
        if bound is not None and bound.family_state_error_bound <= tolerance:
            resolved.append(bound.family_state_error_bound)
            continue
        widths = box.halfwidth
        if not np.any(widths > 0.0):
            return ContinuousGeometryEMCertificate(
                "violated" if bound is not None and np.isfinite(bound.family_state_error_bound) else "indeterminate",
                float(tolerance),
                float("inf") if bound is None else bound.family_state_error_bound,
                processed,
                len(pending),
                box.midpoint.copy() if bound is not None else None,
            )
        if bound is None:
            dimension = int(np.argmax(widths))
        else:
            dimension = int(np.argmax(bound.directional_geometry_scores))
            if bound.directional_geometry_scores[dimension] <= 0.0:
                dimension = int(np.argmax(widths))
        left, right = box.split(dimension)
        pending.extend([right, left])

    if not pending:
        return ContinuousGeometryEMCertificate(
            "certified", float(tolerance), float(max(resolved, default=0.0)), processed, 0, None
        )
    unresolved = []
    for box in pending:
        try:
            unresolved.append(
                bound_geometry_em_family_on_box(
                    model_factory, chart, box, thermal_state=thermal_state, rhs=rhs
                ).family_state_error_bound
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            unresolved.append(float("inf"))
    return ContinuousGeometryEMCertificate(
        "indeterminate", float(tolerance),
        float(max(resolved + unresolved, default=float("inf"))),
        processed, len(pending), None,
    )
