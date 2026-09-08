from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class CertifiedGeometryHeatFeedbackBound:
    """Full-domain bound for the symmetric Joule feedback in the M metric.

    A certified instance asserts, for every declared ``(G,a,U)`` in its domain,

        x^T sym(J_q(G,a,U)) x <= gamma_q x^T M_r(G) x.

    This object is deliberately separate from sampled Jacobian diagnostics.  A
    finite number is not accepted as a proof unless ``certified=True`` and the
    provenance states how the continuous-domain bound was obtained.
    """

    generalized_symmetric_upper_bound: float
    certified: bool
    provenance: str

    def __post_init__(self) -> None:
        value = float(self.generalized_symmetric_upper_bound)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError("heat-feedback bound must be finite and non-negative")
        if self.certified and not str(self.provenance).strip():
            raise ValueError("certified heat-feedback bound requires provenance")
        object.__setattr__(self, "generalized_symmetric_upper_bound", value)


@dataclass(frozen=True)
class ContinuousGeometryContractivityCertificate:
    status: Literal["certified", "indeterminate"]
    thermal_diffusion_margin_lower_bound: float
    heat_feedback_upper_bound: float | None
    contraction_margin_lower_bound: float | None
    center_generalized_thermal_decay: float
    p1_stiffness_ratio_lower: float
    p1_mass_ratio_upper: float
    continuous_geometry_box: bool
    provenance: str

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def certify_geometry_contractivity(
    model,
    *,
    heat_feedback: CertifiedGeometryHeatFeedbackBound | None = None,
) -> ContinuousGeometryContractivityCertificate:
    """Compose a rigorous full-geometry-box M-energy contraction certificate.

    For a fixed geometry, the reduced dynamics satisfy

        M a' = -K a + q(a,U),
        sym(M J_F) = -K + sym(J_q).

    The affine geometry chart proves on the whole box

        K(G) >= alpha_K K_c,   M(G) <= beta_M M_c.

    Hence the purely diffusive generalized decay obeys

        lambda_min(K(G),M(G))
        >= (alpha_K/beta_M) lambda_min(K_c,M_c).

    A caller may then supply an independently certified full-domain bound

        sym(J_q) <= gamma_q M(G).

    The resulting contractivity margin is at least ``lambda_diff-gamma_q``.
    Without that second proof the result is intentionally indeterminate; sampled
    positive margins are never silently upgraded to a continuous-domain claim.
    """

    center = 0.5 * (np.asarray(model.lower, float) + np.asarray(model.upper, float))
    context = model.context(center)
    M = np.asarray(context.M, float)
    K = np.asarray(context.K, float)
    M = 0.5 * (M + M.T)
    K = 0.5 * (K + K.T)
    center_values = scipy.linalg.eigvalsh(K, M, check_finite=True)
    center_decay = float(center_values[0])
    if center_decay <= 0.0:
        raise np.linalg.LinAlgError("center reduced thermal operator is not strictly dissipative")

    alpha_k = float(model.certificate.p1_stiffness_ratio[0])
    beta_m = float(model.certificate.p1_mass_ratio[1])
    if alpha_k <= 0.0 or beta_m <= 0.0 or not np.isfinite(alpha_k + beta_m):
        raise ValueError("geometry chart does not provide positive finite thermal form ratios")
    diffusion = float(np.nextafter((alpha_k / beta_m) * center_decay, -np.inf))

    base = dict(
        thermal_diffusion_margin_lower_bound=diffusion,
        center_generalized_thermal_decay=center_decay,
        p1_stiffness_ratio_lower=alpha_k,
        p1_mass_ratio_upper=beta_m,
        continuous_geometry_box=bool(model.certificate.certified_nondegenerate),
    )
    if heat_feedback is None or not heat_feedback.certified:
        provenance = "thermal geometry-form bound proved; continuous Joule-feedback proof missing"
        if heat_feedback is not None and str(heat_feedback.provenance).strip():
            provenance += f"; supplied non-certified feedback evidence: {heat_feedback.provenance}"
        return ContinuousGeometryContractivityCertificate(
            status="indeterminate",
            heat_feedback_upper_bound=None if heat_feedback is None else heat_feedback.generalized_symmetric_upper_bound,
            contraction_margin_lower_bound=None,
            provenance=provenance,
            **base,
        )

    gamma = heat_feedback.generalized_symmetric_upper_bound
    kappa = float(np.nextafter(diffusion - gamma, -np.inf))
    if kappa <= 0.0:
        return ContinuousGeometryContractivityCertificate(
            status="indeterminate",
            heat_feedback_upper_bound=gamma,
            contraction_margin_lower_bound=kappa,
            provenance=(
                "continuous diffusion and Joule-feedback bounds are certified, but their "
                "difference does not prove positive contractivity; " + heat_feedback.provenance
            ),
            **base,
        )
    return ContinuousGeometryContractivityCertificate(
        status="certified",
        heat_feedback_upper_bound=gamma,
        contraction_margin_lower_bound=kappa,
        provenance=(
            "full geometry-box thermal form ratios + certified generalized Joule-feedback bound; "
            + heat_feedback.provenance
        ),
        **base,
    )
