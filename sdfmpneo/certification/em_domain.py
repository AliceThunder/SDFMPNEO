from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class ParameterBox:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lo = np.asarray(self.lower, dtype=float)
        hi = np.asarray(self.upper, dtype=float)
        if lo.ndim != 1 or hi.shape != lo.shape:
            raise ValueError("lower/upper must be one-dimensional with equal shape")
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
            raise ValueError("parameter box bounds must be finite")
        if np.any(hi < lo):
            raise ValueError("upper must be >= lower componentwise")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def halfwidth(self) -> np.ndarray:
        return 0.5 * (self.upper - self.lower)

    def split(self, dimension: int) -> tuple["ParameterBox", "ParameterBox"]:
        if not 0 <= dimension < self.lower.size:
            raise ValueError("split dimension out of range")
        mid = self.midpoint[dimension]
        left_upper = self.upper.copy()
        left_upper[dimension] = mid
        right_lower = self.lower.copy()
        right_lower[dimension] = mid
        return ParameterBox(self.lower.copy(), left_upper), ParameterBox(right_lower, self.upper.copy())


@dataclass(frozen=True)
class BoxResidualBound:
    box: ParameterBox
    center_residual_dual_norm: float
    residual_upper_bound: float
    reduced_stability_lower_bound: float
    directional_lipschitz_bounds: np.ndarray


@dataclass(frozen=True)
class ContinuousEMResidualCertificate:
    status: Literal["certified", "violated", "indeterminate"]
    tolerance: float
    observed_lower_bound: float
    global_upper_bound: float
    boxes_processed: int
    violating_state: np.ndarray | None
    unresolved_boxes: int

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def _spectral_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(A, dtype=complex), ord=2))


def bound_affine_reduced_residual_on_box(model, box: ParameterBox) -> BoxResidualBound:
    """Bound sup ||b-A(a)V c(a)||_{H^-1} on one affine parameter box.

    The proof uses

        A(a) = A(a_c) + sum_k delta_k A_k,

    Weyl's singular-value perturbation bound for the reduced operator, and an
    explicit bound for the derivative of the Galerkin residual. No parameter
    samples other than the box center are promoted to a certificate.
    """

    problem = model.problem
    if not hasattr(problem, "A_state"):
        raise TypeError("continuous affine certificate requires an affine ParametricEMProblem")
    if box.lower.shape != (problem.n_thermal,):
        raise ValueError("parameter box dimension mismatch")

    # A full-coordinate basis represents every Galerkin solution exactly.
    if model.V.shape[1] == problem.n_em:
        return BoxResidualBound(
            box=box,
            center_residual_dual_norm=0.0,
            residual_upper_bound=0.0,
            reduced_stability_lower_bound=float("inf"),
            directional_lipschitz_bounds=np.zeros(problem.n_thermal),
        )

    center = box.midpoint
    halfwidth = box.halfwidth
    V = np.asarray(model.V, dtype=complex)
    A = np.asarray(problem.operator(center), dtype=complex)
    Ak = np.asarray(problem.A_state, dtype=complex)
    Ar = V.conj().T @ A @ V
    Ark = np.stack([V.conj().T @ block @ V for block in Ak], axis=0)

    beta_center = float(np.min(scipy.linalg.svdvals(Ar)))
    beta_loss = float(sum(halfwidth[k] * _spectral_norm(Ark[k]) for k in range(problem.n_thermal)))
    beta = beta_center - beta_loss

    center_residual = float(model.residual_dual_norm(center))
    if beta <= 0.0:
        return BoxResidualBound(
            box=box,
            center_residual_dual_norm=center_residual,
            residual_upper_bound=float("inf"),
            reduced_stability_lower_bound=beta,
            directional_lipschitz_bounds=np.full(problem.n_thermal, float("inf")),
        )

    br = V.conj().T @ problem.b
    br_norm = float(np.linalg.norm(br))
    c_bound = br_norm / beta

    L = model.riesz.L
    L_inv_A_V = scipy.linalg.solve_triangular(L, A @ V, lower=True, check_finite=True)
    L_inv_Ak_V = np.stack(
        [
            scipy.linalg.solve_triangular(L, block @ V, lower=True, check_finite=True)
            for block in Ak
        ],
        axis=0,
    )

    A_V_bound = _spectral_norm(L_inv_A_V) + sum(
        halfwidth[k] * _spectral_norm(L_inv_Ak_V[k])
        for k in range(problem.n_thermal)
    )

    directional = np.zeros(problem.n_thermal, dtype=float)
    for k in range(problem.n_thermal):
        dc_bound = _spectral_norm(Ark[k]) * br_norm / (beta * beta)
        directional[k] = _spectral_norm(L_inv_Ak_V[k]) * c_bound + A_V_bound * dc_bound

    radius_bound = float(np.dot(halfwidth, directional))
    return BoxResidualBound(
        box=box,
        center_residual_dual_norm=center_residual,
        residual_upper_bound=center_residual + radius_bound,
        reduced_stability_lower_bound=beta,
        directional_lipschitz_bounds=directional,
    )


def certify_affine_reduced_residual_domain(
    model,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
    work_budget: int,
) -> ContinuousEMResidualCertificate:
    """Branch-and-bound continuous-domain residual certification.

    `work_budget` is only a computational safety limit. Exhausting it returns
    `indeterminate`; it can never turn an unresolved domain into a certificate.
    Boxes are split along the direction with the largest proven contribution
    h_k L_k to the residual upper bound. If the reduced stability bound is not
    positive, the widest nonzero parameter direction is split instead.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if work_budget <= 0:
        raise ValueError("work_budget must be positive")

    pending = [ParameterBox(lower, upper)]
    processed = 0
    observed = 0.0
    certified_leaf_uppers: list[float] = []

    while pending and processed < work_budget:
        box = pending.pop()
        bound = bound_affine_reduced_residual_on_box(model, box)
        processed += 1
        observed = max(observed, bound.center_residual_dual_norm)

        if bound.center_residual_dual_norm > tolerance:
            return ContinuousEMResidualCertificate(
                status="violated",
                tolerance=float(tolerance),
                observed_lower_bound=observed,
                global_upper_bound=max(observed, bound.residual_upper_bound),
                boxes_processed=processed,
                violating_state=box.midpoint.copy(),
                unresolved_boxes=len(pending),
            )

        if bound.residual_upper_bound <= tolerance:
            certified_leaf_uppers.append(bound.residual_upper_bound)
            continue

        widths = box.halfwidth
        if not np.any(widths > 0):
            # A point box with center below tolerance is fully resolved.
            certified_leaf_uppers.append(bound.center_residual_dual_norm)
            continue

        if np.all(np.isfinite(bound.directional_lipschitz_bounds)):
            scores = widths * bound.directional_lipschitz_bounds
            dimension = int(np.argmax(scores))
            if scores[dimension] == 0.0:
                dimension = int(np.argmax(widths))
        else:
            dimension = int(np.argmax(widths))

        left, right = box.split(dimension)
        pending.extend([right, left])

    if not pending:
        upper_bound = max(certified_leaf_uppers, default=observed)
        return ContinuousEMResidualCertificate(
            status="certified",
            tolerance=float(tolerance),
            observed_lower_bound=observed,
            global_upper_bound=float(upper_bound),
            boxes_processed=processed,
            violating_state=None,
            unresolved_boxes=0,
        )

    unresolved_upper = []
    for box in pending:
        unresolved_upper.append(bound_affine_reduced_residual_on_box(model, box).residual_upper_bound)
    upper_bound = max(certified_leaf_uppers + unresolved_upper, default=float("inf"))
    return ContinuousEMResidualCertificate(
        status="indeterminate",
        tolerance=float(tolerance),
        observed_lower_bound=observed,
        global_upper_bound=float(upper_bound),
        boxes_processed=processed,
        violating_state=None,
        unresolved_boxes=len(pending),
    )
