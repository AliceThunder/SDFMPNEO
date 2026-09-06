from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .analytic_residual_domain import _series_sup_abs
from .em_domain import ParameterBox


@dataclass(frozen=True)
class GeometryPhysicalLipschitzProof:
    """Certified physical vector-field bounds on one ``[G,a,U]`` box.

    All entries must be theorem-derived upper/lower bounds on the declared box.
    ``geometry_jacobian_entry_bounds[i,k]`` bounds ``|dF_i/dG_k|``;
    ``state_jacobian_norm_bound`` bounds ``||dF/da||_2`` and
    ``operating_jacobian_entry_bounds[i,k]`` bounds ``|dF_i/dU_k|``.
    The class stores proof data only; it never estimates these constants from
    finite differences or sampled trajectories.
    """

    geometry_jacobian_entry_bounds: np.ndarray
    state_jacobian_norm_bound: float
    operating_jacobian_entry_bounds: np.ndarray
    contraction_margin_lower_bound: float
    certified: bool
    provenance: str

    def __post_init__(self) -> None:
        G = np.asarray(self.geometry_jacobian_entry_bounds, dtype=float)
        U = np.asarray(self.operating_jacobian_entry_bounds, dtype=float)
        L = float(self.state_jacobian_norm_bound)
        kappa = float(self.contraction_margin_lower_bound)
        if G.ndim != 2 or U.ndim != 2 or G.shape[0] != U.shape[0]:
            raise ValueError("physical Jacobian bound matrices must share output dimension")
        if np.any(G < 0.0) or np.any(U < 0.0) or np.any(~np.isfinite(G)) or np.any(~np.isfinite(U)):
            raise ValueError("Jacobian entry bounds must be finite and non-negative")
        if L < 0.0 or not np.isfinite(L) or not np.isfinite(kappa):
            raise ValueError("state Jacobian and contraction bounds must be finite")
        if self.certified and not str(self.provenance).strip():
            raise ValueError("certified geometry physical proof requires provenance")
        object.__setattr__(self, "geometry_jacobian_entry_bounds", G)
        object.__setattr__(self, "operating_jacobian_entry_bounds", U)
        object.__setattr__(self, "state_jacobian_norm_bound", L)
        object.__setattr__(self, "contraction_margin_lower_bound", kappa)


@dataclass(frozen=True)
class GeometryAnalyticResidualBoxBound:
    box: ParameterBox
    center_residual_norm: float
    residual_upper_bound: float
    state_variation_bounds: np.ndarray
    derivative_variation_bounds: np.ndarray
    directional_contributions: np.ndarray
    contraction_margin_lower_bound: float
    certified_physics: bool


@dataclass(frozen=True)
class ContinuousGeometryAnalyticResidualCertificate:
    status: Literal["certified", "violated", "indeterminate"]
    tolerance: float
    maximum_residual_bound: float
    boxes_processed: int
    unresolved_boxes: int
    violating_point: np.ndarray | None

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def bound_geometry_analytic_residual_on_box(
    operator,
    box: ParameterBox,
    physical_proof: GeometryPhysicalLipschitzProof,
) -> GeometryAnalyticResidualBoxBound:
    graph = operator.graph
    n = graph.n_modes
    nG = operator.n_geometry
    nU = operator.n_operating
    nstatic = nG + nU
    if box.lower.shape != (n + nstatic + 1,):
        raise ValueError("box must contain [a0,G,U,t]")
    if box.lower[-1] < 0.0:
        raise ValueError("time domain must be non-negative")
    if physical_proof.geometry_jacobian_entry_bounds.shape != (n, nG):
        raise ValueError("geometry physical proof dimension mismatch")
    if physical_proof.operating_jacobian_entry_bounds.shape != (n, nU):
        raise ValueError("operating physical proof dimension mismatch")

    center = box.midpoint
    a0 = center[:n]
    geometry = center[n : n + nG]
    operating = center[n + nG : n + nstatic]
    time = float(center[-1])
    prediction = operator.evaluate(
        time,
        geometry=geometry,
        a0=a0,
        operating=operating,
        stable=True,
    )

    compiled = graph.compile()
    p_lo = box.lower[: n + nstatic]
    p_hi = box.upper[: n + nstatic]
    half = box.halfwidth
    state_dir = np.zeros((n, n + nstatic + 1), dtype=float)
    derivative_dir = np.zeros_like(state_dir)

    for i, series in enumerate(compiled.mode_series):
        dtime = series.derivative(compiled.lambdas)
        ddtime = dtime.derivative(compiled.lambdas)
        state_dir[i, -1] = _series_sup_abs(
            dtime, compiled.lambdas, p_lo, p_hi, box.lower[-1], box.upper[-1]
        )
        derivative_dir[i, -1] = _series_sup_abs(
            ddtime, compiled.lambdas, p_lo, p_hi, box.lower[-1], box.upper[-1]
        )
        for j in range(n + nstatic):
            dp = series.parameter_derivative(j)
            state_dir[i, j] = _series_sup_abs(
                dp, compiled.lambdas, p_lo, p_hi, box.lower[-1], box.upper[-1]
            )
            derivative_dir[i, j] = _series_sup_abs(
                dp.derivative(compiled.lambdas),
                compiled.lambdas,
                p_lo,
                p_hi,
                box.lower[-1],
                box.upper[-1],
            )

    state_var = state_dir @ half
    derivative_var = derivative_dir @ half
    Lstate = float(physical_proof.state_jacobian_norm_bound)
    Gentry = physical_proof.geometry_jacobian_entry_bounds
    Uentry = physical_proof.operating_jacobian_entry_bounds
    g_half = half[n : n + nG]
    u_half = half[n + nG : n + nstatic]
    physical_parameter_drift = float(
        np.linalg.norm(Gentry @ g_half + Uentry @ u_half)
    )
    upper = (
        prediction.residual_norm
        + float(np.linalg.norm(derivative_var))
        + Lstate * float(np.linalg.norm(state_var))
        + physical_parameter_drift
    )

    contributions = np.zeros(n + nstatic + 1, dtype=float)
    for j in range(n + nstatic + 1):
        value = half[j] * (
            float(np.linalg.norm(derivative_dir[:, j]))
            + Lstate * float(np.linalg.norm(state_dir[:, j]))
        )
        if n <= j < n + nG:
            value += half[j] * float(np.linalg.norm(Gentry[:, j - n]))
        elif n + nG <= j < n + nstatic:
            value += half[j] * float(np.linalg.norm(Uentry[:, j - n - nG]))
        contributions[j] = value

    return GeometryAnalyticResidualBoxBound(
        box=box,
        center_residual_norm=float(prediction.residual_norm),
        residual_upper_bound=float(np.nextafter(upper, np.inf)),
        state_variation_bounds=state_var,
        derivative_variation_bounds=derivative_var,
        directional_contributions=contributions,
        contraction_margin_lower_bound=float(physical_proof.contraction_margin_lower_bound),
        certified_physics=bool(physical_proof.certified),
    )


def certify_geometry_analytic_residual_domain(
    operator,
    *,
    initial_lower: np.ndarray,
    initial_upper: np.ndarray,
    geometry_lower: np.ndarray,
    geometry_upper: np.ndarray,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
    time_lower: float,
    time_upper: float,
    tolerance: float,
    work_budget: int,
    physical_proof_factory: Callable[[ParameterBox, np.ndarray, np.ndarray], GeometryPhysicalLipschitzProof],
) -> ContinuousGeometryAnalyticResidualCertificate:
    """Branch proof of ``sup_[a0,G,U,t] ||R||`` without geometry sampling.

    ``physical_proof_factory`` is evaluated for each branch and must provide a
    certified theorem bound on the corresponding physical ``[G,a,U]`` domain.
    If it returns an uncertified proof, that branch can never be marked resolved.
    """

    if tolerance <= 0.0 or work_budget <= 0:
        raise ValueError("tolerance and work_budget must be positive")
    lower = np.concatenate(
        [initial_lower, geometry_lower, operating_lower, [float(time_lower)]]
    )
    upper = np.concatenate(
        [initial_upper, geometry_upper, operating_upper, [float(time_upper)]]
    )
    pending = [ParameterBox(lower, upper)]
    processed = 0
    resolved: list[float] = []
    observed = 0.0
    n = operator.graph.n_modes
    nG = operator.n_geometry
    nU = operator.n_operating

    while pending and processed < work_budget:
        box = pending.pop()
        # First bound analytic state motion; the physical proof factory receives
        # the geometry branch and a conservative state box derived around center.
        center = box.midpoint
        gbox = ParameterBox(box.lower[n : n + nG], box.upper[n : n + nG])
        ubox = ParameterBox(
            box.lower[n + nG : n + nG + nU],
            box.upper[n + nG : n + nG + nU],
        )
        try:
            # A first certified physical proof can be independent of the thermal
            # state box or internally compute its own analytic enclosure.
            proof = physical_proof_factory(gbox, box.lower[:n], box.upper[:n])
            bound = bound_geometry_analytic_residual_on_box(operator, box, proof)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            bound = None
        processed += 1

        if bound is not None:
            observed = max(observed, bound.center_residual_norm)
            if bound.center_residual_norm > tolerance:
                return ContinuousGeometryAnalyticResidualCertificate(
                    "violated", float(tolerance),
                    float(max(observed, bound.residual_upper_bound)),
                    processed, len(pending), center.copy(),
                )
            if bound.certified_physics and bound.residual_upper_bound <= tolerance:
                resolved.append(bound.residual_upper_bound)
                continue

        widths = box.halfwidth
        if not np.any(widths > 0.0):
            pending.append(box)
            break
        if bound is None:
            dimension = int(np.argmax(widths))
        else:
            dimension = int(np.argmax(bound.directional_contributions))
            if bound.directional_contributions[dimension] <= 0.0:
                dimension = int(np.argmax(widths))
        left, right = box.split(dimension)
        pending.extend([right, left])

    if not pending:
        return ContinuousGeometryAnalyticResidualCertificate(
            "certified", float(tolerance), float(max(resolved, default=observed)),
            processed, 0, None,
        )
    return ContinuousGeometryAnalyticResidualCertificate(
        "indeterminate", float(tolerance),
        float(max(resolved, default=float("inf"))) if resolved else float("inf"),
        processed, len(pending), None,
    )
