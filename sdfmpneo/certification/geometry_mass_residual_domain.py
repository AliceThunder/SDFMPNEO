from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from sdfmpneo.analytic.parametric_realization import evaluate_parametric_stable

from .analytic_residual_domain import _series_sup_abs
from .em_domain import ParameterBox


@dataclass(frozen=True)
class GeometryThermalOperatorBounds:
    """Certified thermal-operator bounds on one physical branch.

    Every quantity is with respect to the *physical geometry coordinates* used
    by the public branch box, not the normalized geometry coordinates carried by
    the analytic graph.

    ``mass_geometry_derivative_norm_bounds[k]`` and
    ``stiffness_geometry_derivative_norm_bounds[k]`` bound the spectral norm of
    the corresponding geometry derivative throughout the whole branch.
    """

    minimum_mass_eigenvalue: float
    mass_matrix_norm_bound: float
    stiffness_matrix_norm_bound: float
    mass_geometry_derivative_norm_bounds: np.ndarray
    stiffness_geometry_derivative_norm_bounds: np.ndarray
    certified: bool
    provenance: str

    def __post_init__(self) -> None:
        mmin = float(self.minimum_mass_eigenvalue)
        mnorm = float(self.mass_matrix_norm_bound)
        knorm = float(self.stiffness_matrix_norm_bound)
        dm = np.asarray(self.mass_geometry_derivative_norm_bounds, dtype=float)
        dk = np.asarray(self.stiffness_geometry_derivative_norm_bounds, dtype=float)
        if mmin <= 0.0 or not np.isfinite(mmin):
            raise ValueError("minimum mass eigenvalue must be finite and positive")
        if mnorm <= 0.0 or knorm < 0.0 or not np.isfinite(mnorm + knorm):
            raise ValueError("thermal matrix norm bounds must be finite and non-negative")
        if dm.ndim != 1 or dk.shape != dm.shape:
            raise ValueError("thermal geometry derivative bounds must be matching vectors")
        if np.any(dm < 0.0) or np.any(dk < 0.0) or np.any(~np.isfinite(dm + dk)):
            raise ValueError("thermal geometry derivative bounds must be finite and non-negative")
        if self.certified and not str(self.provenance).strip():
            raise ValueError("certified thermal-operator bounds require provenance")
        object.__setattr__(self, "minimum_mass_eigenvalue", mmin)
        object.__setattr__(self, "mass_matrix_norm_bound", mnorm)
        object.__setattr__(self, "stiffness_matrix_norm_bound", knorm)
        object.__setattr__(self, "mass_geometry_derivative_norm_bounds", dm)
        object.__setattr__(self, "stiffness_geometry_derivative_norm_bounds", dk)


@dataclass(frozen=True)
class GeometryJouleDerivativeBounds:
    """Certified reduced Joule-source derivative bounds on one branch.

    The mass-form heat source is ``q(G,a,U)`` in

        M(G) a' + K(G) a = q(G,a,U).

    ``state_jacobian_norm_bound`` bounds ``||dq/da||_2``.
    Geometry/operating arrays bound the Euclidean norm of each *explicit*
    parameter derivative column, with state dependence excluded because it is
    accounted for separately by the chain rule.
    """

    state_jacobian_norm_bound: float
    geometry_jacobian_column_norm_bounds: np.ndarray
    operating_jacobian_column_norm_bounds: np.ndarray
    certified: bool
    provenance: str

    def __post_init__(self) -> None:
        state = float(self.state_jacobian_norm_bound)
        geometry = np.asarray(self.geometry_jacobian_column_norm_bounds, dtype=float)
        operating = np.asarray(self.operating_jacobian_column_norm_bounds, dtype=float)
        if state < 0.0 or not np.isfinite(state):
            raise ValueError("Joule state-Jacobian norm bound must be finite and non-negative")
        if geometry.ndim != 1 or operating.ndim != 1:
            raise ValueError("Joule parameter derivative bounds must be vectors")
        if np.any(geometry < 0.0) or np.any(operating < 0.0):
            raise ValueError("Joule derivative bounds must be non-negative")
        if np.any(~np.isfinite(geometry)) or np.any(~np.isfinite(operating)):
            raise ValueError("Joule derivative bounds must be finite")
        if self.certified and not str(self.provenance).strip():
            raise ValueError("certified Joule derivative bounds require provenance")
        object.__setattr__(self, "state_jacobian_norm_bound", state)
        object.__setattr__(self, "geometry_jacobian_column_norm_bounds", geometry)
        object.__setattr__(self, "operating_jacobian_column_norm_bounds", operating)


@dataclass(frozen=True)
class GeometryMassResidualPhysicalProof:
    thermal: GeometryThermalOperatorBounds
    joule: GeometryJouleDerivativeBounds
    certified: bool
    provenance: str


def compose_geometry_mass_residual_physical_proof(
    thermal: GeometryThermalOperatorBounds,
    joule: GeometryJouleDerivativeBounds,
    *,
    provenance: str,
) -> GeometryMassResidualPhysicalProof:
    if thermal.mass_geometry_derivative_norm_bounds.shape != joule.geometry_jacobian_column_norm_bounds.shape:
        raise ValueError("thermal/Joule geometry dimensions do not match")
    text = str(provenance).strip()
    certified = bool(thermal.certified and joule.certified)
    if certified and not text:
        raise ValueError("certified mass-residual proof requires provenance")
    return GeometryMassResidualPhysicalProof(
        thermal=thermal,
        joule=joule,
        certified=certified,
        provenance=text,
    )


@dataclass(frozen=True)
class GeometryMassResidualBoxBound:
    box: ParameterBox
    center_mass_residual_norm: float
    center_vector_residual_norm: float
    mass_residual_upper_bound: float
    vector_residual_upper_bound: float
    state_norm_upper_bound: float
    derivative_norm_upper_bound: float
    directional_contributions: np.ndarray
    minimum_mass_eigenvalue: float
    certified_physics: bool


@dataclass(frozen=True)
class ContinuousGeometryMassResidualCertificate:
    status: Literal["certified", "violated", "indeterminate"]
    tolerance: float
    maximum_vector_residual_bound: float
    maximum_mass_residual_bound: float
    boxes_processed: int
    unresolved_boxes: int
    violating_point: np.ndarray | None
    finite_time_only: bool

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def _graph_parameter_box(model, box: ParameterBox):
    graph = model.graph
    n = graph.n_modes
    n_geometry = len(model.geometry_names)
    n_operating = int(model.current_matrix.shape[1])
    expected = n + n_geometry + n_operating + 1
    if box.lower.shape != (expected,):
        raise ValueError("box must contain physical [a0,G,U,t]")
    if box.lower[-1] < 0.0:
        raise ValueError("time domain must be non-negative")

    glo = np.asarray(box.lower[n : n + n_geometry], dtype=float)
    ghi = np.asarray(box.upper[n : n + n_geometry], dtype=float)
    if np.any(glo < np.asarray(model.lower, float)) or np.any(ghi > np.asarray(model.upper, float)):
        raise ValueError("geometry branch lies outside the saved geometry chart")
    zlo = np.asarray(model.normalize(glo), dtype=float)
    zhi = np.asarray(model.normalize(ghi), dtype=float)
    if np.any(zhi < zlo):
        raise ValueError("geometry normalization must preserve coordinate order")

    u_lo = box.lower[n + n_geometry : n + n_geometry + n_operating]
    u_hi = box.upper[n + n_geometry : n + n_geometry + n_operating]
    parameter_lower = np.concatenate([box.lower[:n], zlo, u_lo])
    parameter_upper = np.concatenate([box.upper[:n], zhi, u_hi])
    geometry_scale = 2.0 / (np.asarray(model.upper, float) - np.asarray(model.lower, float))
    return parameter_lower, parameter_upper, geometry_scale


def _analytic_direction_bounds(model, box: ParameterBox):
    graph = model.graph
    n = graph.n_modes
    n_geometry = len(model.geometry_names)
    n_operating = int(model.current_matrix.shape[1])
    n_public = n + n_geometry + n_operating + 1
    parameter_lower, parameter_upper, geometry_scale = _graph_parameter_box(model, box)
    compiled = graph.compile()
    state = np.zeros((n, n_public), dtype=float)
    derivative = np.zeros_like(state)
    tlo, thi = float(box.lower[-1]), float(box.upper[-1])

    for mode, series in enumerate(compiled.mode_series):
        dtime = series.derivative(compiled.lambdas)
        ddtime = dtime.derivative(compiled.lambdas)
        state[mode, -1] = _series_sup_abs(
            dtime, compiled.lambdas, parameter_lower, parameter_upper, tlo, thi
        )
        derivative[mode, -1] = _series_sup_abs(
            ddtime, compiled.lambdas, parameter_lower, parameter_upper, tlo, thi
        )

        # Initial coordinates are already public coordinates.
        for j in range(n):
            dp = series.parameter_derivative(j)
            state[mode, j] = _series_sup_abs(
                dp, compiled.lambdas, parameter_lower, parameter_upper, tlo, thi
            )
            derivative[mode, j] = _series_sup_abs(
                dp.derivative(compiled.lambdas),
                compiled.lambdas,
                parameter_lower,
                parameter_upper,
                tlo,
                thi,
            )

        # The graph stores normalized geometry z; convert dz derivatives back to
        # the physical G coordinates carried by the public certificate box.
        for k in range(n_geometry):
            graph_index = n + k
            public_index = n + k
            scale = abs(float(geometry_scale[k]))
            dp = series.parameter_derivative(graph_index)
            state[mode, public_index] = scale * _series_sup_abs(
                dp, compiled.lambdas, parameter_lower, parameter_upper, tlo, thi
            )
            derivative[mode, public_index] = scale * _series_sup_abs(
                dp.derivative(compiled.lambdas),
                compiled.lambdas,
                parameter_lower,
                parameter_upper,
                tlo,
                thi,
            )

        for k in range(n_operating):
            graph_index = n + n_geometry + k
            public_index = n + n_geometry + k
            dp = series.parameter_derivative(graph_index)
            state[mode, public_index] = _series_sup_abs(
                dp, compiled.lambdas, parameter_lower, parameter_upper, tlo, thi
            )
            derivative[mode, public_index] = _series_sup_abs(
                dp.derivative(compiled.lambdas),
                compiled.lambdas,
                parameter_lower,
                parameter_upper,
                tlo,
                thi,
            )
    return state, derivative


def bound_geometry_mass_residual_on_box(
    model,
    box: ParameterBox,
    physical_proof: GeometryMassResidualPhysicalProof,
) -> GeometryMassResidualBoxBound:
    """Rigorous first-order bound for ``sup_B ||R_v||`` through the mass residual.

    The proof uses

        r_M = M(G) a_dot + K(G) a - q(G,a,U),
        r_v = M(G)^-1 r_M,

    and the mean-value theorem on the complete physical branch ``B``.  No
    collocation or finite-difference estimate is used by this routine.
    """

    if model.graph is None:
        raise ValueError("trained analytic graph is required")
    graph = model.graph
    n = graph.n_modes
    n_geometry = len(model.geometry_names)
    n_operating = int(model.current_matrix.shape[1])
    _graph_parameter_box(model, box)  # validates branch and dimensions

    thermal = physical_proof.thermal
    joule = physical_proof.joule
    if thermal.mass_geometry_derivative_norm_bounds.shape != (n_geometry,):
        raise ValueError("thermal geometry proof dimension mismatch")
    if joule.geometry_jacobian_column_norm_bounds.shape != (n_geometry,):
        raise ValueError("Joule geometry proof dimension mismatch")
    if joule.operating_jacobian_column_norm_bounds.shape != (n_operating,):
        raise ValueError("Joule operating proof dimension mismatch")

    center = box.midpoint
    initial = center[:n]
    geometry = center[n : n + n_geometry]
    operating = center[n + n_geometry : n + n_geometry + n_operating]
    time = float(center[-1])
    static = np.concatenate([model.normalize(geometry), operating])
    state, derivative = evaluate_parametric_stable(
        graph, time, a0=initial, operating=static
    )
    context = model.context(geometry)
    rhs = context.rhs.evaluate(operating)
    heat = context.em.heat_source_for_rhs(state, rhs)
    mass_residual = context.M @ derivative + context.K @ state - heat
    vector_residual = np.linalg.solve(context.M, mass_residual)
    center_mass = float(np.linalg.norm(mass_residual))
    center_vector = float(np.linalg.norm(vector_residual))

    state_direction, derivative_direction = _analytic_direction_bounds(model, box)
    half = box.halfwidth
    state_variation = state_direction @ half
    derivative_variation = derivative_direction @ half
    state_norm = float(np.linalg.norm(state) + np.linalg.norm(state_variation))
    derivative_norm = float(np.linalg.norm(derivative) + np.linalg.norm(derivative_variation))

    mnorm = thermal.mass_matrix_norm_bound
    knorm = thermal.stiffness_matrix_norm_bound
    jqnorm = joule.state_jacobian_norm_bound
    contributions = np.zeros_like(half)
    for j in range(half.size):
        column = (
            mnorm * float(np.linalg.norm(derivative_direction[:, j]))
            + (knorm + jqnorm) * float(np.linalg.norm(state_direction[:, j]))
        )
        if n <= j < n + n_geometry:
            k = j - n
            column += (
                thermal.mass_geometry_derivative_norm_bounds[k] * derivative_norm
                + thermal.stiffness_geometry_derivative_norm_bounds[k] * state_norm
                + joule.geometry_jacobian_column_norm_bounds[k]
            )
        elif n + n_geometry <= j < n + n_geometry + n_operating:
            k = j - n - n_geometry
            column += joule.operating_jacobian_column_norm_bounds[k]
        contributions[j] = half[j] * column

    mass_upper = float(np.nextafter(center_mass + float(np.sum(contributions)), np.inf))
    vector_upper = float(np.nextafter(mass_upper / thermal.minimum_mass_eigenvalue, np.inf))
    return GeometryMassResidualBoxBound(
        box=box,
        center_mass_residual_norm=center_mass,
        center_vector_residual_norm=center_vector,
        mass_residual_upper_bound=mass_upper,
        vector_residual_upper_bound=vector_upper,
        state_norm_upper_bound=float(np.nextafter(state_norm, np.inf)),
        derivative_norm_upper_bound=float(np.nextafter(derivative_norm, np.inf)),
        directional_contributions=contributions,
        minimum_mass_eigenvalue=thermal.minimum_mass_eigenvalue,
        certified_physics=bool(physical_proof.certified),
    )


def certify_geometry_mass_residual_domain(
    model,
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
    physical_proof_factory: Callable[[ParameterBox], GeometryMassResidualPhysicalProof],
) -> ContinuousGeometryMassResidualCertificate:
    """Adaptive continuous-domain proof of the finite-time vector residual.

    The public branch uses physical coordinates ``[a0,G,U,t]``.  A branch is
    certified only when the supplied theorem-derived mass/Joule proof is itself
    certified and the converted vector-residual upper bound is below tolerance.
    Uncertified physics, exhausted work budget, or failed bound construction all
    remain ``indeterminate`` rather than falling back to sampling.
    """

    if tolerance <= 0.0 or work_budget <= 0:
        raise ValueError("tolerance and work_budget must be positive")
    if not np.isfinite(time_lower) or not np.isfinite(time_upper):
        raise ValueError("this certificate covers finite time intervals only")
    if time_lower < 0.0 or time_upper < time_lower:
        raise ValueError("time interval must satisfy 0 <= lower <= upper")
    lower = np.concatenate(
        [initial_lower, geometry_lower, operating_lower, [float(time_lower)]]
    )
    upper = np.concatenate(
        [initial_upper, geometry_upper, operating_upper, [float(time_upper)]]
    )
    pending = [ParameterBox(lower, upper)]
    processed = 0
    resolved_vector: list[float] = []
    resolved_mass: list[float] = []
    observed_vector = 0.0
    observed_mass = 0.0

    while pending and processed < work_budget:
        box = pending.pop()
        center = box.midpoint
        try:
            proof = physical_proof_factory(box)
            bound = bound_geometry_mass_residual_on_box(model, box, proof)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            bound = None
        processed += 1

        if bound is not None:
            observed_vector = max(observed_vector, bound.center_vector_residual_norm)
            observed_mass = max(observed_mass, bound.center_mass_residual_norm)
            if bound.center_vector_residual_norm > tolerance:
                return ContinuousGeometryMassResidualCertificate(
                    status="violated",
                    tolerance=float(tolerance),
                    maximum_vector_residual_bound=float(
                        max(observed_vector, bound.vector_residual_upper_bound)
                    ),
                    maximum_mass_residual_bound=float(
                        max(observed_mass, bound.mass_residual_upper_bound)
                    ),
                    boxes_processed=processed,
                    unresolved_boxes=len(pending),
                    violating_point=center.copy(),
                    finite_time_only=True,
                )
            if bound.certified_physics and bound.vector_residual_upper_bound <= tolerance:
                resolved_vector.append(bound.vector_residual_upper_bound)
                resolved_mass.append(bound.mass_residual_upper_bound)
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
        return ContinuousGeometryMassResidualCertificate(
            status="certified",
            tolerance=float(tolerance),
            maximum_vector_residual_bound=float(max(resolved_vector, default=observed_vector)),
            maximum_mass_residual_bound=float(max(resolved_mass, default=observed_mass)),
            boxes_processed=processed,
            unresolved_boxes=0,
            violating_point=None,
            finite_time_only=True,
        )

    unresolved_vector: list[float] = []
    unresolved_mass: list[float] = []
    for box in pending:
        try:
            proof = physical_proof_factory(box)
            bound = bound_geometry_mass_residual_on_box(model, box, proof)
            unresolved_vector.append(bound.vector_residual_upper_bound)
            unresolved_mass.append(bound.mass_residual_upper_bound)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            unresolved_vector.append(float("inf"))
            unresolved_mass.append(float("inf"))
    return ContinuousGeometryMassResidualCertificate(
        status="indeterminate",
        tolerance=float(tolerance),
        maximum_vector_residual_bound=float(
            max(resolved_vector + unresolved_vector, default=float("inf"))
        ),
        maximum_mass_residual_bound=float(
            max(resolved_mass + unresolved_mass, default=float("inf"))
        ),
        boxes_processed=processed,
        unresolved_boxes=len(pending),
        violating_point=None,
        finite_time_only=True,
    )
