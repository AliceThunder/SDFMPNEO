from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import spsolve

from sdfmpneo.analytic import evaluate_parametric_stable


@dataclass(frozen=True)
class GeometryMassResidualCertificate:
    """Continuous geometry-box equivalence between vector-field and mass residuals.

    For every geometry in the certified affine chart box,

        r_v = da/dt - M(G)^-1[-K(G)a+q],
        r_M = M(G) r_v,

    and ``m_min <= lambda_min(M(G))`` / ``lambda_max(M(G)) <= m_max`` imply

        ||r_M||/m_max <= ||r_v|| <= ||r_M||/m_min.
    """

    minimum_mass_eigenvalue: float
    maximum_mass_eigenvalue: float
    center_minimum_mass_eigenvalue: float
    center_maximum_mass_eigenvalue: float
    p1_mass_ratio_lower: float
    p1_mass_ratio_upper: float
    continuous_geometry_box: bool


@dataclass(frozen=True)
class LocalGeometryDynamicsDiagnostic:
    vector_residual: np.ndarray
    mass_residual: np.ndarray
    vector_residual_norm: float
    mass_residual_norm: float
    vector_norm_lower_from_mass: float
    vector_norm_upper_from_mass: float
    local_mass_contractivity_margin: float
    locally_contractive: bool


@dataclass(frozen=True)
class SampledContractivityReport:
    minimum_mass_contractivity_margin: float
    maximum_mass_contractivity_margin: float
    point_count: int
    all_sampled_points_contractive: bool
    continuous_domain_certified: bool
    scope: str


def certify_geometry_mass_residual_equivalence(model) -> GeometryMassResidualCertificate:
    """Prove mass-residual norm equivalence over the whole saved geometry box.

    ``AffineTetrahedralGeometryChart.certify_box`` already proves P1 mass-form
    ratios on the full parameter box. Restricting the quadratic form to the
    shared thermal subspace preserves those Loewner bounds exactly.
    """

    center = 0.5 * (np.asarray(model.lower, float) + np.asarray(model.upper, float))
    M0 = np.asarray(model.context(center).M, float)
    eig0 = np.linalg.eigvalsh(0.5 * (M0 + M0.T))
    if eig0[0] <= 0.0:
        raise np.linalg.LinAlgError("reduced thermal mass matrix is not positive definite")
    lo_ratio, hi_ratio = map(float, model.certificate.p1_mass_ratio)
    minimum = float(np.nextafter(lo_ratio * eig0[0], -np.inf))
    maximum = float(np.nextafter(hi_ratio * eig0[-1], np.inf))
    if minimum <= 0.0 or not np.isfinite(maximum):
        raise ValueError("geometry chart does not provide a finite positive P1 mass bound")
    return GeometryMassResidualCertificate(
        minimum_mass_eigenvalue=minimum,
        maximum_mass_eigenvalue=maximum,
        center_minimum_mass_eigenvalue=float(eig0[0]),
        center_maximum_mass_eigenvalue=float(eig0[-1]),
        p1_mass_ratio_lower=lo_ratio,
        p1_mass_ratio_upper=hi_ratio,
        continuous_geometry_box=bool(model.certificate.certified_nondegenerate),
    )


def _mass_logarithmic_abscissa(M: np.ndarray, J: np.ndarray) -> float:
    """Largest logarithmic growth rate in the norm ||e||_M."""

    M = np.asarray(M, float)
    J = np.asarray(J, float)
    sym_mj = 0.5 * (M @ J + J.T @ M)
    values = scipy.linalg.eigvalsh(sym_mj, M, check_finite=True)
    return float(values[-1])


def local_geometry_dynamics_diagnostic(
    model,
    *,
    geometry,
    a: np.ndarray,
    operating: np.ndarray,
    da: np.ndarray,
    mass_certificate: GeometryMassResidualCertificate | None = None,
) -> LocalGeometryDynamicsDiagnostic:
    """Exact pointwise residual and M-energy contractivity diagnostic."""

    g = model.geometry_vector(geometry)
    state = np.asarray(a, float)
    current = np.asarray(operating, float)
    derivative = np.asarray(da, float)
    if state.shape != (model.thermal_model.lambdas.size,) or derivative.shape != state.shape:
        raise ValueError("a/da thermal dimensions do not match the saved thermal basis")
    if current.shape != (model.current_matrix.shape[1],):
        raise ValueError("operating dimension mismatch")

    c = model.context(g)
    rhs = c.rhs.evaluate(current)
    q, Jq = c.em.heat_source_and_jacobian_for_rhs(state, rhs)
    F = np.linalg.solve(c.M, -c.K @ state + q)
    J = np.linalg.solve(c.M, -c.K + Jq)
    vector_residual = derivative - F
    mass_residual = c.M @ vector_residual
    vector_norm = float(np.linalg.norm(vector_residual))
    mass_norm = float(np.linalg.norm(mass_residual))

    cert = mass_certificate or certify_geometry_mass_residual_equivalence(model)
    lower = float(mass_norm / cert.maximum_mass_eigenvalue)
    upper = float(mass_norm / cert.minimum_mass_eigenvalue)
    # The chart-wide bounds are conservative, so allow roundoff at the endpoint.
    eps = 32.0 * np.finfo(float).eps * max(1.0, vector_norm, upper)
    if vector_norm < lower - eps or vector_norm > upper + eps:
        raise AssertionError("mass/vector residual norm equivalence was violated")

    margin = -_mass_logarithmic_abscissa(c.M, J)
    return LocalGeometryDynamicsDiagnostic(
        vector_residual=vector_residual,
        mass_residual=mass_residual,
        vector_residual_norm=vector_norm,
        mass_residual_norm=mass_norm,
        vector_norm_lower_from_mass=lower,
        vector_norm_upper_from_mass=upper,
        local_mass_contractivity_margin=float(margin),
        locally_contractive=bool(margin > 0.0),
    )


def sampled_geometry_contractivity(model, *, validation=True, seed=701) -> SampledContractivityReport:
    """Independent numerical stability check; deliberately not a box certificate.

    The report evaluates the exact local M-energy logarithmic norm along the
    analytic surrogate states at deterministic held-out collocation points. A
    positive result is evidence for stability and a useful regression guard, but
    it is not promoted to a continuous-domain proof without interval/analytic
    geometry and nonlinear heat-Jacobian bounds.
    """

    if model.graph is None or model.training_config is None:
        raise ValueError("trained graph and training configuration are required")
    points = model.training_config.points(validation=validation, seed=seed)
    n = model.graph.n_modes
    margins: list[float] = []
    for point in points:
        initial = point[:n]
        static = point[n:-1]
        time = float(point[-1])
        a, _ = evaluate_parametric_stable(model.graph, time, a0=initial, operating=static)
        c, current = model.split(static)
        physical = model.evaluate(a, static)
        margins.append(-_mass_logarithmic_abscissa(c.M, physical.vector_field_jacobian))
    values = np.asarray(margins, float)
    return SampledContractivityReport(
        minimum_mass_contractivity_margin=float(np.min(values)),
        maximum_mass_contractivity_margin=float(np.max(values)),
        point_count=int(values.size),
        all_sampled_points_contractive=bool(np.all(values > 0.0)),
        continuous_domain_certified=False,
        scope="held-out deterministic points on surrogate trajectories; not a continuous-domain proof",
    )


def validate_geometry_trajectory(
    model,
    times,
    *,
    geometry,
    a0,
    operating,
    full_electromagnetics: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """Held-out cross-geometry transient validation, never used for training.

    The reference uses an independent implicit Radau integrator. By default each
    RHS evaluation solves the full sparse electromagnetic equilibrium on the
    queried geometry. The thermal coordinates intentionally remain the same
    saved thermal ROM, so this isolates analytic-evolution and EM-ROM errors; it
    does not claim to validate thermal-rank, mesh or outer-domain truncation.
    """

    if model.graph is None:
        raise ValueError("train or load a geometry model first")
    times = np.asarray(times, float)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times must be a nonempty finite non-negative vector")
    g = model.geometry_vector(geometry)
    initial = np.asarray(a0, float)
    current = np.asarray(operating, float)
    if initial.shape != (model.graph.n_modes,) or current.shape != (model.current_matrix.shape[1],):
        raise ValueError("a0/operating dimensions do not match the model")
    c = model.context(g)
    rhs = c.rhs.evaluate(current)
    problem = c.em.problem

    def reference_rhs(_time, state):
        if full_electromagnetics:
            x = spsolve(problem.operator_sparse(state).tocsc(), rhs)
            q = np.array(
                [
                    np.vdot(x, problem.loss_operator_sparse(j, state) @ x).real
                    for j in range(problem.n_thermal)
                ],
                dtype=float,
            )
        else:
            q = c.em.heat_source_for_rhs(state, rhs)
        return np.linalg.solve(c.M, -c.K @ state + q)

    if float(np.max(times)) == 0.0:
        reference = np.tile(initial, (times.size, 1))
    else:
        solution = solve_ivp(
            reference_rhs,
            (0.0, float(np.max(times))),
            initial,
            method="Radau",
            dense_output=True,
            rtol=float(rtol),
            atol=float(atol),
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        reference = solution.sol(times).T

    predicted = np.array(
        [
            model.predict(
                float(t),
                geometry=g,
                a0=initial,
                operating=current,
                diagnostics=False,
                allow_time_extrapolation=True,
            )["thermal_coordinates"]
            for t in times
        ]
    )
    temperature_error = np.array(
        [model.reference.temperature(p) - model.reference.temperature(r) for p, r in zip(predicted, reference)]
    )
    return {
        "times": times,
        "geometry": dict(zip(model.geometry_names, g)),
        "predicted_coordinates": predicted,
        "reference_coordinates": reference,
        "maximum_coordinate_error": float(np.max(np.linalg.norm(predicted - reference, axis=1))),
        "maximum_temperature_error": float(np.max(np.abs(temperature_error))),
        "reference": "Radau + full sparse EM on queried geometry" if full_electromagnetics else "Radau + shared EM ROM on queried geometry",
        "thermal_reference_scope": "same saved thermal ROM; thermal-rank/mesh/outer-domain errors excluded",
        "used_for_training": False,
    }
