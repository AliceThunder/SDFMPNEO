from __future__ import annotations

import numpy as np
import scipy.linalg

from sdfmpneo.analytic.parametric_realization import evaluate_parametric_stable
from sdfmpneo.em.energy_solver import apsi_physical_energy_metric
from sdfmpneo.em.sparse_solver import CertifiedEnergySparseApsiSolver

from .em_domain import ParameterBox
from .geometry_mass_residual_domain import (
    GeometryJouleDerivativeBounds,
    GeometryThermalOperatorBounds,
    _analytic_direction_bounds,
    compose_geometry_mass_residual_physical_proof,
)
from .nonlinear_em_domain import _material_energy_ratios


_BETA = CertifiedEnergySparseApsiSolver.COERCIVITY_LOWER_BOUND


def _outward_up(value: float) -> float:
    return float(np.nextafter(float(value), np.inf))


def _outward_down(value: float) -> float:
    return float(np.nextafter(float(value), -np.inf))


def _branch_layout(model, box: ParameterBox):
    if model.graph is None:
        raise ValueError("trained analytic graph is required")
    n = int(model.graph.n_modes)
    ng = len(model.geometry_names)
    nu = int(model.current_matrix.shape[1])
    if box.lower.shape != (n + ng + nu + 1,):
        raise ValueError("box must contain physical [a0,G,U,t]")
    gslice = slice(n, n + ng)
    uslice = slice(n + ng, n + ng + nu)
    glo = np.asarray(box.lower[gslice], dtype=float)
    ghi = np.asarray(box.upper[gslice], dtype=float)
    if np.any(glo < model.lower) or np.any(ghi > model.upper):
        raise ValueError("geometry branch lies outside the saved chart")
    return n, ng, nu, gslice, uslice


def _branch_geometry_certificate(model, box: ParameterBox):
    _, _, _, gslice, _ = _branch_layout(model, box)
    certificate = model.chart.certify_box(
        np.asarray(box.lower[gslice], float) - model.geometry_reference,
        np.asarray(box.upper[gslice], float) - model.geometry_reference,
    )
    if not certificate.certified_nondegenerate:
        raise ValueError("geometry branch cannot be certified non-degenerate")
    rho = float(certificate.maximum_relative_jacobian_perturbation)
    if not 0.0 <= rho < 1.0:
        raise ValueError("invalid affine-geometry distortion bound")
    directional = np.asarray(certificate.directional_distortion_bounds, dtype=float)
    # For F=I+E and D_k=dF/dG_k, ||F^-1 D_k|| <= ||D_k||/(1-rho).
    eta = directional / (1.0 - rho)
    return certificate, np.asarray(eta, dtype=float)


def certify_geometry_thermal_operator_bounds(
    model,
    box: ParameterBox,
) -> GeometryThermalOperatorBounds:
    """Prove branch-local M/K and geometry-derivative bounds for the saved P1 ROM.

    P1 mass depends on an affine tetrahedron only through det(J).  Therefore
    |dM/dG_k| <= 3 eta_k M in quadratic-form sense.  P1 stiffness transforms as
    det(F) F^-1 F^-T, giving the conservative exact differential bound
    |dK/dG_k| <= 5 eta_k K.  Projection to the fixed thermal basis preserves
    these Loewner/form inequalities.
    """

    n, ng, _, gslice, _ = _branch_layout(model, box)
    certificate, eta = _branch_geometry_certificate(model, box)
    if eta.shape != (ng,):
        raise ValueError("geometry derivative dimension mismatch")

    center_geometry = np.asarray(box.midpoint[gslice], dtype=float)
    context = model.context(center_geometry)
    M = np.asarray(context.M, dtype=float)
    K = np.asarray(context.K, dtype=float)
    if M.shape != (n, n) or K.shape != (n, n):
        raise ValueError("reduced thermal matrix dimension mismatch")
    M = 0.5 * (M + M.T)
    K = 0.5 * (K + K.T)
    eigM = np.linalg.eigvalsh(M)
    if eigM[0] <= 0.0:
        raise np.linalg.LinAlgError("branch-center thermal mass is not positive definite")

    alpha_m, beta_m = map(float, certificate.p1_mass_ratio)
    _, beta_k = map(float, certificate.p1_stiffness_ratio)
    if alpha_m <= 0.0 or beta_m <= 0.0 or beta_k <= 0.0:
        raise ValueError("geometry chart returned non-positive thermal form ratios")

    mmin = _outward_down(alpha_m * float(eigM[0]))
    mnorm = _outward_up(beta_m * float(np.linalg.norm(M, ord=2)))
    knorm = _outward_up(beta_k * float(np.linalg.norm(K, ord=2)))
    dm = np.nextafter(3.0 * eta * mnorm, np.inf)
    dk = np.nextafter(5.0 * eta * knorm, np.inf)
    return GeometryThermalOperatorBounds(
        minimum_mass_eigenvalue=mmin,
        mass_matrix_norm_bound=mnorm,
        stiffness_matrix_norm_bound=knorm,
        mass_geometry_derivative_norm_bounds=dm,
        stiffness_geometry_derivative_norm_bounds=dk,
        certified=True,
        provenance=(
            "affine-tetra P1 pullback: det(F) mass derivative <=3 eta; "
            "det(F)F^-1F^-T stiffness derivative <=5 eta; fixed thermal projection"
        ),
    )


def _dual_norm(H: np.ndarray, vector: np.ndarray) -> float:
    value = np.asarray(vector, dtype=complex)
    if value.ndim != 1 or value.shape[0] != H.shape[0]:
        raise ValueError("dual vector dimension mismatch")
    solved = scipy.linalg.solve(H, value, assume_a="her", check_finite=True)
    square = max(0.0, float(np.vdot(value, solved).real))
    return _outward_up(np.sqrt(square))


def _induced_thermal_box(model, box: ParameterBox):
    n, ng, nu, gslice, uslice = _branch_layout(model, box)
    center = box.midpoint
    geometry = np.asarray(center[gslice], dtype=float)
    operating = np.asarray(center[uslice], dtype=float)
    initial = np.asarray(center[:n], dtype=float)
    static = np.concatenate([model.normalize(geometry), operating])
    state, _ = evaluate_parametric_stable(
        model.graph, float(center[-1]), a0=initial, operating=static
    )
    state_direction, _ = _analytic_direction_bounds(model, box)
    variation = state_direction @ box.halfwidth
    lower = np.asarray(state - variation, dtype=float)
    upper = np.asarray(state + variation, dtype=float)
    return state, ParameterBox(lower, upper)


def certify_geometry_joule_derivative_bounds(
    model,
    box: ParameterBox,
) -> GeometryJouleDerivativeBounds:
    """Prove continuous reduced-Joule derivatives over one physical branch.

    The proof is performed in the fixed shared reduced EM space.  Full Nedelec
    curl/mass affine-geometry form inequalities survive projection to V.  The
    nonlinear conductivity ratios are theorem-derived by `_material_energy_ratios`.
    Terminal-current geometry dependence is bounded analytically: for a
    normalized P1 surface load, |dw/dG_k| <= 4 eta_k w, hence one two-terminal
    port column has Euclidean derivative norm <= 8 eta_k.  No finite differences
    or sampled Jacobian maxima enter the result.
    """

    n, ng, nu, gslice, uslice = _branch_layout(model, box)
    geometry_cert, eta = _branch_geometry_certificate(model, box)
    center = box.midpoint
    geometry = np.asarray(center[gslice], dtype=float)
    operating = np.asarray(center[uslice], dtype=float)
    context = model.context(geometry)
    em = context.em
    problem = em.problem
    state_center, thermal_box = _induced_thermal_box(model, box)

    mu_material, nu_material, theta = _material_energy_ratios(problem, thermal_box)
    if mu_material <= 0.0 or not np.isfinite(nu_material):
        raise ValueError("constitutive law cannot certify the induced thermal box")
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (n,) or np.any(~np.isfinite(theta)):
        raise ValueError("invalid constitutive thermal derivative bound")

    k_lo, k_hi = map(float, geometry_cert.hcurl_curl_ratio)
    d_lo, d_hi = map(float, geometry_cert.hcurl_mass_ratio)
    mu = min(k_lo, d_lo * float(mu_material))
    nu = max(k_hi, d_hi * float(nu_material))
    if mu <= 0.0 or not np.isfinite(nu):
        raise ValueError("combined geometry/material EM energy ratio is not positive finite")

    # Use the exact physical full-space H at the branch center, then project to
    # the fixed shared V.  Coercivity and every form inequality are preserved.
    A_full = problem.operator_sparse(state_center)
    H_full = apsi_physical_energy_metric(A_full)
    V = np.asarray(em.V, dtype=complex)
    H = np.asarray(V.conj().T @ (H_full @ V), dtype=complex)
    H = 0.5 * (H + H.conj().T)
    eigH = scipy.linalg.eigvalsh(H, check_finite=True)
    if eigH[0] <= 0.0:
        raise np.linalg.LinAlgError("projected physical EM energy metric is not positive definite")
    hmin = _outward_down(float(eigH[0]))
    euclidean_to_dual = _outward_up(1.0 / np.sqrt(hmin))

    # At fixed geometry, the source is affine in physical port currents.  Bound
    # the entire operating box directly in the reduced H_c dual norm.
    u_half = np.asarray(box.halfwidth[uslice], dtype=float)
    reduced_center_source = em.rhs_reduced(context.rhs.evaluate(operating))
    source_dual = _dual_norm(H, reduced_center_source)
    reduced_u_columns = []
    for k in range(nu):
        column = em.rhs_reduced(np.asarray(context.rhs.matrix[:, k], dtype=complex))
        reduced_u_columns.append(column)
        source_dual += u_half[k] * _dual_norm(H, column)

    # Geometry variation of normalized terminal surface loads.  V is in the
    # shared A-psi coordinate topology, so only its scalar-coordinate rows act on
    # terminal RHS columns.
    n_ports = int(np.asarray(model.current_offset).size)
    if n_ports <= 0:
        raise ValueError("at least one physical current port is required")
    scalar_V = V[problem.n_A :, :]
    scalar_projection_norm = float(np.linalg.norm(scalar_V, ord=2))
    port_matrix_geometry_derivative = np.nextafter(
        8.0 * np.sqrt(float(n_ports)) * scalar_projection_norm * eta,
        np.inf,
    )

    port_current_center = np.asarray(model.current_offset, complex) + np.asarray(
        model.current_matrix, complex
    ) @ operating
    port_current_bound = float(np.linalg.norm(port_current_center))
    for k in range(nu):
        port_current_bound += u_half[k] * float(np.linalg.norm(model.current_matrix[:, k]))
    port_current_bound = _outward_up(port_current_bound)

    geometry_half = np.asarray(box.halfwidth[gslice], dtype=float)
    reduced_port_matrix_drift = float(np.dot(geometry_half, port_matrix_geometry_derivative))
    source_dual += reduced_port_matrix_drift * port_current_bound * euclidean_to_dual
    source_dual = _outward_up(source_dual)

    # H(G,a)>=mu H_c.  Physical A-psi coercivity is beta=1/sqrt(2), so the
    # reduced EM state has the following H_c norm bound everywhere in the branch.
    x_bound = _outward_up(source_dual / (_BETA * mu))

    omega = float(problem.omega)
    mode_inf = np.max(np.abs(np.asarray(problem.thermal_test_local, dtype=float)), axis=(1, 2))
    if mode_inf.shape != (n,):
        raise ValueError("thermal test-mode dimension mismatch")

    # |H_j| <= (omega/2)||phi_j||_inf H(G,a) <= C_j H_c.
    Cj = 0.5 * omega * mode_inf * nu

    # Thermal-state derivatives: only conductive A and loss forms vary.  The
    # material theorem gives |dD/da_k|<=theta_k H_c at center geometry; affine
    # geometry contributes at most d_hi to the conductive mass form.
    Ca = d_hi * theta
    Hja = 0.5 * omega * mode_inf[:, None] * (d_hi * theta[None, :])
    J_state = np.zeros((n, n), dtype=float)
    for k in range(n):
        dx = Ca[k] * x_bound / (_BETA * mu)
        J_state[:, k] = 2.0 * Cj * x_bound * dx + Hja[:, k] * x_bound**2
    state_norm = _outward_up(float(np.linalg.norm(J_state, ord="fro")))

    # Explicit geometry derivatives.  Both Nedelec curl-curl and covariant mass
    # forms have differential magnitude <=5 eta_k times the current form.  The
    # same factor applies to each loss form.  The source term includes the exact
    # normalized-terminal load derivative bound above.
    geometry_columns = np.zeros(ng, dtype=float)
    for k in range(ng):
        CAg = 5.0 * eta[k] * nu
        bG_dual = (
            port_matrix_geometry_derivative[k]
            * port_current_bound
            * euclidean_to_dual
        )
        dx = (bG_dual + CAg * x_bound) / (_BETA * mu)
        Hjg = 5.0 * eta[k] * Cj
        column = 2.0 * Cj * x_bound * dx + Hjg * x_bound**2
        geometry_columns[k] = _outward_up(float(np.linalg.norm(column)))

    # Explicit operating derivatives enter only through the affine port-current
    # source.  Account for geometry drift of the port matrix over the branch.
    operating_columns = np.zeros(nu, dtype=float)
    for k in range(nu):
        center_dual = _dual_norm(H, reduced_u_columns[k])
        direction_current_norm = float(np.linalg.norm(model.current_matrix[:, k]))
        drift_dual = (
            reduced_port_matrix_drift
            * direction_current_norm
            * euclidean_to_dual
        )
        dx = (center_dual + drift_dual) / (_BETA * mu)
        column = 2.0 * Cj * x_bound * dx
        operating_columns[k] = _outward_up(float(np.linalg.norm(column)))

    return GeometryJouleDerivativeBounds(
        state_jacobian_norm_bound=state_norm,
        geometry_jacobian_column_norm_bounds=np.nextafter(geometry_columns, np.inf),
        operating_jacobian_column_norm_bounds=np.nextafter(operating_columns, np.inf),
        certified=True,
        provenance=(
            "shared reduced EM energy proof: affine Nedelec curl/mass differential <=5 eta; "
            "certified nonlinear material ratios; projected physical coercivity beta>=1/sqrt(2); "
            "normalized terminal P1 surface-load derivative <=4 eta per terminal"
        ),
    )


def make_uwpt_mass_residual_physical_proof_factory(model):
    """Return the default theorem-only physical provider for a GeometryResearchModel."""

    def factory(box: ParameterBox):
        thermal = certify_geometry_thermal_operator_bounds(model, box)
        joule = certify_geometry_joule_derivative_bounds(model, box)
        return compose_geometry_mass_residual_physical_proof(
            thermal,
            joule,
            provenance=(
                "affine-tetra thermal forms + shared reduced A-psi physical-energy/Joule theorem "
                "+ analytic normalized-terminal source transport"
            ),
        )

    return factory
