from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from sdfmpneo.em.energy_solver import apsi_physical_energy_metric
from sdfmpneo.em.riesz_action import SparseLUReferenceRieszAction
from sdfmpneo.em.sparse_solver import CertifiedEnergySparseApsiSolver

from .em_domain import BoxResidualBound, ContinuousEMResidualCertificate, ParameterBox
from .nonlinear_em_domain import _material_energy_ratios


_BETA = CertifiedEnergySparseApsiSolver.COERCIVITY_LOWER_BOUND


def bound_nonlinear_operating_residual_on_box(
    model,
    box: ParameterBox,
    *,
    source_directions: np.ndarray,
) -> BoxResidualBound:
    """Continuous fixed-geometry certificate over thermal/source/frequency axes.

    The box coordinates are ``[a_1,...,a_r,u_1,...,u_m,omega]`` with
    ``b(u)=b0+B_u u``.  Frequency dependence uses the exact reciprocal A-psi
    structure ``A=K+i(omega/omega0)D``.  This offline proof routine uses the
    reference certified Riesz backend at each branch center; it does not alter
    the scalable online/Riesz path.
    """

    problem = model.problem
    Bsrc = np.asarray(source_directions, dtype=complex)
    if Bsrc.ndim == 1:
        Bsrc = Bsrc[:, None]
    if Bsrc.ndim != 2 or Bsrc.shape[0] != problem.n_em:
        raise ValueError("source_directions must have shape (n_em,n_source)")
    nT = int(problem.n_thermal)
    nU = int(Bsrc.shape[1])
    if box.lower.shape != (nT + nU + 1,):
        raise ValueError("operating box dimension mismatch")

    thermal_box = ParameterBox(box.lower[:nT], box.upper[:nT])
    a = thermal_box.midpoint
    u = box.midpoint[nT : nT + nU]
    omega = float(box.midpoint[-1])
    omega_lo = float(box.lower[-1])
    omega_hi = float(box.upper[-1])
    omega0 = float(problem.omega)
    if omega_lo <= 0.0 or omega <= 0.0 or omega0 <= 0.0:
        raise ValueError("frequency box must be strictly positive")

    A0 = sp.csr_matrix(problem.operator_sparse(a), dtype=complex)
    K = sp.csr_matrix(A0.real)
    D0 = sp.csr_matrix(A0.imag)
    scale = omega / omega0
    A = (K + 1j * scale * D0).tocsr()
    H = (K + scale * D0).astype(complex).tocsr()
    action = SparseLUReferenceRieszAction(H)

    rhs = np.asarray(problem.b, dtype=complex) + Bsrc @ u
    V = np.asarray(model.V, dtype=complex)
    AV = A @ V
    Ar = V.conj().T @ AV
    c = scipy.linalg.solve(Ar, V.conj().T @ rhs, assume_a="gen")
    residual = rhs - AV @ c
    center_residual = float(
        action.decide_dual_norm(residual, threshold=0.0).result.dual_norm_upper_bound
    )
    rhs_dual = float(
        action.decide_dual_norm(rhs, threshold=0.0).result.dual_norm_upper_bound
    )
    source_dual = np.array(
        [
            action.decide_dual_norm(Bsrc[:, j], threshold=0.0).result.dual_norm_upper_bound
            for j in range(nU)
        ],
        dtype=float,
    )

    material_mu, material_nu, thermal_theta = _material_energy_ratios(problem, thermal_box)
    freq_mu = omega_lo / omega
    freq_nu = omega_hi / omega
    mu = min(1.0, material_mu * freq_mu)
    nu = max(1.0, material_nu * freq_nu)
    if mu <= 0.0:
        directional = np.full(box.lower.size, float("inf"))
        return BoxResidualBound(box, center_residual, float("inf"), 0.0, directional)

    source_radius = float(np.dot(box.halfwidth[nT : nT + nU], source_dual))
    rhs_bound = rhs_dual + source_radius
    thermal_L = thermal_theta * freq_nu * rhs_bound * (
        1.0 / (_BETA * mu) + nu / (_BETA * _BETA * mu * mu)
    )
    source_L = source_dual * (1.0 + nu / (_BETA * mu))
    frequency_theta = material_nu / omega
    frequency_L = frequency_theta * rhs_bound * (
        1.0 / (_BETA * mu) + nu / (_BETA * _BETA * mu * mu)
    )
    directional = np.concatenate([thermal_L, source_L, [frequency_L]])
    radius = float(np.dot(box.halfwidth, directional))
    upper = (center_residual + radius) / np.sqrt(mu)
    return BoxResidualBound(
        box=box,
        center_residual_dual_norm=center_residual,
        residual_upper_bound=float(np.nextafter(upper, np.inf)),
        reduced_stability_lower_bound=float(_BETA * mu),
        directional_lipschitz_bounds=directional,
    )


def certify_nonlinear_operating_domain(
    model,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    source_directions: np.ndarray,
    tolerance: float,
    work_budget: int,
) -> ContinuousEMResidualCertificate:
    if tolerance <= 0.0 or work_budget <= 0:
        raise ValueError("tolerance and work_budget must be positive")
    pending = [ParameterBox(lower, upper)]
    processed = 0
    observed = 0.0
    resolved: list[float] = []

    while pending and processed < work_budget:
        box = pending.pop()
        bound = bound_nonlinear_operating_residual_on_box(
            model, box, source_directions=source_directions
        )
        processed += 1
        observed = max(observed, bound.center_residual_dual_norm)
        if bound.center_residual_dual_norm > tolerance:
            return ContinuousEMResidualCertificate(
                "violated", tolerance, observed,
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
        finite = np.where(np.isfinite(scores), scores, -1.0)
        dimension = int(np.argmax(finite))
        if finite[dimension] <= 0.0:
            dimension = int(np.argmax(widths))
        left, right = box.split(dimension)
        pending.extend([right, left])

    if not pending:
        return ContinuousEMResidualCertificate(
            "certified", tolerance, observed,
            float(max(resolved, default=observed)), processed, None, 0,
        )
    unresolved = [
        bound_nonlinear_operating_residual_on_box(
            model, b, source_directions=source_directions
        ).residual_upper_bound
        for b in pending
    ]
    return ContinuousEMResidualCertificate(
        "indeterminate", tolerance, observed,
        float(max(resolved + unresolved, default=float("inf"))),
        processed, None, len(pending),
    )
