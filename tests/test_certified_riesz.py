import numpy as np
import scipy.linalg
import scipy.sparse as sp
import pytest

from sdfmpneo.em import (
    CertifiedPCGRieszAction,
    DiagonalGershgorinEnergyPreconditioner,
)


def diagonally_dominant_spd(n=12):
    diagonal = np.ones(n) * 2.0
    off = np.ones(n - 1) * -0.25
    return sp.diags([off, diagonal, off], offsets=[-1, 0, 1], format="csr", dtype=float)


def energy_norm(H, x):
    return float(np.sqrt(np.real(np.vdot(x, H @ x))))


def test_diagonal_gershgorin_lower_bound_is_below_true_scaled_minimum_eigenvalue():
    H = diagonally_dominant_spd(10)
    preconditioner = DiagonalGershgorinEnergyPreconditioner.build(H)

    d = preconditioner.diagonal
    inv_sqrt = np.diag(1.0 / np.sqrt(d))
    scaled = inv_sqrt @ H.toarray() @ inv_sqrt
    true_minimum = float(np.min(np.linalg.eigvalsh(scaled)))

    assert preconditioner.lower_spectral_equivalence_bound > 0.0
    assert preconditioner.lower_spectral_equivalence_bound <= true_minimum


def test_certified_pcg_riesz_action_bounds_actual_energy_error_without_rtol():
    H = diagonally_dominant_spd(14).astype(complex)
    preconditioner = DiagonalGershgorinEnergyPreconditioner.build(H)
    action = CertifiedPCGRieszAction(H, preconditioner)
    rhs = (np.arange(1, 15) + 0.2j * np.arange(14)).astype(complex)

    requested = 1e-8
    result = action.solve(rhs, requested_energy_action_error=requested)
    exact = scipy.linalg.solve(H.toarray(), rhs, assume_a="her")
    actual_error = energy_norm(H, exact - result.vector)
    exact_dual_norm = energy_norm(H, exact)

    assert result.meets_requested_energy_action_error
    assert result.energy_action_error_bound <= requested
    assert actual_error <= result.energy_action_error_bound * (1.0 + 1e-10) + 1e-14
    assert result.dual_norm_lower_bound <= exact_dual_norm <= result.dual_norm_upper_bound
    assert result.iterations <= H.shape[0]


def test_dual_norm_decision_stops_only_after_certified_interval_separates_threshold():
    H = diagonally_dominant_spd(16).astype(complex)
    preconditioner = DiagonalGershgorinEnergyPreconditioner.build(H)
    action = CertifiedPCGRieszAction(H, preconditioner)
    rhs = np.linspace(0.3, 2.0, H.shape[0]).astype(complex)
    exact = scipy.linalg.solve(H.toarray(), rhs, assume_a="her")
    exact_dual_norm = energy_norm(H, exact)

    below = action.decide_dual_norm(rhs, threshold=1.05 * exact_dual_norm)
    above = action.decide_dual_norm(rhs, threshold=0.95 * exact_dual_norm)

    assert below.relation == "below"
    assert below.result.dual_norm_upper_bound <= below.threshold
    assert above.relation == "above"
    assert above.result.dual_norm_lower_bound > above.threshold
    assert below.result.dual_norm_lower_bound <= exact_dual_norm <= below.result.dual_norm_upper_bound
    assert above.result.dual_norm_lower_bound <= exact_dual_norm <= above.result.dual_norm_upper_bound


def test_gershgorin_preconditioner_refuses_unprovable_positive_equivalence():
    # This matrix is SPD, but its normalized Gershgorin lower bound is negative.
    # The diagonal certificate must refuse it rather than add a fitted shift.
    H = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.6, 0.6],
                [0.6, 1.0, 0.6],
                [0.6, 0.6, 1.0],
            ]
        )
    )
    assert np.min(np.linalg.eigvalsh(H.toarray())) > 0.0
    with pytest.raises(ValueError, match="no positive certified"):
        DiagonalGershgorinEnergyPreconditioner.build(H)
