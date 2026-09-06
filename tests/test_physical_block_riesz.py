import numpy as np
import scipy.linalg
import scipy.sparse as sp
import pytest

from sdfmpneo.em import (
    CertifiedPCGRieszAction,
    DiagonalGershgorinEnergyPreconditioner,
    PhysicalBlockEnergyPreconditioner,
)


def coupled_physical_energy():
    # K_A = 1.  The conductive block is PSD because
    # B-c^T E^-1 c = 1.1 - 3*(0.6)^2 = 0.02 > 0.
    K = sp.csr_matrix([[1.0]])
    c = np.ones(3) * 0.6
    D = np.block(
        [
            [np.array([[1.1]]), c[None, :]],
            [c[:, None], np.eye(3)],
        ]
    )
    H = sp.csr_matrix(D + np.diag([1.0, 0.0, 0.0, 0.0]))
    return H.astype(complex), K.astype(complex)


def test_physical_block_certificate_succeeds_where_diagonal_gershgorin_fails():
    H, K = coupled_physical_energy()

    with pytest.raises(ValueError, match="no positive certified"):
        DiagonalGershgorinEnergyPreconditioner.build(H)

    preconditioner = PhysicalBlockEnergyPreconditioner.build(
        H,
        magnetic_block=K,
        n_A=1,
    )
    assert preconditioner.gamma_upper_bound >= 1.1
    assert preconditioner.lower_spectral_equivalence_bound > 0.0

    # Here P=diag(K,E)=I, so the generalized minimum is simply lambda_min(H).
    true_minimum = float(np.min(np.linalg.eigvalsh(H.toarray().real)))
    assert preconditioner.lower_spectral_equivalence_bound <= true_minimum


def test_physical_block_pcg_action_preserves_riesz_error_certificate():
    H, K = coupled_physical_energy()
    preconditioner = PhysicalBlockEnergyPreconditioner.build(
        H,
        magnetic_block=K,
        n_A=1,
    )
    action = CertifiedPCGRieszAction(H, preconditioner)
    rhs = np.array([1.0 + 0.2j, -0.4, 0.7j, 1.2 - 0.1j])

    result = action.solve(rhs, requested_energy_action_error=1e-10)
    exact = scipy.linalg.solve(H.toarray(), rhs, assume_a="her")
    error = exact - result.vector
    actual_energy_error = float(np.sqrt(np.real(np.vdot(error, H @ error))))

    assert result.meets_requested_energy_action_error
    assert actual_energy_error <= result.energy_action_error_bound * (1.0 + 1e-10) + 1e-14


def test_physical_block_uses_residual_certified_trace_when_magnetic_gershgorin_fails():
    # K is SPD but normalized Gershgorin cannot prove it for this dense coupling.
    K = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.6, 0.6],
                [0.6, 1.0, 0.6],
                [0.6, 0.6, 1.0],
            ]
        )
    )
    Daa = sp.eye(3, format="csr") * 0.1
    H = sp.block_diag((K + Daa, sp.eye(1)), format="csr")
    preconditioner = PhysicalBlockEnergyPreconditioner.build(
        H,
        magnetic_block=K,
        n_A=3,
    )

    assert preconditioner.magnetic_normalized_lower_bound <= 0.0
    assert preconditioner.gamma_certificate_method == "residual_certified_generalized_trace"
    assert np.isfinite(preconditioner.magnetic_inverse_inf_upper_bound)
    assert preconditioner.gamma_upper_bound > 0.0
    assert preconditioner.lower_spectral_equivalence_bound > 0.0


def test_physical_block_rejects_singular_magnetic_energy():
    K = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0]])
    H = sp.block_diag((K + sp.eye(2) * 0.1, sp.eye(1)), format="csr")
    with pytest.raises(ValueError, match="magnetic block factorization failed"):
        PhysicalBlockEnergyPreconditioner.build(
            H,
            magnetic_block=K,
            n_A=2,
        )
