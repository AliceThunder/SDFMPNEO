import numpy as np
import scipy.linalg
import scipy.sparse as sp
import pytest

from sdfmpneo.em import (
    CertifiedPCGRieszAction,
    CoupledPairEnergyPreconditioner,
    DiagonalGershgorinEnergyPreconditioner,
    SparseLUReferenceRieszAction,
)


def strongly_paired_spd_matrix():
    # Diagonal Gershgorin fails in the original coordinates because row 0 has
    # |0.8|+|0.3|>1.  Pairing coordinates (0,1) absorbs the dominant coupling.
    return sp.csr_matrix(
        np.array(
            [
                [1.0, 0.8, 0.3],
                [0.8, 1.0, -0.3],
                [0.3, -0.3, 1.0],
            ],
            dtype=float,
        )
    ).astype(complex)


def block_matrix_from_partition(B, blocks):
    P = np.zeros(B.shape, dtype=complex)
    dense = B.toarray()
    for block in blocks:
        idx = np.array(block, dtype=int)
        P[np.ix_(idx, idx)] = dense[np.ix_(idx, idx)]
    return P


def test_coupled_pair_certificate_succeeds_where_diagonal_gershgorin_fails():
    B = strongly_paired_spd_matrix()
    assert np.min(np.linalg.eigvalsh(B.toarray().real)) > 0.0

    with pytest.raises(ValueError, match="no positive certified"):
        DiagonalGershgorinEnergyPreconditioner.build(B)

    preconditioner = CoupledPairEnergyPreconditioner.build(B)
    assert preconditioner.blocks[0] == (0, 1)
    assert preconditioner.maximum_block_size == 2
    assert preconditioner.paired_block_count == 1
    assert preconditioner.lower_spectral_equivalence_bound > 0.0

    P = block_matrix_from_partition(B, preconditioner.blocks)
    generalized = scipy.linalg.eigvalsh(B.toarray(), P)
    true_lower = float(np.min(generalized.real))
    assert preconditioner.lower_spectral_equivalence_bound <= true_lower


def test_coupled_pair_action_is_the_declared_local_block_inverse():
    B = strongly_paired_spd_matrix()
    preconditioner = CoupledPairEnergyPreconditioner.build(B)
    P = block_matrix_from_partition(B, preconditioner.blocks)
    rhs = np.array([1.0 + 0.2j, -0.4 + 0.1j, 0.7 - 0.6j])

    actual = preconditioner.solve(rhs)
    expected = scipy.linalg.solve(P, rhs, assume_a="her")
    assert np.allclose(actual, expected, rtol=5e-14, atol=5e-14)


def test_coupled_pair_pcg_preserves_riesz_error_certificate():
    B = strongly_paired_spd_matrix()
    preconditioner = CoupledPairEnergyPreconditioner.build(B)
    action = CertifiedPCGRieszAction(B, preconditioner)
    reference_action = SparseLUReferenceRieszAction(B)
    rhs = np.array([1.0 + 0.2j, -0.4, 0.7j])

    result = action.solve(rhs, requested_energy_action_error=1e-10)
    reference = reference_action.solve(rhs, requested_energy_action_error=1e-12)
    difference = reference.vector - result.vector
    difference_energy = float(np.sqrt(np.real(np.vdot(difference, B @ difference))))

    assert result.meets_requested_energy_action_error
    assert reference.meets_requested_energy_action_error
    assert difference_energy <= (
        result.energy_action_error_bound
        + reference.energy_action_error_bound
        + 5e-14
    )
