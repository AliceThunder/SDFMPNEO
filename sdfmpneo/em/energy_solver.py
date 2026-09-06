from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .sparse_solver import (
    CertifiedEnergySparseApsiSolver,
    SparseEnergyLinearSolveCertificate,
)


def apsi_physical_energy_metric(A: sp.spmatrix) -> sp.csr_matrix:
    """Return H=K+D from the reciprocal physical operator A=K+iD.

    This helper is valid for the real-material magnetoquasistatic A-psi operator
    assembled by SDF-MPNEO. It verifies complex symmetry and that real/imaginary
    matrix parts are real symmetric to floating-point backward-error scale; it
    does not reinterpret a generic complex matrix as a physical A-psi operator.
    """

    matrix = sp.csr_matrix(A, dtype=complex)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("A must be square")

    asym = matrix - matrix.T
    asym_norm = float(np.sqrt(np.sum(np.abs(asym.data) ** 2))) if asym.nnz else 0.0
    matrix_norm = float(np.sqrt(np.sum(np.abs(matrix.data) ** 2)))
    backward = np.finfo(float).eps * max(1, n) * max(1.0, matrix_norm)
    if asym_norm > backward:
        raise ValueError("physical A-psi operator must be complex symmetric")

    K = sp.csr_matrix(matrix.real)
    D = sp.csr_matrix(matrix.imag)
    H = (K + D).astype(complex).tocsr()
    H = (0.5 * (H + H.T)).tocsr()
    H.sum_duplicates()
    H.eliminate_zeros()
    return H


class PhysicalEnergySparseApsiSolver(CertifiedEnergySparseApsiSolver):
    """Energy-certified solver specialized to the SDF-MPNEO physical A-psi form."""

    def __init__(self, A: sp.spmatrix) -> None:
        super().__init__(A, apsi_physical_energy_metric(A))


def solve_physical_energy_certified_apsi(
    A: sp.spmatrix,
    b: np.ndarray,
    *,
    requested_energy_state_error: float,
) -> tuple[np.ndarray, SparseEnergyLinearSolveCertificate]:
    solver = PhysicalEnergySparseApsiSolver(A)
    return solver.solve(
        b,
        requested_energy_state_error=requested_energy_state_error,
    )
