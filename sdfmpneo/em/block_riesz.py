from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .certified_riesz import CertifiedPCGRieszAction


def _gamma(operation_count: int) -> float:
    eps = np.finfo(float).eps
    count = max(1, int(operation_count))
    if count * eps >= 1.0:
        raise FloatingPointError("operation count is too large for the gamma_k bound")
    return count * eps / (1.0 - count * eps)


def _positive_real_diagonal(matrix: sp.csr_matrix, name: str) -> np.ndarray:
    diagonal_complex = np.asarray(matrix.diagonal(), dtype=complex)
    n = diagonal_complex.size
    scale = float(np.max(np.abs(diagonal_complex))) if n else 0.0
    imag_bound = _gamma(max(1, 4 * n)) * scale
    if n and float(np.max(np.abs(diagonal_complex.imag))) > imag_bound:
        raise ValueError(f"{name} diagonal is not real to floating-point backward-error scale")
    diagonal = diagonal_complex.real.astype(float)
    if np.any(diagonal <= 0.0):
        raise ValueError(f"{name} must have a strictly positive diagonal")
    return diagonal


def _normalized_gershgorin_lower(
    matrix: sp.csr_matrix,
    diagonal: np.ndarray,
) -> float:
    """Certified lower bound for diag(d)^-1/2 A diag(d)^-1/2.

    This routine is used only when ``diagonal`` is the exact represented
    diagonal of ``matrix``.  Hence every normalized diagonal entry is one in
    arithmetic over the stored coefficients, and only the off-diagonal radius
    requires an outward floating-point enclosure.
    """

    A = sp.csr_matrix(matrix, dtype=complex)
    n = A.shape[0]
    if A.shape != (n, n) or diagonal.shape != (n,):
        raise ValueError("normalized Gershgorin dimensions do not match")

    row_lower = np.empty(n, dtype=float)
    eps = np.finfo(float).eps
    for i in range(n):
        terms: list[float] = []
        start, stop = A.indptr[i], A.indptr[i + 1]
        for j, value in zip(A.indices[start:stop], A.data[start:stop]):
            j = int(j)
            if j == i:
                continue
            denom = math.sqrt(float(diagonal[i] * diagonal[j]))
            terms.append(float(abs(value) / denom))
        radius_hat = math.fsum(terms)
        operation_count = 5 * len(terms) + 2
        if operation_count * eps >= 1.0:
            raise FloatingPointError("matrix row is too large for Gershgorin enclosure")
        gamma = operation_count * eps / (1.0 - operation_count * eps)
        radius_upper = float(np.nextafter(radius_hat / (1.0 - gamma), np.inf))
        row_lower[i] = float(np.nextafter(1.0 - radius_upper, -np.inf))
    return float(np.min(row_lower)) if n else np.inf


def _normalized_gershgorin_upper(
    matrix: sp.csr_matrix,
    reference_diagonal: np.ndarray,
) -> float:
    """Certified upper spectral bound after diagonal congruence scaling."""

    A = sp.csr_matrix(matrix, dtype=complex)
    n = A.shape[0]
    if A.shape != (n, n) or reference_diagonal.shape != (n,):
        raise ValueError("normalized Gershgorin dimensions do not match")

    row_upper = np.empty(n, dtype=float)
    eps = np.finfo(float).eps
    for i in range(n):
        diagonal_term = 0.0
        off_terms: list[float] = []
        start, stop = A.indptr[i], A.indptr[i + 1]
        for j, value in zip(A.indices[start:stop], A.data[start:stop]):
            j = int(j)
            denom = math.sqrt(float(reference_diagonal[i] * reference_diagonal[j]))
            term = float(abs(value) / denom)
            if j == i:
                diagonal_term += term
            else:
                off_terms.append(term)
        radius_hat = math.fsum(off_terms)
        operation_count = 5 * (len(off_terms) + 1) + 2
        if operation_count * eps >= 1.0:
            raise FloatingPointError("matrix row is too large for Gershgorin enclosure")
        gamma = operation_count * eps / (1.0 - operation_count * eps)
        raw = diagonal_term + radius_hat
        row_upper[i] = float(np.nextafter(raw / (1.0 - gamma), np.inf))
    return float(np.max(row_upper)) if n else 0.0


@dataclass(frozen=True)
class PhysicalBlockEnergyPreconditioner:
    """Certified magnetic/scalar-conductive block preconditioner.

    For the physical A-psi energy metric

        H = [[K_A + D_AA, D_Apsi],
             [D_psiA,       D_psipsi]],

    conductivity positivity makes the D block positive semidefinite.  Suppose

        D_AA <= gamma K_A.

    Then Cauchy-Schwarz in the conductive Gram form plus Young's inequality gives

        H >= m(gamma) diag(K_A, D_psipsi),

    with the optimal closed-form coefficient

        m(gamma)
        = 2 / [2 + gamma + sqrt(gamma^2 + 4 gamma)].

    No relaxation/damping parameter is selected by the user.  ``gamma`` is
    bounded from the assembled operators by certified normalized Gershgorin
    enclosures.  If those enclosures cannot prove positivity, construction is
    refused rather than repaired by a shift.

    The current block inverse uses complete sparse LU inside K_A and D_psipsi as
    a correctness implementation.  The theorem and outer PCG interface do not
    depend on that choice; each block can later be replaced by a certified
    auxiliary-space/multilevel action.
    """

    n_A: int
    lower_spectral_equivalence_bound: float
    gamma_upper_bound: float
    magnetic_normalized_lower_bound: float
    scalar_normalized_lower_bound: float
    _lu_magnetic: object
    _lu_scalar: object | None

    @classmethod
    def build(
        cls,
        H: sp.spmatrix,
        *,
        magnetic_block: sp.spmatrix,
        n_A: int,
    ) -> "PhysicalBlockEnergyPreconditioner":
        metric = sp.csr_matrix(H, dtype=complex)
        n = metric.shape[0]
        if metric.shape != (n, n):
            raise ValueError("H must be square")
        n_A = int(n_A)
        if not 0 < n_A <= n:
            raise ValueError("n_A must satisfy 0 < n_A <= H dimension")

        K = sp.csr_matrix(magnetic_block, dtype=complex)
        if K.shape != (n_A, n_A):
            raise ValueError("magnetic_block must have shape (n_A,n_A)")
        K = (0.5 * (K + K.conj().T)).tocsr()
        K.sum_duplicates()
        K.eliminate_zeros()

        K_diag = _positive_real_diagonal(K, "magnetic block")
        k_lower = _normalized_gershgorin_lower(K, K_diag)
        if k_lower <= 0.0:
            raise ValueError(
                "magnetic block has no positive certified normalized Gershgorin lower bound"
            )

        H11 = metric[:n_A, :n_A].tocsr()
        D_AA = (H11 - K).tocsr()
        D_AA = (0.5 * (D_AA + D_AA.conj().T)).tocsr()
        D_AA.sum_duplicates()
        D_AA.eliminate_zeros()
        d_upper = _normalized_gershgorin_upper(D_AA, K_diag)
        gamma_upper = float(np.nextafter(d_upper / k_lower, np.inf))
        if gamma_upper < 0.0 or not np.isfinite(gamma_upper):
            raise ValueError("failed to obtain a finite non-negative D_AA/K_A bound")

        n_scalar = n - n_A
        scalar_lower = np.inf
        lu_scalar = None
        if n_scalar:
            E = metric[n_A:, n_A:].tocsr()
            E = (0.5 * (E + E.conj().T)).tocsr()
            E.sum_duplicates()
            E.eliminate_zeros()
            E_diag = _positive_real_diagonal(E, "scalar conductive block")
            scalar_lower = _normalized_gershgorin_lower(E, E_diag)
            if scalar_lower <= 0.0:
                raise ValueError(
                    "scalar conductive block has no positive certified normalized Gershgorin lower bound"
                )
            lu_scalar = spla.splu(E.tocsc())

        if n_scalar == 0:
            # H11=K+D_AA >= K structurally.
            lower = 1.0
        elif gamma_upper == 0.0:
            # D_AA=0 forces the cross block to vanish for PSD conductive D.
            lower = 1.0
        else:
            root = math.sqrt(gamma_upper * gamma_upper + 4.0 * gamma_upper)
            lower = 2.0 / (2.0 + gamma_upper + root)
            lower = float(np.nextafter(lower, 0.0))
        if lower <= 0.0:
            raise ValueError("physical block spectral-equivalence lower bound is non-positive")

        return cls(
            n_A=n_A,
            lower_spectral_equivalence_bound=lower,
            gamma_upper_bound=gamma_upper,
            magnetic_normalized_lower_bound=float(k_lower),
            scalar_normalized_lower_bound=float(scalar_lower),
            _lu_magnetic=spla.splu(K.tocsc()),
            _lu_scalar=lu_scalar,
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        n_scalar = 0 if self._lu_scalar is None else vector.size - self.n_A
        if vector.ndim != 1 or vector.size < self.n_A:
            raise ValueError("preconditioner rhs dimension mismatch")
        if self._lu_scalar is None and vector.size != self.n_A:
            raise ValueError("preconditioner rhs dimension mismatch")
        left = np.asarray(self._lu_magnetic.solve(vector[: self.n_A]), dtype=complex)
        if not n_scalar:
            return left
        right = np.asarray(self._lu_scalar.solve(vector[self.n_A :]), dtype=complex)
        return np.concatenate([left, right])


def make_physical_block_pcg_riesz_factory(problem):
    """Create a state-independent factory closure for the physical block theorem.

    The magnetic coordinate block depends only on geometry/permeability in the
    present magnetoquasistatic formulation.  The conductive scalar block and
    gamma certificate are rebuilt from each local H(a), so thermal material
    variation remains inside the local certificate.
    """

    required = ("a_basis", "magnetic_stiffness", "n_A")
    if any(not hasattr(problem, name) for name in required):
        raise TypeError("problem does not expose the tetrahedral magnetic/gauge structure")
    R = problem.a_basis
    K_edge = problem.magnetic_stiffness.astype(complex)
    K_A = (R.conj().T @ K_edge @ R).tocsr()
    n_A = int(problem.n_A)

    def factory(H: sp.spmatrix):
        preconditioner = PhysicalBlockEnergyPreconditioner.build(
            H,
            magnetic_block=K_A,
            n_A=n_A,
        )
        return CertifiedPCGRieszAction(H, preconditioner)

    return factory
