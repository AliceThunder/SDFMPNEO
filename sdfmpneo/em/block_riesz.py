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
    A = sp.csr_matrix(matrix, dtype=complex)
    n = A.shape[0]
    if A.shape != (n, n) or diagonal.shape != (n,):
        raise ValueError("normalized Gershgorin dimensions do not match")

    row_lower = np.empty(n, dtype=float)
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
        g = _gamma(5 * len(terms) + 2)
        radius_upper = float(np.nextafter(radius_hat / (1.0 - g), np.inf))
        row_lower[i] = float(np.nextafter(1.0 - radius_upper, -np.inf))
    return float(np.min(row_lower)) if n else np.inf


def _normalized_gershgorin_upper(
    matrix: sp.csr_matrix,
    reference_diagonal: np.ndarray,
) -> float:
    A = sp.csr_matrix(matrix, dtype=complex)
    n = A.shape[0]
    if A.shape != (n, n) or reference_diagonal.shape != (n,):
        raise ValueError("normalized Gershgorin dimensions do not match")

    row_upper = np.empty(n, dtype=float)
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
        g = _gamma(5 * (len(off_terms) + 1) + 2)
        raw = diagonal_term + radius_hat
        row_upper[i] = float(np.nextafter(raw / (1.0 - g), np.inf))
    return float(np.max(row_upper)) if n else 0.0


def _matvec_residual_component_upper(
    A: sp.csr_matrix,
    x: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Componentwise upper bound for the exact represented residual b-Ax."""

    product = np.asarray(A @ x, dtype=complex)
    residual_hat = np.asarray(b, dtype=complex) - product
    out = np.empty(A.shape[0], dtype=float)
    for i in range(A.shape[0]):
        start, stop = A.indptr[i], A.indptr[i + 1]
        indices = A.indices[start:stop]
        values = A.data[start:stop]
        scale = math.fsum(
            float(abs(value) * abs(x[int(j)]))
            for j, value in zip(indices, values)
        )
        # Complex multiply/add plus the final subtraction are enclosed
        # conservatively by a standard gamma_k model.
        g = _gamma(10 * max(1, len(indices)) + 4)
        matvec_error = g * scale
        value = float(abs(residual_hat[i])) + matvec_error
        out[i] = float(np.nextafter(value / (1.0 - _gamma(2)), np.inf))
    return out


def _inflate_sum(value: float, term_count: int) -> float:
    g = _gamma(max(1, term_count))
    return float(np.nextafter(value / (1.0 - g), np.inf))


def _certified_inverse_inf_upper(
    A: sp.csr_matrix,
    lu,
) -> float:
    """A-posteriori upper bound for ||A^-1||_inf without storing a dense inverse.

    Column solves form an implicit approximate inverse X.  The explicitly
    recomputed residual R=I-AX gives

        A^-1 = X (I-R)^-1,
        ||A^-1||_inf <= ||X||_inf / (1-||R||_inf)

    whenever ||R||_inf<1.  Only row sums are retained, so the certificate uses
    O(n) auxiliary memory although it performs n correctness-scale block solves.
    """

    n = A.shape[0]
    x_row_sums = np.zeros(n, dtype=float)
    residual_row_sums = np.zeros(n, dtype=float)
    for j in range(n):
        rhs = np.zeros(n, dtype=complex)
        rhs[j] = 1.0
        x = np.asarray(lu.solve(rhs), dtype=complex)
        x_row_sums += np.abs(x)
        residual_row_sums += _matvec_residual_component_upper(A, x, rhs)

    x_inf = _inflate_sum(float(np.max(x_row_sums)), n)
    residual_inf = _inflate_sum(float(np.max(residual_row_sums)), n)
    if residual_inf >= 1.0:
        raise ValueError("sparse magnetic factorization residual cannot certify an inverse bound")
    upper = x_inf / (1.0 - residual_inf)
    return float(np.nextafter(upper, np.inf))


def _certified_generalized_trace_upper(
    K: sp.csr_matrix,
    D: sp.csr_matrix,
    lu_K,
    inverse_inf_upper: float,
) -> float:
    """Certify lambda_max(K^-1/2 D K^-1/2) by a trace upper bound.

    For physical D>=0,

        lambda_max <= trace(K^-1 D).

    Each diagonal contribution is obtained from a sparse solve K x=D[:,j].
    The solve error is bounded by ||K^-1||_inf times the explicitly certified
    residual infinity norm.  No generalized eigensolver tolerance is introduced.
    """

    n = K.shape[0]
    terms: list[float] = []
    for j in range(n):
        column = np.asarray(D.getcol(j).toarray()).ravel().astype(complex)
        if not np.any(column):
            terms.append(0.0)
            continue
        x = np.asarray(lu_K.solve(column), dtype=complex)
        residual_component = _matvec_residual_component_upper(K, x, column)
        residual_inf = float(np.max(residual_component))
        solution_error_inf = float(
            np.nextafter(inverse_inf_upper * residual_inf, np.inf)
        )
        terms.append(float(np.nextafter(abs(x[j]) + solution_error_inf, np.inf)))

    trace_upper = _inflate_sum(math.fsum(terms), n)
    return float(np.nextafter(trace_upper, np.inf))


@dataclass(frozen=True)
class PhysicalBlockEnergyPreconditioner:
    """Certified magnetic/scalar-conductive block preconditioner.

    For

        H = [[K_A + D_AA, D_Apsi],
             [D_psiA,       D_psipsi]],

    the physical assembly gives K_A>0 after tree-cotree gauge elimination and
    D>=0 from the conductivity Gram form.  If

        D_AA <= gamma K_A,

    conductive Cauchy-Schwarz plus Young's inequality yields

        H >= m(gamma) diag(K_A, D_psipsi),

    where

        m(gamma)=2/[2+gamma+sqrt(gamma^2+4 gamma)].

    The fast certificate first attempts a normalized Gershgorin bound.  When the
    physical magnetic block is not diagonally dominant, a deterministic
    residual-certified sparse-factorization fallback bounds

        gamma <= trace(K_A^-1 D_AA)

    without forming a dense inverse.  Failure of either certificate is reported;
    no diagonal shift or fitted damping is introduced.

    Block inverses currently use complete sparse LU as correctness-scale
    implementations.  The outer theorem only consumes the preconditioner action
    and proved m(gamma), so each block can later be replaced independently by a
    certified auxiliary-space/multilevel action.
    """

    n_A: int
    lower_spectral_equivalence_bound: float
    gamma_upper_bound: float
    gamma_certificate_method: str
    magnetic_normalized_lower_bound: float
    magnetic_inverse_inf_upper_bound: float
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
        try:
            lu_K = spla.splu(K.tocsc())
        except RuntimeError as exc:
            raise ValueError("magnetic block factorization failed") from exc

        H11 = metric[:n_A, :n_A].tocsr()
        D_AA = (H11 - K).tocsr()
        D_AA = (0.5 * (D_AA + D_AA.conj().T)).tocsr()
        D_AA.sum_duplicates()
        D_AA.eliminate_zeros()

        k_lower = _normalized_gershgorin_lower(K, K_diag)
        inverse_inf_upper = np.nan
        if k_lower > 0.0:
            d_upper = _normalized_gershgorin_upper(D_AA, K_diag)
            gamma_upper = float(np.nextafter(d_upper / k_lower, np.inf))
            gamma_method = "normalized_gershgorin"
        else:
            inverse_inf_upper = _certified_inverse_inf_upper(K, lu_K)
            gamma_upper = _certified_generalized_trace_upper(
                K,
                D_AA,
                lu_K,
                inverse_inf_upper,
            )
            gamma_method = "residual_certified_generalized_trace"
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
            # E is a principal block of the physical H>0 and hence SPD.  A
            # positive Gershgorin value is recorded when available, but lack of
            # diagonal dominance is not confused with loss of physical SPD.
            try:
                lu_scalar = spla.splu(E.tocsc())
            except RuntimeError as exc:
                raise ValueError("scalar conductive block factorization failed") from exc

        if n_scalar == 0 or gamma_upper == 0.0:
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
            gamma_certificate_method=gamma_method,
            magnetic_normalized_lower_bound=float(k_lower),
            magnetic_inverse_inf_upper_bound=float(inverse_inf_upper),
            scalar_normalized_lower_bound=float(scalar_lower),
            _lu_magnetic=lu_K,
            _lu_scalar=lu_scalar,
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.ndim != 1 or vector.size < self.n_A:
            raise ValueError("preconditioner rhs dimension mismatch")
        if self._lu_scalar is None and vector.size != self.n_A:
            raise ValueError("preconditioner rhs dimension mismatch")
        left = np.asarray(self._lu_magnetic.solve(vector[: self.n_A]), dtype=complex)
        if self._lu_scalar is None:
            return left
        right = np.asarray(self._lu_scalar.solve(vector[self.n_A :]), dtype=complex)
        return np.concatenate([left, right])


def make_physical_block_pcg_riesz_factory(problem):
    """Create the local-H physical block PCG Riesz factory for tetrahedra."""

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
