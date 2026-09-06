from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class SparseLinearSolveCertificate:
    """A-posteriori algebraic-state certificate for a sparse field solve.

    For A x = b, if a proven lower bound beta <= sigma_min(A) is available,

        ||x-x_h||_2 <= ||b-A x_h||_2 / beta.

    The requested algebraic state error therefore determines the solver's
    absolute residual target. No empirical linear-solver tolerance is introduced.
    The final certificate is always evaluated from the explicitly recomputed true
    residual, not from the iterative solver's internal stopping flag.
    """

    requested_state_error: float
    stability_lower_bound: float
    residual_target: float
    residual_norm: float
    relative_residual_norm: float
    state_error_bound: float
    iterations: int
    iterative_info: int
    method: str
    certified: bool


class ApsiBlockTriangularPreconditioner:
    """Parameter-free exact-block preconditioner for the sparse A-psi system."""

    def __init__(self, A: sp.spmatrix, n_A: int) -> None:
        matrix = sp.csr_matrix(A, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            raise ValueError("A must be square")
        if not 0 < int(n_A) <= n:
            raise ValueError("n_A must satisfy 0 < n_A <= system dimension")
        self.n = n
        self.n_A = int(n_A)
        self.n_scalar = n - self.n_A
        self.A12 = matrix[: self.n_A, self.n_A :].tocsr()
        self._lu11 = spla.splu(matrix[: self.n_A, : self.n_A].tocsc())
        self._lu22 = None
        if self.n_scalar:
            self._lu22 = spla.splu(matrix[self.n_A :, self.n_A :].tocsc())

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        r = np.asarray(rhs, dtype=complex)
        if r.shape != (self.n,):
            raise ValueError("preconditioner rhs dimension mismatch")
        r1 = r[: self.n_A]
        if not self.n_scalar:
            return self._lu11.solve(r1)
        r2 = r[self.n_A :]
        y2 = self._lu22.solve(r2)
        y1 = self._lu11.solve(r1 - self.A12 @ y2)
        return np.concatenate([y1, y2])

    def as_linear_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(
            (self.n, self.n),
            matvec=self.solve,
            dtype=np.dtype(complex),
        )


def _true_residual_certificate(
    matrix: sp.csr_matrix,
    rhs: np.ndarray,
    x: np.ndarray,
    *,
    beta: float,
    error: float,
    residual_target: float,
    iterations: int,
    iterative_info: int,
    method: str,
) -> SparseLinearSolveCertificate:
    true_residual = rhs - matrix @ x
    residual_norm = float(np.linalg.norm(true_residual))
    rhs_norm = float(np.linalg.norm(rhs))
    relative = residual_norm if rhs_norm == 0.0 else residual_norm / rhs_norm
    state_bound = residual_norm / beta
    return SparseLinearSolveCertificate(
        requested_state_error=error,
        stability_lower_bound=beta,
        residual_target=residual_target,
        residual_norm=residual_norm,
        relative_residual_norm=float(relative),
        state_error_bound=float(state_bound),
        iterations=int(iterations),
        iterative_info=int(iterative_info),
        method=method,
        certified=bool(state_bound <= error),
    )


class CertifiedSparseApsiSolver:
    """Reusable certified sparse solver for one assembled electromagnetic state.

    All port/right-hand-side solves at fixed thermal state share the same block
    preconditioner and, if needed, the same complete sparse LU fallback. Thus the
    expensive matrix factorisations are state-dependent, not port-dependent.
    """

    def __init__(
        self,
        A: sp.spmatrix,
        *,
        n_A: int,
        stability_lower_bound: float,
    ) -> None:
        matrix = sp.csr_matrix(A, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            raise ValueError("A must be square")
        beta = float(stability_lower_bound)
        if beta <= 0.0:
            raise ValueError("stability_lower_bound must be positive")
        self.matrix = matrix
        self.n = n
        self.n_A = int(n_A)
        self.beta = beta
        self.preconditioner = ApsiBlockTriangularPreconditioner(matrix, n_A)
        self._full_lu = None

    def _direct_lu(self):
        if self._full_lu is None:
            self._full_lu = spla.splu(self.matrix.tocsc())
        return self._full_lu

    def solve(
        self,
        b: np.ndarray,
        *,
        requested_state_error: float,
    ) -> tuple[np.ndarray, SparseLinearSolveCertificate]:
        rhs = np.asarray(b, dtype=complex)
        if rhs.shape != (self.n,):
            raise ValueError("b dimension mismatch")
        error = float(requested_state_error)
        if error <= 0.0:
            raise ValueError("requested_state_error must be positive")
        residual_target = self.beta * error
        iterations = 0

        def _count(_xk: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        x_iterative, info = spla.bicgstab(
            self.matrix,
            rhs,
            rtol=0.0,
            atol=residual_target,
            maxiter=self.n,
            M=self.preconditioner.as_linear_operator(),
            callback=_count,
        )
        x_iterative = np.asarray(x_iterative, dtype=complex)
        iterative_certificate = _true_residual_certificate(
            self.matrix,
            rhs,
            x_iterative,
            beta=self.beta,
            error=error,
            residual_target=residual_target,
            iterations=iterations,
            iterative_info=int(info),
            method="block_bicgstab",
        )
        if iterative_certificate.certified:
            return x_iterative, iterative_certificate

        x_direct = np.asarray(self._direct_lu().solve(rhs), dtype=complex)
        direct_certificate = _true_residual_certificate(
            self.matrix,
            rhs,
            x_direct,
            beta=self.beta,
            error=error,
            residual_target=residual_target,
            iterations=iterations,
            iterative_info=int(info),
            method="sparse_lu_fallback",
        )
        return x_direct, direct_certificate


def solve_certified_sparse_apsi(
    A: sp.spmatrix,
    b: np.ndarray,
    *,
    n_A: int,
    stability_lower_bound: float,
    requested_state_error: float,
) -> tuple[np.ndarray, SparseLinearSolveCertificate]:
    """One-shot convenience wrapper around `CertifiedSparseApsiSolver`."""

    solver = CertifiedSparseApsiSolver(
        A,
        n_A=n_A,
        stability_lower_bound=stability_lower_bound,
    )
    return solver.solve(b, requested_state_error=requested_state_error)
