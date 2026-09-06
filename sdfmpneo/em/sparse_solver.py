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
    """Parameter-free exact-block preconditioner for the sparse A-psi system.

    For

        A = [A11 A12]
            [A21 A22],

    this applies the inverse of the upper block-triangular matrix

        P = [A11 A12]
            [  0 A22].

    A11 and A22 are factored with sparse LU without drop tolerances or incomplete
    factorisation parameters. Material contrast inside each diagonal block is
    therefore retained exactly up to the sparse direct-factorisation backward
    error; the preconditioner changes convergence speed only and cannot make an
    uncertified solution pass the final residual certificate.
    """

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


def solve_certified_sparse_apsi(
    A: sp.spmatrix,
    b: np.ndarray,
    *,
    n_A: int,
    stability_lower_bound: float,
    requested_state_error: float,
) -> tuple[np.ndarray, SparseLinearSolveCertificate]:
    """Solve a sparse A-psi system and certify its algebraic state error.

    Stage 1 uses block-preconditioned BiCGSTAB with an iteration work bound equal
    to the electromagnetic coordinate dimension. This is dimension-derived, not
    fitted. If that short-recurrence iteration does not satisfy the requested
    a-posteriori error bound, Stage 2 deterministically falls back to a complete
    sparse LU factorisation of the full matrix. The fallback has no drop tolerance
    or empirical fill parameter. It provides a correctness/high-contrast baseline;
    scalable multilevel replacement of that fallback is a separate production
    task.

    Neither stage may relax the requested state error. The returned certificate
    is always computed from the original unpreconditioned matrix and the explicitly
    recomputed true residual.
    """

    matrix = sp.csr_matrix(A, dtype=complex)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("A must be square")
    rhs = np.asarray(b, dtype=complex)
    if rhs.shape != (n,):
        raise ValueError("b dimension mismatch")
    beta = float(stability_lower_bound)
    error = float(requested_state_error)
    if beta <= 0.0:
        raise ValueError("stability_lower_bound must be positive")
    if error <= 0.0:
        raise ValueError("requested_state_error must be positive")

    residual_target = beta * error
    preconditioner = ApsiBlockTriangularPreconditioner(matrix, n_A)
    iterations = 0

    def _count(_xk: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    x_iterative, info = spla.bicgstab(
        matrix,
        rhs,
        rtol=0.0,
        atol=residual_target,
        maxiter=n,
        M=preconditioner.as_linear_operator(),
        callback=_count,
    )
    x_iterative = np.asarray(x_iterative, dtype=complex)
    iterative_certificate = _true_residual_certificate(
        matrix,
        rhs,
        x_iterative,
        beta=beta,
        error=error,
        residual_target=residual_target,
        iterations=iterations,
        iterative_info=int(info),
        method="block_bicgstab",
    )
    if iterative_certificate.certified:
        return x_iterative, iterative_certificate

    full_lu = spla.splu(matrix.tocsc())
    x_direct = np.asarray(full_lu.solve(rhs), dtype=complex)
    direct_certificate = _true_residual_certificate(
        matrix,
        rhs,
        x_direct,
        beta=beta,
        error=error,
        residual_target=residual_target,
        iterations=iterations,
        iterative_info=int(info),
        method="sparse_lu_fallback",
    )
    return x_direct, direct_certificate
