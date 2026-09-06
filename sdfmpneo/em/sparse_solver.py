from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class SparseLinearSolveCertificate:
    """Euclidean a-posteriori algebraic-state certificate."""

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


@dataclass(frozen=True)
class SparseEnergyLinearSolveCertificate:
    """Physical-energy a-posteriori certificate for A=K+iD, K,D>=0.

    With H=K+D,

        |x^H A x| >= (1/sqrt(2)) x^H H x,

    hence

        ||x-x_h||_H <= sqrt(2) ||b-Ax_h||_(H^-1).

    The stability constant is structural and independent of material contrast.
    """

    requested_energy_state_error: float
    coercivity_lower_bound: float
    residual_dual_energy_norm: float
    energy_state_error_bound: float
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


class ApsiEnergyMetric:
    """Sparse physical energy metric H=K+D for A=K+iD."""

    def __init__(self, H: sp.spmatrix) -> None:
        metric = sp.csr_matrix(H, dtype=complex)
        n = metric.shape[0]
        if metric.shape != (n, n):
            raise ValueError("energy metric must be square")
        self.H = metric
        self.n = n
        self._lu = spla.splu(metric.tocsc())

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        r = np.asarray(rhs, dtype=complex)
        if r.shape != (self.n,):
            raise ValueError("energy-metric rhs dimension mismatch")
        return np.asarray(self._lu.solve(r), dtype=complex)

    def norm(self, x: np.ndarray) -> float:
        state = np.asarray(x, dtype=complex)
        if state.shape != (self.n,):
            raise ValueError("energy state dimension mismatch")
        value = float(np.real(np.vdot(state, self.H @ state)))
        backward = np.finfo(float).eps * max(1, self.n) * max(1.0, float(np.linalg.norm(state)) ** 2)
        if value < -backward:
            raise np.linalg.LinAlgError("physical energy metric is not positive")
        return float(np.sqrt(max(0.0, value)))

    def dual_norm(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, dtype=complex)
        y = self.solve(r)
        value = float(np.real(np.vdot(r, y)))
        backward = np.finfo(float).eps * max(1, self.n) * max(1.0, float(np.linalg.norm(r)) * float(np.linalg.norm(y)))
        if value < -backward:
            raise np.linalg.LinAlgError("inverse physical energy metric is not positive")
        return float(np.sqrt(max(0.0, value)))

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
    """Reusable Euclidean certified sparse solver retained for compatibility."""

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


class _EnergyConverged(RuntimeError):
    def __init__(self, x: np.ndarray, iterations: int) -> None:
        super().__init__("energy residual target reached")
        self.x = np.asarray(x, dtype=complex).copy()
        self.iterations = int(iterations)


class CertifiedEnergySparseApsiSolver:
    """Contrast-independent energy-preconditioned certified sparse solver.

    The physical energy metric is also the left preconditioner. In exact
    arithmetic the H-scaled A-psi operator has coercivity at least 1/sqrt(2), so
    the certificate does not deteriorate when copper/seawater conductivity
    contrast grows.
    """

    COERCIVITY_LOWER_BOUND = 1.0 / np.sqrt(2.0)

    def __init__(self, A: sp.spmatrix, H: sp.spmatrix) -> None:
        matrix = sp.csr_matrix(A, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            raise ValueError("A must be square")
        if H.shape != (n, n):
            raise ValueError("H shape mismatch")
        self.matrix = matrix
        self.n = n
        self.energy = ApsiEnergyMetric(H)
        self._full_lu = None

    def _direct_lu(self):
        if self._full_lu is None:
            self._full_lu = spla.splu(self.matrix.tocsc())
        return self._full_lu

    def _certificate(
        self,
        rhs: np.ndarray,
        x: np.ndarray,
        *,
        requested: float,
        iterations: int,
        info: int,
        method: str,
    ) -> SparseEnergyLinearSolveCertificate:
        residual = rhs - self.matrix @ x
        dual = self.energy.dual_norm(residual)
        state_bound = dual / self.COERCIVITY_LOWER_BOUND
        return SparseEnergyLinearSolveCertificate(
            requested_energy_state_error=requested,
            coercivity_lower_bound=self.COERCIVITY_LOWER_BOUND,
            residual_dual_energy_norm=dual,
            energy_state_error_bound=float(state_bound),
            iterations=int(iterations),
            iterative_info=int(info),
            method=method,
            certified=bool(state_bound <= requested),
        )

    def solve(
        self,
        b: np.ndarray,
        *,
        requested_energy_state_error: float,
    ) -> tuple[np.ndarray, SparseEnergyLinearSolveCertificate]:
        rhs = np.asarray(b, dtype=complex)
        if rhs.shape != (self.n,):
            raise ValueError("b dimension mismatch")
        requested = float(requested_energy_state_error)
        if requested <= 0.0:
            raise ValueError("requested_energy_state_error must be positive")
        iterations = 0

        def _callback(xk: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1
            residual = rhs - self.matrix @ xk
            dual = self.energy.dual_norm(residual)
            if dual / self.COERCIVITY_LOWER_BOUND <= requested:
                raise _EnergyConverged(xk, iterations)

        try:
            x_iterative, info = spla.bicgstab(
                self.matrix,
                rhs,
                rtol=0.0,
                atol=0.0,
                maxiter=self.n,
                M=self.energy.as_linear_operator(),
                callback=_callback,
            )
            x_iterative = np.asarray(x_iterative, dtype=complex)
        except _EnergyConverged as reached:
            x_iterative = reached.x
            iterations = reached.iterations
            info = 0

        iterative_certificate = self._certificate(
            rhs,
            x_iterative,
            requested=requested,
            iterations=iterations,
            info=int(info),
            method="energy_bicgstab",
        )
        if iterative_certificate.certified:
            return x_iterative, iterative_certificate

        x_direct = np.asarray(self._direct_lu().solve(rhs), dtype=complex)
        direct_certificate = self._certificate(
            rhs,
            x_direct,
            requested=requested,
            iterations=iterations,
            info=int(info),
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
    solver = CertifiedSparseApsiSolver(
        A,
        n_A=n_A,
        stability_lower_bound=stability_lower_bound,
    )
    return solver.solve(b, requested_state_error=requested_state_error)


def solve_energy_certified_sparse_apsi(
    A: sp.spmatrix,
    H: sp.spmatrix,
    b: np.ndarray,
    *,
    requested_energy_state_error: float,
) -> tuple[np.ndarray, SparseEnergyLinearSolveCertificate]:
    solver = CertifiedEnergySparseApsiSolver(A, H)
    return solver.solve(b, requested_energy_state_error=requested_energy_state_error)
