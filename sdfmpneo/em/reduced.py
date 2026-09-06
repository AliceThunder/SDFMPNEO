from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class RieszFactor:
    """Stable coordinate map for a Hermitian positive-definite Riesz metric.

    If H = L L^H, then the dual norm and Riesz lift are

        ||r||_{H^-1} = ||L^-1 r||_2,
        H^-1 r       = L^-H L^-1 r.

    All residual-Riesz orthogonalisation is carried out in these normalized
    coordinates instead of explicitly solving H w = r and forming V^H H V.
    """

    H: np.ndarray
    L: np.ndarray

    @classmethod
    def build(cls, H: np.ndarray) -> "RieszFactor":
        H = np.asarray(H, dtype=complex)
        L = scipy.linalg.cholesky(H, lower=True, check_finite=True)
        return cls(H=H, L=L)

    def dual_coordinates(self, residual: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve_triangular(
            self.L,
            np.asarray(residual, dtype=complex),
            lower=True,
            check_finite=True,
        )

    def dual_norm(self, residual: np.ndarray) -> float:
        return float(np.linalg.norm(self.dual_coordinates(residual)))

    def riesz_lift(self, residual: np.ndarray) -> np.ndarray:
        y = self.dual_coordinates(residual)
        return scipy.linalg.solve_triangular(
            self.L.conj().T,
            y,
            lower=False,
            check_finite=True,
        )

    def orthonormalized_lift(
        self,
        residual: np.ndarray,
        basis: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the H-normalized residual lift orthogonal to an H-ON basis.

        Linear-dependence detection uses the standard machine-precision
        backward-error scale eps*n*||y||. This is a numerical-algebra safeguard,
        not a scientific model threshold.
        """

        y = self.dual_coordinates(residual)
        original_norm = float(np.linalg.norm(y))

        if basis is not None and basis.size:
            V = np.asarray(basis, dtype=complex)
            Y = self.L.conj().T @ V
            y = y - Y @ (Y.conj().T @ y)

        norm = float(np.linalg.norm(y))
        backward_scale = (
            np.finfo(float).eps
            * max(1, y.size)
            * max(1.0, original_norm)
        )
        if norm <= backward_scale:
            raise np.linalg.LinAlgError(
                "Riesz lift is linearly dependent at the machine-precision backward-error scale"
            )

        return scipy.linalg.solve_triangular(
            self.L.conj().T,
            y / norm,
            lower=False,
            check_finite=True,
        )


@dataclass(frozen=True)
class ParametricEMProblem:
    """Deterministic affine-in-thermal-state electromagnetic operator.

    A(a) = A0 + sum_k a_k A_state[k].
    Reduced heat-source components use
    H_j(a) = H_loss[j] + sum_k a_k H_loss_state[j,k].
    """

    A0: np.ndarray
    A_state: np.ndarray
    b: np.ndarray
    H_metric: np.ndarray
    H_loss: np.ndarray
    H_loss_state: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.A0.shape[0]
        if self.A0.shape != (n, n):
            raise ValueError("A0 must be square")
        if self.A_state.ndim != 3 or self.A_state.shape[1:] != (n, n):
            raise ValueError("A_state must have shape (n_thermal,n_em,n_em)")
        if self.H_loss.shape != self.A_state.shape:
            raise ValueError("H_loss must have shape (n_thermal,n_em,n_em)")
        if self.b.shape != (n,) or self.H_metric.shape != (n, n):
            raise ValueError("b/H_metric shape mismatch")
        if self.H_loss_state is not None:
            expected = (self.n_thermal, self.n_thermal, n, n)
            if self.H_loss_state.shape != expected:
                raise ValueError(f"H_loss_state must have shape {expected}")
        RieszFactor.build(self.H_metric)

    @property
    def n_thermal(self) -> int:
        return self.A_state.shape[0]

    @property
    def n_em(self) -> int:
        return self.A0.shape[0]

    def operator(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.shape != (self.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        return self.A0 + np.tensordot(a, self.A_state, axes=(0, 0))

    def loss_operator(self, output_mode: int, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.shape != (self.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        H = self.H_loss[output_mode]
        if self.H_loss_state is not None:
            H = H + np.tensordot(a, self.H_loss_state[output_mode], axes=(0, 0))
        return H

    def solve_full(self, a: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(self.operator(a), self.b, assume_a="gen")


@dataclass
class ReducedEMModel:
    problem: ParametricEMProblem
    V: np.ndarray
    riesz: RieszFactor | None = None

    def __post_init__(self) -> None:
        if self.riesz is None:
            self.riesz = RieszFactor.build(self.problem.H_metric)

    def operator_reduced(self, a: np.ndarray) -> np.ndarray:
        return self.V.conj().T @ self.problem.operator(a) @ self.V

    def rhs_reduced(self) -> np.ndarray:
        return self.V.conj().T @ self.problem.b

    def solve_coeff(self, a: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(self.operator_reduced(a), self.rhs_reduced(), assume_a="gen")

    def state(self, a: np.ndarray) -> np.ndarray:
        return self.V @ self.solve_coeff(a)

    def full_residual(self, a: np.ndarray) -> np.ndarray:
        x = self.state(a)
        return self.problem.b - self.problem.operator(a) @ x

    def residual_dual_norm(self, a: np.ndarray) -> float:
        return self.riesz.dual_norm(self.full_residual(a))

    def heat_source(self, a: np.ndarray) -> np.ndarray:
        x = self.state(a)
        return np.array(
            [
                np.real(np.vdot(x, self.problem.loss_operator(j, a) @ x))
                for j in range(self.problem.n_thermal)
            ]
        )

    def heat_source_and_jacobian(self, a: np.ndarray):
        a = np.asarray(a, dtype=float)
        A = self.problem.operator(a)
        c = self.solve_coeff(a)
        Ar = self.V.conj().T @ A @ self.V
        x = self.V @ c
        q = self.heat_source(a)
        J = np.zeros((self.problem.n_thermal, self.problem.n_thermal), dtype=float)

        for k, Ak in enumerate(self.problem.A_state):
            Akr = self.V.conj().T @ Ak @ self.V
            dc = scipy.linalg.solve(Ar, -(Akr @ c), assume_a="gen")
            dx = self.V @ dc
            for j in range(self.problem.n_thermal):
                Hj = self.problem.loss_operator(j, a)
                explicit = 0.0
                if self.problem.H_loss_state is not None:
                    explicit = np.real(np.vdot(x, self.problem.H_loss_state[j, k] @ x))
                J[j, k] = 2.0 * np.real(np.vdot(dx, Hj @ x)) + explicit
        return q, J


class ResidualGreedyEMReducer:
    """Snapshot-free residual-Riesz electromagnetic basis construction."""

    def __init__(self, problem: ParametricEMProblem):
        self.problem = problem
        self.riesz = RieszFactor.build(problem.H_metric)

    def initial_basis(self) -> np.ndarray:
        q = self.riesz.orthonormalized_lift(self.problem.b)
        return q[:, None]

    def build(self, candidate_states: Iterable[np.ndarray], tolerance: float) -> ReducedEMModel:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        states = [np.asarray(a, dtype=float) for a in candidate_states]
        if not states:
            raise ValueError("candidate_states cannot be empty")

        V = self.initial_basis()
        while True:
            model = ReducedEMModel(self.problem, V, self.riesz)
            values = np.array([model.residual_dual_norm(a) for a in states])
            idx = int(np.argmax(values))
            if float(values[idx]) <= tolerance or V.shape[1] >= self.problem.n_em:
                return model

            residual = model.full_residual(states[idx])
            try:
                q = self.riesz.orthonormalized_lift(residual, V)
            except np.linalg.LinAlgError:
                return model
            V = np.column_stack([V, q])
