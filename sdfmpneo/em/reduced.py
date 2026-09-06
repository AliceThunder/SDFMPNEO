from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class RieszFactor:
    """Stable coordinate map for a Hermitian positive-definite Riesz metric."""

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
    """Affine electromagnetic problem implementing the generic reduced interface."""

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

    def operator_derivatives(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.shape != (self.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        return self.A_state

    def loss_operator(self, output_mode: int, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.shape != (self.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        H = self.H_loss[output_mode]
        if self.H_loss_state is not None:
            H = H + np.tensordot(a, self.H_loss_state[output_mode], axes=(0, 0))
        return H

    def loss_operator_derivative(
        self,
        output_mode: int,
        state_mode: int,
        a: np.ndarray,
    ) -> np.ndarray:
        if self.H_loss_state is None:
            return np.zeros_like(self.H_loss[output_mode])
        return self.H_loss_state[output_mode, state_mode]

    def solve_full(self, a: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(self.operator(a), self.b, assume_a="gen")


@dataclass
class ReducedEMModel:
    """Reduced solver for any problem implementing the EM operator interface."""

    problem: object
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
        operator_derivatives = self.problem.operator_derivatives(a)

        for k, Ak in enumerate(operator_derivatives):
            Akr = self.V.conj().T @ Ak @ self.V
            dc = scipy.linalg.solve(Ar, -(Akr @ c), assume_a="gen")
            dx = self.V @ dc
            for j in range(self.problem.n_thermal):
                Hj = self.problem.loss_operator(j, a)
                dH = self.problem.loss_operator_derivative(j, k, a)
                explicit = np.real(np.vdot(x, dH @ x))
                J[j, k] = 2.0 * np.real(np.vdot(dx, Hj @ x)) + explicit
        return q, J


class ResidualGreedyEMReducer:
    """Snapshot-free residual-Riesz electromagnetic basis construction."""

    def __init__(self, problem: object):
        self.problem = problem
        self.riesz = RieszFactor.build(problem.H_metric)

    def initial_basis(self) -> np.ndarray:
        q = self.riesz.orthonormalized_lift(self.problem.b)
        return q[:, None]

    def _initial_basis_from_rhs(self, rhs_matrix: np.ndarray) -> np.ndarray:
        B = np.asarray(rhs_matrix, dtype=complex)
        if B.ndim == 1:
            B = B[:, None]
        if B.ndim != 2 or B.shape[0] != self.problem.n_em:
            raise ValueError("rhs_matrix must have shape (n_em,n_rhs)")
        if B.shape[1] == 0:
            raise ValueError("rhs_matrix must contain at least one excitation")

        V = np.empty((self.problem.n_em, 0), dtype=complex)
        for p in range(B.shape[1]):
            try:
                q = self.riesz.orthonormalized_lift(B[:, p], V)
            except np.linalg.LinAlgError:
                continue
            V = np.column_stack([V, q])
        if V.shape[1] == 0:
            raise np.linalg.LinAlgError("all supplied excitations are null/dependent in the Riesz metric")
        return V

    def _rhs_residual(
        self,
        thermal_state: np.ndarray,
        rhs: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        A = self.problem.operator(thermal_state)
        Ar = V.conj().T @ A @ V
        br = V.conj().T @ rhs
        c = scipy.linalg.solve(Ar, br, assume_a="gen")
        return rhs - A @ (V @ c)

    def build_multi_rhs(
        self,
        candidate_states: Iterable[np.ndarray],
        rhs_matrix: np.ndarray,
        tolerance: float,
    ) -> ReducedEMModel:
        """Grow one reduced space for a joint state-by-excitation domain.

        The greedy maximization is taken over every supplied thermal state and
        every RHS column. This is required for multiport impedance/mutual
        inductance: a basis certified only for port 1 is not assumed to span the
        response of port 2.
        """

        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        states = [np.asarray(a, dtype=float) for a in candidate_states]
        if not states:
            raise ValueError("candidate_states cannot be empty")
        B = np.asarray(rhs_matrix, dtype=complex)
        if B.ndim == 1:
            B = B[:, None]
        if B.ndim != 2 or B.shape[0] != self.problem.n_em:
            raise ValueError("rhs_matrix must have shape (n_em,n_rhs)")

        V = self._initial_basis_from_rhs(B)
        while True:
            worst_norm = -1.0
            worst_residual = None
            for state in states:
                for p in range(B.shape[1]):
                    residual = self._rhs_residual(state, B[:, p], V)
                    value = self.riesz.dual_norm(residual)
                    if value > worst_norm:
                        worst_norm = value
                        worst_residual = residual

            if worst_norm <= tolerance or V.shape[1] >= self.problem.n_em:
                return ReducedEMModel(self.problem, V, self.riesz)

            try:
                q = self.riesz.orthonormalized_lift(worst_residual, V)
            except np.linalg.LinAlgError:
                return ReducedEMModel(self.problem, V, self.riesz)
            V = np.column_stack([V, q])

    def build(self, candidate_states: Iterable[np.ndarray], tolerance: float) -> ReducedEMModel:
        return self.build_multi_rhs(candidate_states, self.problem.b[:, None], tolerance)
