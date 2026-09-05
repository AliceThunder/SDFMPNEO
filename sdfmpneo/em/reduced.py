from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg


def _h_inner(x: np.ndarray, y: np.ndarray, H: np.ndarray) -> complex:
    return np.vdot(x, H @ y)


def _h_normalize(v: np.ndarray, H: np.ndarray, basis: Sequence[np.ndarray]) -> np.ndarray:
    w = np.asarray(v, dtype=complex).copy()
    for q in basis:
        w -= q * _h_inner(q, w, H)
    n2 = np.real(_h_inner(w, w, H))
    if n2 <= 0:
        raise np.linalg.LinAlgError("Candidate vector is linearly dependent in the H metric")
    return w / np.sqrt(n2)


@dataclass(frozen=True)
class ParametricEMProblem:
    """First executable full-order deterministic EM operator A(a)=A0+sum a_k A_k."""

    A0: np.ndarray
    A_state: np.ndarray
    b: np.ndarray
    H_metric: np.ndarray
    H_loss: np.ndarray

    def __post_init__(self) -> None:
        n = self.A0.shape[0]
        if self.A0.shape != (n, n):
            raise ValueError("A0 must be square")
        if self.A_state.ndim != 3 or self.A_state.shape[1:] != (n, n):
            raise ValueError("A_state must have shape (n_thermal,n_em,n_em)")
        if self.H_loss.shape != self.A_state.shape:
            raise ValueError("H_loss must match A_state shape")
        if self.b.shape != (n,) or self.H_metric.shape != (n, n):
            raise ValueError("b/H_metric shape mismatch")
        scipy.linalg.cholesky(self.H_metric, lower=True, check_finite=True)

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

    def solve_full(self, a: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(self.operator(a), self.b, assume_a="gen")


@dataclass
class ReducedEMModel:
    problem: ParametricEMProblem
    V: np.ndarray

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
        r = self.full_residual(a)
        z = scipy.linalg.solve(self.problem.H_metric, r, assume_a="her")
        return float(np.sqrt(max(0.0, np.real(np.vdot(r, z)))))

    def heat_source(self, a: np.ndarray) -> np.ndarray:
        x = self.state(a)
        return np.array([np.real(np.vdot(x, H @ x)) for H in self.problem.H_loss])

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
            for j, Hj in enumerate(self.problem.H_loss):
                J[j, k] = 2.0 * np.real(np.vdot(dx, Hj @ x))
        return q, J


class ResidualGreedyEMReducer:
    """Snapshot-free residual-Riesz electromagnetic basis construction."""

    def __init__(self, problem: ParametricEMProblem):
        self.problem = problem

    def initial_basis(self) -> np.ndarray:
        w = scipy.linalg.solve(self.problem.H_metric, self.problem.b, assume_a="her")
        return _h_normalize(w, self.problem.H_metric, [])[:, None]

    def build(self, candidate_states: Iterable[np.ndarray], tolerance: float) -> ReducedEMModel:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        states = [np.asarray(a, dtype=float) for a in candidate_states]
        if not states:
            raise ValueError("candidate_states cannot be empty")
        V = self.initial_basis()
        while True:
            model = ReducedEMModel(self.problem, V)
            values = np.array([model.residual_dual_norm(a) for a in states])
            idx = int(np.argmax(values))
            if float(values[idx]) <= tolerance or V.shape[1] >= self.problem.n_em:
                return model
            r = model.full_residual(states[idx])
            w = scipy.linalg.solve(self.problem.H_metric, r, assume_a="her")
            basis = [V[:, i] for i in range(V.shape[1])]
            try:
                q = _h_normalize(w, self.problem.H_metric, basis)
            except np.linalg.LinAlgError:
                return model
            V = np.column_stack([V, q])
