from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from .energy_solver import apsi_physical_energy_metric
from .riesz_action import (
    CertifiedRieszAction,
    RieszActionFactory,
    SparseLUReferenceRieszAction,
    instantiate_riesz_action,
)
from .sparse_solver import CertifiedEnergySparseApsiSolver


_COERCIVITY = CertifiedEnergySparseApsiSolver.COERCIVITY_LOWER_BOUND


@dataclass(frozen=True)
class SparseEnergyResidualCertificate:
    """A-posteriori reduced-state certificate in the local physical energy.

    The dual norm may be obtained from an inexact Riesz action. Therefore the
    certificate stores a rigorous enclosure; the state-error bound always uses
    the upper endpoint.
    """

    residual_dual_energy_norm_lower_bound: float
    residual_dual_energy_norm_upper_bound: float
    energy_state_error_bound: float

    @property
    def residual_dual_energy_norm(self) -> float:
        return self.residual_dual_energy_norm_upper_bound


@dataclass(frozen=True)
class SparseEnergyReductionCertificate:
    requested_energy_state_error: float
    maximum_energy_state_error_bound: float
    worst_state_index: int
    worst_rhs_index: int
    basis_dimension: int
    certified: bool
    stalled: bool


@dataclass(frozen=True)
class _StateContext:
    state: np.ndarray
    A: sp.csr_matrix
    H: sp.csr_matrix
    riesz_action: CertifiedRieszAction


class SparseEnergyReducedEMModel:
    """Sparse full-order / dense reduced electromagnetic model."""

    def __init__(
        self,
        problem: object,
        basis: np.ndarray,
        *,
        reference_energy_metric: sp.spmatrix,
        reduction_certificate: SparseEnergyReductionCertificate | None = None,
        riesz_action_factory: RieszActionFactory = SparseLUReferenceRieszAction,
    ) -> None:
        V = np.asarray(basis, dtype=complex)
        if V.ndim != 2 or V.shape[0] != problem.n_em:
            raise ValueError("basis must have shape (n_em,n_reduced)")
        if V.shape[1] == 0:
            raise ValueError("basis must contain at least one vector")
        self.problem = problem
        self.V = V
        self.reference_energy_metric = sp.csr_matrix(reference_energy_metric, dtype=complex)
        self.reduction_certificate = reduction_certificate
        self.riesz_action_factory = riesz_action_factory

    @property
    def n_reduced(self) -> int:
        return self.V.shape[1]

    def _operator_sparse(self, a: np.ndarray) -> sp.csr_matrix:
        if not hasattr(self.problem, "operator_sparse"):
            raise TypeError("problem must provide operator_sparse for sparse reduction")
        return sp.csr_matrix(self.problem.operator_sparse(a), dtype=complex)

    def operator_reduced(self, a: np.ndarray) -> np.ndarray:
        A = self._operator_sparse(a)
        return self.V.conj().T @ (A @ self.V)

    def rhs_reduced(self, rhs: np.ndarray | None = None) -> np.ndarray:
        source = self.problem.b if rhs is None else np.asarray(rhs, dtype=complex)
        if source.shape != (self.problem.n_em,):
            raise ValueError("rhs dimension mismatch")
        return self.V.conj().T @ source

    def solve_coeff_for_rhs(self, a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(
            self.operator_reduced(a),
            self.rhs_reduced(rhs),
            assume_a="gen",
        )

    def solve_coeff(self, a: np.ndarray) -> np.ndarray:
        return self.solve_coeff_for_rhs(a, self.problem.b)

    def state_for_rhs(self, a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return self.V @ self.solve_coeff_for_rhs(a, rhs)

    def state(self, a: np.ndarray) -> np.ndarray:
        return self.state_for_rhs(a, self.problem.b)

    def full_residual_for_rhs(self, a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        source = np.asarray(rhs, dtype=complex)
        if source.shape != (self.problem.n_em,):
            raise ValueError("rhs dimension mismatch")
        A = self._operator_sparse(a)
        return source - A @ self.state_for_rhs(a, source)

    def full_residual(self, a: np.ndarray) -> np.ndarray:
        return self.full_residual_for_rhs(a, self.problem.b)

    def residual_certificate_for_rhs(
        self,
        a: np.ndarray,
        rhs: np.ndarray,
    ) -> SparseEnergyResidualCertificate:
        state = np.asarray(a, dtype=float)
        A = self._operator_sparse(state)
        H = apsi_physical_energy_metric(A)
        action = instantiate_riesz_action(
            self.riesz_action_factory,
            H,
            state=state,
        )
        residual = np.asarray(rhs, dtype=complex) - A @ self.state_for_rhs(state, rhs)
        threshold = 0.0
        if self.reduction_certificate is not None:
            threshold = self.reduction_certificate.requested_energy_state_error * _COERCIVITY
        result = action.decide_dual_norm(residual, threshold=threshold).result
        return SparseEnergyResidualCertificate(
            residual_dual_energy_norm_lower_bound=float(result.dual_norm_lower_bound),
            residual_dual_energy_norm_upper_bound=float(result.dual_norm_upper_bound),
            energy_state_error_bound=float(result.dual_norm_upper_bound / _COERCIVITY),
        )

    def residual_certificate(self, a: np.ndarray) -> SparseEnergyResidualCertificate:
        return self.residual_certificate_for_rhs(a, self.problem.b)

    def heat_source_for_rhs(self, a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if not hasattr(self.problem, "loss_operator_sparse"):
            raise TypeError("problem must provide loss_operator_sparse for sparse heat projection")
        x = self.state_for_rhs(a, rhs)
        return np.array(
            [
                np.real(np.vdot(x, self.problem.loss_operator_sparse(j, a) @ x))
                for j in range(self.problem.n_thermal)
            ],
            dtype=float,
        )

    def heat_source(self, a: np.ndarray) -> np.ndarray:
        return self.heat_source_for_rhs(a, self.problem.b)

    def heat_source_and_jacobian_for_rhs(self, a: np.ndarray, rhs: np.ndarray):
        if not hasattr(self.problem, "operator_derivative_sparse"):
            raise TypeError("problem must provide operator_derivative_sparse")
        if not hasattr(self.problem, "loss_operator_derivative_sparse"):
            raise TypeError("problem must provide loss_operator_derivative_sparse")

        state = np.asarray(a, dtype=float)
        source = np.asarray(rhs, dtype=complex)
        if state.shape != (self.problem.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        if source.shape != (self.problem.n_em,):
            raise ValueError("rhs dimension mismatch")

        A = self._operator_sparse(state)
        Ar = self.V.conj().T @ (A @ self.V)
        c = scipy.linalg.solve(Ar, self.V.conj().T @ source, assume_a="gen")
        x = self.V @ c
        q = self.heat_source_for_rhs(state, source)
        J = np.zeros((self.problem.n_thermal, self.problem.n_thermal), dtype=float)

        for k in range(self.problem.n_thermal):
            Ak = sp.csr_matrix(self.problem.operator_derivative_sparse(state, k), dtype=complex)
            Akr = self.V.conj().T @ (Ak @ self.V)
            dc = scipy.linalg.solve(Ar, -(Akr @ c), assume_a="gen")
            dx = self.V @ dc
            for j in range(self.problem.n_thermal):
                Hj = sp.csr_matrix(self.problem.loss_operator_sparse(j, state), dtype=complex)
                dH = sp.csr_matrix(
                    self.problem.loss_operator_derivative_sparse(j, k, state),
                    dtype=complex,
                )
                J[j, k] = (
                    2.0 * np.real(np.vdot(dx, Hj @ x))
                    + np.real(np.vdot(x, dH @ x))
                )
        return q, J

    def heat_source_and_jacobian(self, a: np.ndarray):
        return self.heat_source_and_jacobian_for_rhs(a, self.problem.b)


class SparseEnergyResidualGreedyEMReducer:
    """Snapshot-free residual-Riesz reduction in state-dependent physical energy."""

    def __init__(
        self,
        problem: object,
        *,
        riesz_action_factory: RieszActionFactory = SparseLUReferenceRieszAction,
    ) -> None:
        if not hasattr(problem, "operator_sparse"):
            raise TypeError("problem must provide operator_sparse")
        self.problem = problem
        self.riesz_action_factory = riesz_action_factory
        self.reference_state = np.zeros(problem.n_thermal, dtype=float)
        A0 = sp.csr_matrix(problem.operator_sparse(self.reference_state), dtype=complex)
        self.H0 = apsi_physical_energy_metric(A0)
        self.reference_riesz_action = instantiate_riesz_action(
            self.riesz_action_factory,
            self.H0,
            state=self.reference_state,
        )

    @staticmethod
    def _energy_norm(H: sp.csr_matrix, vector: np.ndarray) -> float:
        v = np.asarray(vector, dtype=complex)
        Hv = H @ v
        value = float(np.real(np.vdot(v, Hv)))
        scale = float(np.linalg.norm(v)) * float(np.linalg.norm(Hv))
        if scale == 0.0:
            return 0.0
        eps = np.finfo(float).eps
        operation_count = max(1, 8 * v.size)
        if operation_count * eps >= 1.0:
            raise FloatingPointError("energy inner product is too large for gamma_k bound")
        gamma = operation_count * eps / (1.0 - operation_count * eps)
        backward = gamma * scale
        if value < -backward:
            raise np.linalg.LinAlgError("physical energy metric is not positive")
        return float(np.sqrt(max(value, 0.0)))

    def _h0_project_out(self, vector: np.ndarray, V: np.ndarray) -> np.ndarray:
        y = np.asarray(vector, dtype=complex).copy()
        if V.size == 0:
            return y
        HV = self.H0 @ V
        gram = V.conj().T @ HV
        rhs = V.conj().T @ (self.H0 @ y)
        coeff = scipy.linalg.solve(gram, rhs, assume_a="her")
        return y - V @ coeff

    def _whiten_h0(self, vectors: np.ndarray) -> np.ndarray:
        W = np.asarray(vectors, dtype=complex)
        if W.ndim != 2 or W.shape[0] != self.problem.n_em or W.shape[1] == 0:
            raise ValueError("vectors must have shape (n_em,n_vectors) with n_vectors>0")
        gram = W.conj().T @ (self.H0 @ W)
        gram = 0.5 * (gram + gram.conj().T)
        L = scipy.linalg.cholesky(gram, lower=True, check_finite=True)
        transform = scipy.linalg.solve_triangular(
            L.conj().T,
            np.eye(W.shape[1], dtype=complex),
            lower=False,
            check_finite=True,
        )
        return W @ transform

    def _append_h0_independent(self, vector: np.ndarray, V: np.ndarray) -> np.ndarray:
        original = np.asarray(vector, dtype=complex)
        y = self._h0_project_out(original, V)
        original_norm = self._energy_norm(self.H0, original)
        norm = self._energy_norm(self.H0, y)
        if original_norm == 0.0:
            raise np.linalg.LinAlgError("candidate Riesz lift has zero H0 energy")
        eps = np.finfo(float).eps
        operation_count = max(1, self.problem.n_em + V.shape[1] + 1)
        if operation_count * eps >= 1.0:
            raise FloatingPointError("basis dimension is too large for gamma_k bound")
        gamma = operation_count * eps / (1.0 - operation_count * eps)
        if norm <= gamma * original_norm:
            raise np.linalg.LinAlgError(
                "candidate lift is H0-dependent at the floating-point backward-error scale"
            )
        return self._whiten_h0(np.column_stack([V, y / norm]))

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
            lift = self.reference_riesz_action.decide_dual_norm(
                B[:, p], threshold=0.0
            ).result.vector
            try:
                V = self._append_h0_independent(lift, V)
            except np.linalg.LinAlgError:
                continue
        if V.shape[1] == 0:
            raise np.linalg.LinAlgError(
                "all supplied excitations are null/dependent in the reference physical energy"
            )
        return V

    def _state_context(self, state: np.ndarray) -> _StateContext:
        a = np.asarray(state, dtype=float)
        if a.shape != (self.problem.n_thermal,):
            raise ValueError("candidate thermal state dimension mismatch")
        A = sp.csr_matrix(self.problem.operator_sparse(a), dtype=complex)
        H = apsi_physical_energy_metric(A)
        action = instantiate_riesz_action(self.riesz_action_factory, H, state=a)
        return _StateContext(a.copy(), A, H, action)

    @staticmethod
    def _residual(context: _StateContext, rhs: np.ndarray, V: np.ndarray) -> np.ndarray:
        AV = context.A @ V
        Ar = V.conj().T @ AV
        br = V.conj().T @ rhs
        c = scipy.linalg.solve(Ar, br, assume_a="gen")
        return rhs - AV @ c

    def build_multi_rhs(
        self,
        candidate_states: Iterable[np.ndarray],
        rhs_matrix: np.ndarray,
        *,
        requested_energy_state_error: float,
    ) -> SparseEnergyReducedEMModel:
        requested = float(requested_energy_state_error)
        if requested <= 0.0:
            raise ValueError("requested_energy_state_error must be positive")

        states = [np.asarray(a, dtype=float) for a in candidate_states]
        if not states:
            raise ValueError("candidate_states cannot be empty")
        contexts = [self._state_context(a) for a in states]

        B = np.asarray(rhs_matrix, dtype=complex)
        if B.ndim == 1:
            B = B[:, None]
        if B.ndim != 2 or B.shape[0] != self.problem.n_em:
            raise ValueError("rhs_matrix must have shape (n_em,n_rhs)")

        V = self._initial_basis_from_rhs(B)
        dual_threshold = requested * _COERCIVITY

        while True:
            maximum_bound = -1.0
            selected_state = -1
            selected_rhs = -1
            selected_lift = None
            all_candidates_certified = True

            for sidx, context in enumerate(contexts):
                for p in range(B.shape[1]):
                    residual = self._residual(context, B[:, p], V)
                    decision = context.riesz_action.decide_dual_norm(
                        residual,
                        threshold=dual_threshold,
                    )
                    result = decision.result
                    bound = float(result.dual_norm_upper_bound / _COERCIVITY)
                    if decision.relation != "below":
                        all_candidates_certified = False
                    if bound > maximum_bound:
                        maximum_bound = bound
                        selected_state = sidx
                        selected_rhs = p
                        selected_lift = result.vector

            if all_candidates_certified:
                certificate = SparseEnergyReductionCertificate(
                    requested_energy_state_error=requested,
                    maximum_energy_state_error_bound=maximum_bound,
                    worst_state_index=selected_state,
                    worst_rhs_index=selected_rhs,
                    basis_dimension=V.shape[1],
                    certified=True,
                    stalled=False,
                )
                return SparseEnergyReducedEMModel(
                    self.problem,
                    V,
                    reference_energy_metric=self.H0,
                    reduction_certificate=certificate,
                    riesz_action_factory=self.riesz_action_factory,
                )

            stalled = V.shape[1] >= self.problem.n_em
            if not stalled and selected_lift is not None:
                try:
                    V = self._append_h0_independent(selected_lift, V)
                    continue
                except np.linalg.LinAlgError:
                    stalled = True

            certificate = SparseEnergyReductionCertificate(
                requested_energy_state_error=requested,
                maximum_energy_state_error_bound=maximum_bound,
                worst_state_index=selected_state,
                worst_rhs_index=selected_rhs,
                basis_dimension=V.shape[1],
                certified=False,
                stalled=stalled,
            )
            return SparseEnergyReducedEMModel(
                self.problem,
                V,
                reference_energy_metric=self.H0,
                reduction_certificate=certificate,
                riesz_action_factory=self.riesz_action_factory,
            )

    def build(
        self,
        candidate_states: Iterable[np.ndarray],
        *,
        requested_energy_state_error: float,
    ) -> SparseEnergyReducedEMModel:
        return self.build_multi_rhs(
            candidate_states,
            self.problem.b[:, None],
            requested_energy_state_error=requested_energy_state_error,
        )
