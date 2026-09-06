from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
import scipy.sparse as sp


@dataclass(frozen=True)
class ThermalTailCertificate:
    """Projection-tail certificate for the full heat equation spectrum.

    For M u_dot + K u = q(t), with M-orthonormal eigenmodes and first omitted
    eigenvalue lambda_{r+1}, if

        ||q(t)||_{M^-1} <= Q

    for the certified operating domain, then the exact omitted modal component
    obeys

        ||u_tail(t)||_M
        <= exp(-lambda_{r+1} t) E0
           + (1-exp(-lambda_{r+1} t)) Q/lambda_{r+1}.

    This certifies spatial projection capability. The additional error caused by
    evaluating the nonlinear electromagnetic source on the reduced state is
    handled separately by the coupled residual/contraction certificate.
    """

    rank: int
    full_dimension: int
    first_omitted_lambda: float
    initial_tail_norm: float
    source_dual_bound: float
    steady_forcing_tail_bound: float
    uniform_projection_tail_bound: float
    requested_state_tolerance: float

    @property
    def certified(self) -> bool:
        return self.uniform_projection_tail_bound <= self.requested_state_tolerance

    def bound_at_time(self, t: float) -> float:
        if t < 0:
            raise ValueError("time must be non-negative")
        if self.rank >= self.full_dimension:
            return 0.0
        decay = np.exp(-self.first_omitted_lambda * float(t))
        return float(
            decay * self.initial_tail_norm
            + (1.0 - decay) * self.steady_forcing_tail_bound
        )


@dataclass(frozen=True)
class ThermalRankSelection:
    model: "ThermalSpectralModel"
    certificate: ThermalTailCertificate


@dataclass(frozen=True)
class ThermalSpectralModel:
    M: np.ndarray
    K: np.ndarray
    Phi: np.ndarray
    lambdas: np.ndarray

    @classmethod
    def build(
        cls,
        M: np.ndarray,
        K: np.ndarray,
        *,
        target_residual: Optional[float] = None,
    ) -> "ThermalSpectralModel":
        """Construct the deterministic mass-orthonormal thermal spectrum.

        The current verification implementation computes the full spectrum and
        can then select a certified retained rank using `select_certified_rank`.
        A future scalable eigensolver will obtain only the required low modes and
        a verified lower bound for the first omitted eigenvalue.
        """

        if sp.issparse(M):
            M = M.toarray()
        if sp.issparse(K):
            K = K.toarray()
        M = np.asarray(M, dtype=float)
        K = np.asarray(K, dtype=float)
        if M.shape != K.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M and K must be square matrices of equal size")

        vals, vecs = scipy.linalg.eigh(K, M, check_finite=True)
        if np.any(vals <= 0):
            raise ValueError("Thermal operator must be positive after boundary treatment")
        if target_residual is not None:
            raise ValueError(
                "rank cannot be selected from an unphysical generic residual target; "
                "use select_certified_rank with an initial state, source dual bound, "
                "and state/output error requirement"
            )
        return cls(M=M, K=K, Phi=vecs, lambdas=vals)

    @property
    def full_dimension(self) -> int:
        return self.M.shape[0]

    @property
    def rank(self) -> int:
        return self.Phi.shape[1]

    @property
    def is_full_spectrum(self) -> bool:
        return self.rank == self.full_dimension

    def project(self, field: np.ndarray) -> np.ndarray:
        field = np.asarray(field, dtype=float)
        if field.shape != (self.full_dimension,):
            raise ValueError("field dimension mismatch")
        return self.Phi.T @ self.M @ field

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        state = np.asarray(a, dtype=float)
        if state.shape != (self.rank,):
            raise ValueError("thermal coordinate dimension mismatch")
        return self.Phi @ state

    def reduced_matrices(self):
        return self.Phi.T @ self.M @ self.Phi, self.Phi.T @ self.K @ self.Phi

    def truncate(self, rank: int) -> "ThermalSpectralModel":
        if not 1 <= rank <= self.rank:
            raise ValueError("rank must satisfy 1 <= rank <= current rank")
        return ThermalSpectralModel(
            M=self.M,
            K=self.K,
            Phi=self.Phi[:, :rank].copy(),
            lambdas=self.lambdas[:rank].copy(),
        )

    def projection_tail_certificate(
        self,
        rank: int,
        *,
        initial_field: np.ndarray,
        source_dual_bound: float,
        requested_state_tolerance: float,
    ) -> ThermalTailCertificate:
        if not self.is_full_spectrum:
            raise ValueError("projection-tail certification requires the full verification spectrum")
        if not 1 <= rank <= self.full_dimension:
            raise ValueError("rank out of range")
        if source_dual_bound < 0:
            raise ValueError("source_dual_bound must be non-negative")
        if requested_state_tolerance <= 0:
            raise ValueError("requested_state_tolerance must be positive")

        u0 = np.asarray(initial_field, dtype=float)
        if u0.shape != (self.full_dimension,):
            raise ValueError("initial_field dimension mismatch")
        modal0 = self.Phi.T @ self.M @ u0

        if rank == self.full_dimension:
            first_omitted = float("inf")
            initial_tail = 0.0
            forcing_tail = 0.0
        else:
            first_omitted = float(self.lambdas[rank])
            initial_tail = float(np.linalg.norm(modal0[rank:]))
            forcing_tail = float(source_dual_bound / first_omitted)

        uniform = float(max(initial_tail, forcing_tail))
        return ThermalTailCertificate(
            rank=rank,
            full_dimension=self.full_dimension,
            first_omitted_lambda=first_omitted,
            initial_tail_norm=initial_tail,
            source_dual_bound=float(source_dual_bound),
            steady_forcing_tail_bound=forcing_tail,
            uniform_projection_tail_bound=uniform,
            requested_state_tolerance=float(requested_state_tolerance),
        )

    def select_certified_rank(
        self,
        *,
        initial_field: np.ndarray,
        source_dual_bound: float,
        requested_state_tolerance: float,
    ) -> ThermalRankSelection:
        """Return the smallest retained rank satisfying the projection-tail bound."""

        if not self.is_full_spectrum:
            raise ValueError("rank selection requires the full verification spectrum")

        for rank in range(1, self.full_dimension + 1):
            cert = self.projection_tail_certificate(
                rank,
                initial_field=initial_field,
                source_dual_bound=source_dual_bound,
                requested_state_tolerance=requested_state_tolerance,
            )
            if cert.certified:
                return ThermalRankSelection(self.truncate(rank), cert)

        raise RuntimeError("full thermal space failed its zero-tail certificate")

    def select_certified_rank_for_output(
        self,
        *,
        initial_field: np.ndarray,
        source_dual_bound: float,
        requested_output_tolerance: float,
        output_lipschitz: float,
    ) -> ThermalRankSelection:
        """Select rank from |Delta Q| <= L_Q ||Delta T||_M."""

        if requested_output_tolerance <= 0:
            raise ValueError("requested_output_tolerance must be positive")
        if output_lipschitz <= 0:
            raise ValueError("output_lipschitz must be positive")
        state_tolerance = requested_output_tolerance / output_lipschitz
        return self.select_certified_rank(
            initial_field=initial_field,
            source_dual_bound=source_dual_bound,
            requested_state_tolerance=state_tolerance,
        )
