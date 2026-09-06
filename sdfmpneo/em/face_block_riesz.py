from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp

from .adaptive_block import AdaptiveAggregateEnergyPreconditioner
from .block_riesz import (
    _certified_generalized_trace_upper,
    _normalized_gershgorin_lower,
    _normalized_gershgorin_upper,
    _positive_real_diagonal,
)
from .certified_riesz import CertifiedEnergyPreconditioner, CertifiedPCGRieszAction
from .magnetic_face_auxiliary import MagneticFaceCirculationEnergyPreconditioner


@dataclass(frozen=True)
class FaceAuxiliaryPhysicalBlockPreconditioner(CertifiedEnergyPreconditioner):
    """Production-oriented physical H preconditioner with face-circulation magnetic action.

    The magnetic action proves

        K_A >= m_F Q_A,

    where Q_A is a certificate-selected local block matrix derived from the
    topology-exact face-circulation lower energy.  Therefore

        K_A^-1 <= (1/m_F) Q_A^-1.

    If direct normalized Gershgorin cannot certify D_AA <= gamma K_A, the
    coupling constant is bounded without a magnetic factorization by

        gamma <= (1/m_F) trace(Q_A^-1 D_AA).

    The trace action and its inverse-norm envelope use exactly the same local
    magnetic Q_A action.  The scalar conductive block is represented by an
    independent certified action satisfying D_psipsi >= m_E P_E.
    """

    n_A: int
    dimension: int
    lower_spectral_equivalence_bound: float
    gamma_upper_bound: float
    gamma_certificate_method: str
    physical_coupling_lower_bound: float
    magnetic_action: MagneticFaceCirculationEnergyPreconditioner
    scalar_action: CertifiedEnergyPreconditioner | None

    @classmethod
    def build(
        cls,
        H: sp.spmatrix,
        *,
        problem,
        scalar_action_factory=None,
    ) -> "FaceAuxiliaryPhysicalBlockPreconditioner":
        metric = sp.csr_matrix(H, dtype=complex)
        n = metric.shape[0]
        if metric.shape != (n, n):
            raise ValueError("H must be square")
        n_A = int(problem.n_A)
        if not 0 < n_A <= n:
            raise ValueError("problem.n_A does not match H")

        R = sp.csr_matrix(problem.a_basis, dtype=complex)
        K = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).tocsr()
        K = (0.5 * (K + K.conj().T)).tocsr()
        K.sum_duplicates()
        K.eliminate_zeros()
        if K.shape != (n_A, n_A):
            raise ValueError("magnetic gauge block dimension mismatch")

        magnetic = MagneticFaceCirculationEnergyPreconditioner.build_from_problem(problem)
        m_F = float(magnetic.lower_spectral_equivalence_bound)
        if m_F <= 0.0:
            raise ValueError("face magnetic action must have a positive certified lower bound")

        H11 = metric[:n_A, :n_A].tocsr()
        D_AA = (0.5 * ((H11 - K) + (H11 - K).conj().T)).tocsr()
        D_AA.sum_duplicates()
        D_AA.eliminate_zeros()

        K_diag = _positive_real_diagonal(K, "magnetic block")
        k_lower = _normalized_gershgorin_lower(K, K_diag)
        if k_lower > 0.0:
            d_upper = _normalized_gershgorin_upper(D_AA, K_diag)
            gamma_upper = float(np.nextafter(d_upper / k_lower, np.inf))
            gamma_method = "normalized_gershgorin"
        else:
            Q_A = magnetic.preconditioner_matrix().tocsr()
            trace_Q = _certified_generalized_trace_upper(
                Q_A,
                D_AA,
                magnetic,
                magnetic.inverse_inf_upper_bound,
            )
            gamma_upper = float(np.nextafter(trace_Q / m_F, np.inf))
            gamma_method = "face_auxiliary_local_trace"

        if gamma_upper < 0.0 or not np.isfinite(gamma_upper):
            raise ValueError("failed to certify a finite non-negative conductive/magnetic coupling bound")

        n_scalar = n - n_A
        scalar = None
        m_E = 1.0
        if n_scalar:
            E = metric[n_A:, n_A:].tocsr()
            E = (0.5 * (E + E.conj().T)).tocsr()
            E.sum_duplicates()
            E.eliminate_zeros()
            factory = AdaptiveAggregateEnergyPreconditioner.build if scalar_action_factory is None else scalar_action_factory
            scalar = factory(E)
            m_E = float(scalar.lower_spectral_equivalence_bound)
            if m_E <= 0.0:
                raise ValueError("scalar conductive action must have a positive certified lower bound")

        if n_scalar == 0 or gamma_upper == 0.0:
            physical_lower = 1.0
        else:
            root = math.sqrt(gamma_upper * gamma_upper + 4.0 * gamma_upper)
            physical_lower = float(np.nextafter(2.0 / (2.0 + gamma_upper + root), 0.0))

        lower = float(np.nextafter(physical_lower * min(m_F, m_E), 0.0))
        if lower <= 0.0:
            raise ValueError("face-auxiliary physical block lower bound is non-positive")

        return cls(
            n_A=n_A,
            dimension=n,
            lower_spectral_equivalence_bound=lower,
            gamma_upper_bound=gamma_upper,
            gamma_certificate_method=gamma_method,
            physical_coupling_lower_bound=physical_lower,
            magnetic_action=magnetic,
            scalar_action=scalar,
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape != (self.dimension,):
            raise ValueError("face-auxiliary physical preconditioner rhs mismatch")
        left = np.asarray(self.magnetic_action.solve(vector[: self.n_A]), dtype=complex)
        if self.scalar_action is None:
            return left
        right = np.asarray(self.scalar_action.solve(vector[self.n_A :]), dtype=complex)
        return np.concatenate([left, right])


def make_face_auxiliary_physical_pcg_riesz_factory(problem, *, scalar_action_factory=None):
    """Create certified local-H Riesz actions using the face-circulation magnetic chain."""

    def factory(H: sp.spmatrix):
        preconditioner = FaceAuxiliaryPhysicalBlockPreconditioner.build(
            H,
            problem=problem,
            scalar_action_factory=scalar_action_factory,
        )
        return CertifiedPCGRieszAction(H, preconditioner)

    return factory
