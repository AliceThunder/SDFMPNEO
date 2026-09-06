from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp

from .block_riesz import (
    _certified_generalized_trace_upper,
    _normalized_gershgorin_lower,
    _normalized_gershgorin_upper,
    _positive_real_diagonal,
)
from .certified_riesz import (
    CertifiedEnergyPreconditioner,
    CertifiedPCGRieszAction,
    DiagonalGershgorinEnergyPreconditioner,
)
from .morse_face_auxiliary import MorseFaceCirculationEnergyPreconditioner


@dataclass(frozen=True)
class MorseAuxiliaryPhysicalBlockPreconditioner(CertifiedEnergyPreconditioner):
    """Factorization-free physical H preconditioner with topology-generated magnetic coarse space.

    For the reciprocal A-psi physical metric

        H = [[K_A + D_AA, D_Apsi],
             [D_psiA,       D_psipsi]],

    the magnetic action constructs a face-circulation auxiliary operator P_A and
    proves exactly

        K_A >= P_A,

    hence m_K=1.  P_A^{-1} is applied by topology-generated fine triangular
    elimination plus a strictly smaller coarse Schur solve.

    The conductive/magnetic coupling is certified without K_A factorization:

        gamma = lambda_max(K_A^-1/2 D_AA K_A^-1/2)
              <= trace(P_A^-1 D_AA).

    For the scalar conductive block this first production candidate accepts only
    a diagonal action for which normalized Gershgorin proves

        D_psipsi >= m_E diag(D_psipsi),  m_E>0.

    If that proof fails the constructor rejects the path; it never silently falls
    back to a complete scalar factorization.
    """

    n_A: int
    dimension: int
    lower_spectral_equivalence_bound: float
    gamma_upper_bound: float
    gamma_certificate_method: str
    physical_coupling_lower_bound: float
    magnetic_action: MorseFaceCirculationEnergyPreconditioner
    scalar_action: CertifiedEnergyPreconditioner | None

    @classmethod
    def build(cls, H: sp.spmatrix, *, problem) -> "MorseAuxiliaryPhysicalBlockPreconditioner":
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

        magnetic = MorseFaceCirculationEnergyPreconditioner.build_from_problem(problem)
        if magnetic.lower_spectral_equivalence_bound != 1.0:
            raise RuntimeError("Morse magnetic auxiliary must certify m_K=1")

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
            P_A = magnetic.auxiliary_matrix().tocsr()
            gamma_upper = _certified_generalized_trace_upper(
                P_A,
                D_AA,
                magnetic,
                magnetic.inverse_inf_upper_bound,
            )
            gamma_method = "morse_face_auxiliary_residual_certified_trace"

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
            scalar = DiagonalGershgorinEnergyPreconditioner.build(E)
            m_E = float(scalar.lower_spectral_equivalence_bound)
            if m_E <= 0.0:
                raise ValueError("scalar conductive action must have a positive certified lower bound")

        if n_scalar == 0 or gamma_upper == 0.0:
            physical_lower = 1.0
        else:
            root = math.sqrt(gamma_upper * gamma_upper + 4.0 * gamma_upper)
            physical_lower = float(np.nextafter(2.0 / (2.0 + gamma_upper + root), 0.0))

        lower = float(np.nextafter(physical_lower * min(1.0, m_E), 0.0))
        if lower <= 0.0:
            raise ValueError("Morse physical block lower bound is non-positive")

        return cls(
            n_A=n_A,
            dimension=n,
            lower_spectral_equivalence_bound=lower,
            gamma_upper_bound=float(gamma_upper),
            gamma_certificate_method=gamma_method,
            physical_coupling_lower_bound=physical_lower,
            magnetic_action=magnetic,
            scalar_action=scalar,
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape != (self.dimension,):
            raise ValueError("Morse physical preconditioner rhs mismatch")
        left = np.asarray(self.magnetic_action.solve(vector[: self.n_A]), dtype=complex)
        if self.scalar_action is None:
            return left
        right = np.asarray(self.scalar_action.solve(vector[self.n_A :]), dtype=complex)
        return np.concatenate([left, right])


def make_morse_auxiliary_physical_pcg_riesz_factory(problem):
    """Create local-H certified PCG Riesz actions with Morse magnetic auxiliary space."""

    def factory(H: sp.spmatrix):
        preconditioner = MorseAuxiliaryPhysicalBlockPreconditioner.build(H, problem=problem)
        return CertifiedPCGRieszAction(H, preconditioner)

    return factory
