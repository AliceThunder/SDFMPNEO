from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .adaptive_block import AdaptiveAggregateEnergyPreconditioner
from .block_riesz import (
    _certified_generalized_trace_upper,
    _certified_inverse_inf_upper,
    _normalized_gershgorin_lower,
    _normalized_gershgorin_upper,
    _positive_real_diagonal,
)
from .certified_riesz import CertifiedEnergyPreconditioner, CertifiedPCGRieszAction
from .magnetic_auxiliary import MagneticCurlSubsetEnergyPreconditioner


@dataclass(frozen=True)
class CurlAuxiliaryPhysicalBlockPreconditioner(CertifiedEnergyPreconditioner):
    """Physical H preconditioner with distinct magnetic and scalar actions.

    The magnetic block is not factorized from K_A.  It uses the physical
    tetrahedral curl factor F_A and a matched subset S:

        K_A = F_A^H F_A >= S^H S = P_A,

    hence m_K=1 exactly at operator level.

    The scalar conductive block uses a separately certified action.  The outer
    A-psi physical theorem remains

        H >= m(gamma) min(m_K,m_E) diag(P_A,P_E).

    Gamma certification is logically separate from the block actions.  The
    current non-diagonally-dominant fallback may still use sparse LU solely to
    construct a rigorous bound on gamma; this class therefore removes magnetic
    *block solve* factorization, not yet the gamma-certificate factorization.
    """

    n_A: int
    dimension: int
    lower_spectral_equivalence_bound: float
    gamma_upper_bound: float
    gamma_certificate_method: str
    physical_coupling_lower_bound: float
    magnetic_action: MagneticCurlSubsetEnergyPreconditioner
    scalar_action: CertifiedEnergyPreconditioner | None

    @classmethod
    def build(
        cls,
        H: sp.spmatrix,
        *,
        problem,
        scalar_action_factory=None,
    ) -> "CurlAuxiliaryPhysicalBlockPreconditioner":
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
            # This is certificate construction only.  It will be removed in the
            # next stage by a factorization-free gamma proof.
            try:
                lu_K_certificate = spla.splu(K.tocsc())
            except RuntimeError as exc:
                raise ValueError("magnetic gamma-certificate factorization failed") from exc
            inverse_inf_upper = _certified_inverse_inf_upper(K, lu_K_certificate)
            gamma_upper = _certified_generalized_trace_upper(
                K,
                D_AA,
                lu_K_certificate,
                inverse_inf_upper,
            )
            gamma_method = "residual_certified_generalized_trace"
        if gamma_upper < 0.0 or not np.isfinite(gamma_upper):
            raise ValueError("failed to certify finite non-negative D_AA/K_A bound")

        magnetic = MagneticCurlSubsetEnergyPreconditioner.build_from_problem(problem)
        magnetic_lower = float(magnetic.lower_spectral_equivalence_bound)
        if magnetic_lower != 1.0:
            raise RuntimeError("physical curl-subset magnetic lower bound must be one")

        n_scalar = n - n_A
        scalar = None
        scalar_lower = 1.0
        if n_scalar:
            E = metric[n_A:, n_A:].tocsr()
            E = (0.5 * (E + E.conj().T)).tocsr()
            E.sum_duplicates()
            E.eliminate_zeros()
            factory = (
                AdaptiveAggregateEnergyPreconditioner.build
                if scalar_action_factory is None
                else scalar_action_factory
            )
            scalar = factory(E)
            scalar_lower = float(scalar.lower_spectral_equivalence_bound)
            if scalar_lower <= 0.0:
                raise ValueError("scalar action must have positive certified lower bound")

        if n_scalar == 0 or gamma_upper == 0.0:
            physical_lower = 1.0
        else:
            root = math.sqrt(gamma_upper * gamma_upper + 4.0 * gamma_upper)
            physical_lower = 2.0 / (2.0 + gamma_upper + root)
            physical_lower = float(np.nextafter(physical_lower, 0.0))

        lower = float(
            np.nextafter(
                physical_lower * min(magnetic_lower, scalar_lower),
                0.0,
            )
        )
        if lower <= 0.0:
            raise ValueError("curl-auxiliary physical block lower bound is non-positive")

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
            raise ValueError("curl-auxiliary physical preconditioner rhs mismatch")
        left = np.asarray(self.magnetic_action.solve(vector[: self.n_A]), dtype=complex)
        if self.scalar_action is None:
            return left
        right = np.asarray(self.scalar_action.solve(vector[self.n_A :]), dtype=complex)
        return np.concatenate([left, right])


def make_curl_auxiliary_physical_pcg_riesz_factory(
    problem,
    *,
    scalar_action_factory=None,
):
    """Create local-H certified PCG Riesz actions with curl magnetic auxiliary space."""

    def factory(H: sp.spmatrix):
        preconditioner = CurlAuxiliaryPhysicalBlockPreconditioner.build(
            H,
            problem=problem,
            scalar_action_factory=scalar_action_factory,
        )
        return CertifiedPCGRieszAction(H, preconditioner)

    return factory
