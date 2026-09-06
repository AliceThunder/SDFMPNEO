from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
import scipy.sparse as sp


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
        """Construct a deterministic mass-orthonormal thermal spectral basis.

        The executable core accepts dense or sparse assembled operators. It
        still keeps the complete spectrum: certified tail-driven rank selection
        has not yet been implemented, and heuristic truncation is deliberately
        disabled. Sparse matrices are therefore converted to dense form only at
        this full-spectrum stage. This is correct for the current verification
        meshes but is not claimed as the scalable production eigensolver.
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
            raise NotImplementedError(
                "Certified thermal-rank selection requires the tail estimator; heuristic truncation is disabled."
            )
        return cls(M=M, K=K, Phi=vecs, lambdas=vals)

    def project(self, field: np.ndarray) -> np.ndarray:
        field = np.asarray(field, dtype=float)
        return self.Phi.T @ self.M @ field

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        return self.Phi @ np.asarray(a, dtype=float)

    def reduced_matrices(self):
        return self.Phi.T @ self.M @ self.Phi, self.Phi.T @ self.K @ self.Phi
