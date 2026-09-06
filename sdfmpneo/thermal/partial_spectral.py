from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .spectral import ThermalSpectralModel, ThermalTailCertificate


@dataclass(frozen=True)
class PartialThermalSpectrum:
    """Low thermal modes plus an independently certified omitted eigenvalue bound."""

    model: ThermalSpectralModel
    first_omitted_lambda_lower_bound: float
    computed_eigenvalues: np.ndarray

    def tail_certificate(
        self,
        *,
        initial_field: np.ndarray,
        source_dual_bound: float,
        requested_state_tolerance: float,
    ) -> ThermalTailCertificate:
        if self.first_omitted_lambda_lower_bound <= 0.0:
            raise ValueError("a positive certified omitted-eigenvalue lower bound is required")
        u0 = np.asarray(initial_field, dtype=float)
        if u0.shape != (self.model.full_dimension,):
            raise ValueError("initial_field dimension mismatch")
        modal = self.model.Phi.T @ self.model.M @ u0
        projected = self.model.Phi @ modal
        tail = u0 - projected
        initial_tail = float(np.sqrt(max(0.0, tail @ (self.model.M @ tail))))
        forcing = float(source_dual_bound / self.first_omitted_lambda_lower_bound)
        uniform = max(initial_tail, forcing)
        return ThermalTailCertificate(
            rank=self.model.rank,
            full_dimension=self.model.full_dimension,
            first_omitted_lambda=float(self.first_omitted_lambda_lower_bound),
            initial_tail_norm=initial_tail,
            source_dual_bound=float(source_dual_bound),
            steady_forcing_tail_bound=forcing,
            uniform_projection_tail_bound=float(uniform),
            requested_state_tolerance=float(requested_state_tolerance),
        )


def build_partial_thermal_spectrum(
    M: sp.spmatrix | np.ndarray,
    K: sp.spmatrix | np.ndarray,
    *,
    rank: int,
    first_omitted_lambda_lower_bound: float,
) -> PartialThermalSpectrum:
    """Compute only the requested low modes; never compute the full spectrum.

    ``first_omitted_lambda_lower_bound`` is deliberately explicit: this routine
    will not reinterpret an approximate Ritz value as a rigorous lower bound.
    It can come from a certified eigensolver, analytic Poincare bound, or an
    external verified eigensolver.  The separation keeps the scalable numerical
    solver simple and the proof obligation visible.
    """

    Msp = sp.csr_matrix(M, dtype=float)
    Ksp = sp.csr_matrix(K, dtype=float)
    n = Msp.shape[0]
    if Msp.shape != (n, n) or Ksp.shape != (n, n):
        raise ValueError("M and K must be square with equal dimensions")
    r = int(rank)
    if not 1 <= r < n:
        raise ValueError("partial rank must satisfy 1 <= rank < dimension")
    lower = float(first_omitted_lambda_lower_bound)
    if lower <= 0.0:
        raise ValueError("first_omitted_lambda_lower_bound must be positive")

    vals, vecs = spla.eigsh(Ksp, k=r, M=Msp, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals = np.asarray(vals[order], dtype=float)
    vecs = np.asarray(vecs[:, order], dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError("thermal low spectrum must be positive")
    gram = vecs.T @ (Msp @ vecs)
    L = np.linalg.cholesky(0.5 * (gram + gram.T))
    vecs = vecs @ np.linalg.inv(L.T)

    model = ThermalSpectralModel(
        M=Msp,
        K=Ksp,
        Phi=vecs,
        lambdas=vals,
    )
    return PartialThermalSpectrum(
        model=model,
        first_omitted_lambda_lower_bound=lower,
        computed_eigenvalues=vals.copy(),
    )
