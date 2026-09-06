from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .spectral import ThermalSpectralModel, ThermalTailCertificate


def li_yau_thermal_eigenvalue_lower_bound(
    mode_index: int,
    *,
    domain_volume: float,
    thermal_conductivity_min: float,
    volumetric_heat_capacity_max: float,
) -> float:
    """Certified Dirichlet lower bound for the ``mode_index``-th heat eigenvalue.

    For ``K u = lambda M u`` in three dimensions,

        R(u) >= kappa_min/(rho c)_max * R_Laplacian(u).

    The Li-Yau inequality for the Dirichlet Laplacian gives

        lambda_k >= (kappa_min/(rho c)_max) * (3/5) * 4*pi^2
                   * [k / ((4*pi/3)|Omega|)]^(2/3).

    The bound is deliberately conservative but requires no computed high modes.
    """

    k = int(mode_index)
    volume = float(domain_volume)
    kappa = float(thermal_conductivity_min)
    rho_cp = float(volumetric_heat_capacity_max)
    if k <= 0:
        raise ValueError("mode_index is one-based and must be positive")
    if volume <= 0.0 or kappa <= 0.0 or rho_cp <= 0.0:
        raise ValueError("volume and material extrema must be positive")
    omega3 = 4.0 * np.pi / 3.0
    laplacian = (3.0 / 5.0) * 4.0 * np.pi**2 * (k / (omega3 * volume)) ** (2.0 / 3.0)
    return float((kappa / rho_cp) * laplacian)


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
        tail = u0 - self.model.Phi @ modal
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
    first_omitted_lambda_lower_bound: float | None = None,
    domain_volume: float | None = None,
    thermal_conductivity_min: float | None = None,
    volumetric_heat_capacity_max: float | None = None,
) -> PartialThermalSpectrum:
    """Compute only low modes and attach a rigorous omitted-mode lower bound.

    The lower bound can be supplied by an external verified eigensolver.  If it
    is omitted, the built-in Li-Yau bound is used from the physical domain volume
    and material extrema. Approximate Ritz values are never promoted to lower
    bounds.
    """

    Msp = sp.csr_matrix(M, dtype=float)
    Ksp = sp.csr_matrix(K, dtype=float)
    n = Msp.shape[0]
    if Msp.shape != (n, n) or Ksp.shape != (n, n):
        raise ValueError("M and K must be square with equal dimensions")
    r = int(rank)
    if not 1 <= r < n:
        raise ValueError("partial rank must satisfy 1 <= rank < dimension")

    if first_omitted_lambda_lower_bound is None:
        if domain_volume is None or thermal_conductivity_min is None or volumetric_heat_capacity_max is None:
            raise ValueError(
                "provide an omitted-eigenvalue lower bound or the Li-Yau volume/material inputs"
            )
        lower = li_yau_thermal_eigenvalue_lower_bound(
            r + 1,
            domain_volume=float(domain_volume),
            thermal_conductivity_min=float(thermal_conductivity_min),
            volumetric_heat_capacity_max=float(volumetric_heat_capacity_max),
        )
    else:
        lower = float(first_omitted_lambda_lower_bound)
    if lower <= 0.0 or not np.isfinite(lower):
        raise ValueError("first_omitted_lambda_lower_bound must be finite and positive")

    vals, vecs = spla.eigsh(Ksp, k=r, M=Msp, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals = np.asarray(vals[order], dtype=float)
    vecs = np.asarray(vecs[:, order], dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError("thermal low spectrum must be positive")
    gram = vecs.T @ (Msp @ vecs)
    L = np.linalg.cholesky(0.5 * (gram + gram.T))
    vecs = vecs @ np.linalg.solve(L.T, np.eye(r))

    model = ThermalSpectralModel(M=Msp, K=Ksp, Phi=vecs, lambdas=vals)
    return PartialThermalSpectrum(
        model=model,
        first_omitted_lambda_lower_bound=lower,
        computed_eigenvalues=vals.copy(),
    )
