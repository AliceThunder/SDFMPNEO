from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class CertifiedEigenvalueInterval:
    center: float
    radius: float

    @property
    def lower(self) -> float:
        return float(self.center - self.radius)

    @property
    def upper(self) -> float:
        return float(self.center + self.radius)


@dataclass(frozen=True)
class SpectralCluster:
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CanonicalThermalAtlasStep:
    clusters: tuple[SpectralCluster, ...]
    aligned_basis: np.ndarray
    reference_intervals: tuple[CertifiedEigenvalueInterval, ...]
    target_intervals: tuple[CertifiedEigenvalueInterval, ...]


def certified_eigenvalue_intervals(model) -> tuple[CertifiedEigenvalueInterval, ...]:
    """Residual-certified intervals for retained generalized eigenpairs.

    With ``phi^T M phi=1``, the dual residual

        eta = ||K phi - lambda M phi||_{M^-1}

    bounds the distance from the Ritz value to the spectrum of the self-adjoint
    heat operator. No arbitrary eigengap tolerance is introduced.
    """

    M = sp.csr_matrix(model.M, dtype=float)
    K = sp.csr_matrix(model.K, dtype=float)
    solve_M = spla.factorized(M.tocsc())
    intervals = []
    for j, value in enumerate(np.asarray(model.lambdas, dtype=float)):
        phi = np.asarray(model.Phi[:, j], dtype=float)
        norm_m = float(np.sqrt(max(0.0, phi @ (M @ phi))))
        if norm_m <= 0.0:
            raise np.linalg.LinAlgError("thermal eigenvector has zero mass norm")
        phi = phi / norm_m
        residual = K @ phi - value * (M @ phi)
        lift = np.asarray(solve_M(np.asarray(residual, dtype=float)), dtype=float)
        eta2 = float(residual @ lift)
        scale = float(np.linalg.norm(residual)) * float(np.linalg.norm(lift))
        eps = np.finfo(float).eps
        backward = 16.0 * max(1, residual.size) * eps * scale
        if eta2 < -backward:
            raise np.linalg.LinAlgError("mass-dual residual norm lost positivity")
        eta = float(np.sqrt(max(0.0, eta2)))
        intervals.append(CertifiedEigenvalueInterval(float(value), eta))
    return tuple(intervals)


def common_certified_clusters(
    reference: tuple[CertifiedEigenvalueInterval, ...],
    target: tuple[CertifiedEigenvalueInterval, ...],
) -> tuple[SpectralCluster, ...]:
    """Form clusters only where spectral separation is proved at both charts."""

    if len(reference) != len(target) or not reference:
        raise ValueError("reference/target interval sets must have equal non-zero length")
    cuts = [0]
    for i in range(len(reference) - 1):
        separated_reference = reference[i].upper < reference[i + 1].lower
        separated_target = target[i].upper < target[i + 1].lower
        if separated_reference and separated_target:
            cuts.append(i + 1)
    cuts.append(len(reference))
    return tuple(SpectralCluster(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1))


def _mass_orthonormalize(basis: np.ndarray, M: sp.csr_matrix) -> np.ndarray:
    B = np.asarray(basis, dtype=float)
    gram = B.T @ (M @ B)
    gram = 0.5 * (gram + gram.T)
    L = scipy.linalg.cholesky(gram, lower=True)
    return B @ scipy.linalg.solve_triangular(
        L.T, np.eye(B.shape[1]), lower=False
    )


def canonicalize_thermal_subspaces(reference_model, target_model) -> CanonicalThermalAtlasStep:
    """Align certified spectral subspaces, never individual crossing eigenvectors."""

    if reference_model.rank != target_model.rank:
        raise ValueError("reference and target retained ranks must match")
    if reference_model.full_dimension != target_model.full_dimension:
        raise ValueError("spectral atlas requires the same reference-domain discrete dimension")
    ref_intervals = certified_eigenvalue_intervals(reference_model)
    tgt_intervals = certified_eigenvalue_intervals(target_model)
    clusters = common_certified_clusters(ref_intervals, tgt_intervals)
    M = sp.csr_matrix(target_model.M, dtype=float)
    aligned = np.zeros_like(np.asarray(target_model.Phi, dtype=float))

    for cluster in clusters:
        sl = slice(cluster.start, cluster.stop)
        R = _mass_orthonormalize(reference_model.Phi[:, sl], M)
        T = _mass_orthonormalize(target_model.Phi[:, sl], M)
        cross = T.T @ (M @ R)
        U, _, Vh = scipy.linalg.svd(cross, full_matrices=False)
        Q = U @ Vh
        aligned[:, sl] = T @ Q

    return CanonicalThermalAtlasStep(
        clusters=clusters,
        aligned_basis=aligned,
        reference_intervals=ref_intervals,
        target_intervals=tgt_intervals,
    )
