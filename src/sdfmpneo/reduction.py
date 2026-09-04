from __future__ import annotations

import torch
from torch import Tensor

from .contracts import BasisBundle, RawBasisBundle, TopologyOperators


class BasisRankError(RuntimeError):
    """Raised when a neural trial space is rank deficient before whitening."""


def _batched_map(matrix: Tensor, coefficients: Tensor) -> Tensor:
    """Apply an unbatched or batched linear map to batched basis columns."""
    if matrix.ndim == 2:
        return torch.einsum("mn,bnr->bmr", matrix, coefficients)
    if matrix.ndim == 3:
        return torch.einsum("bmn,bnr->bmr", matrix, coefficients)
    raise ValueError("matrix must have shape [M,N] or [B,M,N]")


def assemble_solenoidal_basis(
    curl_map: Tensor,
    harmonic_basis: Tensor,
    raw_coefficients: Tensor,
) -> Tensor:
    """Generate an H(div)-admissible basis from exact+harmonic coordinates."""
    n_potential = curl_map.shape[-1]
    n_harmonic = harmonic_basis.shape[-1]
    if raw_coefficients.shape[-2] != n_potential + n_harmonic:
        raise ValueError(
            "raw coefficient dimension must equal potential + harmonic dimensions"
        )
    potential = raw_coefficients[..., :n_potential, :]
    harmonic = raw_coefficients[..., n_potential:, :]
    basis = _batched_map(curl_map, potential)
    if n_harmonic:
        basis = basis + _batched_map(harmonic_basis, harmonic)
    return basis


def metric_orthonormalize(
    basis: Tensor,
    metric: Tensor,
    jitter: float = 1.0e-9,
    rank_rtol: float = 1.0e-10,
) -> Tensor:
    """Return B with B^H M B = I after explicitly checking basis rank.

    Jitter is allowed only to regularise a numerically valid Gram matrix. It is
    not allowed to hide a collapsed neural trial space. Rank is therefore
    checked on the unregularised metric Gram matrix before Cholesky whitening.
    """
    if basis.ndim != 3:
        raise ValueError("basis must have shape [B,N,r]")
    if basis.shape[-2] < basis.shape[-1]:
        raise BasisRankError(
            f"trial space requests rank {basis.shape[-1]} from only "
            f"{basis.shape[-2]} physical/query DOFs"
        )
    if not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie in (0,1)")

    if metric.ndim == 2:
        gram = torch.einsum("bnr,nm,bms->brs", basis.conj(), metric, basis)
    elif metric.ndim == 3:
        gram = torch.einsum("bnr,bnm,bms->brs", basis.conj(), metric, basis)
    else:
        raise ValueError("metric must have shape [N,N] or [B,N,N]")

    eigenvalues = torch.linalg.eigvalsh(gram.detach())
    largest = eigenvalues[..., -1]
    smallest = eigenvalues[..., 0]
    if torch.any(largest <= 0):
        raise BasisRankError("metric Gram matrix has no positive trial-space energy")
    relative_smallest = smallest / largest
    if torch.any(relative_smallest <= rank_rtol):
        worst = float(relative_smallest.min().item())
        raise BasisRankError(
            "neural trial basis is rank deficient or numerically collapsed before "
            f"whitening (min/max Gram eigenvalue={worst:.3e}, threshold={rank_rtol:.3e})"
        )

    rank = gram.shape[-1]
    eye = torch.eye(rank, dtype=gram.dtype, device=gram.device).expand_as(gram)
    chol = torch.linalg.cholesky(gram + jitter * eye)
    whiten = torch.linalg.solve_triangular(
        chol.conj().transpose(-2, -1), eye, upper=True
    )
    return basis @ whiten


def build_physical_bases(
    raw: RawBasisBundle,
    topology: TopologyOperators,
    metrics: dict[str, Tensor],
    jitter: float = 1.0e-9,
    rank_rtol: float = 1.0e-10,
) -> BasisBundle:
    current = assemble_solenoidal_basis(
        topology.curl_current, topology.harmonic_current, raw.current_potential
    )
    velocity = assemble_solenoidal_basis(
        topology.curl_velocity, topology.harmonic_velocity, raw.velocity_potential
    )

    return BasisBundle(
        current=metric_orthonormalize(
            current, metrics["current"], jitter, rank_rtol
        ),
        thermal=metric_orthonormalize(
            raw.thermal, metrics["thermal"], jitter, rank_rtol
        ),
        velocity=metric_orthonormalize(
            velocity, metrics["velocity"], jitter, rank_rtol
        ),
        log_tke=metric_orthonormalize(
            raw.log_tke, metrics["log_tke"], jitter, rank_rtol
        ),
        log_omega=metric_orthonormalize(
            raw.log_omega, metrics["log_omega"], jitter, rank_rtol
        ),
    )


def truncate_bases(bases: BasisBundle, ranks: tuple[int, int, int, int, int]) -> BasisBundle:
    rj, rt, rv, rk, rw = ranks
    return BasisBundle(
        current=bases.current[..., :rj],
        thermal=bases.thermal[..., :rt],
        velocity=bases.velocity[..., :rv],
        log_tke=bases.log_tke[..., :rk],
        log_omega=bases.log_omega[..., :rw],
    )
