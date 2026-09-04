from __future__ import annotations

import torch
from torch import Tensor

from .contracts import (
    BasisBundle,
    PhysicsSeedBundle,
    RawBasisBundle,
    TopologyOperators,
)


class BasisRankError(RuntimeError):
    """Raised when a trial space is rank deficient before whitening."""


def _batched_map(matrix: Tensor, coefficients: Tensor) -> Tensor:
    if matrix.ndim == 2:
        return torch.einsum("mn,bnr->bmr", matrix, coefficients)
    if matrix.ndim == 3:
        return torch.einsum("bmn,bnr->bmr", matrix, coefficients)
    raise ValueError("matrix must have shape [M,N] or [B,M,N]")


def _metric_action(metric: Tensor, basis: Tensor) -> Tensor:
    if metric.ndim == 2:
        return torch.einsum("nm,bmr->bnr", metric, basis)
    if metric.ndim == 3:
        return torch.einsum("bnm,bmr->bnr", metric, basis)
    raise ValueError("metric must have shape [N,N] or [B,N,N]")


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
    """Return B with B^H M B = I after checking unregularised rank."""
    if basis.ndim != 3:
        raise ValueError("basis must have shape [B,N,r]")
    if basis.shape[-1] == 0:
        return basis
    if basis.shape[-2] < basis.shape[-1]:
        raise BasisRankError(
            f"trial space requests rank {basis.shape[-1]} from only "
            f"{basis.shape[-2]} physical/query DOFs"
        )
    if not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie in (0,1)")

    metric_basis = _metric_action(metric, basis)
    gram = basis.conj().transpose(-2, -1) @ metric_basis
    gram = 0.5 * (gram + gram.conj().transpose(-2, -1))

    eigenvalues = torch.linalg.eigvalsh(gram.detach())
    largest = eigenvalues[..., -1]
    smallest = eigenvalues[..., 0]
    if torch.any(largest <= 0):
        raise BasisRankError("metric Gram matrix has no positive trial-space energy")
    relative_smallest = smallest / largest
    if torch.any(relative_smallest <= rank_rtol):
        worst = float(relative_smallest.min().item())
        raise BasisRankError(
            "trial basis is rank deficient or numerically collapsed before whitening "
            f"(min/max Gram eigenvalue={worst:.3e}, threshold={rank_rtol:.3e})"
        )

    rank = gram.shape[-1]
    eye = torch.eye(rank, dtype=gram.dtype, device=gram.device).expand_as(gram)
    chol = torch.linalg.cholesky(gram + jitter * eye)
    whiten = torch.linalg.solve_triangular(
        chol.conj().transpose(-2, -1), eye, upper=True
    )
    return basis @ whiten


def seeded_metric_orthonormalize(
    seed: Tensor,
    neural: Tensor,
    metric: Tensor,
    *,
    total_rank: int,
    jitter: float,
    rank_rtol: float,
) -> Tensor:
    """Preserve physics seed modes and learn only their metric-orthogonal complement.

    The seed columns form the prefix of the returned nested basis. Neural columns
    are projected out of the seed span before their own rank check/whitening.
    """
    if seed.ndim != 3 or neural.ndim != 3:
        raise ValueError("seed and neural bases must have shape [B,N,r]")
    if seed.shape[:2] != neural.shape[:2]:
        raise ValueError("seed and neural bases must share batch and physical DOFs")
    seed_rank = seed.shape[-1]
    if seed_rank > total_rank:
        raise BasisRankError(
            f"physics seed rank {seed_rank} exceeds requested total rank {total_rank}"
        )

    seed_orth = metric_orthonormalize(seed, metric, jitter, rank_rtol)
    neural_needed = total_rank - seed_rank
    if neural_needed == 0:
        return seed_orth
    if neural.shape[-1] < neural_needed:
        raise BasisRankError(
            f"only {neural.shape[-1]} neural candidate modes are available but "
            f"{neural_needed} enrichment modes are required"
        )

    candidate = neural[..., :neural_needed]
    if seed_rank:
        metric_candidate = _metric_action(metric, candidate)
        coefficients = seed_orth.conj().transpose(-2, -1) @ metric_candidate
        candidate = candidate - seed_orth @ coefficients
        # One re-projection after finite-precision subtraction protects the hard
        # seed prefix before whitening the neural complement.
        metric_candidate = _metric_action(metric, candidate)
        coefficients = seed_orth.conj().transpose(-2, -1) @ metric_candidate
        candidate = candidate - seed_orth @ coefficients

    neural_orth = metric_orthonormalize(candidate, metric, jitter, rank_rtol)
    return torch.cat((seed_orth, neural_orth), dim=-1)


def _solenoidal_seed_defect(divergence: Tensor | None, seed: Tensor) -> float:
    if seed.shape[-1] == 0:
        return 0.0
    if divergence is None:
        raise ValueError("divergence operator required to validate vector physics seeds")
    if divergence.ndim != 2:
        raise ValueError("seed validation currently expects an unbatched divergence map")
    residual = torch.einsum("dn,bnr->bdr", divergence, seed)
    numerator = torch.linalg.vector_norm(residual)
    denominator = (
        torch.linalg.matrix_norm(divergence, ord="fro")
        * torch.linalg.vector_norm(seed)
    ).clamp_min(torch.finfo(seed.dtype).eps)
    return float((numerator / denominator).item())


def validate_physics_seeds(
    seeds: PhysicsSeedBundle,
    topology: TopologyOperators,
    *,
    solenoidal_tolerance: float = 1.0e-10,
) -> None:
    current_defect = _solenoidal_seed_defect(
        topology.divergence_current, seeds.current
    )
    velocity_defect = _solenoidal_seed_defect(
        topology.divergence_velocity, seeds.velocity
    )
    if current_defect > solenoidal_tolerance:
        raise ValueError(
            f"EM physics seed violates divergence-free space: {current_defect:.3e}"
        )
    if velocity_defect > solenoidal_tolerance:
        raise ValueError(
            f"velocity physics seed violates divergence-free space: {velocity_defect:.3e}"
        )


def build_physical_bases(
    raw: RawBasisBundle,
    topology: TopologyOperators,
    metrics: dict[str, Tensor],
    seeds: PhysicsSeedBundle,
    jitter: float = 1.0e-9,
    rank_rtol: float = 1.0e-10,
) -> BasisBundle:
    """Build physics-seeded neural trial spaces at their maximum ranks."""
    current_neural = assemble_solenoidal_basis(
        topology.curl_current, topology.harmonic_current, raw.current_potential
    )
    velocity_neural = assemble_solenoidal_basis(
        topology.curl_velocity, topology.harmonic_velocity, raw.velocity_potential
    )
    validate_physics_seeds(seeds, topology)

    return BasisBundle(
        current=seeded_metric_orthonormalize(
            seeds.current,
            current_neural,
            metrics["current"],
            total_rank=raw.current_potential.shape[-1],
            jitter=jitter,
            rank_rtol=rank_rtol,
        ),
        thermal=seeded_metric_orthonormalize(
            seeds.thermal,
            raw.thermal,
            metrics["thermal"],
            total_rank=raw.thermal.shape[-1],
            jitter=jitter,
            rank_rtol=rank_rtol,
        ),
        velocity=seeded_metric_orthonormalize(
            seeds.velocity,
            velocity_neural,
            metrics["velocity"],
            total_rank=raw.velocity_potential.shape[-1],
            jitter=jitter,
            rank_rtol=rank_rtol,
        ),
        log_tke=seeded_metric_orthonormalize(
            seeds.log_tke,
            raw.log_tke,
            metrics["log_tke"],
            total_rank=raw.log_tke.shape[-1],
            jitter=jitter,
            rank_rtol=rank_rtol,
        ),
        log_omega=seeded_metric_orthonormalize(
            seeds.log_omega,
            raw.log_omega,
            metrics["log_omega"],
            total_rank=raw.log_omega.shape[-1],
            jitter=jitter,
            rank_rtol=rank_rtol,
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
