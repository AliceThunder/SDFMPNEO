from __future__ import annotations

import torch
from torch import Tensor

from .contracts import BasisBundle, RawBasisBundle, TopologyOperators


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
    """Generate an H(div)-admissible basis from exact-sequence coordinates.

    raw_coefficients concatenates curl-potential coordinates followed by
    harmonic coordinates. With a compatible discrete de Rham complex,
    div(curl(.)) == 0 identically. Harmonic columns complete the solenoidal
    space for multiply connected domains.
    """
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


def metric_orthonormalize(basis: Tensor, metric: Tensor, jitter: float = 1.0e-9) -> Tensor:
    """Return B with B^H M B = I using a Cholesky whitening map.

    basis:  [B, N, r]
    metric: [B, N, N] or [N, N]
    """
    if metric.ndim == 2:
        gram = torch.einsum("bnr,nm,bms->brs", basis.conj(), metric, basis)
    elif metric.ndim == 3:
        gram = torch.einsum("bnr,bnm,bms->brs", basis.conj(), metric, basis)
    else:
        raise ValueError("metric must have shape [N,N] or [B,N,N]")

    rank = gram.shape[-1]
    eye = torch.eye(rank, dtype=gram.dtype, device=gram.device).expand_as(gram)
    chol = torch.linalg.cholesky(gram + jitter * eye)
    # B_new = B @ L^{-H}, with gram = L L^H.
    whiten = torch.linalg.solve_triangular(
        chol.conj().transpose(-2, -1), eye, upper=True
    )
    return basis @ whiten


def build_physical_bases(
    raw: RawBasisBundle,
    topology: TopologyOperators,
    metrics: dict[str, Tensor],
    jitter: float = 1.0e-9,
) -> BasisBundle:
    current = assemble_solenoidal_basis(
        topology.curl_current, topology.harmonic_current, raw.current_potential
    )
    velocity = assemble_solenoidal_basis(
        topology.curl_velocity, topology.harmonic_velocity, raw.velocity_potential
    )

    return BasisBundle(
        current=metric_orthonormalize(current, metrics["current"], jitter),
        thermal=metric_orthonormalize(raw.thermal, metrics["thermal"], jitter),
        velocity=metric_orthonormalize(velocity, metrics["velocity"], jitter),
        log_tke=metric_orthonormalize(raw.log_tke, metrics["log_tke"], jitter),
        log_omega=metric_orthonormalize(raw.log_omega, metrics["log_omega"], jitter),
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
