from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CubatureRule:
    indices: Tensor
    weights: Tensor
    relative_error: float
    points: int

    def apply(self, element_values: Tensor) -> Tensor:
        """Apply the positive cubature rule along the element dimension."""
        selected = element_values.index_select(0, self.indices.to(element_values.device))
        weights = self.weights.to(dtype=selected.dtype, device=selected.device)
        return torch.tensordot(weights, selected, dims=([0], [0]))


def _projected_nnls(
    a: Tensor,
    target: Tensor,
    *,
    iterations: int = 600,
    tolerance: float = 1.0e-12,
) -> Tensor:
    """Small dense non-negative least-squares solve by projected gradient."""
    if a.ndim != 2 or target.ndim != 1 or a.shape[0] != target.shape[0]:
        raise ValueError("a must be [Nm,K] and target must be [Nm]")
    if a.shape[1] == 0:
        return torch.empty(0, dtype=a.dtype, device=a.device)

    spectral = torch.linalg.matrix_norm(a, ord=2)
    lipschitz = spectral.square().clamp_min(torch.finfo(a.dtype).eps)
    step = 1.0 / lipschitz

    try:
        initial = torch.linalg.lstsq(a, target).solution
        weights = initial.clamp_min(0)
    except RuntimeError:
        weights = torch.zeros(a.shape[1], dtype=a.dtype, device=a.device)

    target_scale = torch.linalg.vector_norm(target).clamp_min(
        torch.finfo(target.dtype).eps
    )
    previous: Tensor | None = None
    for _ in range(iterations):
        residual = a @ weights - target
        gradient = a.transpose(0, 1) @ residual
        candidate = (weights - step * gradient).clamp_min(0)
        error = torch.linalg.vector_norm(a @ candidate - target) / target_scale
        weights = candidate
        if previous is not None:
            scale = torch.maximum(torch.ones_like(error), previous.abs())
            if torch.abs(previous - error) <= tolerance * scale:
                break
        previous = error
    return weights


def build_positive_basis_operator_cubature(
    element_moments: Tensor,
    *,
    relative_tolerance: float = 1.0e-6,
    max_points: int | None = None,
    nnls_iterations: int = 600,
) -> CubatureRule:
    """Construct solution-free positive cubature from operator moments.

    `element_moments[e]` contains the contribution of element `e` to a vector
    of basis/operator moments. No full-order solution snapshot is used.
    """
    if element_moments.ndim != 2:
        raise ValueError("element_moments must have shape [Ne,Nm]")
    if not element_moments.is_floating_point():
        raise ValueError("element_moments must be floating point")
    ne, nm = element_moments.shape
    if ne == 0 or nm == 0:
        raise ValueError("element_moments must be non-empty")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie in (0,1)")

    max_points = min(ne, nm + 8) if max_points is None else min(ne, max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    target = element_moments.sum(dim=0)
    target_norm = torch.linalg.vector_norm(target)
    eps = torch.finfo(element_moments.dtype).eps
    if target_norm <= eps:
        return CubatureRule(
            indices=torch.empty(0, dtype=torch.long, device=element_moments.device),
            weights=torch.empty(0, dtype=element_moments.dtype, device=element_moments.device),
            relative_error=0.0,
            points=0,
        )

    norms = torch.linalg.vector_norm(element_moments, dim=1).clamp_min(eps)
    selected: list[int] = []
    available = torch.ones(ne, dtype=torch.bool, device=element_moments.device)
    weights = torch.empty(0, dtype=element_moments.dtype, device=element_moments.device)
    residual = target.clone()
    relative_error = 1.0

    for _ in range(max_points):
        correlations = (element_moments @ residual) / norms
        correlations = correlations.masked_fill(~available, -torch.inf)
        index = int(torch.argmax(correlations).item())
        if not torch.isfinite(correlations[index]):
            break
        selected.append(index)
        available[index] = False

        indices = torch.tensor(selected, dtype=torch.long, device=element_moments.device)
        a = element_moments.index_select(0, indices).transpose(0, 1)
        weights = _projected_nnls(a, target, iterations=nnls_iterations)
        residual = target - a @ weights
        relative_error = float((torch.linalg.vector_norm(residual) / target_norm).item())
        if relative_error <= relative_tolerance:
            break

    indices = torch.tensor(selected, dtype=torch.long, device=element_moments.device)
    if not selected:
        raise RuntimeError("positive cubature could not select a useful operator moment")
    if torch.any(weights < 0):
        raise RuntimeError("internal error: positive cubature produced negative weights")
    return CubatureRule(
        indices=indices,
        weights=weights,
        relative_error=relative_error,
        points=len(selected),
    )


def independent_certifier_indices(
    element_count: int,
    cubature_indices: Tensor,
    *,
    count: int,
    seed: int = 0,
) -> Tensor:
    """Select a disjoint deterministic-random certification set."""
    if element_count <= 0 or count <= 0:
        raise ValueError("element_count and count must be positive")
    excluded = torch.zeros(element_count, dtype=torch.bool)
    cpu_indices = cubature_indices.detach().to(device="cpu", dtype=torch.long)
    if torch.any(cpu_indices < 0) or torch.any(cpu_indices >= element_count):
        raise ValueError("cubature index out of range")
    excluded[cpu_indices] = True
    candidates = torch.arange(element_count, dtype=torch.long)[~excluded]
    if candidates.numel() < count:
        raise ValueError("not enough elements remain for an independent certifier set")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(candidates.numel(), generator=generator)
    return candidates[permutation[:count]]
