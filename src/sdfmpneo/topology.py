from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import TopologyOperators


@dataclass(frozen=True)
class TopologyValidation:
    current_exact_defect: float
    current_harmonic_defect: float
    velocity_exact_defect: float
    velocity_harmonic_defect: float
    current_generator_rank: int
    velocity_generator_rank: int


def _matrix_rank(matrix: Tensor, rtol: float) -> int:
    if matrix.numel() == 0:
        return 0
    singular = torch.linalg.svdvals(matrix)
    if singular.numel() == 0:
        return 0
    threshold = rtol * singular.max()
    return int((singular > threshold).sum().item())


def _relative_defect(left: Tensor, right: Tensor) -> float:
    product = left @ right
    denominator = (
        torch.linalg.matrix_norm(left, ord="fro")
        * torch.linalg.matrix_norm(right, ord="fro")
    ).clamp_min(torch.finfo(product.dtype).eps)
    return float((torch.linalg.matrix_norm(product, ord="fro") / denominator).item())


def validate_topology(
    topology: TopologyOperators,
    *,
    exact_tolerance: float = 1.0e-10,
    rank_rtol: float = 1.0e-10,
) -> TopologyValidation:
    """Validate de Rham identities and gauge-reduced generator independence.

    This function deliberately does not try to infer topology from geometry. The
    backend owns the reference complex and harmonic construction. The core model
    only accepts generators that satisfy

        D C_g = 0,  D H = 0,

    and whose concatenated exact+harmonic columns are linearly independent.
    This prevents neural modes from being spent on gauge-null coordinates.
    """
    if topology.curl_current.ndim != 2 or topology.harmonic_current.ndim != 2:
        raise ValueError("current topology matrices must be two-dimensional")
    if topology.curl_velocity.ndim != 2 or topology.harmonic_velocity.ndim != 2:
        raise ValueError("velocity topology matrices must be two-dimensional")
    if topology.divergence_current.ndim != 2 or topology.divergence_velocity.ndim != 2:
        raise ValueError("divergence matrices must be two-dimensional")

    nj = topology.curl_current.shape[0]
    nv = topology.curl_velocity.shape[0]
    if topology.harmonic_current.shape[0] != nj:
        raise ValueError("current harmonic and exact generators must share physical DOFs")
    if topology.harmonic_velocity.shape[0] != nv:
        raise ValueError("velocity harmonic and exact generators must share physical DOFs")
    if topology.divergence_current.shape[1] != nj:
        raise ValueError("current divergence map has incompatible DOF dimension")
    if topology.divergence_velocity.shape[1] != nv:
        raise ValueError("velocity divergence map has incompatible DOF dimension")

    current_exact_defect = _relative_defect(
        topology.divergence_current, topology.curl_current
    )
    current_harmonic_defect = _relative_defect(
        topology.divergence_current, topology.harmonic_current
    ) if topology.harmonic_current.shape[1] else 0.0
    velocity_exact_defect = _relative_defect(
        topology.divergence_velocity, topology.curl_velocity
    )
    velocity_harmonic_defect = _relative_defect(
        topology.divergence_velocity, topology.harmonic_velocity
    ) if topology.harmonic_velocity.shape[1] else 0.0

    defects = {
        "current exact": current_exact_defect,
        "current harmonic": current_harmonic_defect,
        "velocity exact": velocity_exact_defect,
        "velocity harmonic": velocity_harmonic_defect,
    }
    failed = {name: value for name, value in defects.items() if value > exact_tolerance}
    if failed:
        formatted = ", ".join(f"{name}={value:.3e}" for name, value in failed.items())
        raise ValueError(f"Topology violates solenoidal exact-sequence constraints: {formatted}")

    current_generator = torch.cat(
        (topology.curl_current, topology.harmonic_current), dim=-1
    )
    velocity_generator = torch.cat(
        (topology.curl_velocity, topology.harmonic_velocity), dim=-1
    )
    current_rank = _matrix_rank(current_generator, rank_rtol)
    velocity_rank = _matrix_rank(velocity_generator, rank_rtol)

    if current_rank != current_generator.shape[1]:
        raise ValueError(
            "Current exact+harmonic generator is column-rank deficient. Remove gauge "
            "null directions before exposing it to the neural basis generator."
        )
    if velocity_rank != velocity_generator.shape[1]:
        raise ValueError(
            "Velocity exact+harmonic generator is column-rank deficient. Remove gauge "
            "null directions before exposing it to the neural basis generator."
        )

    return TopologyValidation(
        current_exact_defect=current_exact_defect,
        current_harmonic_defect=current_harmonic_defect,
        velocity_exact_defect=velocity_exact_defect,
        velocity_harmonic_defect=velocity_harmonic_defect,
        current_generator_rank=current_rank,
        velocity_generator_rank=velocity_rank,
    )
