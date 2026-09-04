from __future__ import annotations

from .config import RankLevel
from .contracts import PhysicsSeedBundle


def physics_seed_ranks(seeds: PhysicsSeedBundle) -> tuple[int, int, int, int, int]:
    return (
        seeds.current.shape[-1],
        seeds.thermal.shape[-1],
        seeds.velocity.shape[-1],
        seeds.log_tke.shape[-1],
        seeds.log_omega.shape[-1],
    )


def validate_seed_prefix_ranks(seeds: PhysicsSeedBundle, first_level: RankLevel) -> None:
    """Ensure every nested rank retains the complete deterministic physics seed."""
    names = ("EM", "thermal", "velocity", "TKE", "omega")
    seed_ranks = physics_seed_ranks(seeds)
    online_ranks = (
        first_level.em,
        first_level.thermal,
        first_level.velocity,
        first_level.tke,
        first_level.omega,
    )
    for name, seed_rank, online_rank in zip(names, seed_ranks, online_ranks, strict=True):
        if seed_rank > online_rank:
            raise ValueError(
                f"{name} physics seed rank {seed_rank} exceeds first nested online "
                f"rank {online_rank}; the physical backbone must never be truncated."
            )
