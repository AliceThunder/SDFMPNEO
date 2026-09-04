from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .config import RankLevel
from .contracts import ReducedState


@dataclass(frozen=True)
class ReducedStateLayout:
    """Canonical slow-state layout z=[a_T,a_v,a_k,a_omega]."""

    thermal: slice
    velocity: slice
    tke: slice
    omega: slice
    size: int

    @classmethod
    def from_rank(cls, rank: RankLevel) -> "ReducedStateLayout":
        i0 = 0
        i1 = i0 + rank.thermal
        i2 = i1 + rank.velocity
        i3 = i2 + rank.tke
        i4 = i3 + rank.omega
        return cls(
            thermal=slice(i0, i1),
            velocity=slice(i1, i2),
            tke=slice(i2, i3),
            omega=slice(i3, i4),
            size=i4,
        )

    def split(self, z: Tensor) -> ReducedState:
        if z.shape[-1] != self.size:
            raise ValueError(f"slow-state size {z.shape[-1]} != expected {self.size}")
        return ReducedState(
            thermal=z[..., self.thermal],
            velocity=z[..., self.velocity],
            log_tke=z[..., self.tke],
            log_omega=z[..., self.omega],
        )

    def pack(self, state: ReducedState) -> Tensor:
        return state.flatten()
