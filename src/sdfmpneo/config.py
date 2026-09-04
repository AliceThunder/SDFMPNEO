from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RankLevel:
    em: int
    thermal: int
    velocity: int
    tke: int
    omega: int


@dataclass(frozen=True)
class RankSchedule:
    levels: tuple[RankLevel, ...] = (
        RankLevel(8, 8, 12, 4, 4),
        RankLevel(16, 16, 20, 8, 8),
        RankLevel(24, 24, 28, 10, 10),
        RankLevel(32, 32, 36, 12, 12),
    )

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("At least one rank level is required.")
        fields = ("em", "thermal", "velocity", "tke", "omega")
        for name in fields:
            values = [getattr(level, name) for level in self.levels]
            if any(v <= 0 for v in values):
                raise ValueError(f"All {name} ranks must be positive.")
            if values != sorted(values):
                raise ValueError(f"Nested {name} ranks must be non-decreasing.")

    @property
    def maximum(self) -> RankLevel:
        return self.levels[-1]


@dataclass(frozen=True)
class SolverConfig:
    max_newton_steps: int = 30
    residual_tolerance: float = 1.0e-8
    initial_pseudo_dt: float = 1.0e-2
    max_pseudo_dt: float = 1.0e8
    pseudo_dt_growth: float = 4.0
    line_search_shrink: float = 0.5
    minimum_step: float = 1.0e-4
    armijo: float = 1.0e-4


@dataclass(frozen=True)
class ModelConfig:
    ranks: RankSchedule = field(default_factory=RankSchedule)
    solver: SolverConfig = field(default_factory=SolverConfig)
    impedance_tolerance: float = 1.0e-3
    thermal_indicator_tolerance: float = 1.0e-2
    flow_indicator_tolerance: float = 1.0e-2
    metric_jitter: float = 1.0e-9
    basis_rank_rtol: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.metric_jitter < 0:
            raise ValueError("metric_jitter must be non-negative")
        if not 0 < self.basis_rank_rtol < 1:
            raise ValueError("basis_rank_rtol must lie in (0,1)")
