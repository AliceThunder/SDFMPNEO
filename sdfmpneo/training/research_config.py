"""Solution-data-free finite-horizon residual training for the fixed analytic network."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.stats import qmc

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork

_HARD_WEIGHT_STRENGTH = 24.0


@dataclass(frozen=True)
class ResearchTrainingConfig:
    initial_lower: tuple[float, ...]
    initial_upper: tuple[float, ...]
    operating_lower: tuple[float, ...]
    operating_upper: tuple[float, ...]
    max_response_time: float
    residual_tolerance: float
    sample_count: int = 64
    validation_count: int = 64
    semigroup_sample_count: int = 8
    semigroup_validation_count: int = 8
    time_sampling: str = "mixed_log"
    time_min: float = 1e-6
    max_network_depth: int | None = None
    max_channels_per_mode: int | None = None
    max_linear_rank: int | None = None
    max_hidden_rank: int | None = None
    max_quadratic_rank: int | None = None
    max_square_rank: int | None = None
    max_cross_rank: int | None = None
    max_state_rank: int | None = None
    jacobian_point_budget: int = 8
    semigroup_jacobian_point_budget: int = 4
    max_iterations: int = 36
    max_validation_epochs: int = 5
    gate_shrink: float = 0.0
    prune_relative_budget: float = 0.10
    prune_rounds: int = 3

    def __post_init__(self):
        for key in ("initial_lower", "initial_upper", "operating_lower", "operating_upper"):
            object.__setattr__(self, key, tuple(float(v) for v in getattr(self, key)))
        if len(self.initial_lower) != len(self.initial_upper) or len(self.operating_lower) != len(self.operating_upper):
            raise ValueError("lower and upper dimensions must match")
        lo = np.asarray(self.initial_lower + self.operating_lower, dtype=float)
        hi = np.asarray(self.initial_upper + self.operating_upper, dtype=float)
        if np.any(~np.isfinite(lo + hi)) or np.any(lo > hi):
            raise ValueError("invalid training parameter box")
        if not np.isfinite(self.max_response_time) or self.max_response_time <= 0.0:
            raise ValueError("max_response_time must be finite and positive")
        if not np.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive")
        if self.time_sampling not in {"linear", "mixed_log"}:
            raise ValueError("time_sampling must be linear or mixed_log")
        if not np.isfinite(self.time_min) or self.time_min <= 0.0 or self.time_min > self.max_response_time:
            raise ValueError("time_min must be positive and no larger than max_response_time")
        for key in (
            "sample_count", "validation_count", "semigroup_sample_count",
            "semigroup_validation_count", "jacobian_point_budget",
            "semigroup_jacobian_point_budget", "max_iterations",
            "max_validation_epochs", "prune_rounds",
        ):
            value = int(getattr(self, key))
            if value != getattr(self, key) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            object.__setattr__(self, key, value)
        if min(self.sample_count, self.validation_count, self.jacobian_point_budget) < 1:
            raise ValueError("sample_count, validation_count and jacobian_point_budget must be positive")
        for key in (
            "max_network_depth", "max_channels_per_mode", "max_linear_rank",
            "max_hidden_rank", "max_quadratic_rank", "max_square_rank", "max_cross_rank", "max_state_rank",
        ):
            value = getattr(self, key)
            if value is not None:
                value = int(value)
                if value < 1:
                    raise ValueError(f"{key} must be positive when specified")
                object.__setattr__(self, key, value)
        if not np.isfinite(self.gate_shrink) or self.gate_shrink < 0.0:
            raise ValueError("gate_shrink must be finite and non-negative")
        if not np.isfinite(self.prune_relative_budget) or self.prune_relative_budget < 0.0:
            raise ValueError("prune_relative_budget must be finite and non-negative")

    def points(self, validation=False, seed=None):
        lo = np.asarray(self.initial_lower + self.operating_lower + (0.0,), dtype=float)
        hi = np.asarray(self.initial_upper + self.operating_upper + (self.max_response_time,), dtype=float)
        count = self.validation_count if validation else self.sample_count
        engine = qmc.Halton(len(lo), scramble=True, seed=(1 if validation else 0) if seed is None else seed)
        unit = engine.random(count)
        points = lo + unit * (hi - lo)
        if self.time_sampling == "mixed_log":
            ids = np.arange(0, count, 2)
            points[ids, -1] = np.exp(
                np.log(self.time_min) + unit[ids, -1] * (np.log(self.max_response_time) - np.log(self.time_min))
            )
        if not validation:
            center = 0.5 * (lo + hi); center[-1] = 0.0
            endpoint = center.copy(); endpoint[-1] = self.max_response_time
            points = np.vstack([points, center, lo, hi, endpoint])
            points[-3, -1] = 0.0
        return points

    def semigroup_points(self, validation=False, seed=None):
        count = self.semigroup_validation_count if validation else self.semigroup_sample_count
        width = len(self.initial_lower) + len(self.operating_lower) + 2
        if count == 0:
            return np.empty((0, width), dtype=float)
        lo = np.asarray(self.initial_lower + self.operating_lower, dtype=float)
        hi = np.asarray(self.initial_upper + self.operating_upper, dtype=float)
        engine = qmc.Halton(len(lo) + 2, scramble=True, seed=(101 if validation else 100) if seed is None else seed)
        unit = engine.random(count)
        static = lo + unit[:, :len(lo)] * (hi - lo)
        t1 = self.max_response_time * unit[:, -2]
        t2 = (self.max_response_time - t1) * unit[:, -1]
        rows = np.column_stack([static, t1, t2])
        if not validation:
            center = 0.5 * (lo + hi)
            rows = np.vstack([
                rows,
                np.concatenate([center, [0.5 * self.max_response_time, 0.5 * self.max_response_time]]),
                np.concatenate([center, [0.25 * self.max_response_time, 0.5 * self.max_response_time]]),
            ])
        return rows


@dataclass(frozen=True)
class ResearchTrainingReport:
    status: str
    active_response_channels: int
    max_response_time: float
    initial_rms_residual: float
    final_rms_residual: float
    maximum_training_residual: float
    maximum_validation_residual: float
    maximum_training_physics_residual: float
    maximum_validation_physics_residual: float
    maximum_training_semigroup_rate_defect: float
    maximum_validation_semigroup_rate_defect: float
    maximum_training_semigroup_defect: float
    maximum_validation_semigroup_defect: float
    objective_history: tuple[float, ...]
    numerical_tolerance_met: bool
    structure: dict

    def to_dict(self):
        return asdict(self)


def _default_capacity(n_modes, n_operating):
    """Choose a fixed maximum capacity that remains tractable at high thermal rank."""
    n_modes = int(n_modes)
    if n_modes >= 96:
        # High-rank thermal ROM: one analytic response per retained thermal mode.
        # Static geometry/current factors still provide nonlinear forcing, while
        # the exact initial thermal decay already carries every retained mode.
        return 1, 1, 4, 1, 8, 4, 1, 1
    if n_modes >= 32:
        return 1, 1, 4, 1, 6, 3, 1, 1
    if n_modes >= 9:
        return 2, 1, 3, 2, 4, 2, 2, 1
    return 2, 1, 2, 1, 2, 2, 1, 1


def make_fixed_network(field, config: ResearchTrainingConfig, *, operating_names=None):
    lambdas = np.asarray(field.thermal_model.lambdas, dtype=float)
    if len(config.initial_lower) != len(lambdas):
        raise ValueError("training initial-state dimension does not match thermal rank")
    if operating_names is None:
        operating_names = tuple(f"u{i}" for i in range(len(config.operating_lower)))
    operating_names = tuple(operating_names)
    if len(operating_names) != len(config.operating_lower):
        raise ValueError("operating_names dimension does not match training domain")
    lo = np.asarray(config.initial_lower + config.operating_lower, dtype=float)
    hi = np.asarray(config.initial_upper + config.operating_upper, dtype=float)
    center = 0.5 * (lo + hi)
    scale = 0.5 * (hi - lo)
    scale[scale <= 0.0] = 1.0
    defaults = _default_capacity(len(lambdas), len(operating_names))
    values = (
        config.max_network_depth, config.max_channels_per_mode,
        config.max_linear_rank, config.max_hidden_rank,
        config.max_quadratic_rank, config.max_square_rank,
        config.max_cross_rank, config.max_state_rank,
    )
    capacity = tuple(default if value is None else int(value) for default, value in zip(defaults, values))
    return FixedAnalyticResponseNetwork(
        lambdas, operating_names,
        max_response_time=config.max_response_time,
        input_center=center, input_scale=scale,
        depth=capacity[0], channels_per_mode=capacity[1],
        linear_rank=capacity[2], hidden_rank=capacity[3],
        quadratic_rank=capacity[4], square_rank=capacity[5],
        cross_rank=capacity[6], state_rank=capacity[7],
    )




__all__ = ["ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network"]
