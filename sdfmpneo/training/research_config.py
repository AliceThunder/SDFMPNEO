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
    source_prefit_count: int = 64
    initial_training_rank: int = 16
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
    network_layer_widths: tuple[int, ...] | None = None
    network_hidden_ranks: tuple[int, ...] | None = None
    network_cross_ranks: tuple[int, ...] | None = None
    network_state_ranks: tuple[int, ...] | None = None
    state_feature_term_budget: int = 24
    jacobian_point_budget: int = 12
    semigroup_jacobian_point_budget: int = 4
    max_iterations: int = 36
    max_iterations_per_layer: int = 12
    max_validation_epochs: int = 5
    max_damping_retries: int = 4
    max_backtracks: int = 6
    backtrack_factor: float = 0.5
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
        integer_keys = (
            "sample_count", "validation_count", "semigroup_sample_count",
            "semigroup_validation_count", "source_prefit_count", "initial_training_rank",
            "state_feature_term_budget", "jacobian_point_budget",
            "semigroup_jacobian_point_budget", "max_iterations",
            "max_iterations_per_layer", "max_validation_epochs",
            "max_damping_retries", "max_backtracks", "prune_rounds",
        )
        for key in integer_keys:
            value = int(getattr(self, key))
            if value != getattr(self, key) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            object.__setattr__(self, key, value)
        if min(
            self.sample_count, self.validation_count, self.source_prefit_count,
            self.initial_training_rank, self.state_feature_term_budget,
            self.jacobian_point_budget, self.max_iterations_per_layer,
            self.max_damping_retries, self.max_backtracks,
        ) < 1:
            raise ValueError("training counts/ranks/retry budgets must be positive")
        for key in (
            "max_network_depth", "max_channels_per_mode", "max_linear_rank",
            "max_hidden_rank", "max_quadratic_rank", "max_square_rank",
            "max_cross_rank", "max_state_rank",
        ):
            value = getattr(self, key)
            if value is not None:
                value = int(value)
                if value < 1:
                    raise ValueError(f"{key} must be positive when specified")
                object.__setattr__(self, key, value)
        for key in (
            "network_layer_widths", "network_hidden_ranks",
            "network_cross_ranks", "network_state_ranks",
        ):
            value = getattr(self, key)
            if value is not None:
                value = tuple(int(v) for v in value)
                minimum = 1 if key == "network_layer_widths" else 0
                if any(v < minimum for v in value):
                    raise ValueError(f"invalid {key}")
                object.__setattr__(self, key, value)
        if not np.isfinite(self.backtrack_factor) or not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must be strictly between zero and one")
        if not np.isfinite(self.gate_shrink) or self.gate_shrink < 0.0:
            raise ValueError("gate_shrink must be finite and non-negative")
        if not np.isfinite(self.prune_relative_budget) or self.prune_relative_budget < 0.0:
            raise ValueError("prune_relative_budget must be finite and non-negative")

    def initial_active_indices(self):
        """Modes used to parameterize restart seeds instead of a Cartesian r-box.

        Automatic restart bounds are usually similar across modes, so sorting by
        span alone can accidentally choose high-index fast modes on ties.  Prefer
        larger spans first and, for equal spans, lower thermal indices.  Thermal
        modes are ordered by increasing decay rate, making this deterministic
        tie-break favor the slow modes that dominate reachable long-time states.
        """
        n = len(self.initial_lower)
        rank = min(n, max(1, int(self.initial_training_rank)))
        span = np.asarray(self.initial_upper, float) - np.asarray(self.initial_lower, float)
        order = np.lexsort((np.arange(n, dtype=int), -span))
        return np.sort(order[:rank].astype(int))

    def _sample_initial_operating(self, count, seed, extra_dimensions=0):
        count = int(count)
        active = self.initial_active_indices()
        n_operating = len(self.operating_lower)
        dimension = len(active) + n_operating + int(extra_dimensions)
        engine = qmc.Halton(max(1, dimension), scramble=True, seed=int(seed))
        unit = engine.random(count)
        cursor = 0

        ilo = np.asarray(self.initial_lower, float)
        ihi = np.asarray(self.initial_upper, float)
        center = 0.5 * (ilo + ihi)
        half = 0.5 * (ihi - ilo)
        initial = np.repeat(center[None, :], count, axis=0)
        if len(active):
            local = 2.0 * unit[:, cursor:cursor + len(active)] - 1.0
            cursor += len(active)
            local /= np.sqrt(max(1, len(active)))
            initial[:, active] += local * half[active][None, :]

        olo = np.asarray(self.operating_lower, float)
        ohi = np.asarray(self.operating_upper, float)
        if n_operating:
            op_unit = unit[:, cursor:cursor + n_operating]
            cursor += n_operating
            operating = olo + op_unit * (ohi - olo)
        else:
            operating = np.empty((count, 0), dtype=float)
        extra = unit[:, cursor:cursor + extra_dimensions] if extra_dimensions else np.empty((count, 0))
        return initial, operating, extra

    def source_points(self, seed=211):
        initial, operating, _ = self._sample_initial_operating(
            self.source_prefit_count, seed, 0
        )
        center_initial = 0.5 * (
            np.asarray(self.initial_lower) + np.asarray(self.initial_upper)
        )
        center_operating = 0.5 * (
            np.asarray(self.operating_lower) + np.asarray(self.operating_upper)
        )
        anchors = [np.concatenate([center_initial, center_operating])]
        if len(center_operating):
            anchors.extend((
                np.concatenate([center_initial, np.asarray(self.operating_lower)]),
                np.concatenate([center_initial, np.asarray(self.operating_upper)]),
            ))
        return np.vstack([np.column_stack([initial, operating]), *anchors])

    def points(self, validation=False, seed=None):
        count = self.validation_count if validation else self.sample_count
        resolved_seed = (1 if validation else 0) if seed is None else int(seed)
        initial, operating, extra = self._sample_initial_operating(
            count, resolved_seed, 1
        )
        unit_time = extra[:, 0]
        time = self.max_response_time * unit_time
        if self.time_sampling == "mixed_log":
            ids = np.arange(0, count, 2)
            time[ids] = np.exp(
                np.log(self.time_min)
                + unit_time[ids] * (
                    np.log(self.max_response_time) - np.log(self.time_min)
                )
            )
        points = np.column_stack([initial, operating, time])
        if not validation:
            center = np.concatenate([
                0.5 * (np.asarray(self.initial_lower) + np.asarray(self.initial_upper)),
                0.5 * (np.asarray(self.operating_lower) + np.asarray(self.operating_upper)),
                [0.0],
            ])
            endpoint = center.copy(); endpoint[-1] = self.max_response_time
            points = np.vstack([points, center, endpoint])
        return points

    def semigroup_points(self, validation=False, seed=None):
        count = self.semigroup_validation_count if validation else self.semigroup_sample_count
        width = len(self.initial_lower) + len(self.operating_lower) + 2
        if count == 0:
            return np.empty((0, width), dtype=float)
        resolved_seed = (101 if validation else 100) if seed is None else int(seed)
        initial, operating, extra = self._sample_initial_operating(
            count, resolved_seed, 2
        )
        t1 = self.max_response_time * extra[:, 0]
        t2 = (self.max_response_time - t1) * extra[:, 1]
        rows = np.column_stack([initial, operating, t1, t2])
        if not validation:
            center = np.concatenate([
                0.5 * (np.asarray(self.initial_lower) + np.asarray(self.initial_upper)),
                0.5 * (np.asarray(self.operating_lower) + np.asarray(self.operating_upper)),
            ])
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
    validation_performed: bool = True
    source_prefit_rms_residual: float | None = None
    source_prefit_max_residual: float | None = None
    trained_response_depth: int = 0

    def to_dict(self):
        return asdict(self)


def _default_capacity(n_modes, n_operating):
    """Legacy scalar view of the default multilayer response capacity."""
    n_modes = int(n_modes)
    n_operating = int(n_operating)
    if n_modes >= 96:
        return 3, 1, 8, 6, 16, 8, 6, 4
    if n_modes >= 32:
        return 3, 1, 6, 4, 12, 6, 4, 2
    if n_modes >= 9:
        return 3, 1, 4, 3, max(6, min(10, n_operating + 1)), 4, 3, 2
    return 2, 1, 3, 2, max(3, min(6, n_operating + 1)), 2, 2, 1


def _default_layer_architecture(n_modes, depth):
    n_modes, depth = int(n_modes), int(depth)
    if depth < 1:
        raise ValueError("response depth must be positive")
    if n_modes >= 96:
        widths = [n_modes, min(64, n_modes), min(32, n_modes)]
        hidden, cross, state = [6, 4], [6, 4], [4, 2]
    elif n_modes >= 32:
        widths = [n_modes, min(32, n_modes), min(16, n_modes)]
        hidden, cross, state = [4, 3], [4, 3], [2, 1]
    else:
        widths = [n_modes, n_modes, max(1, n_modes // 2)]
        hidden, cross, state = [3, 2], [3, 2], [2, 1]
    if depth <= len(widths):
        widths = widths[:depth]
    else:
        widths.extend([widths[-1]] * (depth - len(widths)))
    needed = max(0, depth - 1)
    if needed > len(hidden):
        hidden.extend([hidden[-1]] * (needed - len(hidden)))
        cross.extend([cross[-1]] * (needed - len(cross)))
        state.extend([state[-1]] * (needed - len(state)))
    return tuple(widths), tuple(hidden[:needed]), tuple(cross[:needed]), tuple(state[:needed])


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
    depth = defaults[0] if config.max_network_depth is None else int(config.max_network_depth)
    channels = defaults[1] if config.max_channels_per_mode is None else int(config.max_channels_per_mode)
    if channels != 1:
        raise ValueError("multilayer source-prefit training currently requires one channel per thermal mode")
    widths, hidden_by_layer, cross_by_layer, state_by_layer = _default_layer_architecture(
        len(lambdas), depth
    )
    if config.network_layer_widths is not None:
        widths = tuple(config.network_layer_widths)
        if len(widths) != depth:
            raise ValueError("network_layer_widths must match max_network_depth")
    if config.network_hidden_ranks is not None:
        hidden_by_layer = tuple(config.network_hidden_ranks)
    elif config.max_hidden_rank is not None:
        hidden_by_layer = (int(config.max_hidden_rank),) * max(0, depth - 1)
    if config.network_cross_ranks is not None:
        cross_by_layer = tuple(config.network_cross_ranks)
    elif config.max_cross_rank is not None:
        cross_by_layer = (int(config.max_cross_rank),) * max(0, depth - 1)
    if config.network_state_ranks is not None:
        state_by_layer = tuple(config.network_state_ranks)
    elif config.max_state_rank is not None:
        state_by_layer = (int(config.max_state_rank),) * max(0, depth - 1)
    if not (
        len(hidden_by_layer) == len(cross_by_layer) == len(state_by_layer) == max(0, depth - 1)
    ):
        raise ValueError("per-layer response ranks must contain depth-1 entries")

    linear_rank = defaults[2] if config.max_linear_rank is None else int(config.max_linear_rank)
    quadratic_rank = defaults[4] if config.max_quadratic_rank is None else int(config.max_quadratic_rank)
    square_rank = defaults[5] if config.max_square_rank is None else int(config.max_square_rank)
    return FixedAnalyticResponseNetwork(
        lambdas, operating_names,
        max_response_time=config.max_response_time,
        input_center=center, input_scale=scale,
        depth=depth, channels_per_mode=channels,
        linear_rank=linear_rank,
        hidden_rank=max(hidden_by_layer, default=1),
        quadratic_rank=quadratic_rank,
        square_rank=square_rank,
        cross_rank=max(cross_by_layer, default=1),
        state_rank=max(state_by_layer, default=1),
        layer_widths=widths,
        layer_hidden_ranks=hidden_by_layer,
        layer_cross_ranks=cross_by_layer,
        layer_state_ranks=state_by_layer,
        state_feature_term_budget=config.state_feature_term_budget,
    )


__all__ = [
    "ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network",
    "_default_capacity", "_default_layer_architecture",
]
