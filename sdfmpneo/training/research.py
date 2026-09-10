"""Solution-data-free continuous residual training for the fixed analytic network."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

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
    time_horizon: float
    residual_tolerance: float
    sample_count: int = 64
    validation_count: int = 64
    time_sampling: str = "mixed_log"
    time_min: float = 1e-6
    include_steady_state: bool = True
    max_network_depth: int | None = None
    max_channels_per_mode: int | None = None
    max_quadratic_rank: int | None = None
    max_cross_rank: int | None = None
    max_state_rank: int | None = None
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
        if not np.isfinite(self.time_horizon) or self.time_horizon <= 0.0:
            raise ValueError("time_horizon must be finite and positive")
        if not np.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive")
        if self.time_sampling not in {"linear", "mixed_log"}:
            raise ValueError("time_sampling must be linear or mixed_log")
        if not np.isfinite(self.time_min) or self.time_min <= 0.0 or self.time_min > self.time_horizon:
            raise ValueError("time_min must be positive and no larger than time_horizon")
        for key in ("sample_count", "validation_count", "max_iterations", "max_validation_epochs", "prune_rounds"):
            value = int(getattr(self, key))
            if value != getattr(self, key) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            object.__setattr__(self, key, value)
        if min(self.sample_count, self.validation_count) < 1:
            raise ValueError("sample_count and validation_count must be positive")
        for key in ("max_network_depth", "max_channels_per_mode", "max_quadratic_rank", "max_cross_rank", "max_state_rank"):
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
        hi = np.asarray(self.initial_upper + self.operating_upper + (self.time_horizon,), dtype=float)
        count = self.validation_count if validation else self.sample_count
        engine = qmc.Halton(len(lo), scramble=True, seed=(1 if validation else 0) if seed is None else seed)
        unit = engine.random(count)
        points = lo + unit * (hi - lo)
        if self.time_sampling == "mixed_log":
            ids = np.arange(0, count, 2)
            points[ids, -1] = np.exp(np.log(self.time_min) + unit[ids, -1] * (np.log(self.time_horizon) - np.log(self.time_min)))
        if not validation:
            center = 0.5 * (lo + hi)
            center[-1] = 0.0
            points = np.vstack([points, center, lo, hi])
            points[-2, -1] = 0.0
        if self.include_steady_state:
            steady = points[:count].copy()
            steady[:, -1] = np.inf
            points = np.vstack([points, steady])
        return points


@dataclass(frozen=True)
class ResearchTrainingReport:
    status: str
    active_response_channels: int
    initial_rms_residual: float
    final_rms_residual: float
    maximum_training_residual: float
    maximum_validation_residual: float
    objective_history: tuple[float, ...]
    numerical_tolerance_met: bool
    structure: dict

    def to_dict(self):
        return asdict(self)


def _default_capacity(n_modes, n_operating):
    input_dimension = 1 + int(n_modes) + int(n_operating)
    depth = 5 if n_modes <= 8 else 4
    channels = 2
    quadratic = min(12, max(4, (input_dimension + 1) // 2))
    cross = min(6, max(2, int(np.ceil(np.sqrt(input_dimension)))))
    state = min(4, max(2, int(np.ceil(np.sqrt(max(1, n_modes))))))
    return depth, channels, quadratic, cross, state


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
    values = (config.max_network_depth, config.max_channels_per_mode, config.max_quadratic_rank, config.max_cross_rank, config.max_state_rank)
    capacity = tuple(default if value is None else int(value) for default, value in zip(defaults, values))
    return FixedAnalyticResponseNetwork(lambdas, operating_names, input_center=center, input_scale=scale, depth=capacity[0], channels_per_mode=capacity[1], quadratic_rank=capacity[2], cross_rank=capacity[3], state_rank=capacity[4])


def _evaluate_network(network, field, points, *, jacobian=False, monitor=None):
    points = np.asarray(points, dtype=float)
    n = network.n_modes
    records = []
    prepare = getattr(field, "prepare_training_contexts", None)
    if prepare is not None:
        prepare(points)
    for point in points:
        if monitor is not None:
            monitor.checkpoint()
        initial, operating, time = point[:n], point[n:-1], float(point[-1])
        if jacobian:
            a, da, ja, jda = network.evaluate_parameter_jacobian(time, a0=initial, operating=operating)
            physical = field.evaluate(a, operating)
            records.append(SimpleNamespace(residual=np.asarray(da - physical.vector_field, dtype=float), parameter_jacobian=np.asarray(jda - physical.vector_field_jacobian @ ja, dtype=float)))
        else:
            a, da = network.evaluate(time, a0=initial, operating=operating)
            F = field.vector_field(a, operating) if hasattr(field, "vector_field") else field.evaluate(a, operating).vector_field
            records.append(SimpleNamespace(residual=np.asarray(da - F, dtype=float)))
    return records


def _metrics(records):
    norms = np.asarray([np.linalg.norm(record.residual) for record in records], dtype=float)
    objective = float(np.mean(norms * norms)) if norms.size else 0.0
    return objective, float(np.max(norms, initial=0.0)), norms


def _hard_weights(norms, tolerance):
    values = np.asarray(norms, dtype=float)
    excess = np.maximum(values - float(tolerance), 0.0)
    peak = float(np.max(excess, initial=0.0))
    if peak <= 0.0 or not np.isfinite(peak):
        return np.ones_like(values)
    ratio = excess / peak
    return 1.0 + _HARD_WEIGHT_STRENGTH * ratio * ratio


def _accept(old_norms, new_norms, tolerance, weights):
    old = np.asarray(old_norms, dtype=float)
    new = np.asarray(new_norms, dtype=float)
    protected = old <= tolerance
    margin = 64.0 * np.finfo(float).eps * max(float(tolerance), 1e-30)
    if np.any(new[protected] > float(tolerance) + margin):
        return False
    old_max = float(np.max(old, initial=0.0))
    new_max = float(np.max(new, initial=0.0))
    numerical = 64.0 * np.finfo(float).eps * max(old_max, float(tolerance), 1e-30)
    if new_max < old_max - numerical:
        return True
    if new_max <= old_max + numerical:
        return float(np.dot(weights, new * new)) < float(np.dot(weights, old * old)) - numerical
    return False


def _solve_direction(records, weights, damping, network):
    residual = np.vstack([record.residual for record in records])
    jacobian = np.stack([record.parameter_jacobian for record in records])
    root = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    rw = (root * residual).reshape(-1)
    Jw = (root[:, :, None] * jacobian).reshape(-1, network.parameter_count)
    scales = np.linalg.norm(Jw, axis=0)
    scales[scales < 1e-14] = 1.0
    Jn = Jw / scales
    m, p = Jn.shape
    mu = max(float(damping), 1e-12)
    if m <= p:
        system = Jn @ Jn.T
        system.flat[:: m + 1] += mu
        y = np.linalg.lstsq(system, -rw, rcond=None)[0]
        q = Jn.T @ y
    else:
        system = Jn.T @ Jn
        system.flat[:: p + 1] += mu
        q = np.linalg.lstsq(system, -(Jn.T @ rw), rcond=None)[0]
    delta = q / scales
    trust = 2.0 * max(1.0, float(np.linalg.norm(network.parameters)))
    norm = float(np.linalg.norm(delta))
    if norm > trust:
        delta *= trust / norm
    return delta


def _unique_rows(*arrays, width):
    rows, seen = [], set()
    for array in arrays:
        values = np.asarray(array, dtype=float)
        if values.size == 0:
            continue
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError("collocation dimensions do not match")
        for row in values:
            key = tuple(float(v) for v in row)
            if key not in seen:
                seen.add(key)
                rows.append(row.copy())
    return np.vstack(rows) if rows else np.empty((0, width), dtype=float)


def _prune(network, field, points, tolerance, relative_budget, rounds, monitor):
    current = network
    for _ in range(rounds):
        entries = [entry for entry in current.structure_gate_entries() if entry["value"] != 0.0]
        if not entries:
            break
        records = _evaluate_network(current, field, points, jacobian=True, monitor=monitor)
        _, current_max, _ = _metrics(records)
        jac = np.stack([record.parameter_jacobian for record in records])
        scored = []
        for entry in entries:
            column = jac[:, :, entry["parameter_index"]]
            effect = abs(entry["value"]) * float(np.max(np.linalg.norm(column, axis=1), initial=0.0))
            scored.append((effect, entry["parameter_index"]))
        scored.sort()
        budget = max(0.0, tolerance - current_max) + relative_budget * tolerance
        chosen, used = [], 0.0
        for effect, index in scored:
            if effect == 0.0 or used + effect <= budget:
                chosen.append(index)
                used += effect
            else:
                break
        if not chosen:
            break
        accepted = None
        count = len(chosen)
        while count:
            theta = current.parameters.copy()
            theta[np.asarray(chosen[:count], dtype=int)] = 0.0
            trial = current.with_parameters(theta)
            _, maximum, _ = _metrics(_evaluate_network(trial, field, points, monitor=monitor))
            if maximum <= tolerance:
                accepted = trial
                break
            count //= 2
        if accepted is None:
            break
        current = accepted
    return current


def train_research_network(field, config: ResearchTrainingConfig, *, network=None, operating_names=None, progress=None, monitor=None):
    """Fit all fixed-network parameters jointly; no structural candidate search exists."""
    if network is None:
        network = make_fixed_network(field, config, operating_names=operating_names)
    if not isinstance(network, FixedAnalyticResponseNetwork):
        raise TypeError("only FixedAnalyticResponseNetwork checkpoints are supported")
    if len(config.initial_lower) != network.n_modes or len(config.operating_lower) != len(network.operating_names):
        raise ValueError("training domain does not match the fixed analytic network")
    points = np.asarray(config.points(validation=False), dtype=float)
    checks = np.asarray(config.points(validation=True), dtype=float)
    guards = np.empty((0, points.shape[1]), dtype=float)
    if monitor is not None:
        monitor.retain(network)
        monitor.phase("initial_residual")
    records = _evaluate_network(network, field, points, monitor=monitor)
    objective, maximum, norms = _metrics(records)
    initial_rms = float(np.sqrt(objective))
    history = [objective]
    if monitor is not None:
        monitor.record(network, objective, maximum, len(points))
    damping, epoch, iteration = 1e-3, 0, 0
    validation_max = float("inf")
    status = "stalled"
    final_validation = checks
    while True:
        if maximum <= config.residual_tolerance:
            if monitor is not None:
                monitor.phase("validation")
            final_validation = _unique_rows(guards, checks, width=points.shape[1])
            validation_records = _evaluate_network(network, field, final_validation, monitor=monitor)
            _, validation_max, validation_norms = _metrics(validation_records)
            if monitor is not None:
                monitor.validation(validation_max, len(final_validation))
            if validation_max <= config.residual_tolerance:
                status = "numerically_converged"
                break
            if epoch >= config.max_validation_epochs:
                break
            violating = final_validation[validation_norms > config.residual_tolerance]
            guards = _unique_rows(guards, final_validation[validation_norms <= config.residual_tolerance], width=points.shape[1])
            points = _unique_rows(points, violating, width=points.shape[1])
            epoch += 1
            checks = np.asarray(config.points(validation=True, seed=2 + epoch), dtype=float)
            records = _evaluate_network(network, field, points, monitor=monitor)
            objective, maximum, norms = _metrics(records)
            history.append(objective)
            iteration = 0
            damping = max(damping, 1e-3)
            if monitor is not None:
                monitor.record(network, objective, maximum, len(points), new_points=True)
            continue
        if iteration >= config.max_iterations:
            break
        iteration += 1
        if monitor is not None:
            monitor.phase("weight_refinement")
        linearized = _evaluate_network(network, field, points, jacobian=True, monitor=monitor)
        _, _, old_norms = _metrics(linearized)
        weights = _hard_weights(old_norms, config.residual_tolerance)
        accepted = False
        local_damping = damping
        for _ in range(3):
            delta = _solve_direction(linearized, weights, local_damping, network)
            if not np.all(np.isfinite(delta)) or not np.any(delta):
                local_damping *= 10.0
                continue
            factor = 1.0
            for _ in range(5):
                trial = network.with_parameters(network.parameters + factor * delta)
                if config.gate_shrink:
                    trial = trial.soft_threshold_structure(config.gate_shrink * factor)
                try:
                    trial_records = _evaluate_network(trial, field, points, monitor=monitor)
                    trial_objective, trial_max, trial_norms = _metrics(trial_records)
                except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    trial_records, trial_objective, trial_max = None, float("inf"), float("inf")
                    trial_norms = np.full_like(old_norms, np.inf)
                if trial_records is not None and _accept(old_norms, trial_norms, config.residual_tolerance, weights):
                    network, records = trial, trial_records
                    objective, maximum, norms = trial_objective, trial_max, trial_norms
                    history.append(objective)
                    damping = max(local_damping / 3.0, 1e-12)
                    accepted = True
                    if monitor is not None:
                        monitor.record(network, objective, maximum, len(points))
                    if progress is not None:
                        progress(iteration, float(np.sqrt(objective)), maximum)
                    break
                factor *= 0.5
            if accepted:
                break
            local_damping *= 10.0
        if not accepted:
            break
    if status == "numerically_converged" and config.prune_rounds > 0:
        if monitor is not None:
            monitor.phase("structure_pruning")
        verify = _unique_rows(points, final_validation, width=points.shape[1])
        pruned = _prune(network, field, verify, config.residual_tolerance, config.prune_relative_budget, config.prune_rounds, monitor)
        if pruned is not network:
            network = pruned
            objective, maximum, _ = _metrics(_evaluate_network(network, field, points, monitor=monitor))
            _, validation_max, _ = _metrics(_evaluate_network(network, field, final_validation, monitor=monitor))
            if maximum > config.residual_tolerance or validation_max > config.residual_tolerance:
                raise RuntimeError("validated pruning violated the residual tolerance")
            if monitor is not None:
                monitor.record(network, objective, maximum, len(points))
                monitor.validation(validation_max, len(final_validation))
    structure = network.structure_summary(0.0)
    report = ResearchTrainingReport(status=status, active_response_channels=int(structure["active_response_channels"]), initial_rms_residual=initial_rms, final_rms_residual=float(np.sqrt(objective)), maximum_training_residual=float(maximum), maximum_validation_residual=float(validation_max), objective_history=tuple(float(v) for v in history), numerical_tolerance_met=(status == "numerically_converged"), structure=structure)
    return network, report


__all__ = ["ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network", "train_research_network"]
