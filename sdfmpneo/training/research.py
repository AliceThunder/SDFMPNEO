"""Solution-data-free finite-horizon residual training for the fixed analytic network."""
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
            "semigroup_validation_count", "max_iterations",
            "max_validation_epochs", "prune_rounds",
        ):
            value = int(getattr(self, key))
            if value != getattr(self, key) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            object.__setattr__(self, key, value)
        if min(self.sample_count, self.validation_count) < 1:
            raise ValueError("sample_count and validation_count must be positive")
        for key in (
            "max_network_depth", "max_channels_per_mode", "max_quadratic_rank",
            "max_cross_rank", "max_state_rank",
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
        """Physics-residual collocation points [a0, operating, segment_time]."""
        lo = np.asarray(self.initial_lower + self.operating_lower + (0.0,), dtype=float)
        hi = np.asarray(
            self.initial_upper + self.operating_upper + (self.max_response_time,),
            dtype=float,
        )
        count = self.validation_count if validation else self.sample_count
        engine = qmc.Halton(
            len(lo), scramble=True,
            seed=(1 if validation else 0) if seed is None else seed,
        )
        unit = engine.random(count)
        points = lo + unit * (hi - lo)
        if self.time_sampling == "mixed_log":
            ids = np.arange(0, count, 2)
            points[ids, -1] = np.exp(
                np.log(self.time_min)
                + unit[ids, -1]
                * (np.log(self.max_response_time) - np.log(self.time_min))
            )
        if not validation:
            center = 0.5 * (lo + hi)
            center[-1] = 0.0
            endpoint = center.copy()
            endpoint[-1] = self.max_response_time
            points = np.vstack([points, center, lo, hi, endpoint])
            points[-3, -1] = 0.0
        return points

    def semigroup_points(self, validation=False, seed=None):
        """Restart-consistency points [a0, operating, t1, t2] with t1+t2 <= H."""
        count = (
            self.semigroup_validation_count
            if validation else self.semigroup_sample_count
        )
        width = len(self.initial_lower) + len(self.operating_lower) + 2
        if count == 0:
            return np.empty((0, width), dtype=float)
        lo = np.asarray(self.initial_lower + self.operating_lower, dtype=float)
        hi = np.asarray(self.initial_upper + self.operating_upper, dtype=float)
        engine = qmc.Halton(
            len(lo) + 2, scramble=True,
            seed=(101 if validation else 100) if seed is None else seed,
        )
        unit = engine.random(count)
        static = lo + unit[:, : len(lo)] * (hi - lo)
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
    input_dimension = int(n_modes) + int(n_operating)
    depth = 5 if n_modes <= 8 else 4
    channels = 2
    quadratic = min(12, max(4, (input_dimension + 1) // 2))
    cross = min(6, max(2, int(np.ceil(np.sqrt(max(1, input_dimension))))))
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
    values = (
        config.max_network_depth, config.max_channels_per_mode,
        config.max_quadratic_rank, config.max_cross_rank, config.max_state_rank,
    )
    capacity = tuple(
        default if value is None else int(value)
        for default, value in zip(defaults, values)
    )
    return FixedAnalyticResponseNetwork(
        lambdas,
        operating_names,
        max_response_time=config.max_response_time,
        input_center=center,
        input_scale=scale,
        depth=capacity[0],
        channels_per_mode=capacity[1],
        quadratic_rank=capacity[2],
        cross_rank=capacity[3],
        state_rank=capacity[4],
    )


def _evaluate_network(network, field, points, *, jacobian=False, monitor=None):
    points = np.asarray(points, dtype=float)
    n = network.n_modes
    records = []
    prepare = getattr(field, "prepare_training_contexts", None)
    if prepare is not None and len(points):
        prepare(points)
    for point in points:
        if monitor is not None:
            monitor.checkpoint()
        initial, operating, time = point[:n], point[n:-1], float(point[-1])
        if jacobian:
            a, da, ja, jda = network.evaluate_parameter_jacobian(
                time, a0=initial, operating=operating
            )
            physical = field.evaluate(a, operating)
            records.append(
                SimpleNamespace(
                    residual=np.asarray(da - physical.vector_field, dtype=float),
                    parameter_jacobian=np.asarray(
                        jda - physical.vector_field_jacobian @ ja, dtype=float
                    ),
                )
            )
        else:
            a, da = network.evaluate(time, a0=initial, operating=operating)
            F = (
                field.vector_field(a, operating)
                if hasattr(field, "vector_field")
                else field.evaluate(a, operating).vector_field
            )
            records.append(
                SimpleNamespace(residual=np.asarray(da - F, dtype=float))
            )
    return records


def _evaluate_semigroup(network, rows, *, jacobian=False, monitor=None):
    rows = np.asarray(rows, dtype=float)
    n = network.n_modes
    horizon = float(network.max_response_time)
    records = []
    for row in rows:
        if monitor is not None:
            monitor.checkpoint()
        initial = row[:n]
        operating = row[n:-2]
        t1, t2 = float(row[-2]), float(row[-1])
        total = t1 + t2
        if total > horizon + 64.0 * np.finfo(float).eps * horizon:
            raise ValueError("semigroup sample exceeds max_response_time")
        total = min(total, horizon)
        if jacobian:
            direct, _, Jdirect, _ = network.evaluate_parameter_jacobian(
                total, a0=initial, operating=operating
            )
            first, _, Jfirst, _ = network.evaluate_parameter_jacobian(
                t1, a0=initial, operating=operating
            )
            restarted, _, Jrestart, _ = network.evaluate_parameter_jacobian(
                t2, a0=first, operating=operating
            )
            _, _, Jinitial, _ = network.evaluate_initial_jacobian(
                t2, a0=first, operating=operating
            )
            defect = np.asarray(direct - restarted, dtype=float)
            jac = np.asarray(
                Jdirect - (Jrestart + Jinitial @ Jfirst), dtype=float
            )
            records.append(
                SimpleNamespace(
                    residual=defect / horizon,
                    raw_defect=defect,
                    parameter_jacobian=jac / horizon,
                )
            )
        else:
            direct, _ = network.evaluate(total, a0=initial, operating=operating)
            first, _ = network.evaluate(t1, a0=initial, operating=operating)
            restarted, _ = network.evaluate(t2, a0=first, operating=operating)
            defect = np.asarray(direct - restarted, dtype=float)
            records.append(
                SimpleNamespace(residual=defect / horizon, raw_defect=defect)
            )
    return records


def _metrics(records):
    norms = np.asarray(
        [np.linalg.norm(record.residual) for record in records], dtype=float
    )
    objective = float(np.mean(norms * norms)) if norms.size else 0.0
    return objective, float(np.max(norms, initial=0.0)), norms


def _combined_metrics(physics_records, semigroup_records):
    records = list(physics_records) + list(semigroup_records)
    objective, maximum, norms = _metrics(records)
    _, physics_max, physics_norms = _metrics(physics_records)
    _, semigroup_max, semigroup_norms = _metrics(semigroup_records)
    return SimpleNamespace(
        records=records,
        objective=objective,
        maximum=maximum,
        norms=norms,
        physics_max=physics_max,
        physics_norms=physics_norms,
        semigroup_max=semigroup_max,
        semigroup_norms=semigroup_norms,
    )


def _evaluate_all(network, field, physics_points, semigroup_points, *, jacobian=False, monitor=None):
    physics = _evaluate_network(
        network, field, physics_points, jacobian=jacobian, monitor=monitor
    )
    semigroup = _evaluate_semigroup(
        network, semigroup_points, jacobian=jacobian, monitor=monitor
    )
    return _combined_metrics(physics, semigroup)


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
    numerical = 64.0 * np.finfo(float).eps * max(
        old_max, float(tolerance), 1e-30
    )
    if new_max < old_max - numerical:
        return True
    if new_max <= old_max + numerical:
        return (
            float(np.dot(weights, new * new))
            < float(np.dot(weights, old * old)) - numerical
        )
    return False


def _solve_direction(records, weights, damping, network):
    residual = np.vstack([record.residual for record in records])
    jacobian = np.stack([record.parameter_jacobian for record in records])
    root = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    rw = (root * residual).reshape(-1)
    Jw = (root[:, :, None] * jacobian).reshape(
        -1, network.parameter_count
    )
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
    return (
        np.vstack(rows)
        if rows
        else np.empty((0, width), dtype=float)
    )


def _prune(
    network, field, physics_points, semigroup_points,
    tolerance, relative_budget, rounds, monitor,
):
    current = network
    for _ in range(rounds):
        entries = [
            entry for entry in current.structure_gate_entries()
            if entry["value"] != 0.0
        ]
        if not entries:
            break
        evaluated = _evaluate_all(
            current, field, physics_points, semigroup_points,
            jacobian=True, monitor=monitor,
        )
        jac = np.stack([record.parameter_jacobian for record in evaluated.records])
        scored = []
        for entry in entries:
            column = jac[:, :, entry["parameter_index"]]
            effect = abs(entry["value"]) * float(
                np.max(np.linalg.norm(column, axis=1), initial=0.0)
            )
            scored.append((effect, entry["parameter_index"]))
        scored.sort()
        budget = max(0.0, tolerance - evaluated.maximum) + relative_budget * tolerance
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
            result = _evaluate_all(
                trial, field, physics_points, semigroup_points, monitor=monitor
            )
            if result.maximum <= tolerance:
                accepted = trial
                break
            count //= 2
        if accepted is None:
            break
        current = accepted
    return current


def train_research_network(
    field,
    config: ResearchTrainingConfig,
    *,
    network=None,
    operating_names=None,
    progress=None,
    monitor=None,
):
    """Fit one finite-horizon analytic flow; no structural candidate search exists."""
    if network is None:
        network = make_fixed_network(field, config, operating_names=operating_names)
    if not isinstance(network, FixedAnalyticResponseNetwork):
        raise TypeError("only FixedAnalyticResponseNetwork checkpoints are supported")
    if (
        len(config.initial_lower) != network.n_modes
        or len(config.operating_lower) != len(network.operating_names)
    ):
        raise ValueError("training domain does not match the fixed analytic network")
    if network.max_response_time != config.max_response_time:
        raise ValueError("training max_response_time does not match the saved network")

    points = np.asarray(config.points(validation=False), dtype=float)
    checks = np.asarray(config.points(validation=True), dtype=float)
    semigroup_points = np.asarray(config.semigroup_points(validation=False), dtype=float)
    semigroup_checks = np.asarray(config.semigroup_points(validation=True), dtype=float)
    guards = np.empty((0, points.shape[1]), dtype=float)
    sg_width = len(config.initial_lower) + len(config.operating_lower) + 2
    sg_guards = np.empty((0, sg_width), dtype=float)

    if monitor is not None:
        monitor.retain(network)
        monitor.phase("initial_residual")
    physics_records = _evaluate_network(network, field, points, monitor=monitor)
    evaluated = _combined_metrics(physics_records, [])
    initial_rms = float(np.sqrt(evaluated.objective))
    history = [evaluated.objective]
    if monitor is not None:
        monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

    damping, epoch, iteration = 1e-3, 0, 0
    validation = None
    status = "stalled"
    semigroup_active = False
    final_validation = checks
    final_semigroup_validation = semigroup_checks

    while True:
        if not semigroup_active and evaluated.physics_max <= config.residual_tolerance:
            semigroup_active = True
            if monitor is not None:
                monitor.phase("restart_consistency")
            evaluated = _evaluate_all(
                network, field, points, semigroup_points, monitor=monitor
            )
            history.append(evaluated.objective)
            iteration = 0
            damping = max(damping, 1e-3)
            if monitor is not None:
                monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

        if semigroup_active and evaluated.maximum <= config.residual_tolerance:
            if monitor is not None:
                monitor.phase("validation")
            final_validation = _unique_rows(
                guards, checks, width=points.shape[1]
            )
            final_semigroup_validation = _unique_rows(
                sg_guards, semigroup_checks, width=sg_width
            )
            validation = _evaluate_all(
                network,
                field,
                final_validation,
                final_semigroup_validation,
                monitor=monitor,
            )
            if monitor is not None:
                monitor.validation(validation.maximum, len(final_validation))
            if validation.maximum <= config.residual_tolerance:
                status = "numerically_converged"
                break
            if epoch >= config.max_validation_epochs:
                break

            physics_records = _evaluate_network(
                network, field, final_validation, monitor=monitor
            )
            _, _, physics_norms = _metrics(physics_records)
            semigroup_records = _evaluate_semigroup(
                network, final_semigroup_validation, monitor=monitor
            )
            _, _, sg_norms = _metrics(semigroup_records)
            violating = final_validation[
                physics_norms > config.residual_tolerance
            ]
            violating_sg = final_semigroup_validation[
                sg_norms > config.residual_tolerance
            ]
            guards = _unique_rows(
                guards,
                final_validation[physics_norms <= config.residual_tolerance],
                width=points.shape[1],
            )
            sg_guards = _unique_rows(
                sg_guards,
                final_semigroup_validation[sg_norms <= config.residual_tolerance],
                width=sg_width,
            )
            points = _unique_rows(points, violating, width=points.shape[1])
            semigroup_points = _unique_rows(
                semigroup_points, violating_sg, width=sg_width
            )
            epoch += 1
            checks = np.asarray(
                config.points(validation=True, seed=2 + epoch), dtype=float
            )
            semigroup_checks = np.asarray(
                config.semigroup_points(validation=True, seed=102 + epoch),
                dtype=float,
            )
            evaluated = _evaluate_all(
                network, field, points, semigroup_points, monitor=monitor
            )
            history.append(evaluated.objective)
            iteration = 0
            damping = max(damping, 1e-3)
            if monitor is not None:
                monitor.record(
                    network, evaluated.objective, evaluated.maximum,
                    len(points), new_points=True,
                )
            continue

        if iteration >= config.max_iterations:
            break
        iteration += 1
        if monitor is not None:
            monitor.phase(
                "restart_consistency" if semigroup_active else "weight_refinement"
            )
        if semigroup_active:
            linearized = _evaluate_all(
                network, field, points, semigroup_points,
                jacobian=True, monitor=monitor,
            )
        else:
            physics_linearized = _evaluate_network(
                network, field, points, jacobian=True, monitor=monitor
            )
            linearized = _combined_metrics(physics_linearized, [])
        weights = _hard_weights(
            linearized.norms, config.residual_tolerance
        )
        accepted = False
        local_damping = damping
        for _ in range(3):
            delta = _solve_direction(
                linearized.records, weights, local_damping, network
            )
            if not np.all(np.isfinite(delta)) or not np.any(delta):
                local_damping *= 10.0
                continue
            factor = 1.0
            for _ in range(5):
                trial = network.with_parameters(
                    network.parameters + factor * delta
                )
                if config.gate_shrink:
                    trial = trial.soft_threshold_structure(
                        config.gate_shrink * factor
                    )
                try:
                    if semigroup_active:
                        trial_result = _evaluate_all(
                            trial, field, points, semigroup_points, monitor=monitor
                        )
                    else:
                        trial_physics = _evaluate_network(
                            trial, field, points, monitor=monitor
                        )
                        trial_result = _combined_metrics(trial_physics, [])
                except (
                    ValueError, FloatingPointError, OverflowError,
                    np.linalg.LinAlgError,
                ):
                    trial_result = None
                if (
                    trial_result is not None
                    and _accept(
                        linearized.norms,
                        trial_result.norms,
                        config.residual_tolerance,
                        weights,
                    )
                ):
                    network = trial
                    evaluated = trial_result
                    history.append(evaluated.objective)
                    damping = max(local_damping / 3.0, 1e-12)
                    accepted = True
                    if monitor is not None:
                        monitor.record(
                            network, evaluated.objective,
                            evaluated.maximum, len(points),
                        )
                    if progress is not None:
                        progress(
                            iteration,
                            float(np.sqrt(evaluated.objective)),
                            evaluated.maximum,
                        )
                    break
                factor *= 0.5
            if accepted:
                break
            local_damping *= 10.0
        if not accepted:
            break

    # Always finish with both constraints measured, even if physics never converged.
    evaluated = _evaluate_all(
        network, field, points, semigroup_points, monitor=monitor
    )
    if validation is None:
        validation = _evaluate_all(
            network, field, final_validation,
            final_semigroup_validation, monitor=monitor,
        )

    if status == "numerically_converged" and config.prune_rounds > 0:
        if monitor is not None:
            monitor.phase("structure_pruning")
        verify = _unique_rows(
            points, final_validation, width=points.shape[1]
        )
        verify_sg = _unique_rows(
            semigroup_points,
            final_semigroup_validation,
            width=sg_width,
        )
        pruned = _prune(
            network, field, verify, verify_sg,
            config.residual_tolerance,
            config.prune_relative_budget,
            config.prune_rounds,
            monitor,
        )
        if pruned is not network:
            network = pruned
            evaluated = _evaluate_all(
                network, field, points, semigroup_points, monitor=monitor
            )
            validation = _evaluate_all(
                network, field, final_validation,
                final_semigroup_validation, monitor=monitor,
            )
            if (
                evaluated.maximum > config.residual_tolerance
                or validation.maximum > config.residual_tolerance
            ):
                raise RuntimeError(
                    "validated pruning violated residual/restart tolerance"
                )
            if monitor is not None:
                monitor.record(
                    network, evaluated.objective,
                    evaluated.maximum, len(points),
                )
                monitor.validation(
                    validation.maximum, len(final_validation)
                )

    structure = network.structure_summary(0.0)
    horizon = float(config.max_response_time)
    report = ResearchTrainingReport(
        status=status,
        active_response_channels=int(structure["active_response_channels"]),
        max_response_time=horizon,
        initial_rms_residual=initial_rms,
        final_rms_residual=float(np.sqrt(evaluated.objective)),
        maximum_training_residual=float(evaluated.maximum),
        maximum_validation_residual=float(validation.maximum),
        maximum_training_physics_residual=float(evaluated.physics_max),
        maximum_validation_physics_residual=float(validation.physics_max),
        maximum_training_semigroup_rate_defect=float(evaluated.semigroup_max),
        maximum_validation_semigroup_rate_defect=float(validation.semigroup_max),
        maximum_training_semigroup_defect=float(evaluated.semigroup_max * horizon),
        maximum_validation_semigroup_defect=float(validation.semigroup_max * horizon),
        objective_history=tuple(float(v) for v in history),
        numerical_tolerance_met=(status == "numerically_converged"),
        structure=structure,
    )
    return network, report


__all__ = [
    "ResearchTrainingConfig",
    "ResearchTrainingReport",
    "make_fixed_network",
    "train_research_network",
]
