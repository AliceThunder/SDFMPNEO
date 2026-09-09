from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResearchTrainingContinuation:
    """Persistable collocation state for resumable residual training."""

    points: np.ndarray
    checks: np.ndarray
    guards: np.ndarray
    collocation_epoch: int
    config_signature: str

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float)
        checks = np.asarray(self.checks, dtype=float)
        guards = np.asarray(self.guards, dtype=float)
        if points.ndim != 2 or checks.ndim != 2 or guards.ndim != 2:
            raise ValueError("continuation point arrays must be matrices")
        if points.shape[1] != checks.shape[1] or guards.shape[1] != points.shape[1]:
            raise ValueError("continuation point dimensions must match")
        if int(self.collocation_epoch) != self.collocation_epoch or self.collocation_epoch < 0:
            raise ValueError("collocation_epoch must be a non-negative integer")
        if not str(self.config_signature):
            raise ValueError("config_signature must be non-empty")
        object.__setattr__(self, "points", points.copy())
        object.__setattr__(self, "checks", checks.copy())
        object.__setattr__(self, "guards", guards.copy())
        object.__setattr__(self, "collocation_epoch", int(self.collocation_epoch))


def _config_signature(config) -> str:
    """Hash only fields that define the collocation domain/sampling semantics."""

    payload = {
        "initial_lower": list(config.initial_lower),
        "initial_upper": list(config.initial_upper),
        "operating_lower": list(config.operating_lower),
        "operating_upper": list(config.operating_upper),
        "time_horizon": float(config.time_horizon),
        "sample_count": int(config.sample_count),
        "validation_count": int(config.validation_count),
        "time_sampling": str(config.time_sampling),
        "time_min": float(config.time_min),
        "include_steady_state": bool(config.include_steady_state),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_key(row: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(row, dtype=float))


def _unique_rows(*arrays: np.ndarray, width: int | None = None) -> np.ndarray:
    """Stable exact de-duplication; Halton points are deterministic floats."""

    seen: set[tuple[float, ...]] = set()
    rows: list[np.ndarray] = []
    inferred = width
    for array in arrays:
        values = np.asarray(array, dtype=float)
        if values.ndim != 2:
            raise ValueError("point arrays must be matrices")
        if inferred is None:
            inferred = values.shape[1]
        if values.shape[1] != inferred:
            raise ValueError("point dimensions do not match")
        for row in values:
            key = _row_key(row)
            if key not in seen:
                seen.add(key)
                rows.append(row.copy())
    if inferred is None:
        raise ValueError("width is required for empty input")
    return np.vstack(rows) if rows else np.empty((0, inferred), dtype=float)


def _fresh_checks(config, *, seed: int, excluded: np.ndarray) -> np.ndarray:
    candidate = np.asarray(config.points(validation=True, seed=seed), dtype=float)
    excluded_keys = {_row_key(row) for row in np.asarray(excluded, dtype=float)}
    rows = [row for row in candidate if _row_key(row) not in excluded_keys]
    return np.vstack(rows) if rows else np.empty((0, candidate.shape[1]), dtype=float)


def _select_adaptive_validation_points(records, points, *, tolerance: float):
    """Move only failed checks into training and retain passed checks as guards.

    Every point whose residual norm exceeds the actual convergence tolerance is
    promoted. Passing points are not discarded: they remain in a guard set and
    are re-evaluated at every later validation, so selective refinement cannot
    silently regress previously checked regions.
    """

    values = np.asarray(points, dtype=float)
    norms = np.array([np.linalg.norm(record.residual) for record in records], dtype=float)
    if values.shape[0] != norms.size:
        raise ValueError("validation records and points do not match")
    bad = norms > float(tolerance)
    if not np.any(bad) and norms.size and float(np.max(norms)) > float(tolerance):
        bad[int(np.argmax(norms))] = True
    return values[bad].copy(), values[~bad].copy(), norms


def _initial_continuation(field, config) -> ResearchTrainingContinuation:
    signature = _config_signature(config)
    saved = getattr(field, "training_continuation", None)
    width = len(config.initial_lower) + len(config.operating_lower) + 1
    if isinstance(saved, ResearchTrainingContinuation) and saved.config_signature == signature:
        if saved.points.shape[1] == width:
            return ResearchTrainingContinuation(
                saved.points, saved.checks, saved.guards,
                saved.collocation_epoch, saved.config_signature,
            )
    points = np.asarray(config.points(), dtype=float)
    checks = np.asarray(config.points(validation=True), dtype=float)
    guards = np.empty((0, points.shape[1]), dtype=float)
    return ResearchTrainingContinuation(points, checks, guards, 0, signature)


def _store_continuation(field, state: ResearchTrainingContinuation) -> None:
    try:
        setattr(field, "training_continuation", state)
    except (AttributeError, TypeError):
        pass


def adaptive_train_research_graph(field, config, *, graph=None, progress=None, monitor=None):
    """Original residual-growth algorithm with selective, resumable collocation."""

    from . import research as r

    if graph is None:
        graph = r.ParametricAnalyticEvolutionGraph(
            field.thermal_model.lambdas, [f"u{i}" for i in range(field.n_operating)]
        )
    if len(config.initial_lower) != graph.n_modes or len(config.operating_lower) != field.n_operating:
        raise ValueError("training domain does not match the physical model")
    if len(graph.response_nodes) > config.max_nodes:
        raise ValueError("existing graph exceeds the requested node budget")

    state = _initial_continuation(field, config)
    _store_continuation(field, state)
    points, checks, guards = state.points, state.checks, state.guards
    collocation_epoch = state.collocation_epoch

    if monitor is not None:
        monitor.retain(graph)
        monitor.phase("initial_residual")
    records = r._evaluate(graph, field, points, jacobian=True, monitor=monitor)
    objective, maximum = r._metrics(records)
    initial_rms = np.sqrt(objective)
    history = [objective]
    accepted = len(graph.response_nodes)
    if monitor is not None:
        monitor.record(graph, objective, maximum, len(points))
        monitor.phase("quadratic_seed")

    if graph.response_nodes and maximum > config.residual_tolerance:
        graph = r._refine_weights(
            graph, field, points, monitor=monitor, tolerance=config.residual_tolerance
        )
        records = r._evaluate(graph, field, points, jacobian=True, monitor=monitor)
        objective, maximum = r._metrics(records)
        history.append(objective)
        if monitor is not None:
            monitor.record(graph, objective, maximum, len(points))

    seed = r._quadratic_heating_seed(graph, field, monitor=monitor) if config.max_degree >= 2 else graph
    if len(seed.response_nodes) <= config.max_nodes and seed is not graph:
        try:
            seed_objective, seed_max = r._metrics(r._evaluate(seed, field, points, monitor=monitor))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            seed_objective = float("inf")
        if seed_objective < objective:
            if monitor is not None:
                monitor.record(seed, seed_objective, seed_max, len(points))
            graph = r._refine_weights(
                seed, field, points, monitor=monitor, tolerance=config.residual_tolerance
            )
            records = r._evaluate(graph, field, points, jacobian=True, monitor=monitor)
            objective, maximum = r._metrics(records)
            accepted = len(graph.response_nodes)
            history.append(objective)
            if monitor is not None:
                monitor.record(graph, objective, maximum, len(points))
            if progress is not None:
                progress(accepted, np.sqrt(objective), maximum)

    check_max = float("inf")
    while True:
        if monitor is not None:
            monitor.checkpoint()
        if maximum <= config.residual_tolerance or accepted >= config.max_nodes:
            if monitor is not None:
                monitor.phase("validation")

            validation_points = _unique_rows(guards, checks, width=points.shape[1])
            validation_records = r._evaluate(
                graph, field, validation_points, monitor=monitor
            )
            _, check_max = r._metrics(validation_records)
            if monitor is not None:
                monitor.validation(check_max, len(validation_points))

            if check_max <= config.residual_tolerance and maximum <= config.residual_tolerance:
                status = "numerically_converged"
                break

            violating, passed, _ = _select_adaptive_validation_points(
                validation_records, validation_points, tolerance=config.residual_tolerance
            )
            next_points = _unique_rows(points, violating, width=points.shape[1])
            next_guards = _unique_rows(passed, width=points.shape[1])
            next_epoch = collocation_epoch + 1
            excluded = _unique_rows(next_points, next_guards, width=points.shape[1])
            next_checks = _fresh_checks(
                config, seed=2 + accepted + next_epoch, excluded=excluded
            )
            next_state = ResearchTrainingContinuation(
                next_points, next_checks, next_guards, next_epoch, state.config_signature
            )
            # Store failed validation points even if this run has exhausted its
            # node budget, so a later higher-budget resume starts on the known
            # hard cases instead of forgetting them.
            _store_continuation(field, next_state)

            if accepted >= config.max_nodes:
                status = "budget_exhausted"
                break

            points, checks, guards = next_points, next_checks, next_guards
            collocation_epoch = next_epoch
            state = next_state
            records = r._evaluate(graph, field, points, jacobian=True, monitor=monitor)
            objective, maximum = r._metrics(records)
            history = [objective]
            if monitor is not None:
                monitor.record(graph, objective, maximum, len(points), new_points=True)

        if monitor is not None:
            monitor.phase("candidate_search")
        compiled = [
            r.compile_parametric_realization(graph, a0=record.initial, operating=record.u)
            for record in records
        ]
        names = graph.known_names()
        existing = {(node.target_mode, tuple(sorted(node.parents))) for node in graph.response_nodes}
        scored = []
        response_names = {node.name for node in graph.response_nodes}
        for degree in range(config.max_degree + 1):
            for parents in combinations_with_replacement(names, degree):
                if sum(parent in response_names for parent in parents) > config.max_parent_responses:
                    continue
                dimension = int(
                    np.prod([compiled[0].node_realizations[parent].dimension for parent in parents])
                ) + 1
                if dimension > config.max_realization_dimension:
                    continue
                parent_key = tuple(sorted(parents))
                active_targets = [
                    target for target in range(graph.n_modes)
                    if (target, parent_key) not in existing
                ]
                if not active_targets:
                    continue
                inner, norm2 = r._candidate_tangent_scores(
                    graph, records, compiled, parents, monitor=monitor
                )
                for target in active_targets:
                    if norm2[target] > 0 and np.isfinite(norm2[target] + inner[target]):
                        scored.append(
                            (
                                inner[target] * inner[target] / norm2[target],
                                -inner[target] / norm2[target],
                                target,
                                parents,
                            )
                        )
        scored.sort(key=lambda item: item[0], reverse=True)
        success = False
        for score, weight, target, parents in scored:
            if score <= 0:
                continue
            for _ in range(24):
                trial = r._clone(graph)
                trial.add_product_response(
                    f"response_{len(graph.response_nodes)}", target, parents, weight
                )
                try:
                    trial_records = r._evaluate(trial, field, points, monitor=monitor)
                    trial_objective, trial_max = r._metrics(trial_records)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    trial_objective = float("inf")
                if np.isfinite(trial_objective) and trial_objective < objective:
                    if monitor is not None:
                        monitor.record(trial, trial_objective, trial_max, len(points))
                    graph = r._refine_weights(
                        trial, field, points, monitor=monitor,
                        tolerance=config.residual_tolerance,
                    )
                    objective, maximum = r._metrics(
                        r._evaluate(graph, field, points, monitor=monitor)
                    )
                    history.append(objective)
                    accepted += 1
                    if monitor is not None:
                        monitor.record(graph, objective, maximum, len(points))
                    success = True
                    if progress is not None:
                        progress(accepted, np.sqrt(objective), maximum)
                    break
                weight *= 0.5
            if success:
                break
        if not success:
            status = "stalled"
            if monitor is not None:
                monitor.phase("validation")
            validation_points = _unique_rows(guards, checks, width=points.shape[1])
            _, check_max = r._metrics(r._evaluate(graph, field, validation_points, monitor=monitor))
            if monitor is not None:
                monitor.validation(check_max, len(validation_points))
            break
        records = r._evaluate(graph, field, points, jacobian=True, monitor=monitor)

    report = r.ResearchTrainingReport(
        status,
        accepted,
        float(initial_rms),
        float(np.sqrt(objective)),
        maximum,
        check_max,
        tuple(history),
        status == "numerically_converged",
    )
    return graph, report


def install_adaptive_training() -> None:
    """Install selective collocation before geometry_research imports the trainer."""

    from . import research as training_research

    training_research.train_research_graph = adaptive_train_research_graph
    # ResearchElectroThermalModel imported training_research_graph by value, so
    # update that module binding as well. GeometryResearchModel is imported later
    # by package __init__ and will naturally receive the patched function.
    try:
        from sdfmpneo import research as research_module
        research_module.train_research_graph = adaptive_train_research_graph
    except ImportError:
        pass


def install_geometry_continuation_persistence(model_class) -> None:
    """Persist adaptive collocation state in geometry-family NPZ checkpoints."""

    if getattr(model_class, "_smart_collocation_persistence", False):
        return
    original_save = model_class.save
    original_load = model_class.load.__func__

    def save(self, path):
        result = Path(original_save(self, path))
        state = getattr(self, "training_continuation", None)
        if not isinstance(state, ResearchTrainingContinuation):
            return result
        with np.load(result, allow_pickle=False) as data:
            payload = {name: np.array(data[name], copy=True) for name in data.files}
        metadata = json.loads(str(payload["metadata"]))
        metadata["training_continuation"] = {
            "format_version": 1,
            "collocation_epoch": state.collocation_epoch,
            "config_signature": state.config_signature,
        }
        payload["metadata"] = np.array(json.dumps(metadata))
        payload["continuation_points"] = state.points
        payload["continuation_checks"] = state.checks
        payload["continuation_guards"] = state.guards
        with result.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        return result

    @classmethod
    def load(cls, path):
        out = original_load(cls, path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            saved = metadata.get("training_continuation")
            if saved is not None and saved.get("format_version") == 1:
                out.training_continuation = ResearchTrainingContinuation(
                    data["continuation_points"],
                    data["continuation_checks"],
                    data["continuation_guards"],
                    int(saved["collocation_epoch"]),
                    str(saved["config_signature"]),
                )
        return out

    model_class.save = save
    model_class.load = load
    model_class._smart_collocation_persistence = True
