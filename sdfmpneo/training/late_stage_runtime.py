from __future__ import annotations

from dataclasses import dataclass
import os
from threading import RLock

import numpy as np

from sdfmpneo.analytic.realization import AnalyticRealization
from sdfmpneo.analytic.long_time import realization_action
from sdfmpneo.analytic.parametric_realization import compile_parametric_realization
from .adaptive_runtime import (
    ResearchTrainingContinuation,
    _fresh_checks,
    _initial_continuation,
    _select_adaptive_validation_points,
    _store_continuation,
    _unique_rows,
)
from .parallel_runtime import _ordered_map, training_point_workers


def _iter_admissible_parent_tuples(
    names: tuple[str, ...],
    response_names: set[str],
    degree: int,
    max_parent_responses: int,
):
    """Yield exactly the old filtered combinations without constructing invalid tuples."""
    names = tuple(names)
    response_flags = tuple(name in response_names for name in names)
    n_names = len(names)

    def visit(start: int, remaining: int, used_responses: int, prefix: list[str]):
        if remaining == 0:
            yield tuple(prefix)
            return
        for index in range(start, n_names):
            count = used_responses + int(response_flags[index])
            if count > max_parent_responses:
                continue
            prefix.append(names[index])
            yield from visit(index, remaining - 1, count, prefix)
            prefix.pop()

    yield from visit(0, int(degree), 0, [])


@dataclass(frozen=True)
class _ParentPlan:
    parents: tuple[str, ...]
    initial_indices: tuple[int, ...]
    operating_indices: tuple[int, ...]
    response_parent: str | None
    decay_shift: float


def _parent_plan(graph, parents: tuple[str, ...]) -> _ParentPlan:
    initial_lookup = {name: index for index, name in enumerate(graph.initial_names)}
    operating_lookup = {name: index for index, name in enumerate(graph.operating_names)}
    response_names = {node.name for node in graph.response_nodes}
    initial = []
    operating = []
    response = []
    decay = 0.0
    for parent in parents:
        if parent in initial_lookup:
            index = initial_lookup[parent]
            initial.append(index)
            decay += float(graph.lambdas[index])
        elif parent in operating_lookup:
            operating.append(operating_lookup[parent])
        elif parent in response_names:
            response.append(parent)
        else:
            raise KeyError(parent)
    if len(response) > 1:
        # Historical checkpoints created with a larger response-parent budget
        # keep the generic exact path rather than losing any capability.
        return _ParentPlan(tuple(parents), tuple(initial), tuple(operating), None, float("nan"))
    return _ParentPlan(
        tuple(parents), tuple(initial), tuple(operating),
        None if not response else response[0], decay,
    )


def _source_response_values_structured(graph, record, compiled, plan: _ParentPlan):
    """Exact source/response without Kronecker products for <=1 dynamic parent."""
    if np.isnan(plan.decay_shift):
        from . import research as r
        source = AnalyticRealization.constant(1.0)
        for parent in plan.parents:
            source = source.product(compiled.node_realizations[parent])
        return r._source_response_values(source, graph.lambdas, record.time)

    scale = 1.0
    for index in plan.initial_indices:
        scale *= float(record.initial[index])
    for index in plan.operating_indices:
        scale *= float(record.u[index])

    if plan.response_parent is None:
        source = AnalyticRealization.decay(plan.decay_shift, scale)
    else:
        base = compiled.node_realizations[plan.response_parent]
        shifted = base.A - plan.decay_shift * np.eye(base.dimension, dtype=complex)
        source = AnalyticRealization(shifted, scale * base.b, base.c)

    from . import research as r
    return r._source_response_values(source, graph.lambdas, record.time)


def _candidate_score_one(graph, records, compiled, plan: _ParentPlan):
    n_modes = graph.n_modes
    inner = np.zeros(n_modes, dtype=float)
    norm2 = np.zeros(n_modes, dtype=float)
    identity = np.eye(n_modes, dtype=float)
    linear = np.diag(np.asarray(graph.lambdas, dtype=float))
    for record, realization in zip(records, compiled):
        psi, h = _source_response_values_structured(graph, record, realization, plan)
        tangent = identity * psi - (record.J + linear) * h[np.newaxis, :]
        inner += record.residual @ tangent
        norm2 += np.sum(tangent * tangent, axis=0)
    return inner, norm2


def _score_parent_batch(graph, records, compiled, plans, *, monitor=None):
    """Parallelize over candidates; each task owns its whole collocation sweep."""
    plans = list(plans)
    if not plans:
        return []
    if training_point_workers() <= 1 or len(plans) < 2:
        out = []
        for plan in plans:
            if monitor is not None:
                monitor.checkpoint()
            out.append(_candidate_score_one(graph, records, compiled, plan))
        return out
    return _ordered_map(
        lambda plan: _candidate_score_one(graph, records, compiled, plan),
        plans,
        monitor=monitor,
    )


def _sparse_weight_value_jacobian(graph, point):
    """Exact value/weight Jacobian with derivative vectors only on real ancestry."""
    n_modes = graph.n_modes
    n_weights = len(graph.response_nodes)
    initial = np.asarray(point[:n_modes], dtype=float)
    operating = np.asarray(point[n_modes:-1], dtype=float)
    time = float(point[-1])
    compiled = compile_parametric_realization(graph, a0=initial, operating=operating)

    sparse_db: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in graph.initial_names + graph.operating_names
    }
    for index, node in enumerate(graph.response_nodes):
        source_b = np.array([complex(node.weight)], dtype=complex)
        source_d = {index: np.ones(1, dtype=complex)}
        for parent in node.parents:
            right_b = compiled.node_realizations[parent].b
            right_d = sparse_db[parent]
            keys = source_d.keys() | right_d.keys()
            next_d: dict[int, np.ndarray] = {}
            for key in keys:
                left = source_d.get(key)
                right = right_d.get(key)
                value = np.zeros(source_b.size * right_b.size, dtype=complex)
                if left is not None:
                    value += np.kron(left, right_b)
                if right is not None:
                    value += np.kron(source_b, right)
                next_d[key] = value
            source_b = np.kron(source_b, right_b)
            source_d = next_d
        sparse_db[node.name] = {
            key: np.concatenate([value, np.zeros(1, dtype=complex)])
            for key, value in source_d.items()
        }

    a = np.zeros(n_modes, dtype=float)
    da = np.zeros(n_modes, dtype=float)
    ja = np.zeros((n_modes, n_weights), dtype=float)
    jda = np.zeros_like(ja)

    for mode, name in enumerate(graph.initial_names):
        realization = compiled.node_realizations[name]
        state = realization_action(realization.A, realization.b, time)
        a[mode] += float(np.real(realization.c @ state))
        if not np.isposinf(time):
            da[mode] += float(np.real(realization.c @ (realization.A @ state)))

    for node in graph.response_nodes:
        realization = compiled.node_realizations[node.name]
        derivatives = sparse_db[node.name]
        indices = tuple(sorted(derivatives))
        B = np.column_stack(
            [realization.b] + [derivatives[index] for index in indices]
        )
        state = realization_action(realization.A, B, time)
        values = np.real(realization.c @ state)
        target = node.target_mode
        a[target] += float(values[0])
        if indices:
            ja[target, list(indices)] += values[1:]
        if not np.isposinf(time):
            slopes = np.real(realization.c @ (realization.A @ state))
            da[target] += float(slopes[0])
            if indices:
                jda[target, list(indices)] += slopes[1:]

    return a, da, ja, jda


def _sparse_refine_weights(graph, field, points, max_iterations=12, monitor=None, tolerance=0.0):
    """Joint Gauss--Newton using exact sparse-ancestry weight sensitivities."""
    from . import research as r
    if not graph.response_nodes:
        return graph

    n_modes = graph.n_modes
    for _ in range(max_iterations):
        if monitor is not None:
            monitor.phase("weight_refinement")

        def one(point):
            a, da, ja, jda = _sparse_weight_value_jacobian(graph, point)
            physical = field.evaluate(a, point[n_modes:-1])
            return da - physical.vector_field, jda - physical.vector_field_jacobian @ ja

        pairs = _ordered_map(one, points, monitor=monitor)
        residuals = [item[0] for item in pairs]
        jacobians = [item[1] for item in pairs]
        if max(np.linalg.norm(value) for value in residuals) <= tolerance:
            break

        residual = np.concatenate(residuals)
        J = np.vstack(jacobians)
        scales = np.linalg.norm(J, axis=0)
        scales[scales == 0] = 1.0
        delta = np.linalg.lstsq(J / scales, -residual, rcond=None)[0] / scales
        objective = float(residual @ residual)
        accepted = False

        for _ in range(20):
            trial = r.ParametricAnalyticEvolutionGraph(graph.lambdas, graph.operating_names)
            for node, change in zip(graph.response_nodes, delta):
                trial.add_product_response(
                    node.name, node.target_mode, node.parents, node.weight + change
                )
            try:
                values = r._evaluate(trial, field, points, monitor=monitor)
                trial_objective = sum(
                    float(value.residual @ value.residual) for value in values
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                trial_objective = float("inf")
            if trial_objective < objective:
                graph = trial
                if monitor is not None:
                    obj, maximum = r._metrics(values)
                    monitor.record(graph, obj, maximum, len(points))
                accepted = True
                break
            delta *= 0.5

        if (
            not accepted
            or objective - trial_objective
            <= 1e-8 * max(objective, np.finfo(float).tiny)
        ):
            break
    return graph


def _prepare_working_set(field, *point_sets):
    prepare = getattr(field, "prepare_training_contexts", None)
    if prepare is not None:
        prepare(*point_sets)


def optimized_adaptive_train_research_graph(field, config, *, graph=None, progress=None, monitor=None):
    """Selective resumable training with exact-equivalent late-stage scaling."""
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
    _prepare_working_set(field, points, checks, guards)

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
            _prepare_working_set(field, points, validation_points)
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
            _store_continuation(field, next_state)

            if accepted >= config.max_nodes:
                status = "budget_exhausted"
                break

            points, checks, guards = next_points, next_checks, next_guards
            collocation_epoch = next_epoch
            state = next_state
            _prepare_working_set(field, points, checks, guards)
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
        response_names = {node.name for node in graph.response_nodes}
        existing = {
            (node.target_mode, tuple(sorted(node.parents)))
            for node in graph.response_nodes
        }

        specs = []
        for degree in range(config.max_degree + 1):
            for parents in _iter_admissible_parent_tuples(
                names, response_names, degree, config.max_parent_responses
            ):
                dimension = int(
                    np.prod([
                        compiled[0].node_realizations[parent].dimension
                        for parent in parents
                    ])
                ) + 1
                if dimension > config.max_realization_dimension:
                    continue
                parent_key = tuple(sorted(parents))
                active_targets = tuple(
                    target for target in range(graph.n_modes)
                    if (target, parent_key) not in existing
                )
                if active_targets:
                    specs.append((parents, active_targets, _parent_plan(graph, parents)))

        scores = _score_parent_batch(
            graph, records, compiled, [spec[2] for spec in specs], monitor=monitor
        )
        scored = []
        for (parents, active_targets, _), (inner, norm2) in zip(specs, scores):
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
            _prepare_working_set(field, points, validation_points)
            _, check_max = r._metrics(
                r._evaluate(graph, field, validation_points, monitor=monitor)
            )
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


def install_late_stage_training() -> None:
    """Install exact-equivalent late-stage scaling improvements."""
    from . import research as training_research

    training_research._refine_weights = _sparse_refine_weights
    training_research.train_research_graph = optimized_adaptive_train_research_graph
    try:
        from sdfmpneo import research as research_module
        research_module.train_research_graph = optimized_adaptive_train_research_graph
    except ImportError:
        pass


def install_geometry_working_set_cache(model_class) -> None:
    """Add thread-safe, working-set-sized geometry context caching."""
    if getattr(model_class, "_working_set_cache_installed", False):
        return

    original_init = model_class.__init__
    original_context = model_class.context

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._context_cache_lock = RLock()

    def context(self, geometry):
        lock = getattr(self, "_context_cache_lock", None)
        if lock is None:
            self._context_cache_lock = RLock()
            lock = self._context_cache_lock
        with lock:
            return original_context(self, geometry)

    def prepare_training_contexts(self, *point_sets):
        arrays = []
        for values in point_sets:
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.size:
                arrays.append(array)
        if not arrays:
            return 0

        n_initial = int(self.thermal_model.rank)
        n_geometry = len(self.geometry_names)
        normalized = []
        seen = set()
        for values in arrays:
            if values.ndim != 2:
                raise ValueError("training point sets must be matrices")
            for row in values:
                z = tuple(float(value) for value in row[n_initial:n_initial+n_geometry])
                if z not in seen:
                    seen.add(z)
                    normalized.append(np.asarray(z, dtype=float))

        required = len(normalized)
        maximum = max(1, int(os.environ.get("SDFMPNEO_GEOMETRY_CACHE_MAX", "512")))
        target = min(required, maximum)
        if target > self.cache_size:
            self.cache_size = target

        # If the complete active set fits, build every geometry context once in
        # the main thread so later point-parallel sweeps are cache hits.
        if required <= self.cache_size:
            for z in normalized:
                self.context(self.denormalize(z))
        return required

    model_class.__init__ = init
    model_class.context = context
    model_class.prepare_training_contexts = prepare_training_contexts
    model_class._working_set_cache_installed = True
