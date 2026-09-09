from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from sdfmpneo.analytic.realization import AnalyticRealization
from .parallel_runtime import _ordered_map

# The final convergence criterion is an L-infinity residual bound.  These fixed
# numerical policy constants make the least-squares subproblem emphasize only
# points that currently violate that same bound, without adding a new user-facing
# training stage or relaxing any physical tolerance.
_HARD_WEIGHT_STRENGTH = 24.0
_MAX_STAGNATION_REL = 1.0e-3
_MAX_STAGNATION_STEPS = 2


def _residual_norms(records) -> np.ndarray:
    return np.asarray([np.linalg.norm(value.residual) for value in records], dtype=float)


def hard_point_weights_from_norms(norms, tolerance: float) -> np.ndarray:
    """IRLS weights focused exactly on residual violations above ``tolerance``.

    Passed points keep unit weight.  Among violating points the largest excess
    receives weight 1 + _HARD_WEIGHT_STRENGTH.  A common normalization is not
    applied because absolute row scaling cancels in the Gauss--Newton solution.
    """
    values = np.asarray(norms, dtype=float)
    if values.ndim != 1:
        raise ValueError("residual norms must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=float)
    tol = max(0.0, float(tolerance))
    if tol > 0.0:
        excess = np.maximum(values - tol, 0.0)
    else:
        excess = np.maximum(values, 0.0)
    peak = float(np.max(excess, initial=0.0))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.ones_like(values)
    ratio = excess / peak
    return 1.0 + _HARD_WEIGHT_STRENGTH * ratio * ratio


def hard_point_weights(records, tolerance: float) -> np.ndarray:
    return hard_point_weights_from_norms(_residual_norms(records), tolerance)


def weighted_objective_from_norms(norms, weights) -> float:
    values = np.asarray(norms, dtype=float)
    w = np.asarray(weights, dtype=float)
    if values.shape != w.shape:
        raise ValueError("residual norms and weights do not match")
    return float(np.dot(w, values * values))


def weighted_linear_system(residual_matrix, jacobian_tensor, weights):
    residual = np.asarray(residual_matrix, dtype=float)
    jacobian = np.asarray(jacobian_tensor, dtype=float)
    w = np.asarray(weights, dtype=float)
    if residual.ndim != 2 or jacobian.ndim != 3:
        raise ValueError("batched residual/Jacobian shapes are required")
    if residual.shape[0] != w.size or jacobian.shape[:2] != residual.shape:
        raise ValueError("weighted Gauss--Newton dimensions do not match")
    scale = np.sqrt(w)[:, None]
    return residual * scale, jacobian * scale[:, :, None]


def _protected_points_remain_passed(old_norms, new_norms, tolerance: float) -> bool:
    tol = float(tolerance)
    if tol <= 0.0:
        return True
    old_values = np.asarray(old_norms, dtype=float)
    new_values = np.asarray(new_norms, dtype=float)
    protected = old_values <= tol
    if not np.any(protected):
        return True
    margin = 64.0 * np.finfo(float).eps * max(tol, 1.0e-30)
    return bool(np.all(new_values[protected] <= tol + margin))


def max_first_accept(old_norms, new_norms, tolerance: float, weights) -> bool:
    """Accept a step only if it advances the max criterion or its frozen IRLS merit.

    The ordering is deliberately max-first.  Points that were already below the
    final tolerance are protected from crossing back above it.
    """
    old_values = np.asarray(old_norms, dtype=float)
    new_values = np.asarray(new_norms, dtype=float)
    if old_values.shape != new_values.shape or old_values.ndim != 1:
        raise ValueError("old/new residual norms must have matching vector shapes")
    if not _protected_points_remain_passed(old_values, new_values, tolerance):
        return False
    old_max = float(np.max(old_values, initial=0.0))
    new_max = float(np.max(new_values, initial=0.0))
    numerical = 64.0 * np.finfo(float).eps * max(old_max, float(tolerance), 1.0e-30)
    if new_max < old_max - numerical:
        return True
    if new_max <= old_max + numerical:
        old_merit = weighted_objective_from_norms(old_values, weights)
        new_merit = weighted_objective_from_norms(new_values, weights)
        return new_merit < old_merit - numerical * max(1.0, old_merit)
    return False


def relative_max_improvement(old_max: float, new_max: float, tolerance: float) -> float:
    denominator = max(abs(float(old_max)), abs(float(tolerance)), np.finfo(float).tiny)
    return (float(old_max) - float(new_max)) / denominator


def _linearization(graph, field, points, monitor=None):
    """Use native DAG/GN kernels when admissible, otherwise the exact sparse path."""
    from . import late_stage_runtime as late
    try:
        from .cpp_dag_runtime import _native_value_jacobian_batch
        from sdfmpneo.cpp_dag_backend import gn_linearize, native_threads
        native = _native_value_jacobian_batch(graph, points)
    except Exception:
        native = None
        gn_linearize = None
        native_threads = None

    n_modes = graph.n_modes
    if native is not None:
        a_all, da_all, ja_all, jda_all = native
        points_array = np.asarray(points, dtype=float)

        def physical_one(index):
            if monitor is not None:
                monitor.checkpoint()
            value = field.evaluate(a_all[index], points_array[index, n_modes:-1])
            return value.vector_field, value.vector_field_jacobian

        physical = _ordered_map(physical_one, range(len(points_array)), monitor=monitor)
        F = np.ascontiguousarray(np.vstack([item[0] for item in physical]), dtype=float)
        JF = np.ascontiguousarray(np.stack([item[1] for item in physical]), dtype=float)
        formed = gn_linearize(da_all, F, ja_all, jda_all, JF) if gn_linearize is not None else None
        if formed is None:
            residual = da_all - F
            jacobian = jda_all - np.einsum("pij,pjw->piw", JF, ja_all)
        else:
            residual, jacobian = formed
        return np.asarray(residual), np.asarray(jacobian), native_threads

    def one(point):
        a, da, ja, jda = late._sparse_weight_value_jacobian(graph, point)
        physical = field.evaluate(a, point[n_modes:-1])
        return da - physical.vector_field, jda - physical.vector_field_jacobian @ ja

    pairs = _ordered_map(one, points, monitor=monitor)
    residual = np.vstack([item[0] for item in pairs])
    jacobian = np.stack([item[1] for item in pairs])
    return residual, jacobian, None


def max_residual_refine_weights(graph, field, points, max_iterations=12, monitor=None, tolerance=0.0):
    """Max-aligned IRLS Gauss--Newton with early topology-growth handoff."""
    from . import research as r
    if not graph.response_nodes:
        return graph

    stale_steps = 0
    for _ in range(max_iterations):
        if monitor is not None:
            monitor.phase("weight_refinement")
        residual_matrix, jacobian_tensor, native_threads = _linearization(
            graph, field, points, monitor=monitor
        )
        old_norms = np.linalg.norm(residual_matrix, axis=1)
        old_max = float(np.max(old_norms, initial=0.0))
        if old_max <= tolerance:
            break

        weights = hard_point_weights_from_norms(old_norms, tolerance)
        residual_w, jacobian_w = weighted_linear_system(
            residual_matrix, jacobian_tensor, weights
        )
        residual = residual_w.reshape(-1)
        J = jacobian_w.reshape(-1, len(graph.response_nodes))
        scales = np.linalg.norm(J, axis=0)
        scales[scales == 0.0] = 1.0
        if native_threads is not None:
            with threadpool_limits(limits=native_threads()):
                delta = np.linalg.lstsq(J / scales, -residual, rcond=None)[0] / scales
        else:
            delta = np.linalg.lstsq(J / scales, -residual, rcond=None)[0] / scales

        accepted = False
        accepted_norms = old_norms
        accepted_values = None
        for _ in range(20):
            trial = r.ParametricAnalyticEvolutionGraph(graph.lambdas, graph.operating_names)
            for node, change in zip(graph.response_nodes, delta):
                trial.add_product_response(
                    node.name, node.target_mode, node.parents, node.weight + change
                )
            try:
                values = r._evaluate(trial, field, points, monitor=monitor)
                new_norms = _residual_norms(values)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                values = None
                new_norms = np.full_like(old_norms, np.inf)
            if values is not None and max_first_accept(old_norms, new_norms, tolerance, weights):
                graph = trial
                accepted = True
                accepted_norms = new_norms
                accepted_values = values
                if monitor is not None:
                    objective, maximum = r._metrics(values)
                    monitor.record(graph, objective, maximum, len(points))
                break
            delta *= 0.5

        if not accepted:
            break
        new_max = float(np.max(accepted_norms, initial=0.0))
        if relative_max_improvement(old_max, new_max, tolerance) < _MAX_STAGNATION_REL:
            stale_steps += 1
        else:
            stale_steps = 0
        if stale_steps >= _MAX_STAGNATION_STEPS:
            # Existing topology has stopped improving the actual stopping metric;
            # return immediately so candidate growth can add a new response node.
            break
    return graph


def _weighted_generic_score(graph, records, compiled, plan, point_weights):
    from . import late_stage_batch as batch
    n_modes = graph.n_modes
    identity = np.eye(n_modes, dtype=float)
    linear = np.diag(np.asarray(graph.lambdas, dtype=float))
    inner = np.zeros(n_modes, dtype=float)
    norm2 = np.zeros(n_modes, dtype=float)
    for point_weight, record, realization in zip(point_weights, records, compiled):
        source = AnalyticRealization.constant(1.0)
        for parent in plan.parents:
            source = source.product(realization.node_realizations[parent])
        psi, h = batch._candidate_source_response(source, graph.lambdas, record.time)
        tangent = identity * psi - (record.J + linear) * h[np.newaxis, :]
        inner += point_weight * (record.residual @ tangent)
        norm2 += point_weight * np.sum(tangent * tangent, axis=0)
    return inner, norm2


def weighted_score_parent_batch(graph, records, compiled, plans, *, point_weights=None, monitor=None):
    """Existing shared candidate batching with the same IRLS weights as GN."""
    from . import late_stage_batch as batch
    plans = list(plans)
    if not plans:
        return []
    if point_weights is None:
        point_weights = np.ones(len(records), dtype=float)
    point_weights = np.asarray(point_weights, dtype=float)
    if point_weights.shape != (len(records),):
        raise ValueError("candidate point weights do not match records")

    keys, seen = [], set()
    for plan in plans:
        key = batch._structured_key(plan)
        if key is not None and key not in seen:
            seen.add(key)
            keys.append(key)
    tables = _ordered_map(
        lambda key: batch._unit_candidate_table(graph, records, compiled, key),
        keys,
        monitor=monitor,
    ) if keys else []
    table_by_key = dict(zip(keys, tables))

    if records:
        initial = np.vstack([np.asarray(record.initial, dtype=float) for record in records])
        operating = np.vstack([np.asarray(record.u, dtype=float) for record in records])
    else:
        initial = np.empty((0, graph.n_modes), dtype=float)
        operating = np.empty((0, len(graph.operating_names)), dtype=float)

    out = [None] * len(plans)
    generic, generic_indices = [], []
    for index, plan in enumerate(plans):
        key = batch._structured_key(plan)
        if key is None:
            generic.append(plan)
            generic_indices.append(index)
            continue
        amplitude = batch._plan_scale(plan, initial, operating)
        correlation, local_norm2 = table_by_key[key]
        weighted_amplitude = point_weights * amplitude
        inner = np.sum(weighted_amplitude[:, None] * correlation, axis=0)
        norm2 = np.sum((point_weights * amplitude * amplitude)[:, None] * local_norm2, axis=0)
        out[index] = (inner, norm2)

    if generic:
        values = _ordered_map(
            lambda plan: _weighted_generic_score(
                graph, records, compiled, plan, point_weights
            ),
            generic,
            monitor=monitor,
        )
        for index, value in zip(generic_indices, values):
            out[index] = value
    return out


def _trial_accepts(old_records, new_records, tolerance, weights) -> bool:
    return max_first_accept(
        _residual_norms(old_records), _residual_norms(new_records), tolerance, weights
    )


def max_aligned_train_research_graph(field, config, *, graph=None, progress=None, monitor=None):
    """Adaptive residual growth aligned to the final maximum-residual criterion."""
    from . import research as r
    from . import adaptive_runtime as adaptive
    from . import late_stage_runtime as late

    if graph is None:
        graph = r.ParametricAnalyticEvolutionGraph(
            field.thermal_model.lambdas, [f"u{i}" for i in range(field.n_operating)]
        )
    if len(config.initial_lower) != graph.n_modes or len(config.operating_lower) != field.n_operating:
        raise ValueError("training domain does not match the physical model")
    if len(graph.response_nodes) > config.max_nodes:
        raise ValueError("existing graph exceeds the requested node budget")

    state = adaptive._initial_continuation(field, config)
    adaptive._store_continuation(field, state)
    points, checks, guards = state.points, state.checks, state.guards
    collocation_epoch = state.collocation_epoch
    late._prepare_working_set(field, points, checks, guards)

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
            seed_records = r._evaluate(seed, field, points, monitor=monitor)
            seed_objective, seed_max = r._metrics(seed_records)
            seed_weights = hard_point_weights(records, config.residual_tolerance)
            seed_ok = _trial_accepts(records, seed_records, config.residual_tolerance, seed_weights)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            seed_ok = False
        if seed_ok:
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
            validation_points = adaptive._unique_rows(guards, checks, width=points.shape[1])
            late._prepare_working_set(field, points, validation_points)
            validation_records = r._evaluate(graph, field, validation_points, monitor=monitor)
            _, check_max = r._metrics(validation_records)
            if monitor is not None:
                monitor.validation(check_max, len(validation_points))
            if check_max <= config.residual_tolerance and maximum <= config.residual_tolerance:
                status = "numerically_converged"
                break

            violating, passed, _ = adaptive._select_adaptive_validation_points(
                validation_records, validation_points, tolerance=config.residual_tolerance
            )
            next_points = adaptive._unique_rows(points, violating, width=points.shape[1])
            next_guards = adaptive._unique_rows(passed, width=points.shape[1])
            next_epoch = collocation_epoch + 1
            excluded = adaptive._unique_rows(next_points, next_guards, width=points.shape[1])
            next_checks = adaptive._fresh_checks(
                config, seed=2 + accepted + next_epoch, excluded=excluded
            )
            next_state = adaptive.ResearchTrainingContinuation(
                next_points, next_checks, next_guards, next_epoch, state.config_signature
            )
            adaptive._store_continuation(field, next_state)
            if accepted >= config.max_nodes:
                status = "budget_exhausted"
                break
            points, checks, guards = next_points, next_checks, next_guards
            collocation_epoch = next_epoch
            state = next_state
            late._prepare_working_set(field, points, checks, guards)
            records = r._evaluate(graph, field, points, jacobian=True, monitor=monitor)
            objective, maximum = r._metrics(records)
            history = [objective]
            if monitor is not None:
                monitor.record(graph, objective, maximum, len(points), new_points=True)

        if monitor is not None:
            monitor.phase("candidate_search")

        point_weights = hard_point_weights(records, config.residual_tolerance)
        compiled = [
            r.compile_parametric_realization(graph, a0=record.initial, operating=record.u)
            for record in records
        ]
        names = graph.known_names()
        response_names = {node.name for node in graph.response_nodes}
        existing = {(node.target_mode, tuple(sorted(node.parents))) for node in graph.response_nodes}
        specs = []
        for degree in range(config.max_degree + 1):
            for parents in late._iter_admissible_parent_tuples(
                names, response_names, degree, config.max_parent_responses
            ):
                dimension = int(np.prod([
                    compiled[0].node_realizations[parent].dimension for parent in parents
                ])) + 1
                if dimension > config.max_realization_dimension:
                    continue
                parent_key = tuple(sorted(parents))
                active_targets = tuple(
                    target for target in range(graph.n_modes)
                    if (target, parent_key) not in existing
                )
                if active_targets:
                    specs.append((parents, active_targets, late._parent_plan(graph, parents)))

        scores = weighted_score_parent_batch(
            graph, records, compiled, [spec[2] for spec in specs],
            point_weights=point_weights, monitor=monitor,
        )
        scored = []
        for (parents, active_targets, _), (inner, norm2) in zip(specs, scores):
            for target in active_targets:
                if norm2[target] > 0 and np.isfinite(norm2[target] + inner[target]):
                    scored.append((
                        inner[target] * inner[target] / norm2[target],
                        -inner[target] / norm2[target], target, parents,
                    ))
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
                    acceptable = _trial_accepts(
                        records, trial_records, config.residual_tolerance, point_weights
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    acceptable = False
                if acceptable:
                    if monitor is not None:
                        monitor.record(trial, trial_objective, trial_max, len(points))
                    graph = r._refine_weights(
                        trial, field, points, monitor=monitor,
                        tolerance=config.residual_tolerance,
                    )
                    final_records = r._evaluate(graph, field, points, monitor=monitor)
                    objective, maximum = r._metrics(final_records)
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
            validation_points = adaptive._unique_rows(guards, checks, width=points.shape[1])
            late._prepare_working_set(field, points, validation_points)
            _, check_max = r._metrics(r._evaluate(graph, field, validation_points, monitor=monitor))
            if monitor is not None:
                monitor.validation(check_max, len(validation_points))
            break
        records = r._evaluate(graph, field, points, jacobian=True, monitor=monitor)

    report = r.ResearchTrainingReport(
        status, accepted, float(initial_rms), float(np.sqrt(objective)), maximum,
        check_max, tuple(history), status == "numerically_converged",
    )
    return graph, report


def install_max_residual_training() -> None:
    """Install the max-aligned trainer after native/coverage runtimes are ready."""
    from . import research as r
    from . import adaptive_runtime as adaptive
    from . import late_stage_runtime as late

    r._refine_weights = max_residual_refine_weights
    r.train_research_graph = max_aligned_train_research_graph
    late._score_parent_batch = weighted_score_parent_batch
    adaptive.adaptive_train_research_graph = max_aligned_train_research_graph
    try:
        from sdfmpneo import research as research_module
        research_module.train_research_graph = max_aligned_train_research_graph
    except ImportError:
        pass
