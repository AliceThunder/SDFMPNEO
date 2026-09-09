from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from sdfmpneo.analytic.state_graph import (
    clone_state_graph,
    has_multi_source_states,
    replace_source_weights,
    response_source_count,
    response_sources,
    weight_parameter_count,
)
from sdfmpneo.analytic.state_realization import (
    compile_state_realization,
    state_weight_value_jacobian,
)
from .parallel_runtime import _ordered_map
from .max_residual_runtime import (
    _MAX_STAGNATION_REL,
    _MAX_STAGNATION_STEPS,
    hard_point_weights,
    hard_point_weights_from_norms,
    max_first_accept,
    relative_max_improvement,
    weighted_linear_system,
    weighted_score_parent_batch,
)
from .state_structure_policy import (
    dynamic_response_parent,
    residual_norms,
    select_candidate_action,
)

_INSTALLED = False


def _weight_vector(graph) -> np.ndarray:
    """Flatten every source coefficient without discarding complex components.

    Gauss--Newton computes real coefficient increments because the physical
    residual is real. Existing imaginary components are preserved exactly by
    adding those real increments to the full complex vector.
    """
    return np.asarray(
        [
            source.weight
            for node in graph.response_nodes
            for source in response_sources(graph, node)
        ],
        dtype=np.complex128,
    )


def _state_linearization(graph, field, points, monitor=None):
    # Preserve the C++ DAG/GN fast path for legacy/single-source graphs. The
    # first true enrichment switches to exact source-aware realization AD.
    if not has_multi_source_states(graph):
        from .max_residual_runtime import _linearization
        return _linearization(graph, field, points, monitor=monitor)

    points_array = np.ascontiguousarray(points, dtype=float)
    n_modes = graph.n_modes

    def one(point):
        a, da, ja, jda = state_weight_value_jacobian(graph, point)
        physical = field.evaluate(a, point[n_modes:-1])
        return (
            da - physical.vector_field,
            jda - physical.vector_field_jacobian @ ja,
        )

    pairs = _ordered_map(one, points_array, monitor=monitor)
    residual = np.ascontiguousarray(
        np.vstack([item[0] for item in pairs]), dtype=float
    )
    jacobian = np.ascontiguousarray(
        np.stack([item[1] for item in pairs]), dtype=float
    )
    return residual, jacobian, None


def state_refine_weights(
    graph,
    field,
    points,
    max_iterations=12,
    monitor=None,
    tolerance=0.0,
):
    """Max-aligned IRLS over every active source coefficient in every state."""
    from . import research as r

    if not graph.response_nodes:
        return graph

    stale_steps = 0
    for _ in range(max_iterations):
        if monitor is not None:
            monitor.phase("weight_refinement")
        residual_matrix, jacobian_tensor, native_threads = _state_linearization(
            graph, field, points, monitor=monitor
        )
        old_norms = np.linalg.norm(residual_matrix, axis=1)
        old_max = float(np.max(old_norms, initial=0.0))
        if old_max <= tolerance:
            break

        point_weights = hard_point_weights_from_norms(old_norms, tolerance)
        residual_w, jacobian_w = weighted_linear_system(
            residual_matrix, jacobian_tensor, point_weights
        )
        residual = residual_w.reshape(-1)
        J = jacobian_w.reshape(-1, weight_parameter_count(graph))
        scales = np.linalg.norm(J, axis=0)
        scales[scales == 0.0] = 1.0
        if native_threads is not None:
            with threadpool_limits(limits=native_threads()):
                delta = np.linalg.lstsq(
                    J / scales, -residual, rcond=None
                )[0] / scales
        else:
            delta = np.linalg.lstsq(
                J / scales, -residual, rcond=None
            )[0] / scales

        current = _weight_vector(graph)
        accepted = False
        accepted_norms = old_norms
        for _ in range(20):
            trial = clone_state_graph(graph)
            replace_source_weights(trial, current + delta)
            try:
                values = r._evaluate(trial, field, points, monitor=monitor)
                new_norms = residual_norms(values)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                values = None
                new_norms = np.full_like(old_norms, np.inf)
            if values is not None and max_first_accept(
                old_norms, new_norms, tolerance, point_weights
            ):
                graph = trial
                accepted = True
                accepted_norms = new_norms
                if monitor is not None:
                    objective, maximum = r._metrics(values)
                    monitor.record(graph, objective, maximum, len(points))
                break
            delta *= 0.5

        if not accepted:
            break
        new_max = float(np.max(accepted_norms, initial=0.0))
        if (
            relative_max_improvement(old_max, new_max, tolerance)
            < _MAX_STAGNATION_REL
        ):
            stale_steps += 1
        else:
            stale_steps = 0
        if stale_steps >= _MAX_STAGNATION_STEPS:
            # Fixed active state/source structure has stopped advancing the
            # actual stopping metric; hand control back to column generation.
            break
    return graph


def residual_driven_state_train(
    field, config, *, graph=None, progress=None, monitor=None
):
    """Automatic analytic-state construction driven only by governing residual.

    Source cardinality is not prescribed. Candidate columns either enrich an
    existing dynamic family, grow a genuinely new state, or trigger an exact
    function-preserving split when downstream independent addressability wins the
    actual nonlinear max-residual comparison.
    """
    from . import research as r
    from . import adaptive_runtime as adaptive
    from . import late_stage_runtime as late

    if graph is None:
        graph = r.ParametricAnalyticEvolutionGraph(
            field.thermal_model.lambdas,
            [f"u{i}" for i in range(field.n_operating)],
        )
    if (
        len(config.initial_lower) != graph.n_modes
        or len(config.operating_lower) != field.n_operating
    ):
        raise ValueError("training domain does not match the physical model")
    if len(graph.response_nodes) > config.max_nodes:
        raise ValueError("existing graph exceeds the requested state budget")

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
    if monitor is not None:
        monitor.record(graph, objective, maximum, len(points))
        monitor.phase("quadratic_seed")

    if graph.response_nodes and maximum > config.residual_tolerance:
        graph = r._refine_weights(
            graph,
            field,
            points,
            monitor=monitor,
            tolerance=config.residual_tolerance,
        )
        records = r._evaluate(
            graph, field, points, jacobian=True, monitor=monitor
        )
        objective, maximum = r._metrics(records)
        history.append(objective)
        if monitor is not None:
            monitor.record(graph, objective, maximum, len(points))

    # Existing equation-derived seeding remains valid. It simply initializes
    # single-column states which residual-driven construction may later enrich.
    seed = (
        r._quadratic_heating_seed(graph, field, monitor=monitor)
        if config.max_degree >= 2
        else graph
    )
    if len(seed.response_nodes) <= config.max_nodes and seed is not graph:
        try:
            seed_records = r._evaluate(seed, field, points, monitor=monitor)
            seed_weights = hard_point_weights(
                records, config.residual_tolerance
            )
            seed_ok = max_first_accept(
                residual_norms(records),
                residual_norms(seed_records),
                config.residual_tolerance,
                seed_weights,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            seed_ok = False
        if seed_ok:
            graph = r._refine_weights(
                seed,
                field,
                points,
                monitor=monitor,
                tolerance=config.residual_tolerance,
            )
            records = r._evaluate(
                graph, field, points, jacobian=True, monitor=monitor
            )
            objective, maximum = r._metrics(records)
            history.append(objective)
            if monitor is not None:
                monitor.record(graph, objective, maximum, len(points))
            if progress is not None:
                progress(
                    len(graph.response_nodes), np.sqrt(objective), maximum
                )

    check_max = float("inf")
    status = "stalled"
    while True:
        if monitor is not None:
            monitor.checkpoint()

        if maximum <= config.residual_tolerance:
            if monitor is not None:
                monitor.phase("validation")
            validation_points = adaptive._unique_rows(
                guards, checks, width=points.shape[1]
            )
            late._prepare_working_set(field, points, validation_points)
            validation_records = r._evaluate(
                graph, field, validation_points, monitor=monitor
            )
            _, check_max = r._metrics(validation_records)
            if monitor is not None:
                monitor.validation(check_max, len(validation_points))
            if check_max <= config.residual_tolerance:
                status = "numerically_converged"
                break

            violating, passed, _ = adaptive._select_adaptive_validation_points(
                validation_records,
                validation_points,
                tolerance=config.residual_tolerance,
            )
            next_points = adaptive._unique_rows(
                points, violating, width=points.shape[1]
            )
            next_guards = adaptive._unique_rows(
                passed, width=points.shape[1]
            )
            next_epoch = collocation_epoch + 1
            excluded = adaptive._unique_rows(
                next_points, next_guards, width=points.shape[1]
            )
            next_checks = adaptive._fresh_checks(
                config,
                seed=2 + len(graph.response_nodes) + next_epoch,
                excluded=excluded,
            )
            state = adaptive.ResearchTrainingContinuation(
                next_points,
                next_checks,
                next_guards,
                next_epoch,
                state.config_signature,
            )
            adaptive._store_continuation(field, state)
            points, checks, guards = next_points, next_checks, next_guards
            collocation_epoch = next_epoch
            late._prepare_working_set(field, points, checks, guards)
            records = r._evaluate(
                graph, field, points, jacobian=True, monitor=monitor
            )
            objective, maximum = r._metrics(records)
            history = [objective]
            if monitor is not None:
                monitor.record(
                    graph, objective, maximum, len(points), new_points=True
                )

        if monitor is not None:
            monitor.phase("candidate_search")
        point_weights = hard_point_weights(records, config.residual_tolerance)
        compiled = [
            compile_state_realization(
                graph, a0=record.initial, operating=record.u
            )
            for record in records
        ]
        names = graph.known_names()
        response_names = {node.name for node in graph.response_nodes}
        active = {
            (node.target_mode, tuple(sorted(source.parents)))
            for node in graph.response_nodes
            for source in response_sources(graph, node)
        }

        specs = []
        for degree in range(config.max_degree + 1):
            for parents in late._iter_admissible_parent_tuples(
                names,
                response_names,
                degree,
                config.max_parent_responses,
            ):
                direct_dimension = int(
                    np.prod(
                        [
                            compiled[0].node_realizations[parent].dimension
                            for parent in parents
                        ]
                    )
                ) + 1
                dynamic = dynamic_response_parent(graph, parents)
                split_possible = (
                    isinstance(dynamic, str)
                    and response_source_count(graph, dynamic) > 1
                    and len(graph.response_nodes) < config.max_nodes
                )
                # An over-dimension aggregate parent may still become admissible
                # after an exact split, so never prune it when any state-budget
                # slot remains; the exact split policy enforces the final budget.
                if (
                    direct_dimension > config.max_realization_dimension
                    and not split_possible
                ):
                    continue
                parent_key = tuple(sorted(parents))
                targets = tuple(
                    target
                    for target in range(graph.n_modes)
                    if (target, parent_key) not in active
                )
                if targets:
                    specs.append(
                        (parents, targets, late._parent_plan(graph, parents))
                    )

        scores = weighted_score_parent_batch(
            graph,
            records,
            compiled,
            [spec[2] for spec in specs],
            point_weights=point_weights,
            monitor=monitor,
        )
        scored = []
        for (parents, targets, _), (inner, norm2) in zip(specs, scores):
            for target in targets:
                if norm2[target] > 0 and np.isfinite(
                    norm2[target] + inner[target]
                ):
                    scored.append(
                        (
                            inner[target] * inner[target] / norm2[target],
                            -inner[target] / norm2[target],
                            int(target),
                            tuple(parents),
                        )
                    )
        scored.sort(key=lambda item: item[0], reverse=True)

        success = False
        for score, weight, target, parents in scored:
            if score <= 0:
                continue
            action, trial_records = select_candidate_action(
                graph,
                field,
                points,
                records,
                config,
                target,
                parents,
                weight,
                monitor=monitor,
            )
            if action is None:
                continue
            if monitor is not None:
                trial_objective, trial_max = r._metrics(trial_records)
                monitor.record(
                    action.graph,
                    trial_objective,
                    trial_max,
                    len(points),
                )
            graph = r._refine_weights(
                action.graph,
                field,
                points,
                monitor=monitor,
                tolerance=config.residual_tolerance,
            )
            records = r._evaluate(
                graph, field, points, jacobian=True, monitor=monitor
            )
            objective, maximum = r._metrics(records)
            history.append(objective)
            if monitor is not None:
                monitor.record(graph, objective, maximum, len(points))
            if progress is not None:
                progress(
                    len(graph.response_nodes), np.sqrt(objective), maximum
                )
            success = True
            break

        if not success:
            status = (
                "budget_exhausted"
                if len(graph.response_nodes) >= config.max_nodes
                else "stalled"
            )
            if monitor is not None:
                monitor.phase("validation")
            validation_points = adaptive._unique_rows(
                guards, checks, width=points.shape[1]
            )
            late._prepare_working_set(field, points, validation_points)
            _, check_max = r._metrics(
                r._evaluate(
                    graph, field, validation_points, monitor=monitor
                )
            )
            if monitor is not None:
                monitor.validation(check_max, len(validation_points))
            break

    return graph, r.ResearchTrainingReport(
        status,
        len(graph.response_nodes),
        float(initial_rms),
        float(np.sqrt(objective)),
        maximum,
        check_max,
        tuple(history),
        status == "numerically_converged",
    )


def install_residual_driven_state_training() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import research as r
    from . import adaptive_runtime as adaptive
    from . import cpp_dag_runtime

    # Native C++ currently represents one scalar source per node. Keep that fast
    # path until the first enrichment; then fail closed to exact source-aware AD.
    original_plan = cpp_dag_runtime._plan
    if not getattr(
        cpp_dag_runtime, "_analytic_state_plan_guard_installed", False
    ):
        def safe_plan(graph):
            if has_multi_source_states(graph):
                return None
            return original_plan(graph)

        cpp_dag_runtime._plan = safe_plan
        cpp_dag_runtime._analytic_state_plan_guard_installed = True

    r._refine_weights = state_refine_weights
    r.train_research_graph = residual_driven_state_train
    adaptive.adaptive_train_research_graph = residual_driven_state_train
    try:
        import sdfmpneo.research as research_module
        research_module.train_research_graph = residual_driven_state_train
    except ImportError:
        pass
    _INSTALLED = True
