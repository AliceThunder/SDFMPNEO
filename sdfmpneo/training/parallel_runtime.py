from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from threading import Lock

import numpy as np


_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = Lock()
_INSTALLED = False


def training_point_workers() -> int:
    """Number of independent collocation workers; override with an environment variable."""
    raw = os.environ.get("SDFMPNEO_POINT_WORKERS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError as exc:
            raise ValueError("SDFMPNEO_POINT_WORKERS must be a positive integer") from exc
    return max(1, min(8, os.cpu_count() or 1))


def _executor(workers: int) -> ThreadPoolExecutor:
    with _EXECUTOR_LOCK:
        pool = _EXECUTORS.get(workers)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sdfmpneo-point")
            _EXECUTORS[workers] = pool
        return pool


def _ordered_map(function, items, *, monitor=None):
    items = list(items)
    workers = min(training_point_workers(), len(items)) if items else 1
    if workers <= 1:
        out = []
        for item in items:
            if monitor is not None:
                monitor.checkpoint()
            out.append(function(item))
        return out

    pool = _executor(workers)
    out = []
    # Small ordered batches preserve pause/stop responsiveness while keeping
    # deterministic collection and reduction order.
    batch_size = max(workers, 2 * workers)
    for start in range(0, len(items), batch_size):
        if monitor is not None:
            monitor.checkpoint()
        batch = items[start : start + batch_size]
        out.extend(pool.map(function, batch))
        if monitor is not None:
            monitor.checkpoint()
    return out


def install_training_acceleration() -> None:
    """Install exact-equivalent parallel hot paths into training.research."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import research as r

    original_evaluate = r._evaluate
    original_candidate_scores = r._candidate_tangent_scores
    original_refine_weights = r._refine_weights

    def evaluate(graph, field, points, *, jacobian=False, monitor=None):
        if training_point_workers() <= 1 or len(points) < 2:
            return original_evaluate(graph, field, points, jacobian=jacobian, monitor=monitor)
        n = graph.n_modes

        def one(point):
            initial, u, time = point[:n], point[n:-1], float(point[-1])
            a, da = r.evaluate_parametric_stable(graph, time, a0=initial, operating=u)
            if jacobian:
                physical = field.evaluate(a, u)
                F, J = physical.vector_field, physical.vector_field_jacobian
            elif hasattr(field, "vector_field"):
                F = field.vector_field(a, u)
                J = None
            else:
                rhs = field.rhs(u)
                q = field.em_model.heat_source_for_rhs(a, rhs)
                F = -field.thermal_model.lambdas * a + q + field.thermal_forcing
                J = None
            return r.SimpleNamespace(a=a, residual=da-F, J=J, initial=initial, u=u, time=time)

        return _ordered_map(one, points, monitor=monitor)

    def candidate_tangent_scores(graph, records, compiled, parents, monitor=None):
        if training_point_workers() <= 1 or len(records) < 2:
            return original_candidate_scores(graph, records, compiled, parents, monitor=monitor)
        n = graph.n_modes
        identity = np.eye(n, dtype=float)
        linear = np.diag(np.asarray(graph.lambdas, dtype=float))

        def one(pair):
            record, realization = pair
            source = r.AnalyticRealization.constant(1.0)
            for parent in parents:
                source = source.product(realization.node_realizations[parent])
            psi, h = r._source_response_values(source, graph.lambdas, record.time)
            tangent = identity * psi - (record.J + linear) * h[np.newaxis, :]
            return record.residual @ tangent, np.sum(tangent * tangent, axis=0)

        contributions = _ordered_map(one, zip(records, compiled), monitor=monitor)
        inner = np.zeros(n, dtype=float)
        norm2 = np.zeros(n, dtype=float)
        # Preserve the original left-to-right floating-point accumulation order.
        for local_inner, local_norm2 in contributions:
            inner += local_inner
            norm2 += local_norm2
        return inner, norm2

    def refine_weights(graph, field, points, max_iterations=12, monitor=None, tolerance=0.0):
        if training_point_workers() <= 1 or len(points) < 2:
            return original_refine_weights(
                graph, field, points, max_iterations=max_iterations,
                monitor=monitor, tolerance=tolerance,
            )
        if not graph.response_nodes:
            return graph
        n = graph.n_modes
        for _ in range(max_iterations):
            if monitor is not None:
                monitor.phase("weight_refinement")

            def one(point):
                a, da, ja, jda = r.evaluate_parametric_stable_with_jacobians(
                    graph,
                    float(point[-1]),
                    a0=point[:n],
                    operating=point[n:-1],
                    weight_derivatives=True,
                )
                physical = field.evaluate(a, point[n:-1])
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
                    values = evaluate(trial, field, points, monitor=monitor)
                    trial_objective = sum(float(value.residual @ value.residual) for value in values)
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

    r._evaluate = evaluate
    r._candidate_tangent_scores = candidate_tangent_scores
    r._refine_weights = refine_weights
    _INSTALLED = True
