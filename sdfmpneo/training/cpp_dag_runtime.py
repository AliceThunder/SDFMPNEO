from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from ..analytic.long_time import realization_observation_action
from ..analytic.parametric_realization import compile_parametric_nodes
from ..cpp_dag_backend import (
    backend_info as dag_backend_info,
    dag_batch_sparse_jacobian,
    dag_batch_values,
    gn_linearize,
    native_threads,
)
from .parallel_runtime import _ordered_map

_INSTALLED = False
_CURRENT_SIGNATURE = None
_CURRENT_PLAN = None


@dataclass
class NativeDAGPlan:
    signature: tuple
    structural_nodes: tuple
    _native_arrays: dict[str, np.ndarray]
    total_state: int
    n_modes: int
    n_operating: int
    _row_times: tuple | None = None
    _rows: tuple[np.ndarray, np.ndarray] | None = None

    def observation_rows(self, times) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(float(value) for value in np.asarray(times, dtype=float))
        if key == self._row_times and self._rows is not None:
            return self._rows
        offsets = self._native_arrays["state_offsets"]
        identities = {
            int(dim): np.eye(int(dim), dtype=complex)
            for dim in np.unique(self._native_arrays["dimensions"])
        }

        def one(t):
            values = np.empty(self.total_state, dtype=float)
            slopes = np.empty_like(values)
            for index, realization in enumerate(self.structural_nodes):
                v, s = realization_observation_action(
                    realization.A,
                    identities[int(realization.dimension)],
                    realization.c,
                    float(t),
                )
                lo, hi = int(offsets[index]), int(offsets[index + 1])
                values[lo:hi] = np.real(v)
                slopes[lo:hi] = np.real(s)
            return values, slopes

        packed = _ordered_map(one, key, monitor=None)
        if packed:
            rows = (
                np.ascontiguousarray(np.vstack([item[0] for item in packed])),
                np.ascontiguousarray(np.vstack([item[1] for item in packed])),
            )
        else:
            empty = np.empty((0, self.total_state), dtype=float)
            rows = (empty, empty.copy())
        self._row_times = key
        self._rows = rows
        return rows


def _graph_signature(graph) -> tuple:
    # Weights are intentionally excluded: A/c realization structure depends on
    # topology/decay rates only, so every GN trial reuses the same observation rows.
    return (
        tuple(float(value) for value in graph.lambdas),
        tuple(graph.initial_names),
        tuple(graph.operating_names),
        tuple((node.name, int(node.target_mode), tuple(node.parents)) for node in graph.response_nodes),
    )


def _make_offsets(groups) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    flat = []
    for values in groups:
        flat.extend(int(value) for value in values)
        offsets.append(len(flat))
    return np.asarray(offsets, np.int64), np.asarray(flat, np.int32)


def _real_weights(graph) -> np.ndarray | None:
    values = np.asarray([complex(node.weight) for node in graph.response_nodes], dtype=np.complex128)
    # Historical/custom complex-weight graphs keep the exact Python path. Current
    # residual growth generates real weights, so production graphs use C++.
    if np.any(values.imag != 0.0):
        return None
    return np.ascontiguousarray(values.real, dtype=float)


def _build_plan(graph) -> NativeDAGPlan | None:
    if not graph.response_nodes or _real_weights(graph) is None:
        return None
    initial_lookup = {name: i for i, name in enumerate(graph.initial_names)}
    operating_lookup = {name: i for i, name in enumerate(graph.operating_names)}
    response_lookup = {node.name: i for i, node in enumerate(graph.response_nodes)}
    initial_factors, operating_factors, response_parent, ancestors = [], [], [], []

    for index, node in enumerate(graph.response_nodes):
        initial, operating, response = [], [], []
        for parent in node.parents:
            if parent in initial_lookup:
                initial.append(initial_lookup[parent])
            elif parent in operating_lookup:
                operating.append(operating_lookup[parent])
            elif parent in response_lookup and response_lookup[parent] < index:
                response.append(response_lookup[parent])
            else:
                return None
        if len(response) > 1:
            return None
        parent = -1 if not response else response[0]
        response_parent.append(parent)
        initial_factors.append(tuple(initial))
        operating_factors.append(tuple(operating))
        ancestors.append((index,) if parent < 0 else tuple(ancestors[parent]) + (index,))

    compiled = compile_parametric_nodes(
        graph,
        a0=np.ones(graph.n_modes),
        operating=np.ones(len(graph.operating_names)),
    )
    structural = tuple(compiled.node_realizations[node.name] for node in graph.response_nodes)
    dimensions = np.asarray([item.dimension for item in structural], dtype=np.int32)
    for index, parent in enumerate(response_parent):
        expected = 2 if parent < 0 else int(dimensions[parent]) + 1
        if int(dimensions[index]) != expected:
            return None

    state_offsets = np.zeros(len(structural) + 1, dtype=np.int64)
    state_offsets[1:] = np.cumsum(dimensions, dtype=np.int64)
    ancestor_offsets, ancestor_flat = _make_offsets(ancestors)
    derivative_offsets = np.zeros(len(structural) + 1, dtype=np.int64)
    derivative_offsets[1:] = np.cumsum(
        dimensions.astype(np.int64) * np.diff(ancestor_offsets), dtype=np.int64
    )
    initial_offsets, initial_flat = _make_offsets(initial_factors)
    operating_offsets, operating_flat = _make_offsets(operating_factors)
    arrays = {
        "targets": np.ascontiguousarray([node.target_mode for node in graph.response_nodes], dtype=np.int32),
        "response_parent": np.ascontiguousarray(response_parent, dtype=np.int32),
        "dimensions": np.ascontiguousarray(dimensions, dtype=np.int32),
        "state_offsets": np.ascontiguousarray(state_offsets),
        "derivative_offsets": np.ascontiguousarray(derivative_offsets),
        "ancestor_offsets": np.ascontiguousarray(ancestor_offsets),
        "ancestors": np.ascontiguousarray(ancestor_flat),
        "initial_factor_offsets": np.ascontiguousarray(initial_offsets),
        "initial_factor_indices": np.ascontiguousarray(initial_flat),
        "operating_factor_offsets": np.ascontiguousarray(operating_offsets),
        "operating_factor_indices": np.ascontiguousarray(operating_flat),
    }
    return NativeDAGPlan(
        _graph_signature(graph), structural, arrays, int(state_offsets[-1]),
        graph.n_modes, len(graph.operating_names),
    )


def _plan(graph) -> NativeDAGPlan | None:
    global _CURRENT_SIGNATURE, _CURRENT_PLAN
    signature = _graph_signature(graph)
    if signature == _CURRENT_SIGNATURE:
        # Complex weights can be introduced by custom callers without topology
        # changing, so re-check before returning a cached native plan.
        return _CURRENT_PLAN if _real_weights(graph) is not None else None
    _CURRENT_SIGNATURE = signature
    _CURRENT_PLAN = _build_plan(graph)
    return _CURRENT_PLAN


def _point_arrays(graph, points):
    points = np.ascontiguousarray(points, dtype=float)
    n = graph.n_modes
    initial = np.ascontiguousarray(points[:, :n])
    operating = np.ascontiguousarray(points[:, n:-1])
    times = np.ascontiguousarray(points[:, -1])
    lambdas = np.asarray(graph.lambdas, dtype=float)
    decay = np.empty((len(points), n), dtype=float)
    finite = np.isfinite(times)
    if np.any(finite):
        decay[finite] = np.exp(-times[finite, None] * lambdas[None, :])
    if np.any(~finite):
        decay[~finite] = (lambdas == 0.0)[None, :]
    initial_values = initial * decay
    initial_slopes = -lambdas[None, :] * initial_values
    initial_slopes[~finite] = 0.0
    return initial, operating, times, initial_values, initial_slopes


def _native_values(graph, points):
    plan = _plan(graph)
    weights = _real_weights(graph)
    if plan is None or weights is None or not dag_backend_info(auto_build=True)["available"]:
        return None
    initial, operating, times, initial_values, initial_slopes = _point_arrays(graph, points)
    return dag_batch_values(
        plan, initial, operating, weights, plan.observation_rows(times),
        initial_values, initial_slopes,
    )


def _native_value_jacobian_batch(graph, points):
    plan = _plan(graph)
    weights = _real_weights(graph)
    if plan is None or weights is None or not dag_backend_info(auto_build=True)["available"]:
        return None
    initial, operating, times, initial_values, initial_slopes = _point_arrays(graph, points)
    return dag_batch_sparse_jacobian(
        plan, initial, operating, weights, plan.observation_rows(times),
        initial_values, initial_slopes,
    )


def install_native_dag_training() -> None:
    """Install native analytic-DAG batches and native Gauss-Newton linearization."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import research as r

    original_evaluate = r._evaluate
    original_refine = r._refine_weights

    def evaluate(graph, field, points, *, jacobian=False, monitor=None):
        values = _native_values(graph, points)
        if values is None:
            return original_evaluate(graph, field, points, jacobian=jacobian, monitor=monitor)
        a_all, da_all = values
        points_array = np.asarray(points, dtype=float)
        n = graph.n_modes

        def one(index):
            a = a_all[index]
            u = points_array[index, n:-1]
            if jacobian:
                physical = field.evaluate(a, u)
                F, J = physical.vector_field, physical.vector_field_jacobian
            elif hasattr(field, "vector_field"):
                F, J = field.vector_field(a, u), None
            else:
                rhs = field.rhs(u)
                q = field.em_model.heat_source_for_rhs(a, rhs)
                F = -field.thermal_model.lambdas * a + q + field.thermal_forcing
                J = None
            return SimpleNamespace(
                a=a, residual=da_all[index] - F, J=J,
                initial=points_array[index, :n], u=u,
                time=float(points_array[index, -1]),
            )

        return _ordered_map(one, range(len(points_array)), monitor=monitor)

    def refine(graph, field, points, max_iterations=12, monitor=None, tolerance=0.0):
        if not graph.response_nodes:
            return graph
        if _plan(graph) is None or not dag_backend_info(auto_build=True)["available"]:
            return original_refine(
                graph, field, points, max_iterations=max_iterations,
                monitor=monitor, tolerance=tolerance,
            )
        points_array = np.ascontiguousarray(points, dtype=float)
        n = graph.n_modes
        for _ in range(max_iterations):
            if monitor is not None:
                monitor.phase("weight_refinement")
            analytic = _native_value_jacobian_batch(graph, points_array)
            if analytic is None:
                return original_refine(
                    graph, field, points_array, max_iterations=max_iterations,
                    monitor=monitor, tolerance=tolerance,
                )
            a_all, da_all, ja_all, jda_all = analytic

            def physical_one(index):
                value = field.evaluate(a_all[index], points_array[index, n:-1])
                return value.vector_field, value.vector_field_jacobian

            physical = _ordered_map(physical_one, range(len(points_array)), monitor=monitor)
            F = np.ascontiguousarray(np.vstack([item[0] for item in physical]), dtype=float)
            JF = np.ascontiguousarray(np.stack([item[1] for item in physical]), dtype=float)
            residual_matrix, jacobian_tensor = gn_linearize(da_all, F, ja_all, jda_all, JF)
            norms = np.linalg.norm(residual_matrix, axis=1)
            if (float(np.max(norms)) if norms.size else 0.0) <= tolerance:
                break
            residual = residual_matrix.reshape(-1)
            J = jacobian_tensor.reshape(-1, len(graph.response_nodes))
            scales = np.linalg.norm(J, axis=0)
            scales[scales == 0] = 1.0
            # The GUI historically starts BLAS with one thread. Temporarily lift
            # that limit for the dense least-squares solve; outer point workers are
            # idle here, so this cannot create nested oversubscription.
            with threadpool_limits(limits=native_threads()):
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
                    trial_values = evaluate(trial, field, points_array, monitor=monitor)
                    trial_objective = sum(float(v.residual @ v.residual) for v in trial_values)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    trial_objective = float("inf")
                if trial_objective < objective:
                    graph = trial
                    if monitor is not None:
                        obj, maximum = r._metrics(trial_values)
                        monitor.record(graph, obj, maximum, len(points_array))
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
    r._refine_weights = refine
    _INSTALLED = True


def native_dag_runtime_info() -> dict:
    info = dag_backend_info(auto_build=False)
    return {
        "available": bool(info["available"]),
        "native_threads": int(native_threads()),
        "openmp": bool(info["openmp"]),
        "error": info["error"],
    }
