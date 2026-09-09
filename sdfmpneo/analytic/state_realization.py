from __future__ import annotations

import numpy as np

from .long_time import realization_observation_action
from .realization import AnalyticRealization, CompiledRealizationGraph
from .state_graph import response_sources, weight_parameter_map


def _validated_inputs(graph, a0, operating):
    initial = np.asarray(a0, dtype=float)
    u = np.asarray(operating, dtype=float)
    if initial.shape != (graph.n_modes,):
        raise ValueError("initial coordinate dimension mismatch")
    if u.shape != (len(graph.operating_names),):
        raise ValueError("operating parameter dimension mismatch")
    return initial, u


def _compile_nodes_with_derivatives(
    graph, *, a0, operating, derivative_kind: str | None
):
    initial, u = _validated_inputs(graph, a0, operating)
    if derivative_kind == "weight":
        parameter_map = weight_parameter_map(graph)
        nd = len(parameter_map)
        parameter_lookup = {key: index for index, key in enumerate(parameter_map)}
    elif derivative_kind == "operating":
        nd = u.size
        parameter_lookup = None
    elif derivative_kind is None:
        nd = 0
        parameter_lookup = None
    else:
        raise ValueError("unknown derivative kind")

    nodes: dict[str, AnalyticRealization] = {}
    derivatives: dict[str, np.ndarray] = {}
    sources: dict[str, AnalyticRealization] = {}

    for index, name in enumerate(graph.initial_names):
        realization = AnalyticRealization.decay(graph.lambdas[index], initial[index])
        nodes[name] = realization
        derivatives[name] = np.zeros((realization.dimension, nd), complex)

    for index, name in enumerate(graph.operating_names):
        realization = AnalyticRealization.constant(u[index])
        nodes[name] = realization
        derivative = np.zeros((1, nd), complex)
        if derivative_kind == "operating":
            derivative[0, index] = 1.0
        derivatives[name] = derivative

    for node in graph.response_nodes:
        source_sum = None
        derivative_sum = None
        for source_index, source_spec in enumerate(response_sources(graph, node)):
            term = AnalyticRealization.constant(source_spec.weight)
            derivative = np.zeros((1, nd), complex)
            if derivative_kind == "weight":
                derivative[0, parameter_lookup[(node.name, source_index)]] = 1.0
            for parent in source_spec.parents:
                right = nodes[parent]
                right_derivative = derivatives[parent]
                derivative = (
                    np.einsum("ik,j->ijk", derivative, right.b)
                    + np.einsum("i,jk->ijk", term.b, right_derivative)
                ).reshape(term.dimension * right.dimension, nd)
                term = term.product(right)
            if source_sum is None:
                source_sum = term
                derivative_sum = derivative
            else:
                source_sum = source_sum.add(term)
                derivative_sum = np.vstack([derivative_sum, derivative])
        if source_sum is None or derivative_sum is None:
            raise RuntimeError("analytic state has no active source columns")
        response = source_sum.response(graph.lambdas[node.target_mode])
        nodes[node.name] = response
        sources[node.name] = source_sum
        derivatives[node.name] = np.vstack(
            [derivative_sum, np.zeros((1, nd), complex)]
        )
    return nodes, sources, derivatives, nd


def compile_state_nodes(graph, *, a0, operating):
    from .parametric_realization import CompiledParametricNodeGraph

    nodes, sources, _, _ = _compile_nodes_with_derivatives(
        graph, a0=a0, operating=operating, derivative_kind=None
    )
    return CompiledParametricNodeGraph(
        lambdas=np.asarray(graph.lambdas, dtype=float).copy(),
        node_realizations=nodes,
        source_realizations=sources,
    )


def compile_state_realization(graph, *, a0, operating):
    nodes, sources, _, _ = _compile_nodes_with_derivatives(
        graph, a0=a0, operating=operating, derivative_kind=None
    )
    modes = [AnalyticRealization.zero() for _ in range(graph.n_modes)]
    for mode, name in enumerate(graph.initial_names):
        modes[mode] = modes[mode].add(nodes[name])
    for node in graph.response_nodes:
        modes[node.target_mode] = modes[node.target_mode].add(nodes[node.name])
    return CompiledRealizationGraph(
        lambdas=np.asarray(graph.lambdas, dtype=float).copy(),
        node_realizations=nodes,
        source_realizations=sources,
        mode_realizations=tuple(modes),
    )


def evaluate_state_stable(graph, t, *, a0, operating):
    a, da, _, _ = evaluate_state_stable_with_jacobians(
        graph, t, a0=a0, operating=operating, derivative_kind=None
    )
    return a, da


def evaluate_state_stable_with_jacobians(
    graph,
    t,
    *,
    a0,
    operating,
    weight_derivatives=False,
    derivative_kind=None,
):
    if t < 0 or np.isnan(t):
        raise ValueError("time must be non-negative or positive infinity")
    if derivative_kind is None:
        derivative_kind = "weight" if weight_derivatives else "operating"
    nodes, _, derivatives, nd = _compile_nodes_with_derivatives(
        graph,
        a0=a0,
        operating=operating,
        derivative_kind=derivative_kind,
    )
    a = np.zeros(graph.n_modes)
    da = np.zeros(graph.n_modes)
    ja = np.zeros((graph.n_modes, nd))
    jda = np.zeros_like(ja)

    for mode, name in enumerate(graph.initial_names):
        realization = nodes[name]
        B = np.column_stack([realization.b, derivatives[name]])
        values, slopes = realization_observation_action(
            realization.A, B, realization.c, float(t)
        )
        a[mode] += float(np.real(values[0]))
        da[mode] += float(np.real(slopes[0]))
        if nd:
            ja[mode] += np.real(values[1:])
            jda[mode] += np.real(slopes[1:])

    for node in graph.response_nodes:
        realization = nodes[node.name]
        B = np.column_stack([realization.b, derivatives[node.name]])
        values, slopes = realization_observation_action(
            realization.A, B, realization.c, float(t)
        )
        target = node.target_mode
        a[target] += float(np.real(values[0]))
        da[target] += float(np.real(slopes[0]))
        if nd:
            ja[target] += np.real(values[1:])
            jda[target] += np.real(slopes[1:])
    return a, da, ja, jda


def state_weight_value_jacobian(graph, point):
    point = np.asarray(point, dtype=float)
    n_modes = graph.n_modes
    return evaluate_state_stable_with_jacobians(
        graph,
        float(point[-1]),
        a0=point[:n_modes],
        operating=point[n_modes:-1],
        weight_derivatives=True,
    )
