from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .long_time import realization_observation_action

from .realization import AnalyticRealization, CompiledRealizationGraph


@dataclass(frozen=True)
class CompiledParametricNodeGraph:
    """Lightweight exact DAG compile for hot paths that never use mode direct sums."""
    lambdas: np.ndarray
    node_realizations: dict[str, AnalyticRealization]
    source_realizations: dict[str, AnalyticRealization]


def _validated_inputs(graph, a0, operating):
    initial = np.asarray(a0, dtype=float)
    u = np.asarray(operating, dtype=float)
    if initial.shape != (graph.n_modes,):
        raise ValueError("initial coordinate dimension mismatch")
    if u.shape != (len(graph.operating_names),):
        raise ValueError("operating parameter dimension mismatch")
    return initial, u


def _compile_value_components(graph, *, a0, operating, build_modes):
    initial, u = _validated_inputs(graph, a0, operating)
    nodes: dict[str, AnalyticRealization] = {}
    sources: dict[str, AnalyticRealization] = {}
    modes = [AnalyticRealization.zero() for _ in range(graph.n_modes)] if build_modes else None

    for i, name in enumerate(graph.initial_names):
        realization = AnalyticRealization.decay(graph.lambdas[i], initial[i])
        nodes[name] = realization
        if modes is not None:
            modes[i] = modes[i].add(realization)

    for j, name in enumerate(graph.operating_names):
        nodes[name] = AnalyticRealization.constant(u[j])

    for node in graph.response_nodes:
        source = AnalyticRealization.constant(node.weight)
        for parent in node.parents:
            source = source.product(nodes[parent])
        response = source.response(graph.lambdas[node.target_mode])
        sources[node.name] = source
        nodes[node.name] = response
        if modes is not None:
            modes[node.target_mode] = modes[node.target_mode].add(response)

    return nodes, sources, modes


def _compile_value_only(graph, *, a0, operating):
    """Instantiate the exact full DAG without derivative arrays that are not requested."""
    nodes, sources, modes = _compile_value_components(
        graph, a0=a0, operating=operating, build_modes=True
    )
    return CompiledRealizationGraph(
        lambdas=np.asarray(graph.lambdas, dtype=float).copy(),
        node_realizations=nodes,
        source_realizations=sources,
        mode_realizations=tuple(modes),
    )


def compile_parametric_nodes(graph, *, a0: np.ndarray, operating: np.ndarray) -> CompiledParametricNodeGraph:
    """Compile only node/source realizations for residual, candidate and sparse-Jacobian hot paths."""
    nodes, sources, _ = _compile_value_components(
        graph, a0=a0, operating=operating, build_modes=False
    )
    return CompiledParametricNodeGraph(
        lambdas=np.asarray(graph.lambdas, dtype=float).copy(),
        node_realizations=nodes,
        source_realizations=sources,
    )


def _compile_with_operating_derivatives(graph, *, a0, operating, weight_derivatives=False):
    """Instantiate a parametric analytic DAG as an exact state-space realization.

    Static initial/operating nodes are evaluated only at the requested parameter
    point; all subsequent multiplication and response operations are represented
    exactly by Kronecker sums and augmented linear states. Hence coalescing decay
    rates require no near-resonance threshold.
    """

    initial, u = _validated_inputs(graph, a0, operating)
    nd = len(graph.response_nodes) if weight_derivatives else u.size
    nodes: dict[str, AnalyticRealization] = {}
    derivatives = {}
    sources: dict[str, AnalyticRealization] = {}
    modes = [AnalyticRealization.zero() for _ in range(graph.n_modes)]
    mode_derivatives = [np.zeros((1, nd), complex) for _ in range(graph.n_modes)]

    for i, name in enumerate(graph.initial_names):
        realization = AnalyticRealization.decay(graph.lambdas[i], initial[i])
        nodes[name] = realization
        derivatives[name] = np.zeros((1, nd), complex)
        modes[i] = modes[i].add(realization)
        mode_derivatives[i] = np.vstack([mode_derivatives[i], derivatives[name]])

    for j, name in enumerate(graph.operating_names):
        nodes[name] = AnalyticRealization.constant(u[j])
        derivatives[name] = (np.zeros((1, nd), complex) if weight_derivatives
                             else np.eye(u.size, dtype=complex)[j:j+1])

    for index, node in enumerate(graph.response_nodes):
        source = AnalyticRealization.constant(node.weight)
        db = np.zeros((1, nd), complex)
        if weight_derivatives:
            db[0, index] = 1.
        for parent in node.parents:
            right = nodes[parent]
            db = (np.einsum('ik,j->ijk', db, right.b)
                  + np.einsum('i,jk->ijk', source.b, derivatives[parent])).reshape(
                      source.dimension * right.dimension, nd)
            source = source.product(right)
        response = source.response(graph.lambdas[node.target_mode])
        sources[node.name] = source
        nodes[node.name] = response
        derivatives[node.name] = np.vstack([db, np.zeros((1, nd), complex)])
        modes[node.target_mode] = modes[node.target_mode].add(response)
        mode_derivatives[node.target_mode] = np.vstack([
            mode_derivatives[node.target_mode], derivatives[node.name]])

    return CompiledRealizationGraph(
        lambdas=np.asarray(graph.lambdas, dtype=float).copy(),
        node_realizations=nodes,
        source_realizations=sources,
        mode_realizations=tuple(modes),
    ), mode_derivatives


def compile_parametric_realization(graph, *, a0: np.ndarray, operating: np.ndarray) -> CompiledRealizationGraph:
    return _compile_value_only(graph, a0=a0, operating=operating)


def evaluate_parametric_stable_with_jacobians(graph, t, *, a0, operating, weight_derivatives=False):
    """Stable values and exact operating sensitivities, also at resonance.

    Static parameters enter the realization's initial vector b. Product-rule
    derivatives of b share the same analytic propagator observation rows.
    """
    if t < 0 or np.isnan(t):
        raise ValueError("time must be non-negative or positive infinity")
    compiled, db = _compile_with_operating_derivatives(
        graph, a0=a0, operating=operating, weight_derivatives=weight_derivatives)
    a, da, ja, jda = [], [], [], []
    # Mode realizations are direct sums. Evaluate each DAG node block separately
    # and cache only the output rows rather than complete d-by-d propagators.
    for mode, jac in enumerate(db):
        parts = [compiled.node_realizations[graph.initial_names[mode]]]
        parts += [compiled.node_realizations[node.name] for node in graph.response_nodes
                  if node.target_mode == mode]
        value = np.zeros(1+jac.shape[1])
        slope = np.zeros_like(value)
        offset = 1  # the identically zero initial direct-sum block
        for r in parts:
            block = jac[offset:offset+r.dimension]
            values, slopes = realization_observation_action(
                r.A, np.column_stack([r.b, block]), r.c, t
            )
            value += np.real(values)
            slope += np.real(slopes)
            offset += r.dimension
        a.append(value[0]); da.append(slope[0])
        ja.append(value[1:]); jda.append(slope[1:])
    return tuple(np.asarray(v) for v in (a, da, ja, jda))


def evaluate_parametric_stable(
    graph,
    t: float,
    *,
    a0: np.ndarray,
    operating: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if t < 0 or np.isnan(t):
        raise ValueError('time must be non-negative or positive infinity')
    compiled = compile_parametric_nodes(graph, a0=a0, operating=operating)
    a = np.zeros(graph.n_modes)
    da = np.zeros_like(a)
    for mode, name in enumerate(graph.initial_names):
        r = compiled.node_realizations[name]
        value, slope = realization_observation_action(r.A, r.b, r.c, t)
        a[mode] += np.real(value)
        da[mode] += np.real(slope)
    for node in graph.response_nodes:
        r = compiled.node_realizations[node.name]
        value, slope = realization_observation_action(r.A, r.b, r.c, t)
        a[node.target_mode] += np.real(value)
        da[node.target_mode] += np.real(slope)
    return a, da


def parametric_backend_consistency_defect(
    graph,
    t: float,
    *,
    a0: np.ndarray,
    operating: np.ndarray,
) -> float:
    fast_a, fast_da = graph.evaluate(t, a0=a0, operating=operating)
    stable_a, stable_da = evaluate_parametric_stable(
        graph,
        t,
        a0=a0,
        operating=operating,
    )
    numerator = np.linalg.norm(np.concatenate([fast_a - stable_a, fast_da - stable_da]))
    denominator = max(1.0, float(np.linalg.norm(np.concatenate([stable_a, stable_da]))))
    return float(numerator / denominator)
