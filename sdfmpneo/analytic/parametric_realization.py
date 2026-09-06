from __future__ import annotations

import numpy as np

from .realization import AnalyticRealization, CompiledRealizationGraph


def compile_parametric_realization(graph, *, a0: np.ndarray, operating: np.ndarray) -> CompiledRealizationGraph:
    """Instantiate a parametric analytic DAG as an exact state-space realization.

    Static initial/operating nodes are evaluated only at the requested parameter
    point; all subsequent multiplication and response operations are represented
    exactly by Kronecker sums and augmented linear states. Hence coalescing decay
    rates require no near-resonance threshold.
    """

    initial = np.asarray(a0, dtype=float)
    u = np.asarray(operating, dtype=float)
    if initial.shape != (graph.n_modes,):
        raise ValueError("initial coordinate dimension mismatch")
    if u.shape != (len(graph.operating_names),):
        raise ValueError("operating parameter dimension mismatch")

    nodes: dict[str, AnalyticRealization] = {}
    sources: dict[str, AnalyticRealization] = {}
    modes = [AnalyticRealization.zero() for _ in range(graph.n_modes)]

    for i, name in enumerate(graph.initial_names):
        realization = AnalyticRealization.decay(graph.lambdas[i], initial[i])
        nodes[name] = realization
        modes[i] = modes[i].add(realization)

    for j, name in enumerate(graph.operating_names):
        nodes[name] = AnalyticRealization.constant(u[j])

    for node in graph.response_nodes:
        source = nodes[node.parents[0]]
        for parent in node.parents[1:]:
            source = source.product(nodes[parent])
        source = source.scaled(node.weight)
        response = source.response(graph.lambdas[node.target_mode])
        sources[node.name] = source
        nodes[node.name] = response
        modes[node.target_mode] = modes[node.target_mode].add(response)

    return CompiledRealizationGraph(
        lambdas=np.asarray(graph.lambdas, dtype=float).copy(),
        node_realizations=nodes,
        source_realizations=sources,
        mode_realizations=tuple(modes),
    )


def evaluate_parametric_stable(
    graph,
    t: float,
    *,
    a0: np.ndarray,
    operating: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return compile_parametric_realization(graph, a0=a0, operating=operating).evaluate(t)


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
