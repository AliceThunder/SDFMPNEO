from __future__ import annotations

import numpy as np
import scipy.linalg

from .realization import AnalyticRealization, CompiledRealizationGraph


def _compile_with_operating_derivatives(graph, *, a0, operating, weight_derivatives=False):
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
    return _compile_with_operating_derivatives(graph, a0=a0, operating=operating)[0]


def evaluate_parametric_stable_with_jacobians(graph, t, *, a0, operating, weight_derivatives=False):
    """Stable values and exact operating sensitivities, also at resonance.

    Static parameters enter the realization's initial vector b. Product-rule
    derivatives of b share the same matrix exponential as the state.
    """
    if t < 0 or not np.isfinite(t):
        raise ValueError("time must be finite and non-negative")
    compiled, db = _compile_with_operating_derivatives(
        graph, a0=a0, operating=operating, weight_derivatives=weight_derivatives)
    a, da, ja, jda = [], [], [], []
    for r, jac in zip(compiled.mode_realizations, db):
        x = scipy.linalg.expm(r.A * t) @ np.column_stack([r.b, jac])
        values = np.real(r.c @ x)
        slopes = np.real(r.c @ (r.A @ x))
        a.append(values[0]); da.append(slopes[0])
        ja.append(values[1:]); jda.append(slopes[1:])
    return tuple(np.asarray(v) for v in (a, da, ja, jda))


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
