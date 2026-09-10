from __future__ import annotations

import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.state_graph import has_multi_source_states, response_sources


_ORIGINAL_STATE_LINEARIZATION = None


def _has_dynamic_source_parent(graph) -> bool:
    response_names = {node.name for node in graph.response_nodes}
    return any(
        parent in response_names
        for node in graph.response_nodes
        for source in response_sources(graph, node)
        for parent in source.parents
    )


def scalarize_independent_sources(graph):
    """Expand independent multi-source states only for native GN evaluation.

    When no source depends on another response state, every source column is a
    response with the same physical pole as its aggregate target.  Splitting the
    aggregate into scalar nodes is therefore exactly function preserving.  The
    scalar node order is node-major/source-minor, identical to
    weight_parameter_map(), so the native Jacobian columns retain the optimizer's
    source-weight ordering.
    """
    if not has_multi_source_states(graph) or _has_dynamic_source_parent(graph):
        return None

    scalar = ParametricAnalyticEvolutionGraph(
        np.asarray(graph.lambdas, dtype=float).copy(), graph.operating_names
    )
    known = set(scalar.known_names())
    counter = 0
    for node in graph.response_nodes:
        for source in response_sources(graph, node):
            while True:
                name = f"__source_{counter}"
                counter += 1
                if name not in known:
                    break
            known.add(name)
            scalar.add_product_response(
                name, int(node.target_mode), tuple(source.parents), source.weight
            )
    return scalar


def install_independent_state_native_linearization() -> None:
    """Keep the C++ one-source Jacobian kernel for a coalesced pure seed."""
    global _ORIGINAL_STATE_LINEARIZATION
    from . import residual_state_runtime as runtime
    from .max_residual_runtime import _linearization as native_or_sparse_linearization

    if _ORIGINAL_STATE_LINEARIZATION is not None:
        return
    _ORIGINAL_STATE_LINEARIZATION = runtime._state_linearization

    def state_linearization(graph, field, points, monitor=None):
        scalar = scalarize_independent_sources(graph)
        if scalar is not None:
            residual, jacobian, native_threads = native_or_sparse_linearization(
                scalar, field, points, monitor=monitor
            )
            if jacobian.shape[-1] != graph.weight_parameter_count:
                raise RuntimeError(
                    "scalarized source Jacobian no longer matches active source weights"
                )
            return residual, jacobian, native_threads
        return _ORIGINAL_STATE_LINEARIZATION(
            graph, field, points, monitor=monitor
        )

    runtime._state_linearization = state_linearization


__all__ = [
    "scalarize_independent_sources",
    "install_independent_state_native_linearization",
]
