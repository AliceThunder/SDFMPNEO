from __future__ import annotations

import numpy as np

from sdfmpneo.analytic.realization import AnalyticRealization
from sdfmpneo.analytic.state_graph import response_sources, split_response_source
from sdfmpneo.analytic.state_realization import compile_state_realization
from .late_stage_batch import _candidate_source_response
from .max_residual_runtime import hard_point_weights
from .parallel_runtime import _ordered_map
from .state_structure_policy import (
    dynamic_response_parent,
    next_state_name,
    replace_parent_once,
    residual_norms,
)
from .state_search_runtime import (
    _MIN_PREDICTED_RELATIVE_GAIN,
    _max_aligned_target,
)


# Split cardinality stays unbounded in the model. This is only the number of
# expensive graph-duplication trials retained after exact tangent screening.
_MAX_EXACT_SPLIT_TRIALS = 3

_CACHE_GRAPH = None
_CACHE_RECORDS = None
_CACHE_COMPILED = None
_CACHE_ISOLATED = {}


def _reset_cache(graph, records):
    global _CACHE_GRAPH, _CACHE_RECORDS, _CACHE_COMPILED, _CACHE_ISOLATED
    if graph is _CACHE_GRAPH and records is _CACHE_RECORDS:
        return
    _CACHE_GRAPH = graph
    _CACHE_RECORDS = records
    _CACHE_COMPILED = None
    _CACHE_ISOLATED = {}


def _compiled_for(graph, records, monitor=None):
    global _CACHE_COMPILED
    _reset_cache(graph, records)
    if _CACHE_COMPILED is None:
        _CACHE_COMPILED = _ordered_map(
            lambda record: compile_state_realization(
                graph, a0=record.initial, operating=record.u
            ),
            records,
            monitor=monitor,
        )
    return _CACHE_COMPILED


def _isolated_source_response(graph, compiled, node, source):
    term = AnalyticRealization.constant(source.weight)
    for parent in source.parents:
        term = term.product(compiled.node_realizations[parent])
    return term.response(graph.lambdas[node.target_mode])


def _isolated_for_dynamic(graph, records, compiled, dynamic_node):
    _reset_cache(graph, records)
    cached = _CACHE_ISOLATED.get(dynamic_node.name)
    if cached is not None:
        return cached
    sources = response_sources(graph, dynamic_node)
    values = tuple(
        tuple(
            _isolated_source_response(graph, realization, dynamic_node, source)
            for realization in compiled
        )
        for source in sources
    )
    _CACHE_ISOLATED[dynamic_node.name] = values
    return values


def _split_source_score(
    graph,
    records,
    compiled,
    isolated,
    dynamic_node,
    parents,
    target,
    point_weights,
):
    old_sq = residual_norms(records) ** 2
    linear = np.diag(np.asarray(graph.lambdas, dtype=float))
    correlation = np.zeros(len(records), dtype=float)
    tangent_norm2 = np.zeros(len(records), dtype=float)

    for index, (record, realization) in enumerate(zip(records, compiled)):
        candidate_source = AnalyticRealization.constant(1.0)
        used_dynamic = False
        for parent in parents:
            if parent == dynamic_node.name:
                if used_dynamic:
                    raise ValueError(
                        "split candidate requires exactly one response-parent occurrence"
                    )
                right = isolated[index]
                used_dynamic = True
            else:
                right = realization.node_realizations[parent]
            candidate_source = candidate_source.product(right)
        if not used_dynamic:
            raise ValueError("split candidate does not use the selected dynamic state")

        psi, h = _candidate_source_response(
            candidate_source, graph.lambdas, record.time
        )
        tangent = np.zeros(graph.n_modes, dtype=float)
        tangent[target] = psi
        tangent -= (record.J + linear)[:, target] * h[target]
        correlation[index] = float(record.residual @ tangent)
        tangent_norm2[index] = float(tangent @ tangent)

    return _max_aligned_target(
        old_sq, correlation, tangent_norm2, point_weights
    )


def screened_split_proposals(
    graph,
    records,
    config,
    target,
    parents,
    point_weights,
    *,
    monitor=None,
):
    """Screen source-specific Split tangents before duplicating any DAG branch.

    Under the production one-response-parent rule, isolating source s from an
    aggregate state gives an exact scalar realization h_s. The candidate tangent
    using h_s can therefore be scored without first materializing the
    function-preserving descendant duplication. Compiled current-graph and
    isolated-source realizations are cached for the whole candidate-search
    iteration. Only the best few source choices pay the graph construction and
    nonlinear full-point evaluation cost.
    """
    dynamic = dynamic_response_parent(graph, parents)
    if not isinstance(dynamic, str):
        return []
    dynamic_node = next(
        (node for node in graph.response_nodes if node.name == dynamic), None
    )
    if dynamic_node is None:
        return []
    sources = response_sources(graph, dynamic_node)
    if len(sources) < 2:
        return []

    compiled = _compiled_for(graph, records, monitor=monitor)
    isolated = _isolated_for_dynamic(graph, records, compiled, dynamic_node)
    weights = np.asarray(point_weights, dtype=float)
    if weights.shape != (len(records),):
        weights = hard_point_weights(records, config.residual_tolerance)

    scored = []
    for source_index in range(len(sources)):
        try:
            relative, alpha = _split_source_score(
                graph,
                records,
                compiled,
                isolated[source_index],
                dynamic_node,
                tuple(parents),
                int(target),
                weights,
            )
        except (ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if (
            relative >= _MIN_PREDICTED_RELATIVE_GAIN
            and np.isfinite(alpha)
            and alpha != 0.0
        ):
            scored.append((float(relative), int(source_index), float(alpha)))

    scored.sort(key=lambda item: item[0], reverse=True)
    proposals = []
    for _, source_index, alpha in scored[:_MAX_EXACT_SPLIT_TRIALS]:
        split_name = next_state_name(graph, prefix=f"{dynamic}_split")
        try:
            split_graph = split_response_source(
                graph, dynamic, source_index, split_name
            )
            if len(split_graph.response_nodes) > int(config.max_nodes):
                continue
            split_parents = replace_parent_once(
                tuple(parents), dynamic, split_name
            )
            proposals.append(
                (
                    split_graph,
                    tuple(split_parents),
                    float(alpha),
                    {
                        "split_state": dynamic,
                        "split_source_index": source_index,
                        "split_child": split_name,
                        "split_added_states": (
                            len(split_graph.response_nodes)
                            - len(graph.response_nodes)
                        ),
                    },
                )
            )
        except (ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return proposals


def install_screened_split_policy() -> None:
    # bounded_max_aligned_select_candidate_action resolves _split_proposals from
    # its own module globals at call time, so replacing that binding is enough.
    from . import state_search_runtime as search_runtime

    search_runtime._split_proposals = screened_split_proposals


__all__ = ["screened_split_proposals", "install_screened_split_policy"]
