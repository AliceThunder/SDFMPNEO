from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyticStateSource:
    parents: tuple[str, ...]
    weight: complex

    @classmethod
    def make(cls, parents, weight):
        return cls(tuple(str(value) for value in parents), complex(weight))


def _ensure_sources(graph) -> dict[str, tuple[AnalyticStateSource, ...]]:
    mapping = getattr(graph, "_sdfmpneo_state_sources", None)
    if mapping is None:
        mapping = {
            node.name: (AnalyticStateSource.make(node.parents, node.weight),)
            for node in graph.response_nodes
        }
        graph._sdfmpneo_state_sources = mapping
    else:
        for node in graph.response_nodes:
            mapping.setdefault(
                node.name, (AnalyticStateSource.make(node.parents, node.weight),)
            )
    return mapping


def response_sources(graph, node_or_name) -> tuple[AnalyticStateSource, ...]:
    name = node_or_name if isinstance(node_or_name, str) else node_or_name.name
    mapping = _ensure_sources(graph)
    if name not in mapping:
        raise KeyError(name)
    return tuple(mapping[name])


def response_source_count(graph, node_or_name) -> int:
    return len(response_sources(graph, node_or_name))


def has_multi_source_states(graph) -> bool:
    return any(response_source_count(graph, node) > 1 for node in graph.response_nodes)


def weight_parameter_count(graph) -> int:
    return sum(response_source_count(graph, node) for node in graph.response_nodes)


def weight_parameter_map(graph) -> tuple[tuple[str, int], ...]:
    return tuple(
        (node.name, source_index)
        for node in graph.response_nodes
        for source_index in range(response_source_count(graph, node))
    )


def _node_index(graph, name: str) -> int:
    for index, node in enumerate(graph.response_nodes):
        if node.name == name:
            return index
    raise KeyError(name)


def _refresh_primary_node(graph, name: str) -> None:
    from .parametric import ParametricResponseNode

    index = _node_index(graph, name)
    node = graph.response_nodes[index]
    source = response_sources(graph, name)[0]
    # The legacy dataclass remains a compact compatibility view of the first
    # active column. Source-aware code always reads graph.response_sources().
    graph.response_nodes[index] = ParametricResponseNode(
        node.name, node.target_mode, source.parents, source.weight
    )
    graph._compiled = None


def response_parent_for_source(graph, source: AnalyticStateSource) -> str | None:
    response_names = {node.name for node in graph.response_nodes}
    values = [parent for parent in source.parents if parent in response_names]
    if len(values) > 1:
        raise ValueError(
            "analytic-state sources support at most one response-state parent"
        )
    return None if not values else values[0]


def source_family_parent(graph, sources) -> str | None:
    parents = {response_parent_for_source(graph, source) for source in sources}
    if len(parents) > 1:
        raise ValueError(
            "all sources in one analytic state must share the same response-state parent"
        )
    return next(iter(parents)) if parents else None


def _validate_sources(graph, target_mode: int, sources) -> tuple[AnalyticStateSource, ...]:
    if not 0 <= int(target_mode) < graph.n_modes:
        raise ValueError("target_mode out of range")
    known = set(graph.known_names())
    normalized = tuple(
        value
        if isinstance(value, AnalyticStateSource)
        else AnalyticStateSource.make(value[0], value[1])
        for value in sources
    )
    if not normalized:
        raise ValueError("an analytic state requires at least one source column")
    for source in normalized:
        missing = [parent for parent in source.parents if parent not in known]
        if missing:
            raise ValueError(f"unknown parent nodes: {missing}")
    source_family_parent(graph, normalized)
    return normalized


def add_response_state(graph, name: str, target_mode: int, sources) -> None:
    normalized = _validate_sources(graph, target_mode, sources)
    if name in graph.known_names():
        raise ValueError(f"duplicate node name: {name}")
    original = getattr(type(graph), "_sdfmpneo_original_add_product_response")
    first = normalized[0]
    original(graph, name, int(target_mode), first.parents, first.weight)
    _ensure_sources(graph)[name] = normalized
    _refresh_primary_node(graph, name)


def enrich_response_state(graph, name: str, parents, weight: complex) -> None:
    node = graph.response_nodes[_node_index(graph, name)]
    new_source = AnalyticStateSource.make(parents, weight)
    _validate_sources(graph, node.target_mode, (new_source,))
    values = list(response_sources(graph, name))
    existing_parent = source_family_parent(graph, values)
    new_parent = response_parent_for_source(graph, new_source)
    if existing_parent != new_parent:
        raise ValueError(
            "source enrichment cannot change an analytic state's dynamic family"
        )
    for index, source in enumerate(values):
        if source.parents != new_source.parents:
            continue
        combined = source.weight + new_source.weight
        if combined == 0:
            if len(values) == 1:
                raise ValueError("cannot remove the only source from an analytic state")
            values.pop(index)
        else:
            values[index] = AnalyticStateSource(source.parents, combined)
        break
    else:
        values.append(new_source)
    _ensure_sources(graph)[name] = tuple(values)
    _refresh_primary_node(graph, name)


def replace_source_weights(graph, weights) -> None:
    import numpy as np

    values = np.asarray(weights, dtype=np.complex128).reshape(-1)
    mapping = weight_parameter_map(graph)
    if values.size != len(mapping):
        raise ValueError("weight vector does not match analytic state source count")
    table = _ensure_sources(graph)
    offset = 0
    for node in list(graph.response_nodes):
        sources = list(response_sources(graph, node))
        for index in range(len(sources)):
            sources[index] = AnalyticStateSource(
                sources[index].parents, complex(values[offset])
            )
            offset += 1
        table[node.name] = tuple(sources)
        _refresh_primary_node(graph, node.name)
    graph._compiled = None


def clone_state_graph(graph):
    out = type(graph)(graph.lambdas.copy(), graph.operating_names)
    for node in graph.response_nodes:
        add_response_state(
            out, node.name, node.target_mode, response_sources(graph, node)
        )
    return out


def _replace_parent_once(parents: tuple[str, ...], old: str, new: str):
    values = list(parents)
    positions = [index for index, value in enumerate(values) if value == old]
    if len(positions) != 1:
        raise ValueError(
            "state splitting requires exactly one occurrence of the dynamic parent"
        )
    values[positions[0]] = new
    return tuple(values)


def split_response_source(graph, name: str, source_index: int, new_name: str):
    """Function-preserving split with exact recursive downstream propagation.

    Because production sources contain at most one response-state parent, each
    downstream product is linear in the split parent. Every affected descendant
    is duplicated along the new branch, preserving the represented function
    exactly while making the selected source independently addressable.
    """
    original_position = _node_index(graph, name)
    original_node = graph.response_nodes[original_position]
    original_sources = list(response_sources(graph, name))
    if len(original_sources) < 2:
        raise ValueError("a single-source state cannot be split")
    if not 0 <= int(source_index) < len(original_sources):
        raise IndexError("source_index out of range")
    if new_name in graph.known_names():
        raise ValueError(f"duplicate node name: {new_name}")
    moved = original_sources.pop(int(source_index))

    out = type(graph)(graph.lambdas.copy(), graph.operating_names)
    split_map: dict[str, str] = {}
    used_names = set(graph.known_names()) | {new_name}

    def branch_name(base: str) -> str:
        index = 0
        while True:
            candidate = f"{base}_split_{index}"
            if candidate not in used_names and candidate not in out.known_names():
                used_names.add(candidate)
                return candidate
            index += 1

    for position, node in enumerate(graph.response_nodes):
        sources = response_sources(graph, node)
        if position < original_position:
            add_response_state(out, node.name, node.target_mode, sources)
            continue
        if position == original_position:
            add_response_state(out, node.name, node.target_mode, tuple(original_sources))
            add_response_state(out, new_name, original_node.target_mode, (moved,))
            split_map[node.name] = new_name
            continue

        family_parent = source_family_parent(graph, sources)
        add_response_state(out, node.name, node.target_mode, sources)
        if family_parent is None or family_parent not in split_map:
            continue
        replacement = split_map[family_parent]
        branch_sources = []
        for source in sources:
            if source.parents.count(family_parent) != 1:
                raise ValueError(
                    "split propagation requires one occurrence of the dynamic parent"
                )
            branch_sources.append(
                AnalyticStateSource(
                    _replace_parent_once(source.parents, family_parent, replacement),
                    source.weight,
                )
            )
        clone_name = branch_name(node.name)
        add_response_state(out, clone_name, node.target_mode, tuple(branch_sources))
        split_map[node.name] = clone_name
    return out


def compile_state_series(graph):
    from .parametric import (
        CompiledParametricAnalyticGraph,
        ParametricAnalyticSeries,
        solve_parametric_response_series,
    )

    if graph._compiled is not None:
        return graph._compiled
    n_parameters = graph.n_parameters
    nodes = {}
    modes = [
        ParametricAnalyticSeries.zero(graph.n_modes, n_parameters)
        for _ in range(graph.n_modes)
    ]
    for index, name in enumerate(graph.initial_names):
        series = ParametricAnalyticSeries.initial_decay(
            graph.n_modes, n_parameters, index, index
        )
        nodes[name] = series
        modes[index] = modes[index] + series
    for index, name in enumerate(graph.operating_names):
        nodes[name] = ParametricAnalyticSeries.static_parameter(
            graph.n_modes, n_parameters, graph.n_modes + index
        )
    for node in graph.response_nodes:
        source_sum = ParametricAnalyticSeries.zero(graph.n_modes, n_parameters)
        for source_spec in response_sources(graph, node):
            term = ParametricAnalyticSeries.constant(
                graph.n_modes, n_parameters, source_spec.weight
            )
            for parent in source_spec.parents:
                term = term * nodes[parent]
            source_sum = source_sum + term
        response = solve_parametric_response_series(
            source_sum, node.target_mode, graph.lambdas
        )
        nodes[node.name] = response
        modes[node.target_mode] = modes[node.target_mode] + response
    graph._compiled = CompiledParametricAnalyticGraph(
        lambdas=graph.lambdas.copy(),
        initial_names=graph.initial_names,
        operating_names=graph.operating_names,
        node_series=dict(nodes),
        mode_series=tuple(modes),
    )
    return graph._compiled
