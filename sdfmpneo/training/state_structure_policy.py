from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdfmpneo.analytic.state_graph import (
    clone_state_graph,
    response_source_count,
    response_sources,
    split_response_source,
)
from sdfmpneo.analytic.state_realization import compile_state_realization
from .max_residual_runtime import hard_point_weights, max_first_accept


@dataclass(frozen=True)
class StructuralAction:
    kind: str
    graph: object
    detail: dict


def residual_norms(records) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(value.residual) for value in records], dtype=float
    )


def dynamic_response_parent(graph, parents) -> str | None:
    names = {node.name for node in graph.response_nodes}
    dynamic = [name for name in parents if name in names]
    if len(dynamic) > 1:
        return None
    return None if not dynamic else dynamic[0]


def state_family(graph, target: int, parents):
    return int(target), dynamic_response_parent(graph, parents)


def source_active(graph, target: int, parents) -> bool:
    key = tuple(parents)
    for node in graph.response_nodes:
        if node.target_mode != int(target):
            continue
        if any(source.parents == key for source in response_sources(graph, node)):
            return True
    return False


def family_nodes(graph, target: int, parents):
    family = state_family(graph, target, parents)
    out = []
    for node in graph.response_nodes:
        sources = response_sources(graph, node)
        node_family = (
            node.target_mode,
            dynamic_response_parent(graph, sources[0].parents),
        )
        if node_family == family:
            out.append(node)
    return out


def next_state_name(graph, prefix="response") -> str:
    known = set(graph.known_names())
    index = len(graph.response_nodes)
    while f"{prefix}_{index}" in known:
        index += 1
    return f"{prefix}_{index}"


def replace_parent_once(parents, old, new):
    values = list(parents)
    matches = [index for index, value in enumerate(values) if value == old]
    if len(matches) != 1:
        raise ValueError(
            "split candidate requires exactly one response-parent occurrence"
        )
    values[matches[0]] = new
    return tuple(values)


def candidate_dimension(graph, parents, record) -> int:
    compiled = compile_state_realization(
        graph,
        a0=np.asarray(record.initial, dtype=float),
        operating=np.asarray(record.u, dtype=float),
    )
    dimension = 1
    for parent in parents:
        dimension *= compiled.node_realizations[parent].dimension
    return int(dimension + 1)


def direct_actions(
    graph,
    target,
    parents,
    weight,
    *,
    max_nodes,
    max_realization_dimension,
    record,
):
    parents = tuple(parents)
    if source_active(graph, target, parents):
        return []
    try:
        dimension = candidate_dimension(graph, parents, record)
    except (KeyError, ValueError, FloatingPointError):
        return []
    if dimension > int(max_realization_dimension):
        return []

    existing = family_nodes(graph, target, parents)
    actions = []
    if existing:
        # Same dynamic role: enrich, do not grow a redundant state merely because
        # another source column is required.
        for node in existing:
            trial = clone_state_graph(graph)
            trial.enrich_response_state(node.name, parents, weight)
            actions.append(
                StructuralAction(
                    "enrich",
                    trial,
                    {
                        "state": node.name,
                        "target_mode": int(target),
                        "parents": parents,
                    },
                )
            )
        return actions

    if len(graph.response_nodes) >= int(max_nodes):
        return []
    trial = clone_state_graph(graph)
    name = next_state_name(trial)
    trial.add_response_state(name, int(target), ((parents, weight),))
    return [
        StructuralAction(
            "grow",
            trial,
            {"state": name, "target_mode": int(target), "parents": parents},
        )
    ]


def split_actions(graph, target, parents, weight, *, config, record):
    dynamic = dynamic_response_parent(graph, parents)
    if dynamic is None or response_source_count(graph, dynamic) < 2:
        return []

    actions = []
    for source_index in range(response_source_count(graph, dynamic)):
        split_name = next_state_name(graph, prefix=f"{dynamic}_split")
        try:
            split_graph = split_response_source(
                graph, dynamic, source_index, split_name
            )
            # Exact split propagation can clone several descendants. Those are
            # real dynamic states and therefore consume the existing state budget.
            if len(split_graph.response_nodes) >= int(config.max_nodes):
                continue
            split_parents = replace_parent_once(parents, dynamic, split_name)
            candidates = direct_actions(
                split_graph,
                target,
                split_parents,
                weight,
                max_nodes=config.max_nodes,
                max_realization_dimension=config.max_realization_dimension,
                record=record,
            )
            for action in candidates:
                actions.append(
                    StructuralAction(
                        "split+" + action.kind,
                        action.graph,
                        {
                            **action.detail,
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
        except (ValueError, KeyError, FloatingPointError):
            continue
    return actions


def candidate_actions(graph, target, parents, weight, *, config, record):
    return direct_actions(
        graph,
        target,
        parents,
        weight,
        max_nodes=config.max_nodes,
        max_realization_dimension=config.max_realization_dimension,
        record=record,
    ) + split_actions(
        graph, target, parents, weight, config=config, record=record
    )


def select_candidate_action(
    graph,
    field,
    points,
    records,
    config,
    target,
    parents,
    initial_weight,
    *,
    monitor=None,
):
    """Choose Enrich/Grow/Split by actual max-residual improvement.

    Candidate tangent score only proposes a column. The structural action itself
    is selected by evaluating the true nonlinear residual, so no K or empirical
    enrich-vs-grow threshold is introduced.
    """
    from . import research as r

    frozen_weights = hard_point_weights(records, config.residual_tolerance)
    old_norms = residual_norms(records)
    weight = float(initial_weight)
    sample_record = records[0]
    for _ in range(24):
        best = None
        for action in candidate_actions(
            graph,
            target,
            parents,
            weight,
            config=config,
            record=sample_record,
        ):
            try:
                trial_records = r._evaluate(
                    action.graph, field, points, monitor=monitor
                )
                norms = residual_norms(trial_records)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if not max_first_accept(
                old_norms,
                norms,
                config.residual_tolerance,
                frozen_weights,
            ):
                continue
            key = (
                float(np.max(norms, initial=0.0)),
                float(np.dot(frozen_weights, norms * norms)),
                len(action.graph.response_nodes),
                action.kind,
            )
            if best is None or key < best[0]:
                best = (key, action, trial_records)
        if best is not None:
            return best[1], best[2]
        weight *= 0.5
    return None, None
