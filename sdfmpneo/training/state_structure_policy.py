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
from .max_residual_runtime import (
    hard_point_weights,
    max_first_accept,
    weighted_score_parent_batch,
)


@dataclass(frozen=True)
class StructuralAction:
    kind: str
    graph: object
    detail: dict


def residual_norms(records) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(value.residual) for value in records], dtype=float
    )


def dynamic_response_parent(graph, parents):
    names = {node.name for node in graph.response_nodes}
    dynamic = [name for name in parents if name in names]
    if len(dynamic) > 1:
        # Historical graphs may have a larger response-parent budget. Keep that
        # family distinct and never merge it with the no-dynamic-parent family.
        return tuple(dynamic)
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
    if existing:
        actions = []
        for node in existing:
            trial = clone_state_graph(graph)
            try:
                trial.enrich_response_state(node.name, parents, weight)
            except ValueError:
                continue
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


def _split_proposals(
    graph,
    records,
    config,
    target,
    parents,
    point_weights,
    *,
    monitor=None,
):
    """Return exact split graphs with candidate-specific tangent weights."""
    from . import late_stage_runtime as late

    dynamic = dynamic_response_parent(graph, parents)
    if not isinstance(dynamic, str) or response_source_count(graph, dynamic) < 2:
        return []

    proposals = []
    for source_index in range(response_source_count(graph, dynamic)):
        split_name = next_state_name(graph, prefix=f"{dynamic}_split")
        try:
            split_graph = split_response_source(
                graph, dynamic, source_index, split_name
            )
            # Equality is allowed: an exact split can consume the final state slot
            # and still be followed by Enrich, which does not create another state.
            if len(split_graph.response_nodes) > int(config.max_nodes):
                continue
            split_parents = replace_parent_once(parents, dynamic, split_name)
            compiled = [
                compile_state_realization(
                    split_graph, a0=record.initial, operating=record.u
                )
                for record in records
            ]
            plan = late._parent_plan(split_graph, split_parents)
            inner, norm2 = weighted_score_parent_batch(
                split_graph,
                records,
                compiled,
                [plan],
                point_weights=point_weights,
                monitor=monitor,
            )[0]
            if norm2[target] <= 0 or not np.isfinite(
                norm2[target] + inner[target]
            ):
                continue
            split_weight = -inner[target] / norm2[target]
            proposals.append(
                (
                    split_graph,
                    tuple(split_parents),
                    float(split_weight),
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


def _best_line_search_action(
    base_graph,
    field,
    points,
    old_records,
    config,
    target,
    parents,
    initial_weight,
    frozen_weights,
    *,
    prefix=None,
    detail=None,
    monitor=None,
):
    from . import research as r

    old_norms = residual_norms(old_records)
    sample_record = old_records[0]
    weight = float(initial_weight)
    for _ in range(24):
        best = None
        actions = direct_actions(
            base_graph,
            target,
            parents,
            weight,
            max_nodes=config.max_nodes,
            max_realization_dimension=config.max_realization_dimension,
            record=sample_record,
        )
        for action in actions:
            if prefix is not None:
                action = StructuralAction(
                    prefix + "+" + action.kind,
                    action.graph,
                    {**(detail or {}), **action.detail},
                )
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
            return best
        weight *= 0.5
    return None


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
    """Choose Enrich/Grow/Split using actual nonlinear max residual.

    Direct and split branches each receive their own exact residual-tangent
    coefficient before line search. Therefore Split is not biased by the tangent
    of the pre-split aggregate parent, and no empirical structural threshold or
    fixed K is needed.
    """
    frozen_weights = hard_point_weights(records, config.residual_tolerance)
    candidates = []

    direct = _best_line_search_action(
        graph,
        field,
        points,
        records,
        config,
        target,
        parents,
        initial_weight,
        frozen_weights,
        monitor=monitor,
    )
    if direct is not None:
        candidates.append(direct)

    for split_graph, split_parents, split_weight, detail in _split_proposals(
        graph,
        records,
        config,
        target,
        parents,
        frozen_weights,
        monitor=monitor,
    ):
        candidate = _best_line_search_action(
            split_graph,
            field,
            points,
            records,
            config,
            target,
            split_parents,
            split_weight,
            frozen_weights,
            prefix="split",
            detail=detail,
            monitor=monitor,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]
