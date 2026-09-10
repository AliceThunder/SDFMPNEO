from __future__ import annotations

from dataclasses import replace
from itertools import combinations_with_replacement

import numpy as np

from sdfmpneo.analytic.realization import AnalyticRealization
from sdfmpneo.analytic.state_graph import response_source_count
from .parallel_runtime import _ordered_map
from .state_structure_policy import dynamic_response_parent

# Normal training keeps the cheap one-response-parent dictionary. These rescue
# budgets are activated only after that dictionary and its exact scalar/Split
# fallback have both failed to make progress.
_RESCUE_RESPONSE_STATE_LIMIT = 8
_RESCUE_MAX_BLOCK_SOURCES = 4
_RESCUE_REALIZATION_DIMENSION = 512
_RESCUE_DEGREE_ROUNDS = 2
_RESCUE_REGULARIZATION_PATH = (0.20, 0.08, 0.03)

_ORIGINAL_TRAIN = None
_RESCUE_RECORDS = None
_RESCUE_ATTEMPTED = False


def _rescue_response_names(graph):
    """Small deterministic state set that retains every thermal-mode anchor."""
    nodes = list(graph.response_nodes)
    if len(nodes) <= _RESCUE_RESPONSE_STATE_LIMIT:
        return tuple(node.name for node in nodes)

    chosen = []
    chosen_names = set()

    # Keep the richest state for every thermal target. Geometry seeding normally
    # places the aggregate physical response here, so cross-mode interactions are
    # never lost merely because many later correction states exist.
    for target in range(graph.n_modes):
        local = [node for node in nodes if int(node.target_mode) == target]
        if not local:
            continue
        best = max(
            local,
            key=lambda node: (
                response_source_count(graph, node),
                -nodes.index(node),
            ),
        )
        chosen.append(best.name)
        chosen_names.add(best.name)

    # Recent states encode the residual structures that normal search found most
    # useful immediately before the stall.
    for node in reversed(nodes):
        if len(chosen) >= _RESCUE_RESPONSE_STATE_LIMIT:
            break
        if node.name not in chosen_names:
            chosen.append(node.name)
            chosen_names.add(node.name)

    return tuple(chosen)


def _rescue_parent_pairs(graph):
    names = _rescue_response_names(graph)
    return tuple(combinations_with_replacement(names, 2))


def _pair_dimension(compiled, parents):
    if not compiled:
        return 1
    realization = compiled[0]
    value = 1
    for parent in parents:
        value *= realization.node_realizations[parent].dimension
    return int(value + 1)


def _pair_tangent_table(graph, records, compiled, parents):
    """Exact linearized residual tangents for one two-response source."""
    from . import late_stage_batch as batch

    n_modes = graph.n_modes
    identity = np.eye(n_modes, dtype=float)
    linear = np.diag(np.asarray(graph.lambdas, dtype=float))

    def one(pair):
        record, realization = pair
        source = AnalyticRealization.constant(1.0)
        for parent in parents:
            source = source.product(realization.node_realizations[parent])
        psi, h = batch._candidate_source_response(
            source, graph.lambdas, record.time
        )
        return identity * psi - (record.J + linear) * h[np.newaxis, :]

    return np.asarray(
        _ordered_map(one, zip(records, compiled)),
        dtype=float,
    )


def _interaction_operator(graph, records, compiled, point_weights, config):
    """Build only a tiny dense-in-candidates rescue operator.

    This path is intentionally separate from the normal admissible dictionary:
    it appears only after a proven structural stall and permits two response
    parents, which are required for second-order thermal-state interactions.
    """
    from . import block_sparse_search_runtime as block

    if len(graph.response_nodes) < 1 or not records:
        return None, [], {}

    limit = max(
        int(getattr(config, "max_realization_dimension", 64)),
        _RESCUE_REALIZATION_DIMENSION,
    )
    weights = np.asarray(point_weights, dtype=float)
    candidates = []
    blocks = []

    for parents in _rescue_parent_pairs(graph):
        if _pair_dimension(compiled, parents) > limit:
            continue
        table = _pair_tangent_table(graph, records, compiled, parents)
        for target in range(graph.n_modes):
            tangent = table[:, :, target]
            local_norm2 = np.sum(tangent * tangent, axis=1)
            norm2 = float(np.dot(weights, local_norm2))
            if not np.isfinite(norm2) or norm2 <= np.finfo(float).tiny:
                continue
            norm = float(np.sqrt(norm2))
            candidate = block._Candidate(
                -1,
                int(target),
                tuple(parents),
                (int(target), dynamic_response_parent(graph, parents)),
                ("interaction", tuple(parents)),
                norm,
            )
            index = len(candidates)
            candidates.append(candidate)
            blocks.append(
                block._OperatorBlock(
                    np.asarray([index], dtype=int),
                    np.ones((len(records), 1), dtype=float),
                    np.asarray([norm], dtype=float),
                    tangent,
                )
            )

    if not candidates:
        return None, [], {}
    operator = block._StructuredTangentOperator(
        records, weights, candidates, blocks
    )
    return operator, candidates, block._groups(candidates)


def _trim_proposal(proposal, count):
    if proposal is None:
        return None
    ranked = sorted(
        proposal["sources"],
        key=lambda item: (
            -abs(float(item["normalized_weight"])),
            item["target"],
            item["parents"],
        ),
    )
    sources = tuple(ranked[: max(1, int(count))])
    if not sources:
        return None
    return {**proposal, "sources": sources}


def _interaction_proposals(graph, records, compiled, point_weights, config):
    from . import block_sparse_search_runtime as block

    operator, candidates, groups = _interaction_operator(
        graph, records, compiled, point_weights, config
    )
    if operator is None:
        return ()

    residual = np.vstack(
        [np.asarray(record.residual, dtype=float) for record in records]
    )
    warm = None
    best = None
    for fraction in _RESCUE_REGULARIZATION_PATH:
        warm = block._sparse_group_solve(
            operator, residual, groups, fraction, initial=warm
        )
        proposal = block._proposal_from_z(
            graph,
            records,
            operator,
            candidates,
            groups,
            warm,
            config=config,
        )
        if proposal is not None and (
            best is None
            or proposal["relative_gain"] > best["relative_gain"]
        ):
            best = proposal

    if best is None:
        return ()

    maximum = min(_RESCUE_MAX_BLOCK_SOURCES, len(best["sources"]))
    sizes = []
    for value in (maximum, min(2, maximum), 1):
        if value not in sizes:
            sizes.append(value)
    return tuple(_trim_proposal(best, value) for value in sizes)


def _try_interaction_rescue(
    graph, field, points, records, config, *, monitor=None
):
    """Try a few jointly selected second-order state interactions."""
    from . import block_sparse_search_runtime as block

    if not block._BLOCK_COMPILED:
        return None, None
    point_weights = block._BLOCK_POINT_WEIGHTS
    if point_weights is None:
        return None, None

    # The state source itself may have a larger exact realization than the
    # normal fast dictionary. This is a rescue-only complexity allowance; the
    # dynamic-state count remains governed by max_nodes.
    rescue_config = replace(
        config,
        max_realization_dimension=max(
            int(config.max_realization_dimension),
            _RESCUE_REALIZATION_DIMENSION,
        ),
    )
    proposals = _interaction_proposals(
        graph,
        records,
        block._BLOCK_COMPILED,
        point_weights,
        rescue_config,
    )
    for proposal in proposals:
        if proposal is None:
            continue
        action, trial_records = block._try_block_action(
            graph,
            field,
            points,
            records,
            rescue_config,
            proposal,
            monitor=monitor,
        )
        if action is not None:
            detail = dict(action.detail)
            detail.update(
                {
                    "rescue": "two_response_interaction",
                    "normal_max_parent_responses": int(
                        config.max_parent_responses
                    ),
                }
            )
            return type(action)(
                "interaction_rescue", action.graph, detail
            ), trial_records
    return None, None


def convergence_rescue_select_candidate_action(
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
    """Normal sparse block -> exact fallback -> selective interaction rescue."""
    global _RESCUE_RECORDS, _RESCUE_ATTEMPTED
    from . import block_sparse_search_runtime as block

    if records is not _RESCUE_RECORDS:
        _RESCUE_RECORDS = records
        _RESCUE_ATTEMPTED = False

    if records is not block._BLOCK_RECORDS:
        return block.block_sparse_select_candidate_action(
            graph,
            field,
            points,
            records,
            config,
            target,
            parents,
            initial_weight,
            monitor=monitor,
        )

    if (
        block._BLOCK_PROPOSAL is not None
        and not block._BLOCK_ATTEMPTED
    ):
        block._BLOCK_ATTEMPTED = True
        action, trial_records = block._try_block_action(
            graph,
            field,
            points,
            records,
            config,
            block._BLOCK_PROPOSAL,
            monitor=monitor,
        )
        if action is not None:
            return action, trial_records

    if not block._FALLBACK_ATTEMPTED:
        block._FALLBACK_ATTEMPTED = True
        action, trial_records = block._fallback_best_action(
            graph, field, points, records, config, monitor=monitor
        )
        if action is not None:
            return action, trial_records

    if not _RESCUE_ATTEMPTED:
        _RESCUE_ATTEMPTED = True
        return _try_interaction_rescue(
            graph, field, points, records, config, monitor=monitor
        )

    return None, None


def convergence_required_train(
    field, config, *, graph=None, progress=None, monitor=None
):
    """Do not declare a normal structural stall until rescue dictionaries fail.

    The first pass is exactly the configured model. If it stalls above the
    requested residual, two continuation passes progressively add one static
    polynomial degree while retaining the same physical equations/tolerance.
    Interaction rescue remains selective and only fires after each pass stalls.
    """
    current_graph = graph
    effective = config
    last_report = None

    for rescue_round in range(_RESCUE_DEGREE_ROUNDS + 1):
        current_graph, last_report = _ORIGINAL_TRAIN(
            field,
            effective,
            graph=current_graph,
            progress=progress,
            monitor=monitor,
        )
        if last_report.numerical_tolerance_met:
            return current_graph, last_report
        if last_report.status == "budget_exhausted":
            return current_graph, last_report
        if last_report.status != "stalled":
            return current_graph, last_report
        if rescue_round >= _RESCUE_DEGREE_ROUNDS:
            break

        effective = replace(
            effective,
            max_degree=int(effective.max_degree) + 1,
            max_realization_dimension=max(
                int(effective.max_realization_dimension),
                _RESCUE_REALIZATION_DIMENSION,
            ),
        )

    return current_graph, last_report


def install_convergence_rescue() -> None:
    global _ORIGINAL_TRAIN
    if _ORIGINAL_TRAIN is not None:
        return

    from . import adaptive_runtime as adaptive
    from . import block_sparse_search_runtime as block
    from . import research as research
    from . import residual_state_runtime as runtime

    _ORIGINAL_TRAIN = block._block_sparse_train
    runtime.select_candidate_action = convergence_rescue_select_candidate_action
    research.train_research_graph = convergence_required_train
    adaptive.adaptive_train_research_graph = convergence_required_train
    try:
        import sdfmpneo.research as public_research
        public_research.train_research_graph = convergence_required_train
    except ImportError:
        pass


__all__ = [
    "convergence_rescue_select_candidate_action",
    "convergence_required_train",
    "install_convergence_rescue",
]
