from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .max_residual_runtime import (
    hard_point_weights,
    relative_max_improvement,
    weighted_score_parent_batch as _weighted_l2_score_parent_batch,
)
from .parallel_runtime import _ordered_map
from .state_structure_policy import (
    _best_line_search_action,
    _split_proposals,
    residual_norms,
)


# Structural growth must buy a meaningful decrease of the actual stopping
# metric.  This is intentionally the same scale already used to detect a stale
# fixed-topology Gauss--Newton solve.
_STRUCTURAL_MIN_RELATIVE_GAIN = 1.0e-3
# A clearly useful direct action does not justify enumerating every exact split
# alternative.  Weak direct actions still fall through to Split.
_STRONG_DIRECT_RELATIVE_GAIN = 5.0e-3
# This is a search-work budget, not a source/state cardinality limit.  Candidate
# ranking is max-aligned before this expensive nonlinear stage.
_MAX_NONLINEAR_CANDIDATE_TRIALS = 16
_MAX_ALIGNMENT_HARD_POINTS = 8
_MIN_PREDICTED_RELATIVE_GAIN = 1.0e-4

_LAST_RECORDS_TOKEN = None
_NONLINEAR_TRIALS = 0


def _candidate_alpha_set(old_sq, correlation, tangent_norm2, weights):
    """Small deterministic set containing the relevant 1-D minimax breakpoints."""
    valid = tangent_norm2 > np.finfo(float).tiny
    if not np.any(valid):
        return np.array([0.0], dtype=float)

    order = np.argsort(old_sq)
    hard = order[-min(_MAX_ALIGNMENT_HARD_POINTS, len(order)):]
    values = [0.0]

    denom = float(np.dot(weights, tangent_norm2))
    if denom > np.finfo(float).tiny:
        values.append(-float(np.dot(weights, correlation)) / denom)

    for index in hard:
        if valid[index]:
            values.append(-float(correlation[index]) / float(tangent_norm2[index]))

    # The minimizer of max_i q_i(alpha) is commonly an intersection of two
    # active residual quadratics.  Add the real intersections of only the hard
    # residual set; this stays cheap while aligning ranking to L-infinity.
    for left_pos, left in enumerate(hard):
        for right in hard[left_pos + 1:]:
            A = float(tangent_norm2[left] - tangent_norm2[right])
            B = 2.0 * float(correlation[left] - correlation[right])
            C = float(old_sq[left] - old_sq[right])
            scale = max(abs(A), abs(B), abs(C), np.finfo(float).tiny)
            if abs(A) <= 64.0 * np.finfo(float).eps * scale:
                if abs(B) > 64.0 * np.finfo(float).eps * scale:
                    values.append(-C / B)
                continue
            disc = B * B - 4.0 * A * C
            if disc < 0.0:
                continue
            root = float(np.sqrt(max(0.0, disc)))
            values.extend(((-B - root) / (2.0 * A), (-B + root) / (2.0 * A)))

    base = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if base.size == 0:
        return np.array([0.0], dtype=float)
    # Wild pairwise roots are irrelevant to a local tangent step and can create
    # overflow in the predicted quadratics.  Keep a generous neighborhood of
    # the individual/weighted least-squares minimizers.
    individual = np.asarray(
        [-correlation[i] / tangent_norm2[i] for i in hard if valid[i]], dtype=float
    )
    radius = float(np.max(np.abs(individual), initial=0.0))
    if denom > np.finfo(float).tiny:
        radius = max(radius, abs(-float(np.dot(weights, correlation)) / denom))
    if radius > 0.0 and np.isfinite(radius):
        base = base[np.abs(base) <= 16.0 * radius]
    if base.size == 0:
        return np.array([0.0], dtype=float)
    return np.unique(base)


def _max_aligned_target(old_sq, correlation, tangent_norm2, weights):
    old_max = float(np.sqrt(np.max(old_sq, initial=0.0)))
    if old_max <= 0.0:
        return 0.0, 0.0
    alphas = _candidate_alpha_set(old_sq, correlation, tangent_norm2, weights)
    predicted = (
        old_sq[:, None]
        + 2.0 * correlation[:, None] * alphas[None, :]
        + tangent_norm2[:, None] * (alphas[None, :] ** 2)
    )
    predicted = np.maximum(predicted, 0.0)
    maxima = np.max(predicted, axis=0, initial=0.0)
    best_index = int(np.argmin(maxima))
    best_alpha = float(alphas[best_index])
    new_max = float(np.sqrt(maxima[best_index]))
    relative = max(0.0, (old_max - new_max) / old_max)
    return relative, best_alpha


def max_aligned_score_parent_batch(
    graph, records, compiled, plans, *, point_weights=None, monitor=None
):
    """Rank structured source columns by their linearized L-infinity decrease.

    Existing batching already provides, per collocation point, r.T@T and
    ||T||^2.  Those two quantities are sufficient to predict

        ||r + alpha T||^2

    for any scalar candidate coefficient alpha.  Ranking the resulting minimax
    quadratic is much better aligned with the actual max-first acceptance rule
    than collapsing the same data into a weighted L2 projection first.
    """
    from . import late_stage_batch as batch

    plans = list(plans)
    if not plans:
        return []
    if point_weights is None:
        point_weights = np.ones(len(records), dtype=float)
    point_weights = np.asarray(point_weights, dtype=float)
    if point_weights.shape != (len(records),):
        raise ValueError("candidate point weights do not match records")

    keys, seen = [], set()
    for plan in plans:
        key = batch._structured_key(plan)
        if key is not None and key not in seen:
            seen.add(key)
            keys.append(key)
    tables = (
        _ordered_map(
            lambda key: batch._unit_candidate_table(graph, records, compiled, key),
            keys,
            monitor=monitor,
        )
        if keys
        else []
    )
    table_by_key = dict(zip(keys, tables))

    if records:
        initial = np.vstack([np.asarray(record.initial, dtype=float) for record in records])
        operating = np.vstack([np.asarray(record.u, dtype=float) for record in records])
        old_sq = residual_norms(records) ** 2
    else:
        initial = np.empty((0, graph.n_modes), dtype=float)
        operating = np.empty((0, len(graph.operating_names)), dtype=float)
        old_sq = np.empty(0, dtype=float)

    out = [None] * len(plans)
    generic, generic_indices = [], []
    for index, plan in enumerate(plans):
        key = batch._structured_key(plan)
        if key is None:
            generic.append(plan)
            generic_indices.append(index)
            continue

        amplitude = batch._plan_scale(plan, initial, operating)
        point_correlation, point_norm2 = table_by_key[key]
        inner = np.zeros(graph.n_modes, dtype=float)
        norm2 = np.ones(graph.n_modes, dtype=float)
        for target in range(graph.n_modes):
            correlation = amplitude * point_correlation[:, target]
            tangent_norm2 = amplitude * amplitude * point_norm2[:, target]
            relative, alpha = _max_aligned_target(
                old_sq, correlation, tangent_norm2, point_weights
            )
            if (
                relative < _MIN_PREDICTED_RELATIVE_GAIN
                or not np.isfinite(alpha)
                or alpha == 0.0
            ):
                continue
            # residual_state_runtime consumes the legacy (inner,norm2) contract
            # only through score=inner^2/norm2 and weight=-inner/norm2.  Encode
            # the max-aligned score/weight into that contract without changing
            # any public API.
            synthetic_norm2 = relative / (alpha * alpha)
            synthetic_inner = -alpha * synthetic_norm2
            if np.isfinite(synthetic_norm2 + synthetic_inner) and synthetic_norm2 > 0.0:
                inner[target] = synthetic_inner
                norm2[target] = synthetic_norm2
        out[index] = (inner, norm2)

    if generic:
        # Backward-compatible >1-response-parent graphs keep their exact generic
        # scorer.  Production UWPT uses max_parent_responses=1, so the expensive
        # generic path is not part of the normal search.
        values = _weighted_l2_score_parent_batch(
            graph,
            records,
            compiled,
            generic,
            point_weights=point_weights,
            monitor=monitor,
        )
        for index, value in zip(generic_indices, values):
            out[index] = value
    return out


def _actual_relative_gain(old_records, new_records, tolerance):
    old_max = float(np.max(residual_norms(old_records), initial=0.0))
    new_max = float(np.max(residual_norms(new_records), initial=0.0))
    return relative_max_improvement(old_max, new_max, tolerance)


def bounded_max_aligned_select_candidate_action(
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
    """Bound nonlinear trials and lazily invoke Split only when Direct is weak."""
    global _LAST_RECORDS_TOKEN, _NONLINEAR_TRIALS

    token = id(records)
    if token != _LAST_RECORDS_TOKEN:
        _LAST_RECORDS_TOKEN = token
        _NONLINEAR_TRIALS = 0
    if _NONLINEAR_TRIALS >= _MAX_NONLINEAR_CANDIDATE_TRIALS:
        return None, None
    _NONLINEAR_TRIALS += 1

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
        direct_gain = _actual_relative_gain(
            records, direct[2], config.residual_tolerance
        )
        if direct_gain >= _STRONG_DIRECT_RELATIVE_GAIN:
            return direct[1], direct[2]
        if direct_gain >= _STRUCTURAL_MIN_RELATIVE_GAIN:
            candidates.append(direct)

    # Exact Split can be materially better when an aggregate state needs
    # downstream addressability, but it is too expensive to enumerate for every
    # already-good Direct action.  Evaluate it only in the weak/stalled regime.
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
        if candidate is None:
            continue
        if _actual_relative_gain(records, candidate[2], config.residual_tolerance) >= _STRUCTURAL_MIN_RELATIVE_GAIN:
            candidates.append(candidate)

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def coalesce_unreferenced_state_families(graph):
    """Function-preservingly aggregate a pure equation seed by target family.

    Geometry seeding predates multi-source analytic states and therefore emits
    one scalar response node per equation-derived source.  Before descendants
    exist, all same-target seed responses share the same pole and their sum is
    exactly one analytic state with multiple source columns.
    """
    from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
    from sdfmpneo.analytic.state_graph import response_sources

    if not graph.response_nodes:
        return graph
    response_names = {node.name for node in graph.response_nodes}
    if any(
        parent in response_names
        for node in graph.response_nodes
        for source in response_sources(graph, node)
        for parent in source.parents
    ):
        return graph

    grouped = OrderedDict()
    for node in graph.response_nodes:
        entry = grouped.setdefault(
            int(node.target_mode),
            {"name": node.name, "sources": OrderedDict()},
        )
        for source in response_sources(graph, node):
            entry["sources"][tuple(source.parents)] = (
                entry["sources"].get(tuple(source.parents), 0.0 + 0.0j)
                + source.weight
            )

    out = ParametricAnalyticEvolutionGraph(
        np.asarray(graph.lambdas, dtype=float).copy(), graph.operating_names
    )
    for target, entry in grouped.items():
        sources = [
            (parents, weight)
            for parents, weight in entry["sources"].items()
            if weight != 0
        ]
        if not sources:
            continue
        if len(sources) == 1:
            parents, weight = sources[0]
            out.add_product_response(entry["name"], target, parents, weight)
        else:
            out.add_response_state(entry["name"], target, tuple(sources))
    return out


def install_state_search_policy() -> None:
    from . import residual_state_runtime as runtime

    runtime.weighted_score_parent_batch = max_aligned_score_parent_batch
    runtime.select_candidate_action = bounded_max_aligned_select_candidate_action


def install_geometry_seed_coalescing(geometry_model_cls) -> None:
    if hasattr(geometry_model_cls, "_sdfmpneo_scalar_seed_graph"):
        return
    original = geometry_model_cls.seed_graph
    geometry_model_cls._sdfmpneo_scalar_seed_graph = original

    def seed_graph(self, graph, monitor=None):
        seeded = original(self, graph, monitor=monitor)
        return coalesce_unreferenced_state_families(seeded)

    geometry_model_cls.seed_graph = seed_graph


__all__ = [
    "max_aligned_score_parent_batch",
    "bounded_max_aligned_select_candidate_action",
    "coalesce_unreferenced_state_families",
    "install_state_search_policy",
    "install_geometry_seed_coalescing",
]
