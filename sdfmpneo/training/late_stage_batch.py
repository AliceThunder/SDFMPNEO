from __future__ import annotations

import numpy as np

from sdfmpneo.analytic.long_time import realization_action
from sdfmpneo.analytic.realization import AnalyticRealization
from .parallel_runtime import _ordered_map
from . import late_stage_runtime as _late


_PLAN_GRAPH = None
_PLAN_CACHE = {}
_PLAN_INITIAL_LOOKUP = {}
_PLAN_OPERATING_LOOKUP = {}
_PLAN_RESPONSE_NAMES = set()


def _iter_admissible_parent_tuples(names, response_names, degree, max_parent_responses):
    """Exact old filtered order, with response-suffix branches pruned early."""
    names = tuple(names)
    flags = tuple(name in response_names for name in names)
    n_names = len(names)
    response_start = next((index for index, flag in enumerate(flags) if flag), n_names)

    def visit(start, remaining, used, prefix):
        if remaining == 0:
            yield tuple(prefix)
            return
        if used >= max_parent_responses and start >= response_start:
            return
        stop = n_names if used < max_parent_responses else response_start
        for index in range(start, stop):
            count = used + int(flags[index])
            if count > max_parent_responses:
                continue
            if flags[index] and count >= max_parent_responses and remaining > 1:
                # known_names() places every response after every base node, so
                # all future non-decreasing choices would also be responses.
                continue
            prefix.append(names[index])
            yield from visit(index, remaining - 1, count, prefix)
            prefix.pop()

    yield from visit(0, int(degree), 0, [])


def _reset_parent_plan_cache(graph):
    global _PLAN_GRAPH, _PLAN_CACHE, _PLAN_INITIAL_LOOKUP, _PLAN_OPERATING_LOOKUP, _PLAN_RESPONSE_NAMES
    _PLAN_GRAPH = graph
    _PLAN_CACHE = {}
    _PLAN_INITIAL_LOOKUP = {
        name: index for index, name in enumerate(graph.initial_names)
    }
    _PLAN_OPERATING_LOOKUP = {
        name: index for index, name in enumerate(graph.operating_names)
    }
    _PLAN_RESPONSE_NAMES = {node.name for node in graph.response_nodes}


def _parent_plan(graph, parents):
    global _PLAN_GRAPH
    if graph is not _PLAN_GRAPH:
        _reset_parent_plan_cache(graph)
    parents = tuple(parents)
    cached = _PLAN_CACHE.get(parents)
    if cached is not None:
        return cached

    initial = []
    operating = []
    response = []
    decay = 0.0
    for parent in parents:
        if parent in _PLAN_INITIAL_LOOKUP:
            index = _PLAN_INITIAL_LOOKUP[parent]
            initial.append(index)
            decay += float(graph.lambdas[index])
        elif parent in _PLAN_OPERATING_LOOKUP:
            operating.append(_PLAN_OPERATING_LOOKUP[parent])
        elif parent in _PLAN_RESPONSE_NAMES:
            response.append(parent)
        else:
            raise KeyError(parent)
    if len(response) > 1:
        result = _late._ParentPlan(
            parents, tuple(initial), tuple(operating), None, float("nan")
        )
    else:
        result = _late._ParentPlan(
            parents, tuple(initial), tuple(operating),
            None if not response else response[0], decay,
        )
    _PLAN_CACHE[parents] = result
    return result


def _candidate_source_response(source, lambdas, time):
    """Exact joint source/modal response using the candidate-only propagator LRU."""
    lambdas = np.asarray(lambdas, dtype=float)
    n_source = source.dimension
    n_modes = len(lambdas)
    A = np.zeros((n_source + n_modes, n_source + n_modes), dtype=complex)
    A[:n_source, :n_source] = source.A
    A[n_source:, :n_source] = source.c
    A[n_source:, n_source:] = -np.diag(lambdas)
    b = np.concatenate([source.b, np.zeros(n_modes, dtype=complex)])
    state = realization_action(A, b, time, cache_namespace="candidate")
    psi = float(np.real(source.c @ state[:n_source]))
    responses = np.asarray(np.real(state[n_source:]), dtype=float)
    return psi, responses


def _structured_source_unit(record, compiled, plan):
    if plan.response_parent is None:
        source = AnalyticRealization.decay(plan.decay_shift, 1.0)
    else:
        base = compiled.node_realizations[plan.response_parent]
        shifted = base.A - plan.decay_shift * np.eye(base.dimension, dtype=complex)
        source = AnalyticRealization(shifted, base.b, base.c)
    return _candidate_source_response(source, compiled.lambdas, record.time)


def _structured_key(plan):
    if np.isnan(plan.decay_shift):
        return None
    return plan.response_parent, float(plan.decay_shift)


def _unit_candidate_table(graph, records, compiled, key):
    response_parent, decay_shift = key
    plan = _late._ParentPlan((), (), (), response_parent, decay_shift)
    n_modes = graph.n_modes
    identity = np.eye(n_modes, dtype=float)
    linear = np.diag(np.asarray(graph.lambdas, dtype=float))
    correlation = np.zeros((len(records), n_modes), dtype=float)
    norm2 = np.zeros_like(correlation)
    for index, (record, realization) in enumerate(zip(records, compiled)):
        psi, h = _structured_source_unit(record, realization, plan)
        tangent = identity * psi - (record.J + linear) * h[np.newaxis, :]
        correlation[index] = record.residual @ tangent
        norm2[index] = np.sum(tangent * tangent, axis=0)
    return correlation, norm2


def _plan_scale(plan, initial, operating):
    scale = np.ones(initial.shape[0], dtype=float)
    for index in plan.initial_indices:
        scale *= initial[:, index]
    for index in plan.operating_indices:
        scale *= operating[:, index]
    return scale


def _generic_score(graph, records, compiled, plan):
    """No-capability-loss fallback for old graphs with >1 response parent."""
    n_modes = graph.n_modes
    identity = np.eye(n_modes, dtype=float)
    linear = np.diag(np.asarray(graph.lambdas, dtype=float))
    inner = np.zeros(n_modes, dtype=float)
    norm2 = np.zeros(n_modes, dtype=float)
    for record, realization in zip(records, compiled):
        source = AnalyticRealization.constant(1.0)
        for parent in plan.parents:
            source = source.product(realization.node_realizations[parent])
        psi, h = _candidate_source_response(source, graph.lambdas, record.time)
        tangent = identity * psi - (record.J + linear) * h[np.newaxis, :]
        inner += record.residual @ tangent
        norm2 += np.sum(tangent * tangent, axis=0)
    return inner, norm2


def _score_parent_batch(graph, records, compiled, plans, *, monitor=None):
    """Share exact dynamic responses across all amplitude-only candidate variants."""
    plans = list(plans)
    if not plans:
        return []

    keys = []
    seen = set()
    for plan in plans:
        key = _structured_key(plan)
        if key is not None and key not in seen:
            seen.add(key)
            keys.append(key)

    tables = _ordered_map(
        lambda key: _unit_candidate_table(graph, records, compiled, key),
        keys,
        monitor=monitor,
    ) if keys else []
    table_by_key = dict(zip(keys, tables))

    if records:
        initial = np.vstack([np.asarray(record.initial, dtype=float) for record in records])
        operating = np.vstack([np.asarray(record.u, dtype=float) for record in records])
    else:
        initial = np.empty((0, graph.n_modes), dtype=float)
        operating = np.empty((0, len(graph.operating_names)), dtype=float)

    out = [None] * len(plans)
    generic = []
    generic_indices = []
    for index, plan in enumerate(plans):
        key = _structured_key(plan)
        if key is None:
            generic.append(plan)
            generic_indices.append(index)
            continue
        scale = _plan_scale(plan, initial, operating)
        correlation, local_norm2 = table_by_key[key]
        inner = np.sum(scale[:, np.newaxis] * correlation, axis=0)
        norm2 = np.sum((scale * scale)[:, np.newaxis] * local_norm2, axis=0)
        out[index] = (inner, norm2)

    if generic:
        values = _ordered_map(
            lambda plan: _generic_score(graph, records, compiled, plan),
            generic,
            monitor=monitor,
        )
        for index, value in zip(generic_indices, values):
            out[index] = value
    return out


def _prepare_working_set(field, *point_sets):
    """Keep the expensive training set resident; validation is a one-pass guest."""
    prepare = getattr(field, "prepare_training_contexts", None)
    if prepare is not None and point_sets:
        prepare(point_sets[0])


def install_late_stage_batching() -> None:
    """Patch structural candidate batching into the optimized training runtime."""
    _late._iter_admissible_parent_tuples = _iter_admissible_parent_tuples
    _late._parent_plan = _parent_plan
    _late._score_parent_batch = _score_parent_batch
    _late._prepare_working_set = _prepare_working_set
    from . import adaptive_runtime
    adaptive_runtime.adaptive_train_research_graph = _late.optimized_adaptive_train_research_graph
