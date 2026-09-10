from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdfmpneo.analytic.state_graph import clone_state_graph
from .max_residual_runtime import hard_point_weights, max_first_accept
from .parallel_runtime import _ordered_map
from .state_structure_policy import (
    StructuralAction,
    dynamic_response_parent,
    family_nodes,
    next_state_name,
    residual_norms,
    source_active,
)
from .state_search_runtime import (
    _STRUCTURAL_MIN_RELATIVE_GAIN,
    _max_aligned_target,
    bounded_max_aligned_select_candidate_action as _fallback_select,
    max_aligned_score_parent_batch as _fallback_score,
)

_FISTA_STEPS = 24
_POWER_STEPS = 7
_REGULARIZATION_PATH = (0.30, 0.12, 0.05)
_SOURCE_RELATIVE_CUTOFF = 1.0e-4
_BLOCK_LINE_SEARCH_STEPS = 6
_BLOCK_GOOD_PREDICTED_GAIN = 5.0e-3

_BLOCK_RECORDS = None
_BLOCK_PROPOSAL = None
_BLOCK_ATTEMPTED = False


@dataclass(frozen=True)
class _Candidate:
    plan_index: int
    target: int
    parents: tuple[str, ...]
    family: tuple
    key: tuple
    norm: float


@dataclass
class _OperatorBlock:
    candidate_indices: np.ndarray
    scales: np.ndarray
    norms: np.ndarray
    unit_tangent: np.ndarray


class _StructuredTangentOperator:
    """Matrix-free weighted candidate tangent operator.

    Candidate columns are not materialized as one giant
    ``(points * modes) x candidates`` matrix. Static monomial amplitudes are
    stored in compact scale matrices and multiplied by one shared dynamic
    unit-tangent table per analytic response key.
    """

    def __init__(self, records, point_weights, candidates, blocks):
        self.records = records
        self.weights = np.asarray(point_weights, dtype=float)
        self.sqrt_weights = np.sqrt(self.weights)
        self.candidates = tuple(candidates)
        self.blocks = tuple(blocks)
        residual_width = 0 if not records else len(records[0].residual)
        self.shape = (len(records) * residual_width, len(candidates))

    def physical_matvec(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.shape[1],):
            raise ValueError("candidate coefficient vector has wrong shape")
        if not self.records:
            return np.empty((0, 0), dtype=float)
        out = np.zeros(
            (len(self.records), len(self.records[0].residual)), dtype=float
        )
        for block in self.blocks:
            local = z[block.candidate_indices] / block.norms
            amplitude = block.scales @ local
            out += amplitude[:, None] * block.unit_tangent
        return out

    def matvec(self, z):
        return self.sqrt_weights[:, None] * self.physical_matvec(z)

    def rmatvec(self, y):
        y = np.asarray(y, dtype=float)
        expected = (
            len(self.records),
            0 if not self.records else len(self.records[0].residual),
        )
        if y.shape != expected:
            raise ValueError("weighted tangent residual has wrong shape")
        weighted_y = self.sqrt_weights[:, None] * y
        out = np.zeros(self.shape[1], dtype=float)
        for block in self.blocks:
            local = np.sum(block.unit_tangent * weighted_y, axis=1)
            out[block.candidate_indices] += (block.scales.T @ local) / block.norms
        return out


def _unit_tangent_table(graph, records, compiled, key):
    from . import late_stage_batch as batch
    from . import late_stage_runtime as late

    response_parent, decay_shift = key
    plan = late._ParentPlan((), (), (), response_parent, decay_shift)
    n_modes = graph.n_modes
    identity = np.eye(n_modes, dtype=float)
    linear = np.diag(np.asarray(graph.lambdas, dtype=float))
    out = np.empty((len(records), n_modes, n_modes), dtype=float)
    for index, (record, realization) in enumerate(zip(records, compiled)):
        psi, h = batch._structured_source_unit(record, realization, plan)
        out[index] = identity * psi - (record.J + linear) * h[np.newaxis, :]
    return out


def _build_operator(graph, records, compiled, plans, point_weights, monitor=None):
    from . import late_stage_batch as batch

    plans = list(plans)
    if not plans or not records:
        return None, []

    initial = np.vstack(
        [np.asarray(record.initial, dtype=float) for record in records]
    )
    operating = np.vstack([np.asarray(record.u, dtype=float) for record in records])

    keys, seen = [], set()
    for plan in plans:
        key = batch._structured_key(plan)
        if key is not None and key not in seen:
            seen.add(key)
            keys.append(key)
    if not keys:
        return None, []

    tables = _ordered_map(
        lambda key: _unit_tangent_table(graph, records, compiled, key),
        keys,
        monitor=monitor,
    )
    table_by_key = dict(zip(keys, tables))
    weights = np.asarray(point_weights, dtype=float)

    raw = []
    for plan_index, plan in enumerate(plans):
        key = batch._structured_key(plan)
        if key is None:
            continue
        scale = batch._plan_scale(plan, initial, operating)
        table = table_by_key[key]
        for target in range(graph.n_modes):
            if source_active(graph, target, plan.parents):
                continue
            local_norm2 = np.sum(table[:, :, target] ** 2, axis=1)
            norm2 = float(np.dot(weights, scale * scale * local_norm2))
            if not np.isfinite(norm2) or norm2 <= np.finfo(float).tiny:
                continue
            family = (int(target), dynamic_response_parent(graph, plan.parents))
            raw.append(
                (
                    plan_index,
                    int(target),
                    tuple(plan.parents),
                    family,
                    key,
                    scale,
                    float(np.sqrt(norm2)),
                )
            )
    if not raw:
        return None, []

    candidates = [
        _Candidate(item[0], item[1], item[2], item[3], item[4], item[6])
        for item in raw
    ]
    grouped = {}
    for index, item in enumerate(raw):
        grouped.setdefault((item[4], item[1]), []).append(
            (index, item[5], item[6])
        )

    blocks = []
    for (key, target), values in grouped.items():
        blocks.append(
            _OperatorBlock(
                candidate_indices=np.asarray(
                    [value[0] for value in values], dtype=int
                ),
                scales=np.column_stack([value[1] for value in values]),
                norms=np.asarray([value[2] for value in values], dtype=float),
                unit_tangent=table_by_key[key][:, :, target],
            )
        )
    return _StructuredTangentOperator(records, weights, candidates, blocks), candidates


def _groups(candidates):
    grouped = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(candidate.family, []).append(index)
    return {
        family: np.asarray(indices, dtype=int)
        for family, indices in grouped.items()
    }


def _prox_sparse_group(values, step, lambda_l1, lambda_group, groups):
    out = np.sign(values) * np.maximum(
        np.abs(values) - step * lambda_l1, 0.0
    )
    for indices in groups.values():
        local = out[indices]
        norm = float(np.linalg.norm(local))
        if norm <= 0.0:
            continue
        threshold = step * lambda_group * np.sqrt(len(indices))
        out[indices] *= max(0.0, 1.0 - threshold / norm)
    return out


def _estimate_lipschitz(operator):
    size = operator.shape[1]
    if size == 0:
        return 1.0
    vector = np.full(size, 1.0 / np.sqrt(size), dtype=float)
    value = 1.0
    for _ in range(_POWER_STEPS):
        image = operator.matvec(vector)
        adjoint = operator.rmatvec(image)
        norm = float(np.linalg.norm(adjoint))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            return 1.0
        vector = adjoint / norm
        value = float(np.sum(operator.matvec(vector) ** 2))
    return max(1.0e-12, 1.10 * value)


def _sparse_group_solve(operator, residual, groups, fraction, initial=None):
    weighted_residual = operator.sqrt_weights[:, None] * residual
    gradient0 = operator.rmatvec(weighted_residual)
    if gradient0.size == 0:
        return np.empty(0, dtype=float)

    individual_max = float(np.max(np.abs(gradient0), initial=0.0))
    group_max = max(
        (
            float(np.linalg.norm(gradient0[indices])) / np.sqrt(len(indices))
            for indices in groups.values()
        ),
        default=0.0,
    )
    if max(individual_max, group_max) <= np.finfo(float).tiny:
        return np.zeros_like(gradient0)

    lambda_l1 = float(fraction) * 0.35 * individual_max
    lambda_group = float(fraction) * 0.65 * group_max
    lipschitz = _estimate_lipschitz(operator)

    if initial is None or np.asarray(initial).shape != gradient0.shape:
        x = np.zeros_like(gradient0)
    else:
        x = np.asarray(initial, dtype=float).copy()
    y = x.copy()
    momentum = 1.0

    for _ in range(_FISTA_STEPS):
        Ay = operator.matvec(y)
        smooth_y = 0.5 * float(np.sum((weighted_residual + Ay) ** 2))
        gradient = operator.rmatvec(weighted_residual + Ay)

        local_lipschitz = lipschitz
        while True:
            trial = _prox_sparse_group(
                y - gradient / local_lipschitz,
                1.0 / local_lipschitz,
                lambda_l1,
                lambda_group,
                groups,
            )
            delta = trial - y
            smooth_trial = 0.5 * float(
                np.sum((weighted_residual + operator.matvec(trial)) ** 2)
            )
            majorizer = (
                smooth_y
                + float(np.dot(gradient, delta))
                + 0.5 * local_lipschitz * float(np.dot(delta, delta))
            )
            if smooth_trial <= majorizer + 1.0e-12 * max(1.0, smooth_y):
                break
            local_lipschitz *= 2.0
            if local_lipschitz > lipschitz * 1.0e6:
                break
        lipschitz = local_lipschitz
        next_momentum = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)
        )
        y = trial + ((momentum - 1.0) / next_momentum) * (trial - x)
        if np.linalg.norm(trial - x) <= 1.0e-7 * max(
            1.0, np.linalg.norm(trial)
        ):
            x = trial
            break
        x = trial
        momentum = next_momentum
    return x


def _proposal_from_z(graph, records, operator, candidates, groups, z):
    z = np.asarray(z, dtype=float).copy()
    peak = float(np.max(np.abs(z), initial=0.0))
    if peak <= 0.0:
        return None
    z[np.abs(z) < max(1.0e-12, _SOURCE_RELATIVE_CUTOFF * peak)] = 0.0
    if not np.any(z):
        return None

    direction = operator.physical_matvec(z)
    residual = np.vstack(
        [np.asarray(record.residual, dtype=float) for record in records]
    )
    old_sq = np.sum(residual * residual, axis=1)
    correlation = np.sum(residual * direction, axis=1)
    tangent_norm2 = np.sum(direction * direction, axis=1)
    relative, alpha = _max_aligned_target(
        old_sq, correlation, tangent_norm2, operator.weights
    )
    if (
        relative < _STRUCTURAL_MIN_RELATIVE_GAIN
        or not np.isfinite(alpha)
        or alpha == 0.0
    ):
        return None

    z *= alpha
    selected = []
    for index in np.flatnonzero(z):
        candidate = candidates[int(index)]
        selected.append(
            {
                "plan_index": candidate.plan_index,
                "target": candidate.target,
                "parents": candidate.parents,
                "family": candidate.family,
                "weight": float(z[index] / candidate.norm),
                "normalized_weight": float(z[index]),
            }
        )
    selected.sort(
        key=lambda item: (
            item["family"][0],
            str(item["family"][1]),
            -abs(item["normalized_weight"]),
            item["parents"],
        )
    )
    return {"relative_gain": float(relative), "sources": tuple(selected)}


def _compute_block_proposal(
    graph, records, compiled, plans, point_weights, monitor=None
):
    operator, candidates = _build_operator(
        graph, records, compiled, plans, point_weights, monitor=monitor
    )
    if operator is None or not candidates:
        return None
    groups = _groups(candidates)
    residual = np.vstack(
        [np.asarray(record.residual, dtype=float) for record in records]
    )

    warm = None
    best = None
    for fraction in _REGULARIZATION_PATH:
        warm = _sparse_group_solve(
            operator, residual, groups, fraction, initial=warm
        )
        proposal = _proposal_from_z(
            graph, records, operator, candidates, groups, warm
        )
        if proposal is None:
            continue
        if best is None or proposal["relative_gain"] > best["relative_gain"]:
            best = proposal
        if proposal["relative_gain"] >= _BLOCK_GOOD_PREDICTED_GAIN:
            return proposal
    return best


def block_sparse_score_parent_batch(
    graph, records, compiled, plans, *, point_weights=None, monitor=None
):
    """Solve one sparse block proposal instead of ranking candidates independently."""
    global _BLOCK_RECORDS, _BLOCK_PROPOSAL, _BLOCK_ATTEMPTED
    plans = list(plans)
    if point_weights is None:
        point_weights = np.ones(len(records), dtype=float)

    _BLOCK_RECORDS = records
    _BLOCK_ATTEMPTED = False
    _BLOCK_PROPOSAL = _compute_block_proposal(
        graph, records, compiled, plans, point_weights, monitor=monitor
    )
    if _BLOCK_PROPOSAL is None:
        return _fallback_score(
            graph,
            records,
            compiled,
            plans,
            point_weights=point_weights,
            monitor=monitor,
        )

    out = [
        (
            np.zeros(graph.n_modes, dtype=float),
            np.ones(graph.n_modes, dtype=float),
        )
        for _ in plans
    ]
    block_score = max(float(_BLOCK_PROPOSAL["relative_gain"]), 1.0e-12)
    for rank, item in enumerate(_BLOCK_PROPOSAL["sources"]):
        weight = float(item["weight"])
        if weight == 0.0 or not np.isfinite(weight):
            continue
        desired_score = block_score * (1.0 + 1.0e-9 / (1 + rank))
        norm2 = desired_score / (weight * weight)
        inner = -weight * norm2
        plan_index = int(item["plan_index"])
        target = int(item["target"])
        out[plan_index][0][target] = inner
        out[plan_index][1][target] = norm2
    return out


def _fit_proposal_to_budget(graph, proposal, max_nodes):
    available = max(0, int(max_nodes) - len(graph.response_nodes))
    existing, new = [], {}
    for item in proposal["sources"]:
        if family_nodes(graph, item["target"], item["parents"]):
            existing.append(item)
        else:
            new.setdefault(item["family"], []).append(item)

    if len(new) <= available:
        return proposal

    ranked = []
    for family, values in new.items():
        energy = float(
            np.linalg.norm([value["normalized_weight"] for value in values])
        )
        ranked.append((energy, str(family), family))
    ranked.sort(reverse=True)
    keep = {family for _, _, family in ranked[:available]}
    sources = tuple(
        existing
        + [
            item
            for family, values in new.items()
            if family in keep
            for item in values
        ]
    )
    if not sources:
        return None
    return {**proposal, "sources": sources}


def _add_block_sources(graph, proposal, factor, config):
    trial = clone_state_graph(graph)
    added = 0
    families_added = set()

    for item in proposal["sources"]:
        weight = factor * float(item["weight"])
        if weight == 0.0:
            continue
        target = int(item["target"])
        parents = tuple(item["parents"])
        if source_active(trial, target, parents):
            continue

        existing = family_nodes(trial, target, parents)
        enriched = False
        for node in existing:
            try:
                trial.enrich_response_state(node.name, parents, weight)
                enriched = True
                break
            except ValueError:
                continue

        if not enriched:
            if len(trial.response_nodes) >= int(config.max_nodes):
                return None
            name = next_state_name(trial)
            try:
                trial.add_response_state(name, target, ((parents, weight),))
            except ValueError:
                return None
            families_added.add(
                (target, dynamic_response_parent(trial, parents))
            )
        added += 1

    if added == 0:
        return None
    return StructuralAction(
        "block_sparse",
        trial,
        {
            "source_count": added,
            "new_family_count": len(families_added),
            "predicted_relative_gain": float(proposal["relative_gain"]),
        },
    )


def _try_block_action(
    graph, field, points, records, config, proposal, monitor=None
):
    from . import research as r

    proposal = _fit_proposal_to_budget(graph, proposal, config.max_nodes)
    if proposal is None:
        return None, None

    old_norms = residual_norms(records)
    frozen_weights = hard_point_weights(records, config.residual_tolerance)
    factor = 1.0
    for _ in range(_BLOCK_LINE_SEARCH_STEPS):
        action = _add_block_sources(graph, proposal, factor, config)
        if action is None:
            return None, None
        try:
            trial_records = r._evaluate(
                action.graph, field, points, monitor=monitor
            )
            new_norms = residual_norms(trial_records)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            trial_records = None
            new_norms = np.full_like(old_norms, np.inf)

        if trial_records is not None and max_first_accept(
            old_norms,
            new_norms,
            config.residual_tolerance,
            frozen_weights,
        ):
            old_max = float(np.max(old_norms, initial=0.0))
            new_max = float(np.max(new_norms, initial=0.0))
            denominator = max(
                old_max,
                float(config.residual_tolerance),
                np.finfo(float).tiny,
            )
            if (
                (old_max - new_max) / denominator
                >= _STRUCTURAL_MIN_RELATIVE_GAIN
            ):
                return action, trial_records
        factor *= 0.5
    return None, None


def block_sparse_select_candidate_action(
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
    """Validate one whole sparse block; fall back to scalar/Split search."""
    global _BLOCK_ATTEMPTED

    if (
        records is _BLOCK_RECORDS
        and _BLOCK_PROPOSAL is not None
        and not _BLOCK_ATTEMPTED
    ):
        _BLOCK_ATTEMPTED = True
        action, trial_records = _try_block_action(
            graph,
            field,
            points,
            records,
            config,
            _BLOCK_PROPOSAL,
            monitor=monitor,
        )
        if action is not None:
            return action, trial_records

    return _fallback_select(
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


def install_block_sparse_state_search() -> None:
    from . import residual_state_runtime as runtime

    runtime.weighted_score_parent_batch = block_sparse_score_parent_batch
    runtime.select_candidate_action = block_sparse_select_candidate_action


__all__ = [
    "block_sparse_score_parent_batch",
    "block_sparse_select_candidate_action",
    "install_block_sparse_state_search",
]
