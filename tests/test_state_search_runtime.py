from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.state_realization import evaluate_state_stable
from sdfmpneo.training import residual_state_runtime
from sdfmpneo.training.block_sparse_search_runtime import (
    _Candidate,
    _OperatorBlock,
    _StructuredTangentOperator,
    _fit_proposal_to_budget,
    _groups,
    _sparse_group_solve,
    block_sparse_score_parent_batch,
    block_sparse_select_candidate_action,
)
from sdfmpneo.training.state_linearization_runtime import scalarize_independent_sources
from sdfmpneo.training.state_search_runtime import (
    _max_aligned_target,
    bounded_max_aligned_select_candidate_action,
    coalesce_unreferenced_state_families,
    max_aligned_score_parent_batch,
)
from sdfmpneo.training.state_split_runtime import screened_split_proposals
import sdfmpneo.training.state_search_runtime as state_search_runtime


def _scalar_seed_graph():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.1], ["g", "u"])
    graph.add_product_response("a", 0, ("g",), 0.3)
    graph.add_product_response("b", 0, ("u",), -0.2)
    graph.add_product_response("c", 0, ("g", "u"), 0.08)
    graph.add_product_response("d", 1, ("u",), 0.15)
    return graph


def _assert_same_response(left, right):
    a0 = np.array([0.1, -0.05])
    operating = np.array([0.4, 1.3])
    for time in (0.0, 1.0e-5, 0.2, 20.0, np.inf):
        before = evaluate_state_stable(
            left, time, a0=a0, operating=operating
        )
        after = evaluate_state_stable(
            right, time, a0=a0, operating=operating
        )
        assert np.allclose(after[0], before[0], rtol=3e-12, atol=3e-13)
        assert np.allclose(after[1], before[1], rtol=3e-12, atol=3e-13)


def test_geometry_seed_family_coalescing_is_function_preserving():
    graph = _scalar_seed_graph()
    collapsed = coalesce_unreferenced_state_families(graph)
    assert len(graph.response_nodes) == 4
    assert len(collapsed.response_nodes) == 2
    assert collapsed.response_source_count("a") == 3
    assert collapsed.response_source_count("d") == 1
    _assert_same_response(graph, collapsed)


def test_coalesced_independent_states_scalarize_for_native_gn_without_change():
    scalar = _scalar_seed_graph()
    collapsed = coalesce_unreferenced_state_families(scalar)
    expanded = scalarize_independent_sources(collapsed)
    assert expanded is not None
    assert len(expanded.response_nodes) == collapsed.weight_parameter_count
    _assert_same_response(collapsed, expanded)


def test_seed_coalescing_and_scalarization_refuse_dynamic_descendants():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.1], ["u"])
    graph.add_product_response("a", 0, ("u",), 0.3)
    graph.add_response_state(
        "b",
        1,
        ((("a", "u"), -0.2), (("a",), 0.05)),
    )
    assert coalesce_unreferenced_state_families(graph) is graph
    assert scalarize_independent_sources(graph) is None


def test_minimax_candidate_weight_targets_the_actual_maximum():
    old_sq = np.array([4.0, 1.0])
    correlation = np.array([-2.0, 0.0])
    tangent_norm2 = np.array([1.0, 0.0])
    relative, alpha = _max_aligned_target(
        old_sq, correlation, tangent_norm2, np.ones(2)
    )
    assert 1.0 <= alpha <= 3.0
    assert np.isclose(relative, 0.5)


def _toy_operator():
    records = [
        SimpleNamespace(residual=np.array([1.0, 0.0])),
        SimpleNamespace(residual=np.array([0.0, 1.0])),
    ]
    candidates = (
        _Candidate(0, 0, ("u",), (0, None), (None, 0.0), 1.0),
        _Candidate(1, 0, ("u", "u"), (0, None), (None, 0.0), 1.0),
    )
    block = _OperatorBlock(
        candidate_indices=np.array([0, 1]),
        scales=np.eye(2),
        norms=np.ones(2),
        unit_tangent=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    return records, _StructuredTangentOperator(
        records, np.array([2.0, 3.0]), candidates, (block,)
    ), candidates


def test_matrix_free_candidate_operator_has_exact_adjoint():
    _, operator, _ = _toy_operator()
    z = np.array([0.7, -0.4])
    y = np.array([[0.2, -0.3], [0.5, 0.8]])
    left = float(np.sum(operator.matvec(z) * y))
    right = float(np.dot(z, operator.rmatvec(y)))
    assert np.isclose(left, right, rtol=2e-13, atol=2e-13)


def test_sparse_group_solve_selects_a_joint_residual_reducing_block():
    records, operator, candidates = _toy_operator()
    residual = np.vstack([record.residual for record in records])
    groups = _groups(candidates)
    z = _sparse_group_solve(operator, residual, groups, 0.05)
    assert np.count_nonzero(z) == 2
    before = np.linalg.norm(operator.sqrt_weights[:, None] * residual)
    after = np.linalg.norm(
        operator.sqrt_weights[:, None] * residual + operator.matvec(z)
    )
    assert after < before
    assert np.all(z < 0.0)


def test_block_proposal_respects_dynamic_state_budget_not_source_count():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.1], ["u"])
    proposal = {
        "relative_gain": 0.1,
        "sources": (
            {
                "plan_index": 0,
                "target": 0,
                "parents": ("u",),
                "family": (0, None),
                "weight": -0.2,
                "normalized_weight": -2.0,
            },
            {
                "plan_index": 1,
                "target": 0,
                "parents": ("u", "u"),
                "family": (0, None),
                "weight": 0.05,
                "normalized_weight": 0.5,
            },
            {
                "plan_index": 2,
                "target": 1,
                "parents": ("u",),
                "family": (1, None),
                "weight": -0.1,
                "normalized_weight": -1.0,
            },
        ),
    }
    fitted = _fit_proposal_to_budget(graph, proposal, max_nodes=1)
    assert fitted is not None
    assert {item["family"] for item in fitted["sources"]} == {(0, None)}
    assert len(fitted["sources"]) == 2


def test_package_uses_block_sparse_search_with_exact_fallbacks_available():
    assert residual_state_runtime.weighted_score_parent_batch is block_sparse_score_parent_batch
    assert (
        residual_state_runtime.select_candidate_action
        is block_sparse_select_candidate_action
    )
    assert state_search_runtime._split_proposals is screened_split_proposals
    assert callable(max_aligned_score_parent_batch)
    assert callable(bounded_max_aligned_select_candidate_action)
