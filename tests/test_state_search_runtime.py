from __future__ import annotations

import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.state_realization import evaluate_state_stable
from sdfmpneo.training import residual_state_runtime
from sdfmpneo.training.state_search_runtime import (
    _max_aligned_target,
    coalesce_unreferenced_state_families,
    max_aligned_score_parent_batch,
    bounded_max_aligned_select_candidate_action,
)
from sdfmpneo.training.state_split_runtime import screened_split_proposals
import sdfmpneo.training.state_search_runtime as state_search_runtime


def test_geometry_seed_family_coalescing_is_function_preserving():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.1], ["g", "u"])
    graph.add_product_response("a", 0, ("g",), 0.3)
    graph.add_product_response("b", 0, ("u",), -0.2)
    graph.add_product_response("c", 0, ("g", "u"), 0.08)
    graph.add_product_response("d", 1, ("u",), 0.15)

    collapsed = coalesce_unreferenced_state_families(graph)
    assert len(graph.response_nodes) == 4
    assert len(collapsed.response_nodes) == 2
    assert collapsed.response_source_count("a") == 3
    assert collapsed.response_source_count("d") == 1

    a0 = np.array([0.1, -0.05])
    operating = np.array([0.4, 1.3])
    for time in (0.0, 1.0e-5, 0.2, 20.0, np.inf):
        before = evaluate_state_stable(
            graph, time, a0=a0, operating=operating
        )
        after = evaluate_state_stable(
            collapsed, time, a0=a0, operating=operating
        )
        assert np.allclose(after[0], before[0], rtol=3e-12, atol=3e-13)
        assert np.allclose(after[1], before[1], rtol=3e-12, atol=3e-13)


def test_seed_coalescing_refuses_graphs_with_dynamic_descendants():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.1], ["u"])
    graph.add_product_response("a", 0, ("u",), 0.3)
    graph.add_product_response("b", 1, ("a", "u"), -0.2)
    assert coalesce_unreferenced_state_families(graph) is graph


def test_minimax_candidate_weight_targets_the_actual_maximum():
    # q0(alpha)=4-4a+a^2 is minimized at a=2; q1(alpha)=1 is fixed.
    # Therefore min max(q0,q1)=1 and the residual maximum drops 2 -> 1.
    old_sq = np.array([4.0, 1.0])
    correlation = np.array([-2.0, 0.0])
    tangent_norm2 = np.array([1.0, 0.0])
    relative, alpha = _max_aligned_target(
        old_sq, correlation, tangent_norm2, np.ones(2)
    )
    assert np.isclose(alpha, 2.0)
    assert np.isclose(relative, 0.5)


def test_package_uses_bounded_max_aligned_state_search():
    assert residual_state_runtime.weighted_score_parent_batch is max_aligned_score_parent_batch
    assert (
        residual_state_runtime.select_candidate_action
        is bounded_max_aligned_select_candidate_action
    )
    assert state_search_runtime._split_proposals is screened_split_proposals
