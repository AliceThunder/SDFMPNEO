import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.parametric_realization import (
    compile_parametric_nodes,
    compile_parametric_realization,
)
from sdfmpneo.training import research as training_research
from sdfmpneo.training import late_stage_runtime as late_stage


def test_node_only_compile_matches_full_compile_nodes_and_sources():
    graph = ParametricAnalyticEvolutionGraph([0.2, 0.35], ["u", "g"])
    graph.add_product_response("r0", 0, ["u"], 0.4)
    graph.add_product_response("r1", 1, ["a0_0", "g"], -0.2)
    graph.add_product_response("r2", 0, ["g", "r0"], 0.15)
    graph.add_product_response("r3", 1, ["r2", "a0_1"], 0.08)
    a0 = np.array([0.17, -0.09])
    operating = np.array([1.7, -0.35])

    light = compile_parametric_nodes(graph, a0=a0, operating=operating)
    full = compile_parametric_realization(graph, a0=a0, operating=operating)
    assert light.node_realizations.keys() == full.node_realizations.keys()
    assert light.source_realizations.keys() == full.source_realizations.keys()
    for name in light.node_realizations:
        left, right = light.node_realizations[name], full.node_realizations[name]
        assert np.array_equal(left.A, right.A)
        assert np.array_equal(left.b, right.b)
        assert np.array_equal(left.c, right.c)
    for name in light.source_realizations:
        left, right = light.source_realizations[name], full.source_realizations[name]
        assert np.array_equal(left.A, right.A)
        assert np.array_equal(left.b, right.b)
        assert np.array_equal(left.c, right.c)


def test_legacy_node_only_compile_hot_paths_remain_installed_beneath_fixed_trainer():
    from sdfmpneo.training.fixed_network_runtime import train_fixed_analytic_response_network

    assert training_research.compile_parametric_realization is compile_parametric_nodes
    assert late_stage.compile_parametric_realization is compile_parametric_nodes
    assert callable(late_stage._sparse_refine_weights)
    assert callable(late_stage.optimized_adaptive_train_research_graph)
    assert training_research.train_research_graph is train_fixed_analytic_response_network
