from __future__ import annotations

import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.state_graph import (
    clone_state_graph,
    replace_source_weights,
    response_source_count,
    response_sources,
    split_response_source,
    weight_parameter_count,
)
from sdfmpneo.analytic.state_realization import (
    evaluate_state_stable,
    state_weight_value_jacobian,
)
from sdfmpneo.training.state_persistence_runtime import (
    load_graph_from_metadata,
    serialize_graph,
)


def _multi_source_graph():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.2], ["u0", "u1"])
    graph.add_response_state(
        "h0",
        0,
        (
            (("u0",), 0.6),
            (("u1",), -0.25),
            (("u0", "u1"), 0.12),
        ),
    )
    graph.add_response_state(
        "h1",
        1,
        (
            (("h0", "u0"), 0.35),
            (("h0", "u1"), -0.18),
        ),
    )
    graph.add_product_response("h2", 0, ("h1", "u0"), 0.08)
    return graph


def test_enrich_is_an_unbounded_active_source_set_not_a_fixed_k_block():
    graph = ParametricAnalyticEvolutionGraph([0.9], ["u0", "u1"])
    graph.add_product_response("state", 0, (), 0.2)
    columns = [
        ("u0",),
        ("u1",),
        ("u0", "u0"),
        ("u0", "u1"),
        ("u1", "u1"),
        ("a0_0",),
        ("a0_0", "u0"),
        ("a0_0", "u1"),
    ]
    for index, parents in enumerate(columns, start=1):
        graph.enrich_response_state("state", parents, 0.01 * index)
    assert response_source_count(graph, "state") == 1 + len(columns)
    assert weight_parameter_count(graph) == 1 + len(columns)


def test_multi_source_state_equals_sum_of_scalar_response_nodes():
    aggregated = ParametricAnalyticEvolutionGraph([0.8], ["u0", "u1"])
    aggregated.add_response_state(
        "state", 0, ((("u0",), 0.4), (("u1",), -0.3))
    )
    scalar = ParametricAnalyticEvolutionGraph([0.8], ["u0", "u1"])
    scalar.add_product_response("a", 0, ("u0",), 0.4)
    scalar.add_product_response("b", 0, ("u1",), -0.3)
    for time in (0.0, 0.03, 2.0, 1.0e5, np.inf):
        actual = evaluate_state_stable(
            aggregated, time, a0=np.array([0.2]), operating=np.array([1.1, 0.7])
        )
        expected = evaluate_state_stable(
            scalar, time, a0=np.array([0.2]), operating=np.array([1.1, 0.7])
        )
        assert np.allclose(actual[0], expected[0], rtol=2e-12, atol=2e-13)
        assert np.allclose(actual[1], expected[1], rtol=2e-12, atol=2e-13)


def test_recursive_split_preserves_the_complete_downstream_dag_exactly():
    graph = _multi_source_graph()
    split = split_response_source(graph, "h0", 1, "h0_branch")
    assert len(split.response_nodes) > len(graph.response_nodes)
    # h1 and h2 must also be branched to preserve their dependence on the old
    # aggregate h0/h1 states while exposing the selected source independently.
    assert any(node.name.startswith("h1_split_") for node in split.response_nodes)
    assert any(node.name.startswith("h2_split_") for node in split.response_nodes)

    a0 = np.array([0.15, -0.05])
    u = np.array([1.2, 0.65])
    for time in (0.0, 0.02, 0.7, 25.0, np.inf):
        before = evaluate_state_stable(graph, time, a0=a0, operating=u)
        after = evaluate_state_stable(split, time, a0=a0, operating=u)
        assert np.allclose(after[0], before[0], rtol=3e-12, atol=3e-13)
        assert np.allclose(after[1], before[1], rtol=3e-12, atol=3e-13)


def test_exact_multi_source_weight_jacobian_matches_central_difference():
    graph = _multi_source_graph()
    point = np.array([0.1, -0.08, 1.1, 0.75, 0.4])
    a, da, ja, jda = state_weight_value_jacobian(graph, point)
    assert ja.shape == (2, weight_parameter_count(graph))
    assert jda.shape == ja.shape

    weights = np.array(
        [source.weight.real for node in graph.response_nodes for source in response_sources(graph, node)]
    )
    step = 1.0e-6
    for index in range(weights.size):
        plus = clone_state_graph(graph)
        minus = clone_state_graph(graph)
        wp, wm = weights.copy(), weights.copy()
        wp[index] += step
        wm[index] -= step
        replace_source_weights(plus, wp)
        replace_source_weights(minus, wm)
        ap, dap = evaluate_state_stable(
            plus, point[-1], a0=point[:2], operating=point[2:4]
        )
        am, dam = evaluate_state_stable(
            minus, point[-1], a0=point[:2], operating=point[2:4]
        )
        assert np.allclose(ja[:, index], (ap - am) / (2 * step), rtol=2e-7, atol=2e-9)
        assert np.allclose(jda[:, index], (dap - dam) / (2 * step), rtol=2e-7, atol=2e-9)


def test_state_metadata_roundtrip_preserves_every_active_source():
    graph = _multi_source_graph()
    nodes = serialize_graph(graph)
    restored = load_graph_from_metadata(graph.lambdas, graph.operating_names, nodes)
    assert serialize_graph(restored) == nodes
    for time in (0.1, 10.0, np.inf):
        left = evaluate_state_stable(
            graph, time, a0=np.array([0.2, -0.1]), operating=np.array([0.9, 1.3])
        )
        right = evaluate_state_stable(
            restored, time, a0=np.array([0.2, -0.1]), operating=np.array([0.9, 1.3])
        )
        assert np.allclose(left[0], right[0], rtol=2e-12, atol=2e-13)
        assert np.allclose(left[1], right[1], rtol=2e-12, atol=2e-13)


def test_legacy_single_source_with_multiple_response_parents_stays_supported():
    graph = ParametricAnalyticEvolutionGraph([0.6, 1.1], ["u"])
    graph.add_product_response("left", 0, ("u",), 0.3)
    graph.add_product_response("right", 1, ("u",), -0.2)
    # Historical max_parent_responses > 1 graphs are still legal when this is a
    # single source column. The one-dynamic-parent restriction only applies to
    # aggregation of several source columns into one analytic state.
    graph.add_product_response("mixed", 0, ("left", "right"), 0.07)

    cloned = clone_state_graph(graph)
    restored = load_graph_from_metadata(
        graph.lambdas, graph.operating_names, serialize_graph(graph)
    )
    assert response_source_count(cloned, "mixed") == 1
    assert response_source_count(restored, "mixed") == 1
    for time in (0.03, 0.8, 50.0, np.inf):
        expected = evaluate_state_stable(graph, time, a0=[0.1, -0.05], operating=[1.2])
        for candidate in (cloned, restored):
            actual = evaluate_state_stable(
                candidate, time, a0=[0.1, -0.05], operating=[1.2]
            )
            assert np.allclose(actual[0], expected[0], rtol=3e-12, atol=3e-13)
            assert np.allclose(actual[1], expected[1], rtol=3e-12, atol=3e-13)


def test_fixed_research_checkpoint_roundtrip_keeps_multi_source_states(tmp_path):
    from sdfmpneo import ResearchElectroThermalModel, demo_research_model

    model = demo_research_model()
    graph = ParametricAnalyticEvolutionGraph(
        model.core.thermal_model.lambdas, ["u0", "u1"]
    )
    graph.add_response_state(
        "heating",
        0,
        (
            (("u0",), 2.0e-5),
            (("u1",), -3.0e-5),
            (("u0", "u1"), 1.0e-8),
        ),
    )
    model.graph = graph
    model.training_config = None
    model.training_report = None

    path = model.save(tmp_path / "multi_source_fixed.npz")
    loaded = ResearchElectroThermalModel.load(path)
    assert serialize_graph(loaded.graph) == serialize_graph(graph)
    assert loaded.graph.response_source_count("heating") == 3

    for time in (0.0, 0.2, 1000.0, np.inf):
        expected = model.predict(
            time, a0=[0.4], operating=[20.0, 8.0], diagnostics=False
        )
        actual = loaded.predict(
            time, a0=[0.4], operating=[20.0, 8.0], diagnostics=False
        )
        assert np.allclose(
            actual["thermal_coordinates"],
            expected["thermal_coordinates"],
            rtol=2e-12,
            atol=2e-13,
        )
        assert np.allclose(
            actual["thermal_derivative"],
            expected["thermal_derivative"],
            rtol=2e-12,
            atol=2e-13,
        )


def test_package_routes_training_to_residual_driven_state_runtime():
    from sdfmpneo.training import research as training_research
    from sdfmpneo.training.residual_state_runtime import (
        residual_driven_state_train,
        state_refine_weights,
    )

    assert training_research._refine_weights is state_refine_weights
    assert training_research.train_research_graph is residual_driven_state_train
