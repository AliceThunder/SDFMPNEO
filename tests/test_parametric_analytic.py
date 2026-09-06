import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph


def test_parametric_graph_has_exact_initial_condition_for_arbitrary_operating_point():
    graph = ParametricAnalyticEvolutionGraph([1.0, 2.0], ["current", "load"])
    graph.add_product_response(
        "coupling",
        0,
        ("current", "a0_0", "a0_1"),
        0.3,
    )

    for a0, u in [
        (np.array([0.4, -0.2]), np.array([1.0, 10.0])),
        (np.array([-0.1, 0.7]), np.array([2.5, 25.0])),
    ]:
        a, _ = graph.evaluate(0.0, a0=a0, operating=u)
        assert np.allclose(a, a0)


def test_static_operating_node_is_inside_exact_analytic_response():
    graph = ParametricAnalyticEvolutionGraph([1.0], ["u"])
    graph.add_product_response(
        "u_times_initial",
        0,
        ("u", "a0_0"),
        2.0,
    )

    a0 = np.array([0.4])
    u = np.array([3.0])
    t = 0.7
    a, da = graph.evaluate(t, a0=a0, operating=u)

    expected = a0[0] * np.exp(-t) + 2.0 * u[0] * a0[0] * t * np.exp(-t)
    expected_da = -a0[0] * np.exp(-t) + 2.0 * u[0] * a0[0] * (1.0 - t) * np.exp(-t)
    assert np.allclose(a[0], expected, rtol=1e-13, atol=1e-14)
    assert np.allclose(da[0], expected_da, rtol=1e-13, atol=1e-14)


def test_operating_jacobian_is_analytic_not_finite_difference():
    graph = ParametricAnalyticEvolutionGraph([1.0], ["u"])
    graph.add_product_response("n1", 0, ("u", "a0_0"), 2.0)
    graph.add_product_response("n2", 0, ("u", "u", "a0_0"), -0.5)

    compiled = graph.compile()
    a0 = np.array([0.6])
    u = np.array([1.7])
    t = 0.4
    J = compiled.operating_jacobian(t, a0=a0, operating=u)

    expected = (2.0 * a0[0] - u[0] * a0[0]) * t * np.exp(-t)
    assert np.allclose(J[0, 0], expected, rtol=1e-13, atol=1e-14)


def test_one_topology_queries_multiple_initial_and_operating_conditions():
    graph = ParametricAnalyticEvolutionGraph([0.8], ["power"])
    graph.add_product_response("feedback", 0, ("power", "a0_0"), 0.25)

    result_1 = graph.evaluate(1.0, a0=np.array([0.5]), operating=np.array([1.0]))[0]
    result_2 = graph.evaluate(1.0, a0=np.array([0.2]), operating=np.array([4.0]))[0]
    assert not np.allclose(result_1, result_2)
