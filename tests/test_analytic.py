import numpy as np

from sdfmpneo.analytic import AnalyticEvolutionGraph, AnalyticSeries, DeferredResponseSeries, solve_response_series


def test_series_product_and_derivative():
    lam = np.array([2.0, 3.0])
    a = AnalyticSeries.decay(2, 0, 2.0)
    b = AnalyticSeries.decay(2, 1, 3.0)
    p = a * b
    t = 0.4
    expected = 6.0 * np.exp(-5.0 * t)
    assert np.allclose(p.evaluate(t, lam), expected)
    assert np.allclose(p.derivative(lam).evaluate(t, lam), -5.0 * expected)


def test_exact_resonant_response():
    lam = np.array([2.0])
    src = AnalyticSeries.decay(1, 0, 4.0)
    resp = DeferredResponseSeries.from_source(src, 0)
    compiled = solve_response_series(src, 0, lam)
    t = 0.3
    expected = 4.0 * t * np.exp(-2.0 * t)
    assert np.allclose(resp.evaluate(t, lam), expected)
    assert np.allclose(compiled.evaluate(t, lam), expected)
    lhs = compiled.derivative(lam).evaluate(t, lam) + lam[0] * compiled.evaluate(t, lam)
    assert np.allclose(lhs, src.evaluate(t, lam))


def test_nonresonant_polynomial_response_identity():
    lam = np.array([1.25, 2.5])
    src = AnalyticSeries(2, {(2, (0, 1)): 0.7})
    compiled = solve_response_series(src, 0, lam)
    for t in [0.0, 0.2, 0.9, 2.0]:
        lhs = compiled.derivative(lam).evaluate(t, lam) + lam[0] * compiled.evaluate(t, lam)
        assert np.allclose(lhs, src.evaluate(t, lam), rtol=1e-11, atol=1e-11)
    assert np.allclose(compiled.evaluate(0.0, lam), 0.0)


def test_graph_initial_condition():
    graph = AnalyticEvolutionGraph(np.array([1.0, 2.0]), np.array([0.4, -0.2]))
    graph.add_product_response("n1", 0, ("base_0", "base_1"), 0.3)
    a, _ = graph.evaluate(0.0)
    assert np.allclose(a, [0.4, -0.2])


def test_second_generation_response_can_use_response_parent():
    lam = np.array([1.0, 2.0])
    graph = AnalyticEvolutionGraph(lam, np.array([0.5, 0.4]))
    graph.add_product_response("n1", 0, ("base_0", "base_1"), 0.3)
    graph.add_product_response("n2", 1, ("n1", "base_0"), 0.5)

    compiled = graph.compile()
    n1 = compiled.series_for_node("n1")
    n2 = compiled.series_for_node("n2")
    src2 = compiled.source_for_node("n2")

    for t in [0.0, 0.1, 0.7, 1.3]:
        lhs2 = n2.derivative(lam).evaluate(t, lam) + lam[1] * n2.evaluate(t, lam)
        assert np.allclose(lhs2, src2.evaluate(t, lam), rtol=1e-11, atol=1e-11)

    assert np.allclose(n1.evaluate(0.0, lam), 0.0)
    assert np.allclose(n2.evaluate(0.0, lam), 0.0)
    a0, _ = graph.evaluate(0.0)
    assert np.allclose(a0, [0.5, 0.4])


def test_third_generation_and_compile_cache_invalidation():
    lam = np.array([0.8, 1.7])
    graph = AnalyticEvolutionGraph(lam, np.array([0.6, -0.25]))
    graph.add_product_response("n1", 0, ("base_0", "base_1"), 0.2)
    first = graph.compile()
    graph.add_product_response("n2", 1, ("n1", "base_0"), -0.4)
    second = graph.compile()
    assert first is not second
    graph.add_product_response("n3", 0, ("n2", "n1"), 0.15)
    third = graph.compile()

    n3 = third.series_for_node("n3")
    src3 = third.source_for_node("n3")
    for t in [0.05, 0.4, 1.1]:
        lhs = n3.derivative(lam).evaluate(t, lam) + lam[0] * n3.evaluate(t, lam)
        assert np.allclose(lhs, src3.evaluate(t, lam), rtol=2e-10, atol=2e-11)
    assert third.term_counts()["n3"] > 0
