import numpy as np

from sdfmpneo.analytic import AnalyticEvolutionGraph, AnalyticSeries, DeferredResponseSeries


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
    t = 0.3
    expected = 4.0 * t * np.exp(-2.0 * t)
    assert np.allclose(resp.evaluate(t, lam), expected)
    lhs = resp.derivative_value(t, lam) + lam[0] * resp.evaluate(t, lam)
    assert np.allclose(lhs, src.evaluate(t, lam))


def test_graph_initial_condition():
    graph = AnalyticEvolutionGraph(np.array([1.0, 2.0]), np.array([0.4, -0.2]))
    graph.add_product_response("n1", 0, ("base_0", "base_1"), 0.3)
    a, _ = graph.evaluate(0.0)
    assert np.allclose(a, [0.4, -0.2])
