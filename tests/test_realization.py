import numpy as np

from sdfmpneo.analytic import AnalyticEvolutionGraph, AnalyticRealization


def test_exact_resonance_is_handled_without_rate_difference_branch():
    lam = 1.0
    source = AnalyticRealization.decay(lam, 1.0)
    response = source.response(lam)

    for t in [0.0, 0.2, 1.0, 3.0]:
        expected = t * np.exp(-lam * t)
        assert np.allclose(response.evaluate(t), expected, rtol=1e-13, atol=1e-14)
        lhs = response.derivative_value(t) + lam * response.evaluate(t)
        assert np.allclose(lhs, np.exp(-lam * t), rtol=1e-13, atol=1e-14)


def test_near_resonance_remains_accurate_without_closeness_threshold():
    lam = 1.0
    rho = 1.0 - 1e-12
    source = AnalyticRealization.decay(rho, 1.0)
    response = source.response(lam)

    t = 1.0
    delta = lam - rho
    expected = np.exp(-lam * t) * np.expm1(delta * t) / delta
    assert np.allclose(response.evaluate(t), expected, rtol=2e-14, atol=2e-15)

    lhs = response.derivative_value(t) + lam * response.evaluate(t)
    assert np.allclose(lhs, np.exp(-rho * t), rtol=2e-14, atol=2e-15)


def test_product_and_response_realization_closure():
    f = AnalyticRealization.decay(0.7, 1.2)
    g = AnalyticRealization.decay(1.1, -0.4)
    source = f.product(g)
    response = source.response(0.9)

    for t in [0.0, 0.3, 1.2]:
        product_value = f.evaluate(t) * g.evaluate(t)
        assert np.allclose(source.evaluate(t), product_value, rtol=1e-13, atol=1e-14)
        lhs = response.derivative_value(t) + 0.9 * response.evaluate(t)
        assert np.allclose(lhs, product_value, rtol=1e-12, atol=1e-13)


def test_graph_stable_backend_detects_canonical_near_resonance_conditioning():
    lam = np.array([1.0, 1.0 - 1e-12])
    graph = AnalyticEvolutionGraph(lam, np.array([0.0, 1.0]))
    graph.add_product_response("near_resonant", 0, ("base_1",), 1.0)

    t = 1.0
    stable_a, stable_da = graph.evaluate_stable(t)
    delta = lam[0] - lam[1]
    expected = np.exp(-lam[0] * t) * np.expm1(delta * t) / delta
    expected_da = np.exp(-lam[1] * t) - lam[0] * expected

    assert np.allclose(stable_a[0], expected, rtol=2e-14, atol=2e-15)
    assert np.allclose(stable_da[0], expected_da, rtol=2e-14, atol=2e-15)

    # The fast canonical backend is intentionally retained for deployment, but
    # this independent realization backend exposes cancellation when decay rates
    # nearly coalesce. No hand-chosen closeness threshold is involved.
    assert graph.backend_consistency_defect(t) > 1e-10
