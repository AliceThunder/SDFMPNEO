import numpy as np

from sdfmpneo.analytic import (
    ParametricAnalyticEvolutionGraph,
    evaluate_parametric_stable,
    parametric_backend_consistency_defect,
)


def test_parametric_stable_backend_handles_near_resonance_without_threshold():
    lambdas = np.array([1.0, 1.0 - 1e-12])
    graph = ParametricAnalyticEvolutionGraph(lambdas, ["gain"])
    graph.add_product_response(
        "near_resonant",
        0,
        ("gain", "a0_1"),
        1.0,
    )

    a0 = np.array([0.0, 1.0])
    u = np.array([2.0])
    t = 1.0
    stable_a, stable_da = evaluate_parametric_stable(
        graph,
        t,
        a0=a0,
        operating=u,
    )

    delta = lambdas[0] - lambdas[1]
    expected_h = u[0] * np.exp(-lambdas[0] * t) * np.expm1(delta * t) / delta
    expected_dh = u[0] * np.exp(-lambdas[1] * t) - lambdas[0] * expected_h
    assert np.allclose(stable_a[0], expected_h, rtol=2e-14, atol=2e-15)
    assert np.allclose(stable_da[0], expected_dh, rtol=2e-14, atol=2e-15)

    assert parametric_backend_consistency_defect(
        graph,
        t,
        a0=a0,
        operating=u,
    ) > 1e-10


def test_parametric_stable_backend_preserves_arbitrary_initial_condition_at_zero():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.4], ["u"])
    graph.add_product_response("n1", 0, ("u", "a0_1"), 0.3)
    graph.add_product_response("n2", 1, ("n1", "u"), -0.2)

    a0 = np.array([0.4, -0.6])
    stable_a, _ = evaluate_parametric_stable(
        graph,
        0.0,
        a0=a0,
        operating=np.array([1.5]),
    )
    assert np.allclose(stable_a, a0)
