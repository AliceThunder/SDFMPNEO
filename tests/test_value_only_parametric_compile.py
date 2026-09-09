import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.parametric_realization import (
    _compile_with_operating_derivatives,
    compile_parametric_realization,
)


def test_value_only_compile_matches_derivative_compiler_structure_and_values():
    graph = ParametricAnalyticEvolutionGraph([0.2, 0.35], ["u", "g"])
    graph.add_product_response("r0", 0, ["u"], 0.4)
    graph.add_product_response("r1", 1, ["a0_0", "g"], -0.2)
    graph.add_product_response("r2", 0, ["g", "r0"], 0.15)
    graph.add_product_response("r3", 1, ["r2", "a0_1"], 0.08)

    a0 = np.array([0.17, -0.09])
    operating = np.array([1.7, -0.35])
    fast = compile_parametric_realization(graph, a0=a0, operating=operating)
    reference, _ = _compile_with_operating_derivatives(
        graph, a0=a0, operating=operating, weight_derivatives=False
    )

    assert fast.node_realizations.keys() == reference.node_realizations.keys()
    assert fast.source_realizations.keys() == reference.source_realizations.keys()
    for name in fast.node_realizations:
        left, right = fast.node_realizations[name], reference.node_realizations[name]
        assert np.array_equal(left.A, right.A)
        assert np.array_equal(left.b, right.b)
        assert np.array_equal(left.c, right.c)
    for name in fast.source_realizations:
        left, right = fast.source_realizations[name], reference.source_realizations[name]
        assert np.array_equal(left.A, right.A)
        assert np.array_equal(left.b, right.b)
        assert np.array_equal(left.c, right.c)
    for left, right in zip(fast.mode_realizations, reference.mode_realizations):
        assert np.array_equal(left.A, right.A)
        assert np.array_equal(left.b, right.b)
        assert np.array_equal(left.c, right.c)
