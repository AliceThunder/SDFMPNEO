import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.em import ParametricEMProblem, ReducedEMModel
from sdfmpneo.training import AffineOperatingRHSMap, ParametricElectroThermalResidual


def test_parametric_physical_residual_and_operating_derivative_are_exact():
    # One-dimensional electromagnetic problem: A=2, H_loss=1, b(U)=U.
    # Therefore x=U/2 and q=U^2/4.
    problem = ParametricEMProblem(
        A0=np.array([[2.0]], dtype=complex),
        A_state=np.zeros((1, 1, 1), dtype=complex),
        b=np.array([0.0], dtype=complex),
        H_metric=np.array([[1.0]], dtype=complex),
        H_loss=np.array([[[1.0]]], dtype=complex),
    )
    em = ReducedEMModel(problem, np.array([[1.0]], dtype=complex))

    # a(t)=a0 exp(-t)+U(1-exp(-t)), hence da+a=U exactly.
    graph = ParametricAnalyticEvolutionGraph([1.0], ["U"])
    graph.add_product_response("forcing", 0, ("U",), 1.0)

    rhs_map = AffineOperatingRHSMap(
        offset=np.array([0.0], dtype=complex),
        matrix=np.array([[1.0]], dtype=complex),
    )
    residual_model = ParametricElectroThermalResidual(graph, em, rhs_map)

    u = np.array([1.2])
    sample = residual_model.evaluate(0.7, a0=np.array([0.3]), operating=u)

    expected_residual = u[0] - u[0] ** 2 / 4.0
    expected_derivative = 1.0 - u[0] / 2.0
    assert np.allclose(sample.residual[0], expected_residual, rtol=1e-13, atol=1e-14)
    assert np.allclose(
        sample.residual_operating_jacobian[0, 0],
        expected_derivative,
        rtol=1e-13,
        atol=1e-14,
    )

    # Independent regression check; the model derivative above is analytic.
    h = 1e-6
    plus = residual_model.evaluate(0.7, a0=np.array([0.3]), operating=u + h).residual[0]
    minus = residual_model.evaluate(0.7, a0=np.array([0.3]), operating=u - h).residual[0]
    finite = (plus - minus) / (2 * h)
    assert np.allclose(sample.residual_operating_jacobian[0, 0], finite, rtol=1e-8, atol=1e-10)
