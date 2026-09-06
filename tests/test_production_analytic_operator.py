import numpy as np

from sdfmpneo import CertifiedAnalyticEvolutionOperator, CertifiedElectroThermalVectorField
from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.em import ParametricEMProblem, ReducedEMModel
from sdfmpneo.thermal import ThermalSpectralModel
from sdfmpneo.training import AffineOperatingRHSMap


def test_analytic_operator_solves_closed_reduced_physics_without_time_marching():
    problem = ParametricEMProblem(
        A0=np.array([[1.0 + 0.0j]]),
        A_state=np.zeros((1, 1, 1), dtype=complex),
        b=np.zeros(1, dtype=complex),
        H_metric=np.eye(1),
        H_loss=np.ones((1, 1, 1), dtype=complex),
    )
    em = ReducedEMModel(problem, np.eye(1, dtype=complex))
    thermal = ThermalSpectralModel(
        M=np.eye(1), K=np.array([[2.0]]), Phi=np.eye(1), lambdas=np.array([2.0])
    )
    rhs_map = AffineOperatingRHSMap(np.zeros(1, dtype=complex), np.ones((1, 1), dtype=complex))
    field = CertifiedElectroThermalVectorField(thermal, em, rhs_map=rhs_map)

    graph = ParametricAnalyticEvolutionGraph([2.0], ["u"])
    graph.add_product_response("u2", 0, ["u", "u"], 1.0)
    operator = CertifiedAnalyticEvolutionOperator(graph, field)

    prediction = operator.evaluate(0.7, a0=np.array([0.3]), operating=np.array([1.2]))
    expected = 0.3 * np.exp(-1.4) + (1.2**2 / 2.0) * (1.0 - np.exp(-1.4))
    assert np.allclose(prediction.state, [expected], rtol=1e-11, atol=1e-12)
    assert prediction.residual_norm < 1e-10
    ir = operator.canonical_ir()
    assert ir.compression_error_bound == 0.0
    assert len(ir.mode_terms) == 1
