import numpy as np

from sdfmpneo import (
    ParametricAnalyticEvolutionGraph,
    ParametricExecutableSDFMPNEOModel,
    ThermalSpectralModel,
)
from sdfmpneo.em import ParametricEMProblem, ReducedEMModel
from sdfmpneo.training import AffineOperatingRHSMap


def make_model():
    thermal = ThermalSpectralModel.build(np.array([[1.0]]), np.array([[1.0]]))
    graph = ParametricAnalyticEvolutionGraph(thermal.lambdas, ["source"])
    graph.add_product_response("source_response", 0, ("source",), 1.0)

    problem = ParametricEMProblem(
        A0=np.array([[2.0]], dtype=complex),
        A_state=np.zeros((1, 1, 1), dtype=complex),
        b=np.array([0.0], dtype=complex),
        H_metric=np.array([[1.0]], dtype=complex),
        H_loss=np.array([[[1.0]]], dtype=complex),
    )
    em = ReducedEMModel(problem, np.array([[1.0]], dtype=complex))
    rhs_map = AffineOperatingRHSMap(
        offset=np.array([0.0], dtype=complex),
        matrix=np.array([[1.0]], dtype=complex),
    )
    return ParametricExecutableSDFMPNEOModel(
        evolution=graph,
        thermal_model=thermal,
        electromagnetic_model=em,
        rhs_map=rhs_map,
    )


def test_parametric_online_model_directly_queries_a0_U_and_time():
    model = make_model()
    t = 0.8
    a0 = np.array([0.25])
    u = np.array([1.2])
    result = model.evaluate(t, a0=a0, operating=u)

    expected_a = a0[0] * np.exp(-t) + u[0] * (1.0 - np.exp(-t))
    expected_residual = u[0] - u[0] ** 2 / 4.0
    expected_residual_du = 1.0 - u[0] / 2.0

    assert np.allclose(result.thermal_coordinates[0], expected_a)
    assert np.allclose(result.temperature_field[0], expected_a)
    assert np.allclose(result.drive_rhs, u.astype(complex))
    assert np.allclose(result.physical_residual[0], expected_residual)
    assert np.allclose(result.residual_operating_jacobian[0, 0], expected_residual_du)


def test_same_online_model_accepts_different_initial_and_operating_conditions():
    model = make_model()
    first = model.evaluate(1.0, a0=np.array([0.1]), operating=np.array([0.5]))
    second = model.evaluate(1.0, a0=np.array([0.7]), operating=np.array([1.5]))
    assert not np.allclose(first.thermal_coordinates, second.thermal_coordinates)
    assert not np.allclose(first.reduced_heat_source, second.reduced_heat_source)
