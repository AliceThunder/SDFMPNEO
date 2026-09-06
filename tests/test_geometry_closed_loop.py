import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.geometry_operator import GeometryConditionedAnalyticEvolutionOperator
from sdfmpneo.electrothermal import CertifiedElectroThermalVectorField
from sdfmpneo.em import ParametricEMProblem, ReducedEMModel
from sdfmpneo.geometry_family import CertifiedGeometryElectroThermalFamily
from sdfmpneo.thermal import ThermalSpectralModel
from sdfmpneo.training.geometry_residual import GeometryElectroThermalResidual
from sdfmpneo.training.parametric_growth import (
    ParametricGrowthCandidate,
    ParametricQuadratureSample,
    ParametricTangentResidualGrower,
)
from sdfmpneo.training.parametric_residual import AffineOperatingRHSMap


def _thermal(angle: float, lam: float) -> ThermalSpectralModel:
    c, s = np.cos(angle), np.sin(angle)
    Q = np.array([[c, -s], [s, c]])
    M = np.eye(2)
    K = lam * np.eye(2)
    return ThermalSpectralModel(M=M, K=K, Phi=Q, lambdas=np.array([lam, lam]))


def _em_model():
    problem = ParametricEMProblem(
        A0=np.eye(2, dtype=complex),
        A_state=np.zeros((2, 2, 2), dtype=complex),
        b=np.zeros(2, dtype=complex),
        H_metric=np.eye(2),
        H_loss=np.zeros((2, 2, 2), dtype=complex),
    )
    return ReducedEMModel(problem, np.eye(2, dtype=complex))


def test_one_graph_handles_geometry_rotating_a_degenerate_thermal_subspace():
    reference = _thermal(0.0, 1.0)
    em = _em_model()
    rhs_map = AffineOperatingRHSMap(np.zeros(2), np.zeros((2, 1)))

    def field_factory(g):
        value = float(g[0])
        local = _thermal(value, 1.0 + 0.5 * value)
        return CertifiedElectroThermalVectorField(local, em, rhs_map=rhs_map)

    family = CertifiedGeometryElectroThermalFamily(reference, ("g",), field_factory)
    chart = family.chart(np.array([0.4]))
    # Degenerate spectral-subspace Procrustes removes the arbitrary local rotation.
    assert np.allclose(chart.canonical_to_local.T @ chart.canonical_to_local, np.eye(2))
    assert len(chart.spectral_clusters) == 1
    assert chart.spectral_clusters[0].size == 2

    graph = ParametricAnalyticEvolutionGraph(reference.lambdas, ("g", "u"))
    operator = GeometryConditionedAnalyticEvolutionOperator(
        graph, family, physical_operating_names=("u",)
    )
    pred = operator.evaluate(
        1.0,
        geometry=np.array([0.4]),
        a0=np.array([1.0, 0.0]),
        operating=np.array([0.0]),
    )
    assert pred.residual_norm > 0.0

    residual = GeometryElectroThermalResidual(graph, family, n_physical_operating=1)
    sample = ParametricQuadratureSample(
        time=1.0,
        initial=np.array([1.0, 0.0]),
        operating=np.array([0.4, 0.0]),
        weight=1.0,
    )
    grower = ParametricTangentResidualGrower(graph, residual, (sample,))
    score = grower.score(ParametricGrowthCandidate(0, ("a0_0", "g")))
    assert score.predicted_decrease > 0.0
    assert score.proposed_weight < 0.0
