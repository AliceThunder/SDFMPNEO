from dataclasses import dataclass

import numpy as np

from sdfmpneo import CertifiedElectroThermalVectorField
from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.em import ParametricEMProblem, ReducedEMModel
from sdfmpneo.thermal import ThermalSpectralModel
from sdfmpneo.training import (
    AffineOperatingRHSMap,
    ParametricQuadratureSample,
    SolutionDataFreeClosedLoopTrainer,
)


@dataclass(frozen=True)
class _Certificate:
    status: str


def _trainer(graph):
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
    samples = (
        ParametricQuadratureSample(0.4, np.array([0.2]), np.array([0.8]), 1.0),
        ParametricQuadratureSample(0.8, np.array([-0.1]), np.array([1.2]), 1.0),
    )
    return SolutionDataFreeClosedLoopTrainer(graph, field, samples)


def test_training_stops_only_from_external_continuous_certificate():
    graph = ParametricAnalyticEvolutionGraph([2.0], ["u"])
    graph.add_product_response("u2", 0, ["u", "u"], 1.0)
    result = _trainer(graph).train(lambda active: _Certificate("certified"), work_budget=1)
    assert result.status == "certified"
    assert result.accepted_nodes == 0


def test_training_budget_cannot_create_false_certificate():
    graph = ParametricAnalyticEvolutionGraph([2.0], ["u"])
    result = _trainer(graph).train(lambda active: _Certificate("indeterminate"), work_budget=1)
    assert result.status == "indeterminate"
    assert result.work_used == 1
