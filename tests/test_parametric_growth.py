import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.em import ParametricEMProblem, ReducedEMModel
from sdfmpneo.training import (
    AffineOperatingRHSMap,
    ParametricElectroThermalResidual,
    ParametricQuadratureSample,
    ParametricTangentResidualGrower,
    parametric_product_candidates,
)


def make_growth_problem():
    problem = ParametricEMProblem(
        A0=np.array([[2.0]], dtype=complex),
        A_state=np.zeros((1, 1, 1), dtype=complex),
        b=np.array([0.0], dtype=complex),
        H_metric=np.array([[1.0]], dtype=complex),
        H_loss=np.array([[[1.0]]], dtype=complex),
    )
    em = ReducedEMModel(problem, np.array([[1.0]], dtype=complex))
    graph = ParametricAnalyticEvolutionGraph([1.0], ["U"])
    rhs_map = AffineOperatingRHSMap(
        offset=np.array([0.0], dtype=complex),
        matrix=np.array([[1.0]], dtype=complex),
    )
    residual = ParametricElectroThermalResidual(graph, em, rhs_map)
    return graph, residual


def test_parametric_residual_growth_identifies_exact_U_squared_source():
    graph, residual = make_growth_problem()

    # q_em=U^2/4 while the base analytic graph has da+a=0. The exact missing
    # source is therefore (1/4) U^2. With a0=0, all degree-two candidates that
    # contain a0 vanish, so the residual criterion must select U*U.
    samples = [
        ParametricQuadratureSample(0.0, np.array([0.0]), np.array([0.5]), 1.0),
        ParametricQuadratureSample(0.4, np.array([0.0]), np.array([1.0]), 1.0),
        ParametricQuadratureSample(1.1, np.array([0.0]), np.array([1.5]), 1.0),
    ]
    grower = ParametricTangentResidualGrower(graph, residual, samples)
    candidates = parametric_product_candidates(graph, degree=2)
    best = grower.best(candidates)

    assert best.candidate.target_mode == 0
    assert best.candidate.parents == ("U", "U")
    assert np.allclose(best.proposed_weight, 0.25, rtol=1e-13, atol=1e-14)

    proposal = grower.propose(candidates, name="learned_U2")
    assert proposal.accepted_by_actual_residual
    assert proposal.trial_objective < 1e-24
    assert proposal.trial_objective < proposal.current_objective
