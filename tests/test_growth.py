import numpy as np

from sdfmpneo.analytic import AnalyticEvolutionGraph
from sdfmpneo.training import AnalyticGrowthCandidate, TangentResidualGrower, product_candidates


class LinearHeatSource:
    def __init__(self, B):
        self.B = np.asarray(B, dtype=float)

    def heat_source_and_jacobian(self, a):
        a = np.asarray(a, dtype=float)
        return self.B @ a, self.B.copy()


def test_product_candidate_dictionary_is_deterministic_and_excludes_existing():
    graph = AnalyticEvolutionGraph([1.0, 2.0], [0.4, 0.2])
    graph.add_product_response("n1", 0, ("base_0", "base_1"), 0.1)
    candidates = product_candidates(graph, degree=2)
    keys = [(candidate.target_mode, candidate.parents) for candidate in candidates]
    assert (0, ("base_0", "base_1")) not in keys
    assert keys == [
        (candidate.target_mode, candidate.parents)
        for candidate in product_candidates(graph, degree=2)
    ]


def test_tangent_growth_is_exact_for_linear_heat_source():
    graph = AnalyticEvolutionGraph([1.0, 1.8], [0.5, 0.35])
    em = LinearHeatSource([[0.15, 0.08], [0.02, 0.1]])
    times = np.array([0.0, 0.2, 0.7, 1.4])
    weights = np.array([0.1, 0.4, 0.35, 0.15])
    grower = TangentResidualGrower(graph, em, times, weights)
    candidate = AnalyticGrowthCandidate(0, ("base_0", "base_1"))
    score = grower.score(candidate)
    assert score.predicted_decrease >= 0.0

    proposal = grower.propose([candidate], "g1")
    assert proposal.accepted_by_actual_residual
    assert proposal.trial_objective <= proposal.current_objective + 1e-14

    actual_decrease = proposal.current_objective - proposal.trial_objective
    assert np.allclose(actual_decrease, score.predicted_decrease, rtol=1e-11, atol=1e-12)
