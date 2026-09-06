from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterable, Sequence, Tuple

import numpy as np

from sdfmpneo.analytic import AnalyticEvolutionGraph, AnalyticSeries, solve_response_series


@dataclass(frozen=True)
class AnalyticGrowthCandidate:
    target_mode: int
    parents: Tuple[str, ...]


@dataclass(frozen=True)
class GrowthScore:
    candidate: AnalyticGrowthCandidate
    proposed_weight: float
    predicted_decrease: float
    tangent_norm2: float
    residual_inner_tangent: float


@dataclass(frozen=True)
class GrowthProposal:
    score: GrowthScore
    current_objective: float
    trial_objective: float
    accepted_by_actual_residual: bool
    trial_graph: AnalyticEvolutionGraph


def product_candidates(graph: AnalyticEvolutionGraph, degree: int) -> Tuple[AnalyticGrowthCandidate, ...]:
    """Return the complete commutative product-dictionary layer of a given degree."""
    if degree <= 0:
        raise ValueError("degree must be positive")
    names = [node.name for node in graph.base_nodes] + [node.name for node in graph.response_nodes]
    existing = {(node.target_mode, tuple(sorted(node.parents))) for node in graph.response_nodes}
    out = []
    for parents in combinations_with_replacement(names, degree):
        canonical = tuple(sorted(parents))
        for target in range(graph.n_modes):
            if (target, canonical) not in existing:
                out.append(AnalyticGrowthCandidate(target, canonical))
    return tuple(out)


class TangentResidualGrower:
    """Deterministic residual-sensitivity enrichment on a supplied quadrature rule.

    For a unit-weight candidate response h in target mode i, the exact thermal
    linear part contributes e_i*psi because (d/dt+lambda_i)h=psi. The first
    variation of the complete nonlinear residual is therefore

        D = e_i*psi - J_g(a)[:, i]*h.

    The scalar candidate weight minimizing the quadratic tangent residual is
    obtained in closed form. A trial graph is then evaluated with the full
    nonlinear residual, so a tangent prediction is never silently reported as
    an actual residual decrease.
    """

    def __init__(
        self,
        graph: AnalyticEvolutionGraph,
        em_model,
        times: Sequence[float],
        quadrature_weights: Sequence[float] | None = None,
    ):
        self.graph = graph
        self.em_model = em_model
        self.times = np.asarray(times, dtype=float)
        if self.times.ndim != 1 or self.times.size == 0:
            raise ValueError("times must be a non-empty one-dimensional sequence")
        if np.any(self.times < 0):
            raise ValueError("times must be non-negative")
        if quadrature_weights is None:
            self.weights = np.ones_like(self.times)
        else:
            self.weights = np.asarray(quadrature_weights, dtype=float)
            if self.weights.shape != self.times.shape:
                raise ValueError("quadrature_weights must match times")
            if np.any(self.weights <= 0):
                raise ValueError("quadrature_weights must be positive")

    def _residual_and_jacobian(self, graph: AnalyticEvolutionGraph, t: float):
        a, da = graph.evaluate(float(t))
        g, J = self.em_model.heat_source_and_jacobian(a)
        g = np.asarray(g, dtype=float)
        J = np.asarray(J, dtype=float)
        residual = da + graph.lambdas * a - g
        return residual, J

    def objective(self, graph: AnalyticEvolutionGraph | None = None) -> float:
        active = self.graph if graph is None else graph
        total = 0.0
        for q, t in enumerate(self.times):
            residual, _ = self._residual_and_jacobian(active, float(t))
            total += self.weights[q] * float(residual @ residual)
        return total

    def _candidate_series(self, candidate: AnalyticGrowthCandidate):
        compiled = self.graph.compile()
        source = AnalyticSeries.constant(self.graph.n_modes, 1.0)
        for parent in candidate.parents:
            source = source * compiled.series_for_node(parent)
        response = solve_response_series(source, candidate.target_mode, self.graph.lambdas)
        return source, response

    def score(self, candidate: AnalyticGrowthCandidate) -> GrowthScore:
        if not 0 <= candidate.target_mode < self.graph.n_modes:
            raise ValueError("candidate target_mode out of range")
        known = set(self.graph.compile().node_series)
        missing = [parent for parent in candidate.parents if parent not in known]
        if missing:
            raise ValueError(f"Unknown candidate parents: {missing}")

        source, response = self._candidate_series(candidate)
        inner = 0.0
        norm2 = 0.0
        e = np.zeros(self.graph.n_modes, dtype=float)
        e[candidate.target_mode] = 1.0

        for q, t in enumerate(self.times):
            residual, J = self._residual_and_jacobian(self.graph, float(t))
            psi = float(np.real(source.evaluate(float(t), self.graph.lambdas)))
            h = float(np.real(response.evaluate(float(t), self.graph.lambdas)))
            tangent = e * psi - J[:, candidate.target_mode] * h
            weight_q = self.weights[q]
            inner += weight_q * float(residual @ tangent)
            norm2 += weight_q * float(tangent @ tangent)

        if norm2 == 0.0:
            proposed_weight = 0.0
            decrease = 0.0
        else:
            proposed_weight = -inner / norm2
            decrease = (inner * inner) / norm2

        return GrowthScore(candidate, proposed_weight, decrease, norm2, inner)

    def best(self, candidates: Iterable[AnalyticGrowthCandidate]) -> GrowthScore:
        scored = [self.score(candidate) for candidate in candidates]
        if not scored:
            raise ValueError("candidates cannot be empty")
        return max(scored, key=lambda item: item.predicted_decrease)

    def propose(self, candidates: Iterable[AnalyticGrowthCandidate], name: str) -> GrowthProposal:
        score = self.best(candidates)
        current = self.objective(self.graph)
        trial = self.graph.clone()
        trial.add_product_response(
            name,
            score.candidate.target_mode,
            score.candidate.parents,
            score.proposed_weight,
        )
        actual = self.objective(trial)
        return GrowthProposal(score, current, actual, actual <= current, trial)
