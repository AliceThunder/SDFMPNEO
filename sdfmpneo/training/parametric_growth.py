from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterable, Sequence, Tuple

import numpy as np

from sdfmpneo.analytic import (
    ParametricAnalyticEvolutionGraph,
    ParametricAnalyticSeries,
    solve_parametric_response_series,
)

from .parametric_residual import ParametricElectroThermalResidual


@dataclass(frozen=True)
class ParametricQuadratureSample:
    time: float
    initial: np.ndarray
    operating: np.ndarray
    weight: float

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError("time must be non-negative")
        if self.weight <= 0:
            raise ValueError("quadrature weight must be positive")
        object.__setattr__(self, "initial", np.asarray(self.initial, dtype=float))
        object.__setattr__(self, "operating", np.asarray(self.operating, dtype=float))


@dataclass(frozen=True)
class ParametricGrowthCandidate:
    target_mode: int
    parents: Tuple[str, ...]


@dataclass(frozen=True)
class ParametricGrowthScore:
    candidate: ParametricGrowthCandidate
    proposed_weight: float
    predicted_decrease: float
    tangent_norm2: float
    residual_inner_tangent: float


@dataclass(frozen=True)
class ParametricGrowthProposal:
    score: ParametricGrowthScore
    current_objective: float
    trial_objective: float
    accepted_by_actual_residual: bool
    trial_graph: ParametricAnalyticEvolutionGraph


def parametric_product_candidates(
    graph: ParametricAnalyticEvolutionGraph,
    degree: int,
) -> Tuple[ParametricGrowthCandidate, ...]:
    if degree <= 0:
        raise ValueError("degree must be positive")
    names = graph.known_names()
    existing = {
        (node.target_mode, tuple(sorted(node.parents)))
        for node in graph.response_nodes
    }
    out = []
    for parents in combinations_with_replacement(names, degree):
        canonical = tuple(sorted(parents))
        for target in range(graph.n_modes):
            if (target, canonical) not in existing:
                out.append(ParametricGrowthCandidate(target, canonical))
    return tuple(out)


class ParametricTangentResidualGrower:
    """Residual-driven topology growth on an externally supplied domain rule."""

    def __init__(
        self,
        graph: ParametricAnalyticEvolutionGraph,
        residual_model: ParametricElectroThermalResidual,
        samples: Sequence[ParametricQuadratureSample],
    ) -> None:
        if not samples:
            raise ValueError("samples cannot be empty")
        self.graph = graph
        self.residual_model = residual_model
        self.samples = tuple(samples)
        for sample in self.samples:
            if sample.initial.shape != (graph.n_modes,):
                raise ValueError("sample initial-state dimension mismatch")
            if sample.operating.shape != (len(graph.operating_names),):
                raise ValueError("sample operating dimension mismatch")

    @staticmethod
    def _clone(graph: ParametricAnalyticEvolutionGraph) -> ParametricAnalyticEvolutionGraph:
        out = ParametricAnalyticEvolutionGraph(graph.lambdas.copy(), graph.operating_names)
        for node in graph.response_nodes:
            out.add_product_response(node.name, node.target_mode, node.parents, node.weight)
        return out

    def objective(self, graph: ParametricAnalyticEvolutionGraph | None = None) -> float:
        active = self.graph if graph is None else graph
        residual_model = (
            self.residual_model
            if active is self.graph
            else ParametricElectroThermalResidual(
                active,
                self.residual_model.em_model,
                self.residual_model.rhs_map,
                thermal_forcing=self.residual_model.thermal_forcing,
            )
        )
        total = 0.0
        for sample in self.samples:
            value = residual_model.evaluate(
                sample.time,
                a0=sample.initial,
                operating=sample.operating,
            )
            total += sample.weight * float(value.residual @ value.residual)
        return total

    def _candidate_series(self, candidate: ParametricGrowthCandidate):
        compiled = self.graph.compile()
        n_parameters = self.graph.n_parameters
        source = ParametricAnalyticSeries.constant(
            self.graph.n_modes, n_parameters, 1.0
        )
        for parent in candidate.parents:
            source = source * compiled.node_series[parent]
        response = solve_parametric_response_series(
            source, candidate.target_mode, self.graph.lambdas
        )
        return source, response

    def score(self, candidate: ParametricGrowthCandidate) -> ParametricGrowthScore:
        if not 0 <= candidate.target_mode < self.graph.n_modes:
            raise ValueError("candidate target_mode out of range")
        known = set(self.graph.known_names())
        missing = [parent for parent in candidate.parents if parent not in known]
        if missing:
            raise ValueError(f"unknown candidate parents: {missing}")

        source, response = self._candidate_series(candidate)
        inner = 0.0
        norm2 = 0.0
        e = np.zeros(self.graph.n_modes, dtype=float)
        e[candidate.target_mode] = 1.0
        compiled = self.graph.compile()

        for sample in self.samples:
            state = self.residual_model.evaluate(
                sample.time,
                a0=sample.initial,
                operating=sample.operating,
            )
            _, J_g_a = self.residual_model.em_model.heat_source_and_jacobian_for_rhs(
                state.a, state.rhs
            )
            parameters = compiled.parameter_vector(sample.initial, sample.operating)
            psi = float(np.real(source.evaluate(sample.time, self.graph.lambdas, parameters)))
            h = float(np.real(response.evaluate(sample.time, self.graph.lambdas, parameters)))
            tangent = e * psi - np.asarray(J_g_a)[:, candidate.target_mode] * h
            inner += sample.weight * float(state.residual @ tangent)
            norm2 += sample.weight * float(tangent @ tangent)

        if norm2 == 0.0:
            proposed = 0.0
            decrease = 0.0
        else:
            proposed = -inner / norm2
            decrease = inner * inner / norm2
        return ParametricGrowthScore(candidate, proposed, decrease, norm2, inner)

    def best(self, candidates: Iterable[ParametricGrowthCandidate]) -> ParametricGrowthScore:
        scores = [self.score(candidate) for candidate in candidates]
        if not scores:
            raise ValueError("candidates cannot be empty")
        return max(scores, key=lambda item: item.predicted_decrease)

    def propose(
        self,
        candidates: Iterable[ParametricGrowthCandidate],
        name: str,
    ) -> ParametricGrowthProposal:
        score = self.best(candidates)
        current = self.objective(self.graph)
        trial = self._clone(self.graph)
        trial.add_product_response(
            name,
            score.candidate.target_mode,
            score.candidate.parents,
            score.proposed_weight,
        )
        actual = self.objective(trial)
        return ParametricGrowthProposal(
            score=score,
            current_objective=current,
            trial_objective=actual,
            accepted_by_actual_residual=(actual <= current),
            trial_graph=trial,
        )
