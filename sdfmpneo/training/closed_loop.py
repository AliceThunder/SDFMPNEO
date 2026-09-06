from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .parametric_growth import (
    ParametricQuadratureSample,
    ParametricTangentResidualGrower,
    parametric_product_candidates,
)
from .parametric_residual import ParametricElectroThermalResidual


@dataclass(frozen=True)
class ClosedLoopTrainingResult:
    graph: object
    status: str
    accepted_nodes: int
    certificate: object
    dictionary_degree: int
    work_used: int


class SolutionDataFreeClosedLoopTrainer:
    """Residual-grown analytic graph driven only by the closed physical model.

    ``physical_model`` can be the fixed-geometry production vector field or any
    residual model implementing ``evaluate/with_graph/vector_field_state_jacobian``
    (including the cross-geometry residual).  The same tangent grower is used in
    both cases.  Samples construct candidates only; the caller-supplied continuous
    domain certificate is the sole stopping criterion.
    """

    def __init__(self, graph, physical_model, samples: Sequence[ParametricQuadratureSample]):
        self.graph = graph
        self.physical_model = physical_model
        self.samples = tuple(samples)
        if all(hasattr(physical_model, name) for name in ("evaluate", "with_graph", "vector_field_state_jacobian")):
            self._base_residual = physical_model
        else:
            vector_field = physical_model
            if getattr(vector_field, "rhs_map", None) is None:
                raise ValueError("closed-loop parametric training requires vector_field.rhs_map")
            self._base_residual = ParametricElectroThermalResidual.from_vector_field(
                graph, vector_field
            )

    def _residual_for(self, graph):
        if graph is self.graph and getattr(self._base_residual, "graph", None) is graph:
            return self._base_residual
        return self._base_residual.with_graph(graph)

    def train(
        self,
        certify: Callable[[object], object],
        *,
        work_budget: int,
    ) -> ClosedLoopTrainingResult:
        if work_budget <= 0:
            raise ValueError("work_budget must be positive")
        graph = self.graph
        degree = 1
        accepted = 0
        work = 0

        certificate = certify(graph)
        if getattr(certificate, "status", None) == "certified":
            return ClosedLoopTrainingResult(graph, "certified", 0, certificate, degree, 0)

        while work < work_budget:
            residual_model = self._residual_for(graph)
            grower = ParametricTangentResidualGrower(graph, residual_model, self.samples)
            candidates = parametric_product_candidates(graph, degree)
            if not candidates:
                degree += 1
                work += 1
                continue
            proposal = grower.propose(candidates, name=f"r{accepted}_d{degree}")
            work += 1
            if proposal.score.predicted_decrease <= 0.0 or not proposal.accepted_by_actual_residual:
                degree += 1
                continue

            graph = proposal.trial_graph
            accepted += 1
            certificate = certify(graph)
            if getattr(certificate, "status", None) == "certified":
                self.graph = graph
                self._base_residual = self._base_residual.with_graph(graph)
                return ClosedLoopTrainingResult(
                    graph, "certified", accepted, certificate, degree, work
                )

        self.graph = graph
        self._base_residual = self._base_residual.with_graph(graph)
        certificate = certify(graph)
        return ClosedLoopTrainingResult(
            graph,
            "certified" if getattr(certificate, "status", None) == "certified" else "indeterminate",
            accepted,
            certificate,
            degree,
            work,
        )
