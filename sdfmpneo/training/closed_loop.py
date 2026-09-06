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

    Quadrature samples construct tangent candidates; they never certify the final
    graph.  A caller-supplied continuous-domain certificate is evaluated after
    each accepted topology update. Exhausting ``work_budget`` returns
    ``indeterminate`` and cannot create a false training certificate.
    """

    def __init__(self, graph, vector_field, samples: Sequence[ParametricQuadratureSample]):
        self.graph = graph
        self.vector_field = vector_field
        self.samples = tuple(samples)
        if vector_field.rhs_map is None:
            raise ValueError("closed-loop parametric training requires vector_field.rhs_map")
        self.residual_model = ParametricElectroThermalResidual.from_vector_field(
            graph, vector_field
        )

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
            residual_model = ParametricElectroThermalResidual.from_vector_field(
                graph, self.vector_field
            )
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
                return ClosedLoopTrainingResult(
                    graph, "certified", accepted, certificate, degree, work
                )

        self.graph = graph
        certificate = certify(graph)
        return ClosedLoopTrainingResult(
            graph,
            "certified" if getattr(certificate, "status", None) == "certified" else "indeterminate",
            accepted,
            certificate,
            degree,
            work,
        )
