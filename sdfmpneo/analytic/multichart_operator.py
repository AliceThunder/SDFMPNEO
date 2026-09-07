from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MultiChartGeometryAnalyticPrediction:
    chart_name: str
    prediction: object


class MultiChartGeometryAnalyticEvolutionOperator:
    """Dispatch certified analytic operators over a discrete geometry atlas.

    Each chart keeps its own reference thermal spectrum and analytic graph.  This
    is intentional: a remeshing/topology change is a discrete chart transition,
    not a fictitious continuous parameter.  Within each chart the existing
    ``GeometryConditionedAnalyticEvolutionOperator`` still supplies one fixed DAG
    over ``(G,U,a0,t)``.
    """

    def __init__(self, operators: Mapping[str, object]) -> None:
        mapping = {str(name): operator for name, operator in operators.items()}
        if not mapping or any(not name.strip() for name in mapping):
            raise ValueError("multi-chart analytic operator requires named charts")
        modes = {int(operator.graph.n_modes) for operator in mapping.values()}
        operating = {int(operator.n_operating) for operator in mapping.values()}
        operating_names = {
            tuple(getattr(operator, "physical_operating_names", ()))
            for operator in mapping.values()
        }
        if len(modes) != 1:
            raise ValueError("all chart operators must expose the same retained thermal rank")
        if len(operating) != 1 or len(operating_names) != 1:
            raise ValueError("all chart operators must expose the same physical operating interface")
        self._operators = mapping
        self._n_modes = modes.pop()
        self._n_operating = operating.pop()
        self._operating_names = operating_names.pop()

    @property
    def chart_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operators))

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def n_operating(self) -> int:
        return self._n_operating

    @property
    def physical_operating_names(self) -> tuple[str, ...]:
        return self._operating_names

    def operator(self, chart_name: str):
        name = str(chart_name)
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"unknown analytic geometry chart {name!r}") from exc

    def evaluate(
        self,
        chart_name: str,
        t: float,
        *,
        geometry: np.ndarray,
        a0: np.ndarray,
        operating: np.ndarray,
        stable: bool = True,
    ) -> MultiChartGeometryAnalyticPrediction:
        prediction = self.operator(chart_name).evaluate(
            t,
            geometry=geometry,
            a0=a0,
            operating=operating,
            stable=stable,
        )
        return MultiChartGeometryAnalyticPrediction(str(chart_name), prediction)
