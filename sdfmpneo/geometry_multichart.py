from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MultiChartGeometrySelection:
    chart_name: str
    geometry: np.ndarray
    chart: object
    field: object


class CertifiedMultiChartGeometryFamily:
    """Finite union of independently certified geometry/operator charts.

    A topology or remeshing change is not forced into one continuous affine
    chart.  Instead each shape/topology family owns its own certified reference
    mesh, thermal atlas and geometry coordinates.  The union is discrete in
    ``chart_name`` and continuous only inside each chart, matching the actual
    mathematical structure

        G = union_s G_s.

    Different charts may therefore have different full FE dimensions.  They must
    expose the same retained thermal rank so downstream state/output interfaces
    remain compatible.  Cross-chart interpolation is deliberately absent; a
    caller must select one certified chart explicitly.
    """

    def __init__(self, charts: Mapping[str, object]) -> None:
        mapping = {str(name): family for name, family in charts.items()}
        if not mapping or any(not name.strip() for name in mapping):
            raise ValueError("multi-chart geometry family requires named charts")
        ranks = {int(family.n_modes) for family in mapping.values()}
        if len(ranks) != 1:
            raise ValueError("all geometry charts must expose the same retained thermal rank")
        self._charts = mapping
        self._n_modes = ranks.pop()

    @property
    def chart_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._charts))

    @property
    def n_modes(self) -> int:
        return self._n_modes

    def family(self, chart_name: str):
        name = str(chart_name)
        try:
            return self._charts[name]
        except KeyError as exc:
            raise KeyError(f"unknown geometry chart {name!r}") from exc

    def geometry_names(self, chart_name: str) -> tuple[str, ...]:
        return tuple(self.family(chart_name).geometry_names)

    def select(self, chart_name: str, geometry: np.ndarray) -> MultiChartGeometrySelection:
        family = self.family(chart_name)
        g = np.asarray(geometry, dtype=float)
        if g.shape != (family.n_geometry,):
            raise ValueError("geometry parameter dimension does not match selected chart")
        chart = family.chart(g)
        return MultiChartGeometrySelection(str(chart_name), g.copy(), chart, chart.field)

    def field(self, chart_name: str, geometry: np.ndarray):
        return self.select(chart_name, geometry).field
