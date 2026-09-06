from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryResidualSample:
    time: float
    geometry: np.ndarray
    physical_operating: np.ndarray
    a: np.ndarray
    da: np.ndarray
    rhs: np.ndarray
    g_em: np.ndarray
    residual: np.ndarray
    norm: float


class GeometryElectroThermalResidual:
    """Production residual for one analytic graph across geometry and operating U.

    Graph static parameters are ordered as ``[G,U]``.  The physical target field
    is built deterministically by ``CertifiedGeometryElectroThermalFamily`` and
    transformed onto the canonical thermal atlas.  The graph may retain the
    reference decay spectrum; all target-spectrum/operator changes are therefore
    present in the exact physical residual ``da-F_G(a,U)``.
    """

    def __init__(self, graph, geometry_family, n_physical_operating: int) -> None:
        self.graph = graph
        self.geometry_family = geometry_family
        self.n_physical_operating = int(n_physical_operating)
        if self.n_physical_operating < 0:
            raise ValueError("n_physical_operating must be non-negative")
        expected = geometry_family.n_geometry + self.n_physical_operating
        if len(graph.operating_names) != expected:
            raise ValueError("graph static parameter dimension does not match [G,U]")
        if graph.n_modes != geometry_family.n_modes:
            raise ValueError("graph/geometry-family thermal rank mismatch")

    def with_graph(self, graph) -> "GeometryElectroThermalResidual":
        return GeometryElectroThermalResidual(
            graph,
            self.geometry_family,
            self.n_physical_operating,
        )

    def split_static(self, static_parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(static_parameters, dtype=float)
        expected = self.geometry_family.n_geometry + self.n_physical_operating
        if p.shape != (expected,):
            raise ValueError("static parameter dimension mismatch")
        nG = self.geometry_family.n_geometry
        return p[:nG], p[nG:]

    def vector_field_state_jacobian(self, a: np.ndarray, operating: np.ndarray) -> np.ndarray:
        geometry, u = self.split_static(operating)
        physical = self.geometry_family.field(geometry)
        return np.asarray(physical.evaluate(np.asarray(a, dtype=float), u).vector_field_jacobian, dtype=float)

    def evaluate(
        self,
        t: float,
        *,
        a0: np.ndarray,
        operating: np.ndarray,
    ) -> GeometryResidualSample:
        time = float(t)
        if time < 0.0:
            raise ValueError("time must be non-negative")
        static = np.asarray(operating, dtype=float)
        geometry, u = self.split_static(static)
        initial = np.asarray(a0, dtype=float)
        a, da = self.graph.evaluate(time, a0=initial, operating=static)
        physical = self.geometry_family.field(geometry)
        evaluation = physical.evaluate(a, u)
        residual = np.asarray(da, dtype=float) - np.asarray(evaluation.vector_field, dtype=float)
        return GeometryResidualSample(
            time=time,
            geometry=geometry.copy(),
            physical_operating=u.copy(),
            a=np.asarray(a, dtype=float),
            da=np.asarray(da, dtype=float),
            rhs=np.asarray(evaluation.rhs, dtype=complex),
            g_em=np.asarray(evaluation.heat_source, dtype=float),
            residual=residual,
            norm=float(np.linalg.norm(residual)),
        )
