from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parametric_realization import evaluate_parametric_stable


@dataclass(frozen=True)
class GeometryAnalyticPrediction:
    time: float
    geometry: np.ndarray
    initial: np.ndarray
    operating: np.ndarray
    state: np.ndarray
    derivative: np.ndarray
    residual: np.ndarray
    residual_norm: float


class GeometryConditionedAnalyticEvolutionOperator:
    """One fixed analytic DAG for ``(G,U,a0,t)->a(t)``.

    The graph keeps the reference thermal decay spectrum. Geometry parameters are
    ordinary static analytic nodes, placed before physical operating parameters.
    The physical vector field is evaluated on the canonical thermal atlas at the
    requested geometry, so the exact target-spectrum/operator change appears in
    the physical residual and can be reduced by the same residual-grown graph.

    No network predicts another network's weights and no time stepping is used.
    """

    def __init__(
        self,
        graph,
        geometry_family,
        *,
        physical_operating_names,
    ) -> None:
        self.graph = graph
        self.geometry_family = geometry_family
        self.physical_operating_names = tuple(str(v) for v in physical_operating_names)
        expected = tuple(geometry_family.geometry_names) + self.physical_operating_names
        if tuple(graph.operating_names) != expected:
            raise ValueError(
                "graph static nodes must be geometry_names followed by physical_operating_names"
            )
        if graph.n_modes != geometry_family.n_modes:
            raise ValueError("graph/reference thermal atlas rank mismatch")
        ref = np.asarray(geometry_family.reference_thermal_model.lambdas, dtype=float)
        if not np.array_equal(np.asarray(graph.lambdas, dtype=float), ref):
            raise ValueError("cross-geometry graph must use the reference thermal spectrum")

    @property
    def n_geometry(self) -> int:
        return self.geometry_family.n_geometry

    @property
    def n_operating(self) -> int:
        return len(self.physical_operating_names)

    def static_parameters(self, geometry: np.ndarray, operating: np.ndarray) -> np.ndarray:
        g = np.asarray(geometry, dtype=float)
        u = np.asarray(operating, dtype=float)
        if g.shape != (self.n_geometry,):
            raise ValueError("geometry parameter dimension mismatch")
        if u.shape != (self.n_operating,):
            raise ValueError("physical operating dimension mismatch")
        return np.concatenate([g, u])

    def evaluate(
        self,
        t: float,
        *,
        geometry: np.ndarray,
        a0: np.ndarray,
        operating: np.ndarray,
        stable: bool = True,
    ) -> GeometryAnalyticPrediction:
        time = float(t)
        if time < 0.0:
            raise ValueError("time must be non-negative")
        g = np.asarray(geometry, dtype=float)
        initial = np.asarray(a0, dtype=float)
        u = np.asarray(operating, dtype=float)
        static = self.static_parameters(g, u)
        if initial.shape != (self.graph.n_modes,):
            raise ValueError("initial coordinate dimension mismatch")

        if stable:
            a, da = evaluate_parametric_stable(
                self.graph, time, a0=initial, operating=static
            )
        else:
            a, da = self.graph.evaluate(time, a0=initial, operating=static)

        physical = self.geometry_family.field(g)
        if physical.n_operating != self.n_operating:
            raise ValueError("geometry field/physical operating dimensions do not match")
        residual = physical.residual(a, da, u)
        return GeometryAnalyticPrediction(
            time=time,
            geometry=g.copy(),
            initial=initial.copy(),
            operating=u.copy(),
            state=np.asarray(a, dtype=float),
            derivative=np.asarray(da, dtype=float),
            residual=np.asarray(residual, dtype=float),
            residual_norm=float(np.linalg.norm(residual)),
        )
