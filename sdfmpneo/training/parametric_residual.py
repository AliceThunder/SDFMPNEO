from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineOperatingRHSMap:
    """Exact electromagnetic excitation map b(U)=b0+B_U U for real U."""

    offset: np.ndarray
    matrix: np.ndarray

    def __post_init__(self) -> None:
        b0 = np.asarray(self.offset, dtype=complex)
        B = np.asarray(self.matrix, dtype=complex)
        if b0.ndim != 1:
            raise ValueError("offset must be one-dimensional")
        if B.ndim != 2 or B.shape[0] != b0.size:
            raise ValueError("matrix must have shape (n_em,n_operating)")
        object.__setattr__(self, "offset", b0)
        object.__setattr__(self, "matrix", B)

    @property
    def n_em(self) -> int:
        return self.offset.size

    @property
    def n_operating(self) -> int:
        return self.matrix.shape[1]

    def evaluate(self, operating: np.ndarray) -> np.ndarray:
        u = np.asarray(operating, dtype=float)
        if u.shape != (self.n_operating,):
            raise ValueError("operating parameter dimension mismatch")
        return self.offset + self.matrix @ u


@dataclass(frozen=True)
class ParametricResidualSample:
    time: float
    a: np.ndarray
    da: np.ndarray
    rhs: np.ndarray
    g_em: np.ndarray
    residual: np.ndarray
    norm: float
    state_operating_jacobian: np.ndarray
    derivative_operating_jacobian: np.ndarray
    heat_source_operating_jacobian: np.ndarray
    residual_operating_jacobian: np.ndarray


class ParametricElectroThermalResidual:
    """Exact residual sensitivity for static U entering an affine EM RHS.

    The residual is the same production equation used online:

        R = da + Lambda a - g_em(a,U) - f_T.
    """

    def __init__(
        self,
        graph,
        electromagnetic_model,
        rhs_map: AffineOperatingRHSMap,
        thermal_forcing: np.ndarray | None = None,
    ):
        self.graph = graph
        self.em_model = electromagnetic_model
        self.rhs_map = rhs_map
        if rhs_map.n_em != electromagnetic_model.problem.n_em:
            raise ValueError("rhs map/electromagnetic dimension mismatch")
        if rhs_map.n_operating != len(graph.operating_names):
            raise ValueError("rhs map/analytic operating dimension mismatch")
        if graph.n_modes != electromagnetic_model.problem.n_thermal:
            raise ValueError("analytic/electromagnetic thermal dimension mismatch")
        forcing = np.zeros(graph.n_modes) if thermal_forcing is None else np.asarray(thermal_forcing, dtype=float)
        if forcing.shape != (graph.n_modes,):
            raise ValueError("thermal_forcing dimension mismatch")
        self.thermal_forcing = forcing

    @classmethod
    def from_vector_field(cls, graph, vector_field) -> "ParametricElectroThermalResidual":
        if vector_field.rhs_map is None:
            raise ValueError("parametric residual requires a vector field with rhs_map")
        return cls(
            graph,
            vector_field.em_model,
            vector_field.rhs_map,
            thermal_forcing=vector_field.thermal_forcing,
        )

    def with_graph(self, graph) -> "ParametricElectroThermalResidual":
        return ParametricElectroThermalResidual(
            graph,
            self.em_model,
            self.rhs_map,
            thermal_forcing=self.thermal_forcing,
        )

    def vector_field_state_jacobian(self, a: np.ndarray, operating: np.ndarray) -> np.ndarray:
        rhs = self.rhs_map.evaluate(np.asarray(operating, dtype=float))
        _, Jg = self.em_model.heat_source_and_jacobian_for_rhs(np.asarray(a, dtype=float), rhs)
        return np.asarray(Jg, dtype=float) - np.diag(np.asarray(self.graph.lambdas, dtype=float))

    def _graph_operating_jacobians(
        self,
        t: float,
        a0: np.ndarray,
        operating: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        compiled = self.graph.compile()
        p = compiled.parameter_vector(a0, operating)
        offset = len(compiled.initial_names)
        n_modes = len(compiled.mode_series)
        n_u = len(compiled.operating_names)
        J_a = np.zeros((n_modes, n_u), dtype=float)
        J_da = np.zeros((n_modes, n_u), dtype=float)

        for i, series in enumerate(compiled.mode_series):
            for j in range(n_u):
                dseries = series.parameter_derivative(offset + j)
                J_a[i, j] = np.real(dseries.evaluate(t, compiled.lambdas, p))
                J_da[i, j] = np.real(
                    dseries.derivative(compiled.lambdas).evaluate(t, compiled.lambdas, p)
                )
        return J_a, J_da

    def _explicit_heat_source_operating_jacobian(
        self,
        a: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        x = self.em_model.state_for_rhs(a, rhs)
        n_out = self.em_model.problem.n_thermal
        n_u = self.rhs_map.n_operating
        J = np.zeros((n_out, n_u), dtype=float)

        for ell in range(n_u):
            direction_rhs = self.rhs_map.matrix[:, ell]
            dx = self.em_model.state_for_rhs(a, direction_rhs)
            for j in range(n_out):
                H = self.em_model.problem.loss_operator(j, a)
                J[j, ell] = 2.0 * np.real(np.vdot(dx, H @ x))
        return J

    def evaluate(
        self,
        t: float,
        *,
        a0: np.ndarray,
        operating: np.ndarray,
    ) -> ParametricResidualSample:
        u = np.asarray(operating, dtype=float)
        initial = np.asarray(a0, dtype=float)
        a, da = self.graph.evaluate(float(t), a0=initial, operating=u)
        rhs = self.rhs_map.evaluate(u)

        g, J_g_a = self.em_model.heat_source_and_jacobian_for_rhs(a, rhs)
        g = np.asarray(g, dtype=float)
        J_g_a = np.asarray(J_g_a, dtype=float)
        J_a, J_da = self._graph_operating_jacobians(float(t), initial, u)
        J_g_u_explicit = self._explicit_heat_source_operating_jacobian(a, rhs)

        residual = da + self.graph.lambdas * a - g - self.thermal_forcing
        J_residual = (
            J_da
            + self.graph.lambdas[:, None] * J_a
            - J_g_a @ J_a
            - J_g_u_explicit
        )

        return ParametricResidualSample(
            time=float(t),
            a=np.asarray(a, dtype=float),
            da=np.asarray(da, dtype=float),
            rhs=rhs,
            g_em=g,
            residual=np.asarray(residual, dtype=float),
            norm=float(np.linalg.norm(residual)),
            state_operating_jacobian=J_a,
            derivative_operating_jacobian=J_da,
            heat_source_operating_jacobian=J_g_a @ J_a + J_g_u_explicit,
            residual_operating_jacobian=J_residual,
        )
