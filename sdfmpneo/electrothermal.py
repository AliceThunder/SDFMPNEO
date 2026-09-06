from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElectroThermalFieldEvaluation:
    state: np.ndarray
    operating: np.ndarray
    rhs: np.ndarray
    heat_source: np.ndarray
    heat_source_jacobian: np.ndarray
    vector_field: np.ndarray
    vector_field_jacobian: np.ndarray


class CertifiedElectroThermalVectorField:
    """Single reduced electrothermal dynamics callable.

    The object owns no transient solver. It only evaluates

        da/dt = -Lambda a + g_em(a,U) + f_T.

    ``rhs_map`` may be omitted for a fixed excitation or may provide an exact
    static operating map through ``rhs_map.evaluate(U)``.
    """

    def __init__(
        self,
        thermal_model,
        electromagnetic_model,
        *,
        rhs_map=None,
        fixed_rhs: np.ndarray | None = None,
        thermal_forcing: np.ndarray | None = None,
    ) -> None:
        self.thermal_model = thermal_model
        self.electromagnetic_model = electromagnetic_model
        self.em_model = electromagnetic_model
        self.rhs_map = rhs_map
        n = int(len(thermal_model.lambdas))
        if electromagnetic_model.problem.n_thermal != n:
            raise ValueError("thermal/electromagnetic reduced dimensions do not match")
        if rhs_map is not None and fixed_rhs is not None:
            raise ValueError("use rhs_map or fixed_rhs, not both")
        self.fixed_rhs = (
            np.asarray(electromagnetic_model.problem.b, dtype=complex)
            if rhs_map is None and fixed_rhs is None
            else None if fixed_rhs is None else np.asarray(fixed_rhs, dtype=complex)
        )
        if self.fixed_rhs is not None and self.fixed_rhs.shape != (electromagnetic_model.problem.n_em,):
            raise ValueError("fixed_rhs dimension mismatch")
        forcing = np.zeros(n) if thermal_forcing is None else np.asarray(thermal_forcing, dtype=float)
        if forcing.shape != (n,):
            raise ValueError("thermal_forcing dimension mismatch")
        self.thermal_forcing = forcing

    @property
    def n_modes(self) -> int:
        return len(self.thermal_model.lambdas)

    @property
    def n_operating(self) -> int:
        return 0 if self.rhs_map is None else int(self.rhs_map.n_operating)

    def rhs(self, operating: np.ndarray | None = None) -> np.ndarray:
        if self.rhs_map is None:
            if operating is not None and np.asarray(operating).size:
                raise ValueError("this vector field has no operating parameters")
            return self.fixed_rhs.copy()
        if operating is None:
            raise ValueError("operating parameters are required")
        return np.asarray(self.rhs_map.evaluate(np.asarray(operating, dtype=float)), dtype=complex)

    def evaluate(
        self,
        state: np.ndarray,
        operating: np.ndarray | None = None,
    ) -> ElectroThermalFieldEvaluation:
        a = np.asarray(state, dtype=float)
        if a.shape != (self.n_modes,):
            raise ValueError("state dimension mismatch")
        u = np.empty(0, dtype=float) if operating is None else np.asarray(operating, dtype=float)
        source = self.rhs(None if self.rhs_map is None else u)
        q, Jq = self.electromagnetic_model.heat_source_and_jacobian_for_rhs(a, source)
        q = np.asarray(q, dtype=float)
        Jq = np.asarray(Jq, dtype=float)
        lambdas = np.asarray(self.thermal_model.lambdas, dtype=float)
        F = -lambdas * a + q + self.thermal_forcing
        JF = Jq - np.diag(lambdas)
        return ElectroThermalFieldEvaluation(
            state=a.copy(),
            operating=u.copy(),
            rhs=source,
            heat_source=q,
            heat_source_jacobian=Jq,
            vector_field=F,
            vector_field_jacobian=JF,
        )

    def residual(
        self,
        state: np.ndarray,
        derivative: np.ndarray,
        operating: np.ndarray | None = None,
    ) -> np.ndarray:
        da = np.asarray(derivative, dtype=float)
        evaluation = self.evaluate(state, operating)
        if da.shape != evaluation.vector_field.shape:
            raise ValueError("derivative dimension mismatch")
        return da - evaluation.vector_field

    def contraction_margin(self, state: np.ndarray, operating: np.ndarray | None = None) -> float:
        J = self.evaluate(state, operating).vector_field_jacobian
        symmetric = 0.5 * (J + J.T)
        return float(-np.max(np.linalg.eigvalsh(symmetric)))
