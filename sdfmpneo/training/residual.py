from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualSample:
    a: np.ndarray
    da: np.ndarray
    g_em: np.ndarray
    residual: np.ndarray
    norm: float


class ElectroThermalResidual:
    """Reduced physical residual R = da + Lambda*a - g_em(a; rhs)."""

    def __init__(self, lambdas: np.ndarray, em_model, rhs: np.ndarray | None = None):
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.em_model = em_model
        self.rhs = None if rhs is None else np.asarray(rhs, dtype=complex)

    def evaluate(
        self,
        a: np.ndarray,
        da: np.ndarray,
        rhs: np.ndarray | None = None,
    ) -> ResidualSample:
        a = np.asarray(a, dtype=float)
        da = np.asarray(da, dtype=float)
        source = self.rhs if rhs is None else np.asarray(rhs, dtype=complex)

        if source is None:
            g = np.asarray(self.em_model.heat_source(a), dtype=float)
        else:
            g = np.asarray(self.em_model.heat_source_for_rhs(a, source), dtype=float)

        R = da + self.lambdas * a - g
        return ResidualSample(a=a, da=da, g_em=g, residual=R, norm=float(np.linalg.norm(R)))
