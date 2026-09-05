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
    """Reduced physical residual R = da + Lambda*a - g_em(a)."""

    def __init__(self, lambdas: np.ndarray, em_model):
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.em_model = em_model

    def evaluate(self, a: np.ndarray, da: np.ndarray) -> ResidualSample:
        a = np.asarray(a, dtype=float)
        da = np.asarray(da, dtype=float)
        g = np.asarray(self.em_model.heat_source(a), dtype=float)
        R = da + self.lambdas * a - g
        return ResidualSample(a=a, da=da, g_em=g, residual=R, norm=float(np.linalg.norm(R)))
