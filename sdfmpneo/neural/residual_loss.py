from __future__ import annotations

import torch


class ElectroThermalPhysicsLoss:
    """Loss without solution snapshots.

    The objective is the electrothermal governing equation residual.
    """

    def __init__(self, contraction_weight: float = 1e-3):
        self.contraction_weight = contraction_weight

    def __call__(self, derivative, state, heat_source, lambdas):
        residual = derivative + lambdas * state - heat_source
        return torch.mean(residual ** 2)
