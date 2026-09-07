from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhysicsTrainingResult:
    loss_history: list[float]
    final_loss: float


class PhysicsResidualTrainer:
    """Data-free optimizer contract for the evolution operator.

    Training minimizes governing equation residuals only. No FEM transient
    snapshots are required.
    """

    def __init__(self, operator, optimizer, residual_weight: float = 1.0):
        self.operator = operator
        self.optimizer = optimizer
        self.residual_weight = float(residual_weight)

    def residual_loss(self, a0, operating, time):
        prediction = self.operator(a0, operating, time)
        residual = self.operator.physics_residual(prediction, a0, operating, time)
        return self.residual_weight * (residual ** 2).mean()

    def train(self, samples):
        history = []
        for a0, u, t in samples:
            self.optimizer.zero_grad()
            loss = self.residual_loss(a0, u, t)
            loss.backward()
            self.optimizer.step()
            history.append(float(loss.detach()))
        return PhysicsTrainingResult(history, history[-1] if history else 0.0)
