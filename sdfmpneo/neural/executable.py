from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NeuralOnlineState:
    time: float
    thermal_coordinates: np.ndarray
    thermal_derivative: np.ndarray
    geometry_latent: object | None = None


class NeuralElectroThermalEvolution:
    """Adapter connecting physics neural evolution with SDF-MPNEO models.

    The adapter keeps the original certified analytic evolution path available
    while allowing the neural correction operator to replace only the evolution
    part. Electromagnetic and thermal certification modules remain unchanged.
    """

    def __init__(self, analytic_evolution, neural_operator=None, geometry_encoder=None):
        self.analytic_evolution = analytic_evolution
        self.neural_operator = neural_operator
        self.geometry_encoder = geometry_encoder

    def evaluate(self, t, *, a0=None, lambdas=None, geometry=None):
        if self.neural_operator is None:
            return self.analytic_evolution.evaluate(float(t))

        if a0 is None or lambdas is None:
            raise ValueError("neural evolution requires a0 and lambdas")

        g = None
        if geometry is not None:
            g = self.geometry_encoder(geometry) if self.geometry_encoder else geometry

        state = self.neural_operator.evaluate_state(
            np.asarray(a0),
            np.asarray([t]),
            np.asarray(lambdas),
            g,
        )
        return state.state.detach().cpu().numpy(), state.derivative.detach().cpu().numpy()
