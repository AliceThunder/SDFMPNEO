from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .operator import PhysicsNeuralEvolutionOperator


@dataclass(frozen=True)
class NeuralElectroThermalState:
    state: np.ndarray
    derivative: np.ndarray
    correction: np.ndarray
    residual: float


class NeuralElectroThermalWrapper:
    """Couple the physics evolution model with the electrothermal residual path.

    The neural model only supplies a correction term. The analytical thermal
    decay and electromagnetic heat source remain the governing backbone.
    """

    def __init__(self, operator: PhysicsNeuralEvolutionOperator, vector_field):
        self.operator = operator
        self.vector_field = vector_field

    def evaluate(self, t, *, initial_state, operating=None, geometry=None):
        prediction = self.operator(t, initial_state, operating=operating, geometry=geometry)
        a = np.asarray(prediction, dtype=float)
        field = self.vector_field.evaluate(a, operating)
        residual = np.linalg.norm(field.vector_field - self.operator.time_derivative(t, initial_state, operating=operating, geometry=geometry))
        return NeuralElectroThermalState(
            state=a,
            derivative=np.asarray(field.vector_field, dtype=float),
            correction=a - np.asarray(initial_state, dtype=float),
            residual=float(residual),
        )
