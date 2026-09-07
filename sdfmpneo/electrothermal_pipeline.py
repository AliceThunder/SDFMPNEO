from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrothermal import CertifiedElectroThermalVectorField


@dataclass(frozen=True)
class ElectroThermalSurrogateState:
    """State container returned by the production surrogate entry point.

    The object intentionally contains only reduced physical coordinates. Time
    evolution is delegated to the analytic evolution operator, so this layer
    never performs transient marching.
    """

    coordinates: np.ndarray
    time: float


class ElectroThermalSurrogate:
    """Unified callable connecting physical field dynamics and evolution.

    This closes the software boundary between the certified electrothermal
    vector field and the continuous-time analytic evolution backend.
    """

    def __init__(self, vector_field: CertifiedElectroThermalVectorField, evolution_operator):
        self.vector_field = vector_field
        self.evolution_operator = evolution_operator

        if getattr(evolution_operator, "n_modes", vector_field.n_modes) != vector_field.n_modes:
            raise ValueError("evolution/vector-field reduced dimensions do not match")

    @property
    def n_modes(self) -> int:
        return self.vector_field.n_modes

    def state(self, initial_state: np.ndarray, time: float, operating=None) -> ElectroThermalSurrogateState:
        if time < 0:
            raise ValueError("time must be non-negative")
        coordinates = self.evolution_operator.evaluate(
            np.asarray(initial_state, dtype=float),
            float(time),
            operating,
        )
        return ElectroThermalSurrogateState(
            coordinates=np.asarray(coordinates, dtype=float),
            time=float(time),
        )

    def residual(self, state: np.ndarray, operating=None) -> np.ndarray:
        return self.vector_field.evaluate(state, operating).vector_field
