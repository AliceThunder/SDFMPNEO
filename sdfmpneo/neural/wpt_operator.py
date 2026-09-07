from __future__ import annotations

import numpy as np


class WPTGeometryConditionedOperator:
    """High-level interface for underwater WPT multi-physics inference.

    Geometry/material descriptors are kept explicit and are encoded before
    entering the evolution operator. No geometry-specific neural weights are
    generated.
    """

    def __init__(self, geometry_encoder, evolution_operator):
        self.geometry_encoder = geometry_encoder
        self.evolution_operator = evolution_operator

    def predict(self, t, initial_state, geometry, operating=None):
        latent = self.geometry_encoder(geometry)
        return self.evolution_operator(
            t,
            initial_state,
            operating=operating,
            geometry=latent,
        )

    def impedance_outputs(self, result):
        return {
            "state": np.asarray(result),
            "available": True,
        }
