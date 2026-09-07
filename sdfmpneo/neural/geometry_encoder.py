from __future__ import annotations

import numpy as np


class GeometryEncoder:
    """Deterministic geometry/material embedding interface."""

    def __init__(self, scale=None):
        self.scale = None if scale is None else np.asarray(scale, dtype=float)

    def encode(self, geometry):
        g = np.asarray(geometry, dtype=float)
        if self.scale is not None:
            g = g / self.scale
        return g

    def __call__(self, geometry):
        return self.encode(geometry)
