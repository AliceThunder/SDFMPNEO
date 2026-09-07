from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhysicsCollocationBatch:
    time: np.ndarray
    state: np.ndarray
    operating: np.ndarray | None


class PhysicsCollocationSampler:
    """Generate collocation points without FEM transient snapshots."""

    def __init__(self, state_dimension, time_range=(0.0, 1.0), seed=0):
        self.state_dimension = int(state_dimension)
        self.time_range = tuple(time_range)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size, operating_dimension=0):
        t = self.rng.uniform(self.time_range[0], self.time_range[1], int(batch_size))
        a = self.rng.normal(0.0, 1.0, (int(batch_size), self.state_dimension))
        u = None
        if operating_dimension:
            u = self.rng.uniform(-1.0, 1.0, (int(batch_size), operating_dimension))
        return PhysicsCollocationBatch(t, a, u)
