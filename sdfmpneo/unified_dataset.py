"""Compact operator/residual sampling metadata for solution-label-free Maxwell training.

The dataset stores geometry/material states only. Sparse Maxwell operators are
assembled from hard physics by the trainer; no dense reduced ``Q/S`` matrices
and no Maxwell solution labels are persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .unified_geometry import UnifiedUWPTGeometry


@dataclass
class MaxwellResidualDataset:
    geometries: tuple
    states: tuple
    split: np.ndarray
    residual_steps: int = 3
    seed: int = 0

    def __post_init__(self):
        if len(self.geometries) != len(self.states) or len(self.geometries) != len(self.split):
            raise ValueError("geometry/state/split sample counts must match")
        if len(self.geometries) < 10:
            raise ValueError("at least ten operator samples are required")
        if int(self.residual_steps) < 1:
            raise ValueError("residual_steps must be positive")
        self.split = np.asarray(self.split, np.int8)
        if np.any(~np.isin(self.split, [0, 1, 2])):
            raise ValueError("dataset split contains an unknown partition")

    def indices(self, name):
        return np.flatnonzero(self.split == {"train": 0, "validation": 1, "test": 2}[name])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        geometry_json = np.asarray([
            json.dumps(
                (g if isinstance(g, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(g)).to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for g in self.geometries
        ])
        state_json = np.asarray([
            json.dumps(s, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if s is not None else "null"
            for s in self.states
        ])
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                geometry_json=geometry_json,
                state_json=state_json,
                split=self.split,
                residual_steps=np.array(int(self.residual_steps)),
                seed=np.array(int(self.seed)),
            )
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                tuple(json.loads(str(v)) for v in data["geometry_json"]),
                tuple(json.loads(str(v)) for v in data["state_json"]),
                data["split"],
                int(data["residual_steps"]),
                int(data["seed"]),
            )


def generate_residual_dataset(background, geometry_samples, state_samples, *, seed=0, residual_steps=3, monitor=None):
    pairs = list(zip(geometry_samples, state_samples))
    if len(pairs) < 10:
        raise ValueError("at least ten operator samples are required")
    geometries, states = [], []
    for i, (geometry, state) in enumerate(pairs):
        if monitor is not None:
            monitor.checkpoint()
        g = background.validate_geometry(geometry)
        geometries.append(g.to_mapping())
        states.append(None if state is None else dict(state))
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(phase="maxwell_operator_samples", training_points=i + 1)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    split = np.zeros(len(pairs), np.int8)
    n_validation = max(1, int(round(0.15 * len(pairs))))
    n_test = max(1, int(round(0.15 * len(pairs))))
    split[order[:n_validation]] = 1
    split[order[n_validation:n_validation + n_test]] = 2
    return MaxwellResidualDataset(tuple(geometries), tuple(states), split, int(residual_steps), int(seed))


__all__ = ["MaxwellResidualDataset", "generate_residual_dataset"]
