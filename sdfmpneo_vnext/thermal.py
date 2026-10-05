from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class StableThermalModel:
    capacity: np.ndarray
    conductance: np.ndarray
    source: np.ndarray
    decoder: np.ndarray
    reference_temperature: float = 293.15

    def __post_init__(self):
        C = np.asarray(self.capacity, dtype=float)
        G = np.asarray(self.conductance, dtype=float)
        B = np.asarray(self.source, dtype=float)
        Phi = np.asarray(self.decoder, dtype=float)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or G.shape != C.shape:
            raise ValueError("capacity/conductance must be square and equal-sized")
        if B.ndim != 2 or B.shape[0] != C.shape[0]:
            raise ValueError("source has incompatible shape")
        if Phi.ndim != 2 or Phi.shape[1] != C.shape[0]:
            raise ValueError("decoder has incompatible shape")
        if not np.allclose(C, C.T, atol=1e-12) or np.min(np.linalg.eigvalsh(C)) <= 0:
            raise ValueError("capacity must be symmetric positive definite")
        if not np.allclose(G, G.T, atol=1e-12) or np.min(np.linalg.eigvalsh(G)) < -1e-12:
            raise ValueError("conductance must be symmetric positive semidefinite")
        object.__setattr__(self, "capacity", C)
        object.__setattr__(self, "conductance", G)
        object.__setattr__(self, "source", B)
        object.__setattr__(self, "decoder", Phi)

    @property
    def n_states(self):
        return self.capacity.shape[0]

    def steady_state(self, power) -> np.ndarray:
        p = np.asarray(power, dtype=float)
        rhs = self.source @ p
        return np.linalg.solve(self.conductance, rhs)

    def advance_constant_power(self, state, power, dt: float) -> np.ndarray:
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        z = np.asarray(state, dtype=float)
        p = np.asarray(power, dtype=float)
        if z.shape != (self.n_states,):
            raise ValueError("state has wrong shape")
        A = -np.linalg.solve(self.capacity, self.conductance)
        f = np.linalg.solve(self.capacity, self.source @ p)
        if dt == 0:
            return z.copy()
        aug = np.zeros((self.n_states + 1, self.n_states + 1), dtype=float)
        aug[:self.n_states, :self.n_states] = A
        aug[:self.n_states, -1] = f
        E = expm(aug * dt)
        return E[:self.n_states, :self.n_states] @ z + E[:self.n_states, -1]

    def temperature(self, state) -> np.ndarray:
        z = np.asarray(state, dtype=float)
        return self.reference_temperature + self.decoder @ z

    def energy(self, state) -> float:
        z = np.asarray(state, dtype=float)
        return float(0.5 * z @ self.capacity @ z)
