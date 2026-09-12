"""Neural tensor surrogate: ordinary MLP coefficients plus hard physics decoder."""
from __future__ import annotations

import numpy as np

from .physical_layer import decode_heat_source_numpy, decode_heat_source_torch
from .pod import TensorPOD


class NeuralTensorSurrogate:
    """Map ``(thermal state, geometry)`` to Joule heat through POD coefficients.

    The network predicts normalized POD coefficients only.  Current variables
    never enter the MLP; they enter exclusively through the quadratic physics
    layer after coefficient de-normalization and POD decoding.
    """

    def __init__(
        self,
        network,
        pod: TensorPOD,
        *,
        state_dimension: int,
        geometry_dimension: int,
        coefficient_mean: np.ndarray,
        coefficient_scale: np.ndarray,
    ) -> None:
        self.network = network
        self.pod = pod
        self.state_dimension = int(state_dimension)
        self.geometry_dimension = int(geometry_dimension)
        mean = np.asarray(coefficient_mean, dtype=float).reshape(-1)
        scale = np.asarray(coefficient_scale, dtype=float).reshape(-1)
        if mean.shape != (pod.rank,) or scale.shape != (pod.rank,):
            raise ValueError("coefficient normalization dimensions do not match POD rank")
        if np.any(~np.isfinite(mean + scale)) or np.any(scale <= 0.0):
            raise ValueError("coefficient normalization must be finite with positive scale")
        if self.state_dimension != pod.thermal_rank or self.geometry_dimension < 0:
            raise ValueError("surrogate state/geometry dimensions are invalid")
        self.coefficient_mean = mean
        self.coefficient_scale = scale

    @property
    def input_dimension(self) -> int:
        return self.state_dimension + self.geometry_dimension

    def _input_numpy(self, state, geometry) -> np.ndarray:
        a = np.asarray(state, dtype=float).reshape(-1)
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if a.shape != (self.state_dimension,) or g.shape != (self.geometry_dimension,):
            raise ValueError("state/geometry dimensions do not match surrogate")
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(g)):
            raise ValueError("state and geometry must be finite")
        return np.concatenate([a, g])

    def predict_coefficients_numpy(self, state, geometry) -> np.ndarray:
        import torch

        x = self._input_numpy(state, geometry)
        parameter = next(self.network.parameters())
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device)
            normalized = self.network(xt).detach().cpu().numpy().astype(float)
        return self.coefficient_mean + self.coefficient_scale * normalized

    def predict_packed_numpy(self, state, geometry) -> np.ndarray:
        return self.pod.decode(self.predict_coefficients_numpy(state, geometry))

    def heat_source_numpy(self, state, geometry, operating) -> np.ndarray:
        beta = self.predict_coefficients_numpy(state, geometry)
        return decode_heat_source_numpy(beta, self.pod, operating)

    def coefficients_torch(self, state, geometry):
        import torch

        if state.shape[-1] != self.state_dimension or geometry.shape[-1] != self.geometry_dimension:
            raise ValueError("state/geometry tensor dimensions do not match surrogate")
        x = torch.cat([state, geometry], dim=-1)
        normalized = self.network(x)
        mean = torch.as_tensor(self.coefficient_mean, dtype=normalized.dtype, device=normalized.device)
        scale = torch.as_tensor(self.coefficient_scale, dtype=normalized.dtype, device=normalized.device)
        return mean + scale * normalized

    def heat_source_torch(self, state, geometry, operating):
        import torch

        beta = self.coefficients_torch(state, geometry)
        mean = torch.as_tensor(self.pod.mean, dtype=beta.dtype, device=beta.device)
        basis = torch.as_tensor(self.pod.basis, dtype=beta.dtype, device=beta.device)
        return decode_heat_source_torch(beta, mean, basis, self.pod.thermal_rank, operating)


__all__ = ["NeuralTensorSurrogate"]
