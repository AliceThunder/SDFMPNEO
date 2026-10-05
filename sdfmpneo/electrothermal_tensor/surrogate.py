"""Neural tensor surrogate: ordinary MLP coefficients plus hard physics decoder."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .physical_layer import (
    decode_heat_source_batch_numpy,
    decode_heat_source_torch,
)
from .pod import TensorPOD
from .symmetric import quadratic_feature


@dataclass(frozen=True)
class PreparedOperatingQuadraticLayer:
    """Exact operating-specific contraction of the POD Joule tensor.

    For fixed operating parameters ``u`` the expensive current-feature
    contraction is state independent:

        q(a,g,u) = mean_q(u) + basis_q(u) beta(a,g).

    A trajectory can therefore reuse these two small arrays for every ETD stage.
    """

    operating: np.ndarray
    mean_heat: np.ndarray
    coefficient_to_heat: np.ndarray

    def evaluate(self, coefficients: np.ndarray) -> np.ndarray:
        beta = np.asarray(coefficients, dtype=float).reshape(-1)
        if beta.shape != (self.coefficient_to_heat.shape[1],) or np.any(~np.isfinite(beta)):
            raise ValueError("prepared operating layer coefficient dimension mismatch")
        return self.mean_heat + self.coefficient_to_heat @ beta


class NeuralTensorSurrogate:
    """Map ``(thermal state, geometry)`` to Joule heat through POD coefficients.

    The network predicts normalized POD coefficients only. Current variables
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
        self._operating_cache: OrderedDict[tuple[float, ...], PreparedOperatingQuadraticLayer] = OrderedDict()
        self._operating_cache_size = 16

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

    def _input_batch_numpy(self, states, geometries) -> np.ndarray:
        a = np.asarray(states, dtype=float)
        g = np.asarray(geometries, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.state_dimension:
            raise ValueError("batched states have wrong shape")
        if g.ndim == 1:
            if g.shape != (self.geometry_dimension,):
                raise ValueError("shared batch geometry has wrong shape")
            g = np.repeat(g[None, :], len(a), axis=0)
        if g.ndim != 2 or g.shape != (len(a), self.geometry_dimension):
            raise ValueError("batched geometries have wrong shape")
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(g)):
            raise ValueError("batched state/geometry values must be finite")
        return np.hstack([a, g])

    def predict_coefficients_numpy(self, state, geometry) -> np.ndarray:
        import torch

        x = self._input_numpy(state, geometry)
        parameter = next(self.network.parameters())
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device)
            normalized = self.network(xt).detach().cpu().numpy().astype(float)
        return self.coefficient_mean + self.coefficient_scale * normalized

    def predict_coefficients_batch_numpy(self, states, geometries) -> np.ndarray:
        """One network forward for a batch of state/geometry points."""
        import torch

        x = self._input_batch_numpy(states, geometries)
        parameter = next(self.network.parameters())
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device)
            normalized = self.network(xt).detach().cpu().numpy().astype(float)
        return self.coefficient_mean[None, :] + self.coefficient_scale[None, :] * normalized

    def predict_packed_numpy(self, state, geometry) -> np.ndarray:
        return self.pod.decode(self.predict_coefficients_numpy(state, geometry))

    def prepare_operating_numpy(self, operating) -> PreparedOperatingQuadraticLayer:
        """Precontract the exact quadratic-current POD layer for one operating point."""
        u = np.asarray(operating, dtype=float).reshape(-1)
        if u.shape != (self.pod.current_dimension,) or np.any(~np.isfinite(u)):
            raise ValueError("operating/current dimension mismatch")
        key = tuple(float(v) for v in u)
        cached = self._operating_cache.get(key)
        if cached is not None:
            self._operating_cache.move_to_end(key)
            return cached
        feature = quadratic_feature(u)
        n_sym = self.pod.packed_symmetric_size
        mean_modes = self.pod.mean.reshape(self.pod.thermal_rank, n_sym)
        basis_modes = self.pod.basis.reshape(self.pod.thermal_rank, n_sym, self.pod.rank)
        prepared = PreparedOperatingQuadraticLayer(
            operating=u.copy(),
            mean_heat=np.asarray(mean_modes @ feature, dtype=float),
            coefficient_to_heat=np.asarray(
                np.einsum("rsk,s->rk", basis_modes, feature, optimize=True),
                dtype=float,
            ),
        )
        self._operating_cache[key] = prepared
        if len(self._operating_cache) > self._operating_cache_size:
            self._operating_cache.popitem(last=False)
        return prepared

    def heat_source_numpy(self, state, geometry, operating) -> np.ndarray:
        beta = self.predict_coefficients_numpy(state, geometry)
        return self.prepare_operating_numpy(operating).evaluate(beta)

    def heat_source_batch_numpy(self, states, geometries, operating) -> np.ndarray:
        """Vectorized heat source for aligned state/geometry/operating batches."""
        beta = self.predict_coefficients_batch_numpy(states, geometries)
        u = np.asarray(operating, dtype=float)
        if u.ndim != 2 or u.shape != (len(beta), self.pod.current_dimension):
            raise ValueError("batched operating values have wrong shape")
        if np.any(~np.isfinite(u)):
            raise ValueError("batched operating values must be finite")
        return decode_heat_source_batch_numpy(beta, self.pod, u)

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


__all__ = ["NeuralTensorSurrogate", "PreparedOperatingQuadraticLayer"]
