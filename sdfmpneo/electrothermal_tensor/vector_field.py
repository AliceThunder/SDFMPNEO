"""Structure-preserving electrothermal vector field with hard thermal operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ReducedThermalOperator:
    mass: np.ndarray
    stiffness: np.ndarray

    def __post_init__(self) -> None:
        M = np.asarray(self.mass, dtype=float)
        K = np.asarray(self.stiffness, dtype=float)
        if M.ndim != 2 or K.shape != M.shape or M.shape[0] != M.shape[1]:
            raise ValueError("mass and stiffness must be square matrices of equal size")
        if np.any(~np.isfinite(M)) or np.any(~np.isfinite(K)):
            raise ValueError("thermal operators must be finite")
        if not np.allclose(M, M.T, rtol=1e-11, atol=1e-13):
            raise ValueError("reduced thermal mass must be symmetric")
        if not np.allclose(K, K.T, rtol=1e-11, atol=1e-13):
            raise ValueError("reduced thermal stiffness must be symmetric")
        np.linalg.cholesky(0.5 * (M + M.T))
        np.linalg.cholesky(0.5 * (K + K.T))
        object.__setattr__(self, "mass", M)
        object.__setattr__(self, "stiffness", K)

    @property
    def rank(self) -> int:
        return self.mass.shape[0]


class FixedThermalOperatorFamily:
    def __init__(self, mass: np.ndarray, stiffness: np.ndarray, *, geometry_dimension: int = 0):
        self._operator = ReducedThermalOperator(mass, stiffness)
        self.geometry_dimension = int(geometry_dimension)
        if self.geometry_dimension < 0:
            raise ValueError("geometry_dimension must be non-negative")

    def operator(self, geometry: np.ndarray) -> ReducedThermalOperator:
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.shape != (self.geometry_dimension,):
            raise ValueError("geometry dimension mismatch")
        return self._operator


class CallableThermalOperatorFamily:
    """Adapter around an application-owned geometry-to-(M,K) callback."""

    def __init__(self, geometry_dimension: int, callback: Callable[[np.ndarray], object]):
        self.geometry_dimension = int(geometry_dimension)
        self.callback = callback

    def operator(self, geometry: np.ndarray) -> ReducedThermalOperator:
        g = np.asarray(geometry, dtype=float).reshape(-1)
        if g.shape != (self.geometry_dimension,) or np.any(~np.isfinite(g)):
            raise ValueError("geometry dimension mismatch")
        value = self.callback(g)
        if isinstance(value, ReducedThermalOperator):
            return value
        if hasattr(value, "M") and hasattr(value, "K"):
            return ReducedThermalOperator(value.M, value.K)
        if isinstance(value, tuple) and len(value) == 2:
            return ReducedThermalOperator(value[0], value[1])
        raise TypeError("thermal operator callback must return ReducedThermalOperator, (M,K), or an object with M/K")


class NeuralElectroThermalVectorField:
    """Hard ``M a_dot = -K a + q_theta + f_T`` composition.

    ``thermal_rhs_forcing`` is a deterministic reduced thermal RHS term.  It is
    deliberately outside the neural surrogate and outside the Joule tensor so
    EM diagnostics can compare ``q_theta`` against the physical Joule source
    without mixing in non-electromagnetic heating/boundary forcing.
    """

    def __init__(self, surrogate, thermal_operators, *, thermal_rhs_forcing=None) -> None:
        self.surrogate = surrogate
        self.thermal_operators = thermal_operators
        if int(thermal_operators.geometry_dimension) != int(surrogate.geometry_dimension):
            raise ValueError("surrogate and thermal operator geometry dimensions differ")
        forcing = (
            np.zeros(int(surrogate.state_dimension), dtype=float)
            if thermal_rhs_forcing is None
            else np.asarray(thermal_rhs_forcing, dtype=float).reshape(-1)
        )
        if forcing.shape != (int(surrogate.state_dimension),) or np.any(~np.isfinite(forcing)):
            raise ValueError("thermal_rhs_forcing dimension mismatch or non-finite values")
        self.thermal_rhs_forcing = forcing

    @property
    def thermal_rank(self) -> int:
        return self.surrogate.state_dimension

    @property
    def geometry_dimension(self) -> int:
        return self.surrogate.geometry_dimension

    @property
    def current_dimension(self) -> int:
        return self.surrogate.pod.current_dimension

    def _validate(self, state, geometry, operating):
        a = np.asarray(state, dtype=float).reshape(-1)
        g = np.asarray(geometry, dtype=float).reshape(-1)
        u = np.asarray(operating, dtype=float).reshape(-1)
        if a.shape != (self.thermal_rank,):
            raise ValueError("thermal state dimension mismatch")
        if g.shape != (self.geometry_dimension,):
            raise ValueError("geometry dimension mismatch")
        if u.shape != (self.current_dimension,):
            raise ValueError("operating/current dimension mismatch")
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(g)) or np.any(~np.isfinite(u)):
            raise ValueError("vector-field inputs must be finite")
        return a, g, u

    def heat_source(self, state, geometry, operating) -> np.ndarray:
        """Neural Joule source only, excluding deterministic thermal forcing."""
        a, g, u = self._validate(state, geometry, operating)
        q = np.asarray(self.surrogate.heat_source_numpy(a, g, u), dtype=float)
        if q.shape != (self.thermal_rank,) or np.any(~np.isfinite(q)):
            raise FloatingPointError("neural heat source is invalid")
        return q

    def thermal_source(self, state, geometry, operating) -> np.ndarray:
        """Total reduced thermal RHS source ``q_theta + f_T``."""
        return self.heat_source(state, geometry, operating) + self.thermal_rhs_forcing

    def rhs(self, state, geometry, operating) -> np.ndarray:
        a, g, u = self._validate(state, geometry, operating)
        operator = self.thermal_operators.operator(g)
        if operator.rank != self.thermal_rank:
            raise ValueError("thermal operator rank does not match surrogate")
        return -operator.stiffness @ a + self.thermal_source(a, g, u)

    def vector_field(self, state, geometry, operating) -> np.ndarray:
        a, g, u = self._validate(state, geometry, operating)
        operator = self.thermal_operators.operator(g)
        rhs = -operator.stiffness @ a + self.thermal_source(a, g, u)
        return np.linalg.solve(operator.mass, rhs)

    def state_jacobian(self, state, geometry, operating) -> np.ndarray:
        """Differentiate only the neural Joule source; hard forcing has zero Jacobian."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("install sdfmpneo[neural] for neural state Jacobians") from exc
        a, g, u = self._validate(state, geometry, operating)
        parameter = next(self.surrogate.network.parameters())
        at = torch.tensor(a, dtype=parameter.dtype, device=parameter.device, requires_grad=True)
        gt = torch.tensor(g, dtype=parameter.dtype, device=parameter.device)
        ut = torch.tensor(u, dtype=parameter.dtype, device=parameter.device)

        def source(x):
            return self.surrogate.heat_source_torch(x, gt, ut)

        jac_q = torch.autograd.functional.jacobian(source, at).detach().cpu().numpy().astype(float)
        operator = self.thermal_operators.operator(g)
        return np.linalg.solve(operator.mass, -operator.stiffness + jac_q)


__all__ = [
    "CallableThermalOperatorFamily",
    "FixedThermalOperatorFamily",
    "NeuralElectroThermalVectorField",
    "ReducedThermalOperator",
]