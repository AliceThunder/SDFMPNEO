from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np
import scipy.linalg

from .long_time import realization_action


_STRUCTURE_CACHE_SIZE = max(0, int(os.environ.get("SDFMPNEO_STRUCTURE_CACHE_ENTRIES", "1024")))


def _matrix_from_bytes(shape: tuple[int, int], data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.complex128).reshape(shape)


def _vector_from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.complex128)


@lru_cache(maxsize=_STRUCTURE_CACHE_SIZE)
def _cached_add_structure(shape1, A1_bytes, c1_bytes, shape2, A2_bytes, c2_bytes):
    A1 = _matrix_from_bytes(shape1, A1_bytes)
    A2 = _matrix_from_bytes(shape2, A2_bytes)
    c1 = _vector_from_bytes(c1_bytes)
    c2 = _vector_from_bytes(c2_bytes)
    return scipy.linalg.block_diag(A1, A2), np.concatenate([c1, c2])


@lru_cache(maxsize=_STRUCTURE_CACHE_SIZE)
def _cached_product_structure(shape1, A1_bytes, c1_bytes, shape2, A2_bytes, c2_bytes):
    A1 = _matrix_from_bytes(shape1, A1_bytes)
    A2 = _matrix_from_bytes(shape2, A2_bytes)
    c1 = _vector_from_bytes(c1_bytes)
    c2 = _vector_from_bytes(c2_bytes)
    I1 = np.eye(shape1[0], dtype=complex)
    I2 = np.eye(shape2[0], dtype=complex)
    A = np.kron(A1, I2) + np.kron(I1, A2)
    c = np.kron(c1, c2)
    return A, c


@lru_cache(maxsize=_STRUCTURE_CACHE_SIZE)
def _cached_response_structure(shape, A_bytes, c_bytes, decay_rate):
    source_A = _matrix_from_bytes(shape, A_bytes)
    source_c = _vector_from_bytes(c_bytes)
    n = shape[0]
    A = np.zeros((n + 1, n + 1), dtype=complex)
    A[:n, :n] = source_A
    A[n, :n] = source_c
    A[n, n] = -float(decay_rate)
    c = np.zeros(n + 1, dtype=complex)
    c[n] = 1.0
    return A, c


def clear_realization_structure_cache() -> None:
    _cached_add_structure.cache_clear()
    _cached_product_structure.cache_clear()
    _cached_response_structure.cache_clear()


@dataclass(frozen=True)
class AnalyticRealization:
    """Finite-dimensional exact realization f(t)=c^T exp(A t) b.

    This representation is closed under addition, scalar multiplication,
    products and first-order response operators. Repeated/near-repeated decay
    rates are handled by the matrix exponential through confluent/Jordan
    structure; no near-resonance threshold is required.
    """

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=complex)
        b = np.asarray(self.b, dtype=complex)
        c = np.asarray(self.c, dtype=complex)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if b.shape != (A.shape[0],) or c.shape != (A.shape[0],):
            raise ValueError("b and c must match realization dimension")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)

    @property
    def dimension(self) -> int:
        return self.A.shape[0]

    @staticmethod
    def zero() -> "AnalyticRealization":
        return AnalyticRealization(
            np.zeros((1, 1), dtype=complex),
            np.zeros(1, dtype=complex),
            np.ones(1, dtype=complex),
        )

    @staticmethod
    def constant(value: complex) -> "AnalyticRealization":
        return AnalyticRealization(
            np.zeros((1, 1), dtype=complex),
            np.array([complex(value)]),
            np.ones(1, dtype=complex),
        )

    @staticmethod
    def decay(rate: float, amplitude: complex = 1.0) -> "AnalyticRealization":
        if rate < 0:
            raise ValueError("decay rate must be non-negative")
        return AnalyticRealization(
            np.array([[-float(rate)]], dtype=complex),
            np.array([complex(amplitude)]),
            np.ones(1, dtype=complex),
        )

    def scaled(self, value: complex) -> "AnalyticRealization":
        return AnalyticRealization(self.A, self.b, complex(value) * self.c)

    def add(self, other: "AnalyticRealization") -> "AnalyticRealization":
        if not isinstance(other, AnalyticRealization):
            raise TypeError("other must be AnalyticRealization")
        A, c = _cached_add_structure(
            self.A.shape, self.A.tobytes(), self.c.tobytes(),
            other.A.shape, other.A.tobytes(), other.c.tobytes(),
        )
        b = np.concatenate([self.b, other.b])
        # Copy cached structure arrays so public realization arrays remain
        # independently mutable exactly as before this optimization.
        return AnalyticRealization(A.copy(), b, c.copy())

    def product(self, other: "AnalyticRealization") -> "AnalyticRealization":
        """Exact product via the Kronecker-sum realization."""

        if not isinstance(other, AnalyticRealization):
            raise TypeError("other must be AnalyticRealization")
        A, c = _cached_product_structure(
            self.A.shape, self.A.tobytes(), self.c.tobytes(),
            other.A.shape, other.A.tobytes(), other.c.tobytes(),
        )
        b = np.kron(self.b, other.b)
        return AnalyticRealization(A.copy(), b, c.copy())

    def response(self, decay_rate: float) -> "AnalyticRealization":
        """Exact y'+lambda*y=f, y(0)=0 response realization."""

        lam = float(decay_rate)
        if lam < 0:
            raise ValueError("response decay rate must be non-negative")
        A, c = _cached_response_structure(
            self.A.shape, self.A.tobytes(), self.c.tobytes(), lam
        )
        b = np.concatenate([self.b, np.zeros(1, dtype=complex)])
        return AnalyticRealization(A.copy(), b, c.copy())

    def evaluate(self, t: float) -> complex:
        if t < 0:
            raise ValueError("time must be non-negative")
        state = realization_action(self.A, self.b, t)
        return complex(self.c @ state)

    def derivative_value(self, t: float) -> complex:
        if t < 0:
            raise ValueError("time must be non-negative")
        state = realization_action(self.A, self.b, t)
        return 0j if np.isposinf(t) else complex(self.c @ (self.A @ state))


@dataclass(frozen=True)
class CompiledRealizationGraph:
    lambdas: np.ndarray
    node_realizations: dict[str, AnalyticRealization]
    source_realizations: dict[str, AnalyticRealization]
    mode_realizations: tuple[AnalyticRealization, ...]

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([np.real(r.evaluate(t)) for r in self.mode_realizations], dtype=float)
        da = np.array([np.real(r.derivative_value(t)) for r in self.mode_realizations], dtype=float)
        return a, da

    def node(self, name: str) -> AnalyticRealization:
        return self.node_realizations[name]

    def source(self, name: str) -> AnalyticRealization:
        return self.source_realizations[name]

    @property
    def total_mode_state_dimension(self) -> int:
        return sum(r.dimension for r in self.mode_realizations)
