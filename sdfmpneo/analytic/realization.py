from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .long_time import realization_action


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
        A = scipy.linalg.block_diag(self.A, other.A)
        b = np.concatenate([self.b, other.b])
        c = np.concatenate([self.c, other.c])
        return AnalyticRealization(A, b, c)

    def product(self, other: "AnalyticRealization") -> "AnalyticRealization":
        """Exact product via the Kronecker-sum realization."""

        if not isinstance(other, AnalyticRealization):
            raise TypeError("other must be AnalyticRealization")
        I1 = np.eye(self.dimension, dtype=complex)
        I2 = np.eye(other.dimension, dtype=complex)
        A = np.kron(self.A, I2) + np.kron(I1, other.A)
        b = np.kron(self.b, other.b)
        c = np.kron(self.c, other.c)
        return AnalyticRealization(A, b, c)

    def response(self, decay_rate: float) -> "AnalyticRealization":
        """Exact y'+lambda*y=f, y(0)=0 response realization."""

        lam = float(decay_rate)
        if lam < 0:
            raise ValueError("response decay rate must be non-negative")
        n = self.dimension
        A = np.zeros((n + 1, n + 1), dtype=complex)
        A[:n, :n] = self.A
        A[n, :n] = self.c
        A[n, n] = -lam
        b = np.concatenate([self.b, np.zeros(1, dtype=complex)])
        c = np.zeros(n + 1, dtype=complex)
        c[n] = 1.0
        return AnalyticRealization(A, b, c)

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
