from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.special import hyp1f1

Signature = Tuple[int, ...]
TermKey = Tuple[int, Signature]


def _sig_add(a: Signature, b: Signature) -> Signature:
    if len(a) != len(b):
        raise ValueError("Decay signatures must have the same dimension")
    return tuple(x + y for x, y in zip(a, b))


def unit_signature(n_modes: int, i: int) -> Signature:
    s = [0] * n_modes
    s[i] = 1
    return tuple(s)


@dataclass(frozen=True)
class AnalyticSeries:
    """Sparse finite sum of c*t^m*exp(-(n dot lambda)t)."""

    n_modes: int
    terms: Dict[TermKey, complex]

    def __post_init__(self) -> None:
        clean: Dict[TermKey, complex] = {}
        for (power, sig), coeff in self.terms.items():
            if power < 0:
                raise ValueError("Polynomial power must be non-negative")
            if len(sig) != self.n_modes:
                raise ValueError("Decay signature dimension mismatch")
            if coeff != 0:
                clean[(int(power), tuple(int(x) for x in sig))] = complex(coeff)
        object.__setattr__(self, "terms", clean)

    @staticmethod
    def zero(n_modes: int) -> "AnalyticSeries":
        return AnalyticSeries(n_modes, {})

    @staticmethod
    def constant(n_modes: int, value: complex) -> "AnalyticSeries":
        return AnalyticSeries(n_modes, {(0, (0,) * n_modes): complex(value)})

    @staticmethod
    def decay(n_modes: int, mode: int, coeff: complex = 1.0) -> "AnalyticSeries":
        return AnalyticSeries(n_modes, {(0, unit_signature(n_modes, mode)): complex(coeff)})

    def __add__(self, other: "AnalyticSeries") -> "AnalyticSeries":
        if self.n_modes != other.n_modes:
            raise ValueError("Mode dimension mismatch")
        out = dict(self.terms)
        for key, value in other.terms.items():
            out[key] = out.get(key, 0.0) + value
            if out[key] == 0:
                del out[key]
        return AnalyticSeries(self.n_modes, out)

    def __sub__(self, other: "AnalyticSeries") -> "AnalyticSeries":
        return self + (-1.0) * other

    def __rmul__(self, scalar: complex) -> "AnalyticSeries":
        return AnalyticSeries(self.n_modes, {k: scalar * v for k, v in self.terms.items()})

    def __mul__(self, other):
        if np.isscalar(other):
            return other * self
        if not isinstance(other, AnalyticSeries):
            return NotImplemented
        if self.n_modes != other.n_modes:
            raise ValueError("Mode dimension mismatch")
        out: Dict[TermKey, complex] = {}
        for (m1, n1), c1 in self.terms.items():
            for (m2, n2), c2 in other.terms.items():
                key = (m1 + m2, _sig_add(n1, n2))
                out[key] = out.get(key, 0.0) + c1 * c2
        return AnalyticSeries(self.n_modes, out)

    def derivative(self, lambdas: np.ndarray) -> "AnalyticSeries":
        lambdas = np.asarray(lambdas, dtype=float)
        if lambdas.shape != (self.n_modes,):
            raise ValueError("lambdas must have shape (n_modes,)")
        out: Dict[TermKey, complex] = {}
        for (m, sig), c in self.terms.items():
            rho = float(np.dot(np.asarray(sig, dtype=float), lambdas))
            if m > 0:
                key = (m - 1, sig)
                out[key] = out.get(key, 0.0) + c * m
            key = (m, sig)
            out[key] = out.get(key, 0.0) - c * rho
        return AnalyticSeries(self.n_modes, out)

    def evaluate(self, t: float, lambdas: np.ndarray) -> complex:
        lambdas = np.asarray(lambdas, dtype=float)
        if lambdas.shape != (self.n_modes,):
            raise ValueError("lambdas must have shape (n_modes,)")
        total = 0.0 + 0.0j
        for (m, sig), c in self.terms.items():
            rho = float(np.dot(np.asarray(sig, dtype=float), lambdas))
            total += c * (t ** m) * np.exp(-rho * t)
        return total


@dataclass(frozen=True)
class ResponseTerm:
    target_mode: int
    power: int
    signature: Signature
    coeff: complex

    def evaluate(self, t: float, lambdas: np.ndarray) -> complex:
        lam = float(lambdas[self.target_mode])
        rho = float(np.dot(np.asarray(self.signature, dtype=float), lambdas))
        delta = lam - rho
        m = self.power
        integral = (t ** (m + 1)) * hyp1f1(m + 1, m + 2, delta * t) / (m + 1)
        return self.coeff * np.exp(-lam * t) * integral

    def derivative_value(self, t: float, lambdas: np.ndarray) -> complex:
        lam = float(lambdas[self.target_mode])
        rho = float(np.dot(np.asarray(self.signature, dtype=float), lambdas))
        forcing = self.coeff * (t ** self.power) * np.exp(-rho * t)
        return forcing - lam * self.evaluate(t, lambdas)


@dataclass(frozen=True)
class DeferredResponseSeries:
    n_modes: int
    terms: Tuple[ResponseTerm, ...]

    @staticmethod
    def from_source(source: AnalyticSeries, target_mode: int) -> "DeferredResponseSeries":
        return DeferredResponseSeries(
            source.n_modes,
            tuple(ResponseTerm(target_mode, m, sig, c) for (m, sig), c in source.terms.items()),
        )

    def evaluate(self, t: float, lambdas: np.ndarray) -> complex:
        return sum(term.evaluate(t, lambdas) for term in self.terms)

    def derivative_value(self, t: float, lambdas: np.ndarray) -> complex:
        return sum(term.derivative_value(t, lambdas) for term in self.terms)
