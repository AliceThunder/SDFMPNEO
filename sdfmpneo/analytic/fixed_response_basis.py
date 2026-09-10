from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class _LogicalNode:
    name: str
    target_mode: int


def _merge_signatures(a, b):
    """Merge two sorted modal-index signatures without O(n_modes) count vectors."""
    a = tuple(a); b = tuple(b)
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    if i < len(a): out.extend(a[i:])
    if j < len(b): out.extend(b[j:])
    return tuple(out)


class _ExpPoly:
    """Finite sums c*t^k*exp(-sum(lambda[idx])*t), with optional dense tangents.

    Exponential signatures store only participating modal indices.  This keeps
    symbolic bookkeeping proportional to nonlinear polynomial degree instead of
    proportional to the full thermal rank.
    """

    def __init__(self, n_modes, nd=0):
        self.n_modes = int(n_modes)
        self.nd = int(nd)
        self.terms = {}

    @classmethod
    def zero(cls, n_modes, nd=0):
        return cls(n_modes, nd)

    @classmethod
    def modal_term(cls, n_modes, mode, coefficient, *, nd=0, derivative=None):
        out = cls(n_modes, nd)
        out.add_term((int(mode),), 0, coefficient, derivative)
        return out

    @classmethod
    def constant_term(cls, n_modes, coefficient, *, nd=0, derivative=None):
        out = cls(n_modes, nd)
        out.add_term((), 0, coefficient, derivative)
        return out

    def add_term(self, signature, power, coefficient, derivative=None):
        signature = tuple(int(v) for v in signature)
        power = int(power)
        coefficient = float(coefficient)
        if signature != tuple(sorted(signature)) or any(v < 0 or v >= self.n_modes for v in signature):
            raise ValueError("invalid modal exponential signature")
        gradient = np.zeros(self.nd) if derivative is None else np.asarray(derivative, float)
        if power < 0 or gradient.shape != (self.nd,):
            raise ValueError("invalid exponential-polynomial term")
        key = signature, power
        old_c, old_g = self.terms.get(key, (0.0, np.zeros(self.nd)))
        coefficient += old_c
        gradient = gradient + old_g
        if coefficient == 0.0 and not np.any(gradient):
            self.terms.pop(key, None)
        else:
            self.terms[key] = coefficient, gradient
        return self

    def add_scaled(self, other, scale, parameter_index=None):
        if self.n_modes != other.n_modes or self.nd != other.nd:
            raise ValueError("signal dimensions do not match")
        scale = float(scale)
        for (signature, power), (coefficient, gradient) in other.terms.items():
            local = scale * gradient
            if parameter_index is not None:
                local = local.copy()
                local[int(parameter_index)] += coefficient
            self.add_term(signature, power, scale * coefficient, local)
        return self

    def product(self, other):
        if self.n_modes != other.n_modes or self.nd != other.nd:
            raise ValueError("signal dimensions do not match")
        out = _ExpPoly.zero(self.n_modes, self.nd)
        for (sa, pa), (va, ga) in self.terms.items():
            for (sb, pb), (vb, gb) in other.terms.items():
                out.add_term(
                    _merge_signatures(sa, sb),
                    pa + pb,
                    va * vb,
                    ga * vb + gb * va,
                )
        return out

    def _rate(self, signature, rates):
        if not signature:
            return 0.0
        return float(np.sum(rates[np.asarray(signature, dtype=int)]))

    def response(self, target, lambdas):
        rates = np.asarray(lambdas, float)
        target = int(target)
        lam = float(rates[target])
        target_signature = (target,)
        out = _ExpPoly.zero(self.n_modes, self.nd)
        for (signature, power), (coefficient, gradient) in self.terms.items():
            mu = self._rate(signature, rates)
            # Same pole (including numerically equal degenerate thermal modes):
            # integral t^k exp(-lambda t) gives t^(k+1)/(k+1).
            if lam == mu:
                factor = 1.0 / (power + 1)
                out.add_term(signature, power + 1, factor * coefficient, factor * gradient)
                continue
            delta = lam - mu
            factorial = math.factorial(power)
            for m in range(power + 1):
                p = power - m
                factor = ((-1.0) ** m) * factorial / math.factorial(p) / delta ** (m + 1)
                out.add_term(signature, p, factor * coefficient, factor * gradient)
            tail = -(((-1.0) ** power) * factorial / delta ** (power + 1))
            out.add_term(target_signature, 0, tail * coefficient, tail * gradient)
        return out

    def evaluate(self, t, lambdas):
        t = float(t)
        if not np.isfinite(t) or t < 0:
            raise ValueError("analytic segment time must be finite and non-negative")
        rates = np.asarray(lambdas, float)
        value = slope = 0.0
        gradient = np.zeros(self.nd)
        slope_gradient = np.zeros(self.nd)
        for (signature, power), (coefficient, dcoefficient) in self.terms.items():
            mu = self._rate(signature, rates)
            if t == 0.0:
                basis = 1.0 if power == 0 else 0.0
                dbasis = 1.0 if power == 1 else (-mu if power == 0 else 0.0)
            else:
                log_basis = power * math.log(t) - mu * t
                basis = 0.0 if log_basis < -745.0 else (
                    math.inf if log_basis > 709.0 else math.exp(log_basis)
                )
                dbasis = (
                    basis * (power / t - mu)
                    if np.isfinite(basis)
                    else math.copysign(math.inf, power / t - mu)
                )
            value += coefficient * basis
            slope += coefficient * dbasis
            gradient += dcoefficient * basis
            slope_gradient += dcoefficient * dbasis
        return value, slope, gradient, slope_gradient


def _weighted_sum(signals, weights, parameter_indices, nd):
    if len(signals) == 0:
        raise ValueError("weighted sum requires at least one signal")
    out = _ExpPoly.zero(signals[0].n_modes, nd)
    for i, (signal, weight) in enumerate(zip(signals, np.asarray(weights).reshape(-1))):
        index = None if parameter_indices is None else int(parameter_indices[i])
        out.add_scaled(signal, weight, index)
    return out


def _gated(signal, value, parameter_index, nd):
    out = _ExpPoly.zero(signal.n_modes, nd)
    out.add_scaled(signal, value, parameter_index)
    return out


