from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .algebra import AnalyticSeries, TermKey


@dataclass(frozen=True)
class CompiledAnalyticKernel:
    """Vectorized executable form of mode-wise analytic series.

    Every unique (power, decay-signature) basis function is evaluated once.
    Mode outputs are produced by one coefficient-matrix multiplication.
    """

    lambdas: np.ndarray
    keys: Tuple[TermKey, ...]
    coefficients: np.ndarray
    powers: np.ndarray
    rates: np.ndarray

    @staticmethod
    def build(mode_series: Sequence[AnalyticSeries], lambdas: np.ndarray) -> "CompiledAnalyticKernel":
        lambdas = np.asarray(lambdas, dtype=float)
        if not mode_series:
            raise ValueError("mode_series cannot be empty")
        n_modes = len(mode_series)
        if lambdas.shape != (n_modes,):
            raise ValueError("lambdas and mode_series must have equal mode dimension")
        if any(series.n_modes != n_modes for series in mode_series):
            raise ValueError("All mode series must share the same n_modes")

        key_set = set()
        for series in mode_series:
            key_set.update(series.terms)
        keys = tuple(sorted(key_set, key=lambda item: (item[0], item[1])))

        coefficients = np.zeros((n_modes, len(keys)), dtype=complex)
        index = {key: j for j, key in enumerate(keys)}
        for i, series in enumerate(mode_series):
            for key, coeff in series.terms.items():
                coefficients[i, index[key]] = coeff

        powers = np.array([key[0] for key in keys], dtype=int)
        rates = np.array(
            [float(np.dot(np.asarray(key[1], dtype=float), lambdas)) for key in keys],
            dtype=float,
        )
        return CompiledAnalyticKernel(lambdas.copy(), keys, coefficients, powers, rates)

    def basis(self, t: float) -> np.ndarray:
        if not self.keys:
            return np.empty(0, dtype=float)
        return np.power(t, self.powers) * np.exp(-self.rates * t)

    def basis_derivative(self, t: float) -> np.ndarray:
        if not self.keys:
            return np.empty(0, dtype=float)
        exp_part = np.exp(-self.rates * t)
        values = -self.rates * np.power(t, self.powers) * exp_part
        mask = self.powers > 0
        if np.any(mask):
            values[mask] += self.powers[mask] * np.power(t, self.powers[mask] - 1) * exp_part[mask]
        return values

    def evaluate(self, t: float):
        phi = self.basis(t)
        dphi = self.basis_derivative(t)
        a = np.real(self.coefficients @ phi)
        da = np.real(self.coefficients @ dphi)
        return np.asarray(a, dtype=float), np.asarray(da, dtype=float)

    @property
    def unique_term_count(self) -> int:
        return len(self.keys)
