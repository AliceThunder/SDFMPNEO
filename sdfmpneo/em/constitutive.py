from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class ConductivityLaw:
    """Scalar analytic conductivity law interface."""

    def evaluate(self, temperature: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, temperature: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantConductivity(ConductivityLaw):
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("conductivity must be non-negative")

    def evaluate(self, temperature: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(temperature, dtype=float), self.sigma, dtype=float)

    def derivative(self, temperature: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(temperature, dtype=float))


@dataclass(frozen=True)
class AffineConductivity(ConductivityLaw):
    sigma_ref: float
    beta: float
    temperature_ref: float

    def evaluate(self, temperature: np.ndarray) -> np.ndarray:
        T = np.asarray(temperature, dtype=float)
        sigma = self.sigma_ref + self.beta * (T - self.temperature_ref)
        if np.any(sigma < 0):
            raise ValueError("affine conductivity law left its physically admissible non-negative domain")
        return sigma

    def derivative(self, temperature: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(temperature, dtype=float), self.beta, dtype=float)


@dataclass(frozen=True)
class ReciprocalLinearResistivity(ConductivityLaw):
    """Exact conductivity induced by a linear resistivity law.

    rho(T) = rho_ref [1 + alpha (T-T_ref)]
    sigma(T) = sigma_ref / [1 + alpha (T-T_ref)].

    This is not a polynomial approximation: evaluate() and derivative() use the
    rational law directly. The admissible domain is the set where the
    denominator is strictly positive.
    """

    sigma_ref: float
    alpha: float
    temperature_ref: float

    def __post_init__(self) -> None:
        if self.sigma_ref <= 0:
            raise ValueError("sigma_ref must be positive")

    def _denominator(self, temperature: np.ndarray) -> np.ndarray:
        T = np.asarray(temperature, dtype=float)
        d = 1.0 + self.alpha * (T - self.temperature_ref)
        if np.any(d <= 0):
            raise ValueError("linear-resistivity conductivity law left its positive-resistivity domain")
        return d

    def evaluate(self, temperature: np.ndarray) -> np.ndarray:
        d = self._denominator(temperature)
        return self.sigma_ref / d

    def derivative(self, temperature: np.ndarray) -> np.ndarray:
        d = self._denominator(temperature)
        return -self.sigma_ref * self.alpha / (d * d)


@dataclass(frozen=True)
class ConductivityRegion:
    name: str
    mask: np.ndarray
    law: ConductivityLaw


class CompositeCellConductivity:
    """Disjoint material-region conductivity field on one cell grid."""

    def __init__(
        self,
        shape_cells: tuple[int, int, int],
        regions: Sequence[ConductivityRegion],
    ):
        self.shape_cells = tuple(shape_cells)
        self.regions = tuple(regions)
        occupied = np.zeros(self.shape_cells, dtype=bool)
        for region in self.regions:
            mask = np.asarray(region.mask, dtype=bool)
            if mask.shape != self.shape_cells:
                raise ValueError(f"region {region.name!r} mask shape mismatch")
            if np.any(occupied & mask):
                raise ValueError("conductivity regions must be disjoint")
            occupied |= mask

    def evaluate(self, temperature_cell: np.ndarray) -> np.ndarray:
        T = np.asarray(temperature_cell, dtype=float)
        if T.shape != self.shape_cells:
            raise ValueError("temperature_cell shape mismatch")
        sigma = np.zeros(self.shape_cells, dtype=float)
        for region in self.regions:
            mask = np.asarray(region.mask, dtype=bool)
            sigma[mask] = region.law.evaluate(T[mask])
        return sigma

    def derivative(self, temperature_cell: np.ndarray) -> np.ndarray:
        T = np.asarray(temperature_cell, dtype=float)
        if T.shape != self.shape_cells:
            raise ValueError("temperature_cell shape mismatch")
        derivative = np.zeros(self.shape_cells, dtype=float)
        for region in self.regions:
            mask = np.asarray(region.mask, dtype=bool)
            derivative[mask] = region.law.derivative(T[mask])
        return derivative

    def region_masks(self) -> dict[str, np.ndarray]:
        return {region.name: np.asarray(region.mask, dtype=bool).copy() for region in self.regions}
