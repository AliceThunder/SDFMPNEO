from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .geometry import SuperellipseSpiral

MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class ConductorMaterial:
    conductivity: float
    relative_permeability: float = 1.0
    resistance_temperature_coefficient: float = 0.0
    reference_temperature: float = 293.15

    def __post_init__(self):
        if self.conductivity <= 0 or not np.isfinite(self.conductivity):
            raise ValueError("conductivity must be positive")
        if self.relative_permeability <= 0 or not np.isfinite(self.relative_permeability):
            raise ValueError("relative_permeability must be positive")
        if (
            not np.isfinite(self.resistance_temperature_coefficient)
            or self.resistance_temperature_coefficient < 0
        ):
            raise ValueError(
                "resistance_temperature_coefficient must be finite and nonnegative"
            )
        if not np.isfinite(self.reference_temperature):
            raise ValueError("reference_temperature must be finite")

    def conductivity_at(self, temperature: float) -> float:
        factor = 1.0 + self.resistance_temperature_coefficient * (
            float(temperature) - self.reference_temperature
        )
        if factor <= 0 or not np.isfinite(factor):
            raise ValueError(
                "temperature is outside the linear-resistivity material domain"
            )
        return float(self.conductivity / factor)

    def at_temperature(self, temperature: float) -> "ConductorMaterial":
        return ConductorMaterial(
            self.conductivity_at(temperature),
            self.relative_permeability,
            0.0,
            float(temperature),
        )


@dataclass(frozen=True)
class HomogeneousMedium:
    relative_permittivity: float = 1.0
    relative_permeability: float = 1.0
    conductivity: float = 0.0

    def __post_init__(self):
        if self.relative_permittivity <= 0 or self.relative_permeability <= 0 or self.conductivity < 0:
            raise ValueError("invalid passive homogeneous medium")

    @property
    def permeability(self) -> float:
        return MU0 * self.relative_permeability


@dataclass(frozen=True)
class CoilObject:
    geometry: SuperellipseSpiral
    material: ConductorMaterial
    name: str = "coil"


@dataclass(frozen=True)
class Scene:
    coils: Tuple[CoilObject, ...]
    medium: HomogeneousMedium = HomogeneousMedium()

    def __post_init__(self):
        object.__setattr__(self, "coils", tuple(self.coils))
        if not self.coils:
            raise ValueError("scene must contain at least one coil")
        if not all(
            isinstance(
                coil,
                CoilObject,
            )
            for coil in self.coils
        ):
            raise TypeError(
                "scene coils must be CoilObject instances"
            )
        if not isinstance(
            self.medium,
            HomogeneousMedium,
        ):
            raise TypeError(
                "vNext MVP currently supports only HomogeneousMedium; "
                "heterogeneous electromagnetic media require the post-MVP "
                "SIE/VIE extension"
            )
        if not np.isclose(
            self.medium.conductivity,
            0.0,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "vNext MVP electromagnetic solvers currently require a "
                "lossless homogeneous background (medium.conductivity == 0); "
                "lossy media require an explicit dielectric/environment "
                "dissipation channel and are not silently approximated"
            )
