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

    def __post_init__(self):
        if self.conductivity <= 0 or not np.isfinite(self.conductivity):
            raise ValueError("conductivity must be positive")
        if self.relative_permeability <= 0 or not np.isfinite(self.relative_permeability):
            raise ValueError("relative_permeability must be positive")


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
