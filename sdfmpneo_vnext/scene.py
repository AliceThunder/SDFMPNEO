from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .geometry import SuperellipseSpiral
from .package_geometry import SuperquadricPackageGeometry

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

    def complex_permittivity(
        self,
        frequency_hz: float,
    ) -> complex:
        """Passive e^{+j omega t} complex permittivity.

        The conductivity term is represented as -j*sigma/omega. A conductive
        homogeneous background therefore has no finite complex-permittivity
        representation at exactly DC; the static conduction problem is a
        separate physical limit.
        """
        if (
            not np.isfinite(frequency_hz)
            or frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        epsilon = (
            EPS0
            * self.relative_permittivity
        )
        if self.conductivity == 0.0:
            return complex(
                epsilon
            )
        if frequency_hz == 0.0:
            raise ValueError(
                "conductive homogeneous background has no finite "
                "complex-permittivity representation at DC"
            )
        omega = (
            2.0
            * np.pi
            * float(
                frequency_hz
            )
        )
        return complex(
            epsilon
            - 1j
            * self.conductivity
            / omega
        )

    def loss_conductivity(
        self,
        frequency_hz: float,
    ) -> float:
        if (
            not np.isfinite(frequency_hz)
            or frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        return float(
            self.conductivity
        )


@dataclass(frozen=True)
class IsotropicMaterial:
    relative_permittivity: float = 1.0
    relative_permeability: float = 1.0
    conductivity: float = 0.0
    thermal_conductivity: float | None = None
    density: float | None = None
    heat_capacity: float | None = None

    def __post_init__(self):
        if (
            not np.isfinite(self.relative_permittivity)
            or self.relative_permittivity <= 0.0
            or not np.isfinite(self.relative_permeability)
            or self.relative_permeability <= 0.0
            or not np.isfinite(self.conductivity)
            or self.conductivity < 0.0
        ):
            raise ValueError(
                "invalid passive isotropic electromagnetic material"
            )
        thermal = (
            self.thermal_conductivity,
            self.density,
            self.heat_capacity,
        )
        if any(
            value is not None
            for value in thermal
        ):
            if not all(
                value is not None
                and np.isfinite(value)
                and value > 0.0
                for value in thermal
            ):
                raise ValueError(
                    "thermal_conductivity, density, and heat_capacity "
                    "must be supplied together as positive finite values"
                )

    @property
    def permeability(
        self,
    ) -> float:
        return (
            MU0
            * self.relative_permeability
        )

    def relative_permittivity_at(
        self,
        frequency_hz: float,
    ) -> complex:
        return (
            self.complex_permittivity(
                frequency_hz
            )
            / EPS0
        )

    def loss_conductivity(
        self,
        frequency_hz: float,
    ) -> float:
        if (
            not np.isfinite(
                frequency_hz
            )
            or frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        return float(
            self.conductivity
        )

    def complex_permittivity(
        self,
        frequency_hz: float,
    ) -> complex:
        if (
            not np.isfinite(frequency_hz)
            or frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        epsilon = (
            EPS0
            * self.relative_permittivity
        )
        if self.conductivity == 0.0:
            return complex(
                epsilon
            )
        if frequency_hz == 0.0:
            raise ValueError(
                "conductive material has no finite complex-permittivity "
                "representation at DC"
            )
        omega = (
            2.0
            * np.pi
            * frequency_hz
        )
        return complex(
            epsilon
            - 1j
            * self.conductivity
            / omega
        )


@dataclass(frozen=True)
class DebyeMaterial:
    relative_permittivity_static: float
    relative_permittivity_infinite: float
    relaxation_time: float
    relative_permeability: float = 1.0
    conductivity: float = 0.0
    thermal_conductivity: float | None = None
    density: float | None = None
    heat_capacity: float | None = None

    def __post_init__(
        self,
    ):
        if (
            not np.isfinite(
                self.relative_permittivity_static
            )
            or not np.isfinite(
                self.relative_permittivity_infinite
            )
            or self.relative_permittivity_infinite
            <= 0.0
            or self.relative_permittivity_static
            < self.relative_permittivity_infinite
            or not np.isfinite(
                self.relaxation_time
            )
            or self.relaxation_time
            <= 0.0
            or not np.isfinite(
                self.relative_permeability
            )
            or self.relative_permeability
            <= 0.0
            or not np.isfinite(
                self.conductivity
            )
            or self.conductivity
            < 0.0
        ):
            raise ValueError(
                "invalid passive Debye material parameters"
            )
        thermal = (
            self.thermal_conductivity,
            self.density,
            self.heat_capacity,
        )
        if any(
            value is not None
            for value in thermal
        ):
            if not all(
                value is not None
                and np.isfinite(
                    value
                )
                and value > 0.0
                for value in thermal
            ):
                raise ValueError(
                    "thermal_conductivity, density, and heat_capacity "
                    "must be supplied together as positive finite values"
                )

    @property
    def permeability(
        self,
    ) -> float:
        return (
            MU0
            * self.relative_permeability
        )

    @property
    def relative_permittivity(
        self,
    ) -> float:
        """Static relative permittivity for backward-compatible metadata."""
        return float(
            self.relative_permittivity_static
        )

    def relative_permittivity_at(
        self,
        frequency_hz: float,
    ) -> complex:
        return (
            self.complex_permittivity(
                frequency_hz
            )
            / EPS0
        )

    def complex_permittivity(
        self,
        frequency_hz: float,
    ) -> complex:
        if (
            not np.isfinite(
                frequency_hz
            )
            or frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        if frequency_hz == 0.0:
            if self.conductivity > 0.0:
                raise ValueError(
                    "conductive Debye material has no finite "
                    "complex-permittivity representation at DC"
                )
            return complex(
                EPS0
                * self.relative_permittivity_static
            )
        omega = (
            2.0
            * np.pi
            * float(
                frequency_hz
            )
        )
        susceptibility = (
            self.relative_permittivity_static
            - self.relative_permittivity_infinite
        ) / (
            1.0
            + 1j
            * omega
            * self.relaxation_time
        )
        relative = (
            self.relative_permittivity_infinite
            + susceptibility
        )
        return complex(
            EPS0
            * relative
            - 1j
            * self.conductivity
            / omega
        )

    def loss_conductivity(
        self,
        frequency_hz: float,
    ) -> float:
        if (
            not np.isfinite(
                frequency_hz
            )
            or frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        if frequency_hz == 0.0:
            return float(
                self.conductivity
            )
        omega = (
            2.0
            * np.pi
            * float(
                frequency_hz
            )
        )
        epsilon = (
            self.complex_permittivity(
                frequency_hz
            )
        )
        effective = (
            -omega
            * float(
                np.imag(
                    epsilon
                )
            )
        )
        return float(
            max(
                effective,
                0.0,
            )
        )


@dataclass(frozen=True)
class PackageObject:
    geometry: SuperquadricPackageGeometry
    material: IsotropicMaterial | DebyeMaterial
    name: str = "package"

    def __post_init__(self):
        if not isinstance(
            self.geometry,
            SuperquadricPackageGeometry,
        ):
            raise TypeError(
                "package geometry must be SuperquadricPackageGeometry"
            )
        if not isinstance(
            self.material,
            (
                IsotropicMaterial,
                DebyeMaterial,
            ),
        ):
            raise TypeError(
                "package material must be a supported passive isotropic material"
            )


@dataclass(frozen=True)
class CoilObject:
    geometry: SuperellipseSpiral
    material: ConductorMaterial
    name: str = "coil"


@dataclass(frozen=True)
class Scene:
    coils: Tuple[CoilObject, ...]
    medium: HomogeneousMedium | IsotropicMaterial | DebyeMaterial = HomogeneousMedium()
    packages: Tuple[PackageObject, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "coils", tuple(self.coils))
        object.__setattr__(self, "packages", tuple(self.packages))
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
            (
                HomogeneousMedium,
                IsotropicMaterial,
                DebyeMaterial,
            ),
        ):
            raise TypeError(
                "scene medium must be a supported homogeneous passive "
                "electromagnetic material"
            )
        if not all(
            isinstance(
                package,
                PackageObject,
            )
            for package in self.packages
        ):
            raise TypeError(
                "scene packages must be PackageObject instances"
            )

    def require_mvp_electromagnetic_scope(
        self,
    ) -> None:
        if not np.isclose(
            self.medium.loss_conductivity(
                0.0
            ),
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
        if self.packages:
            raise NotImplementedError(
                "package geometry is represented by Scene, but electromagnetic "
                "package coupling requires the post-MVP SIE/VIE backend and "
                "must not be silently ignored"
            )
