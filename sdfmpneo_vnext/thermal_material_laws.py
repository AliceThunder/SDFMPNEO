from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .scene import EPS0, MU0, TensorElectricMaterial


class MaterialTemperatureLaw(Protocol):
    """Explicit constitutive law that rebuilds one EM material at temperature."""

    def at_temperature(self, temperature: float):
        ...


def _linear_factor(
    coefficient: float,
    temperature: float,
    reference_temperature: float,
    *,
    name: str,
    allow_zero: bool,
) -> float:
    coefficient = float(coefficient)
    temperature = float(temperature)
    reference_temperature = float(reference_temperature)
    if (
        not np.isfinite(coefficient)
        or not np.isfinite(temperature)
        or not np.isfinite(reference_temperature)
    ):
        raise ValueError(f"{name} temperature law parameters must be finite")
    value = 1.0 + coefficient * (temperature - reference_temperature)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(
            f"temperature is outside the linear {name} law domain; "
            f"scale factor must remain {qualifier}"
        )
    return float(value)


def _validate_law_parameters(reference_temperature: float, coefficients):
    if not np.isfinite(reference_temperature):
        raise ValueError("reference_temperature must be finite")
    if any(not np.isfinite(float(value)) for value in coefficients):
        raise ValueError("temperature coefficients must be finite")


@dataclass(frozen=True)
class TemperatureAdjustedIsotropicMaterial:
    """Passive isotropic material view with explicit temperature scaling.

    The underlying material keeps its frequency-dispersion model. Temperature
    scales only the real electric coefficient, the total effective electric
    loss conductivity and isotropic magnetic permeability. This avoids
    inventing Debye poles or other hidden dispersion parameters.
    """

    base_material: object
    temperature: float
    reference_temperature: float = 293.15
    permittivity_temperature_coefficient: float = 0.0
    loss_temperature_coefficient: float = 0.0
    permeability_temperature_coefficient: float = 0.0

    def __post_init__(self):
        required = (
            "relative_permeability",
            "conductivity",
            "relative_permittivity_at",
            "complex_permittivity",
            "loss_conductivity",
        )
        if isinstance(self.base_material, TensorElectricMaterial):
            raise TypeError(
                "TemperatureAdjustedIsotropicMaterial cannot wrap "
                "TensorElectricMaterial; use LinearTensorTemperatureLaw"
            )
        if any(not hasattr(self.base_material, name) for name in required):
            raise TypeError("base_material does not satisfy the passive material contract")
        _validate_law_parameters(
            self.reference_temperature,
            (
                self.permittivity_temperature_coefficient,
                self.loss_temperature_coefficient,
                self.permeability_temperature_coefficient,
            ),
        )
        # Validate the requested temperature immediately, so an invalid warm
        # scene cannot be constructed and fail later inside an EM solve.
        self._factors()

    def _factors(self):
        return (
            _linear_factor(
                self.permittivity_temperature_coefficient,
                self.temperature,
                self.reference_temperature,
                name="permittivity",
                allow_zero=False,
            ),
            _linear_factor(
                self.loss_temperature_coefficient,
                self.temperature,
                self.reference_temperature,
                name="loss conductivity",
                allow_zero=True,
            ),
            _linear_factor(
                self.permeability_temperature_coefficient,
                self.temperature,
                self.reference_temperature,
                name="permeability",
                allow_zero=False,
            ),
        )

    @property
    def relative_permeability(self) -> float:
        _, _, factor = self._factors()
        return float(self.base_material.relative_permeability) * factor

    @property
    def permeability(self) -> float:
        return MU0 * self.relative_permeability

    @property
    def conductivity(self) -> float:
        _, factor, _ = self._factors()
        return max(float(self.base_material.conductivity) * factor, 0.0)

    def loss_conductivity(self, frequency_hz: float) -> float:
        frequency_hz = float(frequency_hz)
        if not np.isfinite(frequency_hz) or frequency_hz < 0.0:
            raise ValueError("frequency_hz must be finite and nonnegative")
        _, factor, _ = self._factors()
        return max(
            float(self.base_material.loss_conductivity(frequency_hz)) * factor,
            0.0,
        )

    def _base_relative_real(self, frequency_hz: float) -> float:
        frequency_hz = float(frequency_hz)
        try:
            relative = self.base_material.relative_permittivity_at(frequency_hz)
            value = float(np.real(relative))
        except (ValueError, NotImplementedError):
            if frequency_hz != 0.0 or not hasattr(
                self.base_material, "relative_permittivity"
            ):
                raise
            value = float(self.base_material.relative_permittivity)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("base material has invalid real relative permittivity")
        return value

    def relative_permittivity_at(self, frequency_hz: float) -> complex:
        frequency_hz = float(frequency_hz)
        if not np.isfinite(frequency_hz) or frequency_hz < 0.0:
            raise ValueError("frequency_hz must be finite and nonnegative")
        epsilon_factor, _, _ = self._factors()
        real_relative = epsilon_factor * self._base_relative_real(frequency_hz)
        if frequency_hz == 0.0:
            return complex(real_relative)
        omega = 2.0 * np.pi * frequency_hz
        loss = self.loss_conductivity(frequency_hz)
        return complex(real_relative - 1j * loss / (omega * EPS0))

    def complex_permittivity(self, frequency_hz: float) -> complex:
        return EPS0 * self.relative_permittivity_at(frequency_hz)

    def __getattr__(self, name):
        # Preserve optional thermal metadata and any passive-material metadata
        # without duplicating all currently supported dielectric model classes.
        return getattr(self.base_material, name)


@dataclass(frozen=True)
class LinearIsotropicTemperatureLaw:
    material: object
    reference_temperature: float = 293.15
    permittivity_temperature_coefficient: float = 0.0
    loss_temperature_coefficient: float = 0.0
    permeability_temperature_coefficient: float = 0.0

    def __post_init__(self):
        if isinstance(self.material, TensorElectricMaterial):
            raise TypeError(
                "LinearIsotropicTemperatureLaw requires an isotropic material"
            )
        _validate_law_parameters(
            self.reference_temperature,
            (
                self.permittivity_temperature_coefficient,
                self.loss_temperature_coefficient,
                self.permeability_temperature_coefficient,
            ),
        )

    def at_temperature(self, temperature: float) -> TemperatureAdjustedIsotropicMaterial:
        return TemperatureAdjustedIsotropicMaterial(
            self.material,
            float(temperature),
            reference_temperature=float(self.reference_temperature),
            permittivity_temperature_coefficient=float(
                self.permittivity_temperature_coefficient
            ),
            loss_temperature_coefficient=float(self.loss_temperature_coefficient),
            permeability_temperature_coefficient=float(
                self.permeability_temperature_coefficient
            ),
        )


@dataclass(frozen=True)
class LinearTensorTemperatureLaw:
    material: TensorElectricMaterial
    reference_temperature: float = 293.15
    permittivity_temperature_coefficient: float = 0.0
    loss_temperature_coefficient: float = 0.0
    permeability_temperature_coefficient: float = 0.0

    def __post_init__(self):
        if not isinstance(self.material, TensorElectricMaterial):
            raise TypeError(
                "LinearTensorTemperatureLaw requires TensorElectricMaterial"
            )
        _validate_law_parameters(
            self.reference_temperature,
            (
                self.permittivity_temperature_coefficient,
                self.loss_temperature_coefficient,
                self.permeability_temperature_coefficient,
            ),
        )

    def at_temperature(self, temperature: float) -> TensorElectricMaterial:
        epsilon_factor = _linear_factor(
            self.permittivity_temperature_coefficient,
            temperature,
            self.reference_temperature,
            name="tensor permittivity",
            allow_zero=False,
        )
        loss_factor = _linear_factor(
            self.loss_temperature_coefficient,
            temperature,
            self.reference_temperature,
            name="tensor loss conductivity",
            allow_zero=True,
        )
        permeability_factor = _linear_factor(
            self.permeability_temperature_coefficient,
            temperature,
            self.reference_temperature,
            name="tensor permeability",
            allow_zero=False,
        )
        return TensorElectricMaterial(
            relative_permittivity_tensor=(
                epsilon_factor
                * np.asarray(self.material.relative_permittivity_tensor, dtype=float)
            ),
            conductivity_tensor=(
                loss_factor
                * np.asarray(self.material.conductivity_tensor, dtype=float)
            ),
            relative_permeability=(
                permeability_factor * float(self.material.relative_permeability)
            ),
            thermal_conductivity=getattr(
                self.material, "thermal_conductivity", None
            ),
            density=getattr(self.material, "density", None),
            heat_capacity=getattr(self.material, "heat_capacity", None),
            thermal_conductivity_tensor=getattr(
                self.material, "thermal_conductivity_tensor", None
            ),
        )
