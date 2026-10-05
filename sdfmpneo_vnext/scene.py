from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable
import numpy as np
from .geometry import SuperellipseSpiral
from .package_geometry import SuperquadricPackageGeometry

MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12


def _validated_optional_thermal_tensor(
    scalar,
    tensor,
    density,
    heat_capacity,
):
    if (
        scalar is None
        and tensor is None
        and density is None
        and heat_capacity is None
    ):
        return None
    if (
        scalar is not None
        and tensor is not None
    ):
        raise ValueError(
            "thermal_conductivity and thermal_conductivity_tensor are "
            "mutually exclusive"
        )
    if (
        density is None
        or heat_capacity is None
        or (
            scalar is None
            and tensor is None
        )
        or not np.isfinite(
            density
        )
        or density <= 0.0
        or not np.isfinite(
            heat_capacity
        )
        or heat_capacity <= 0.0
    ):
        raise ValueError(
            "thermal conductivity, density, and heat_capacity must be "
            "supplied together as positive finite properties"
        )
    if scalar is not None:
        if (
            not np.isfinite(
                scalar
            )
            or scalar <= 0.0
        ):
            raise ValueError(
                "thermal_conductivity must be positive and finite"
            )
        return None
    value = np.asarray(
        tensor,
        dtype=float,
    )
    if (
        value.shape != (
            3,
            3,
        )
        or np.any(
            ~np.isfinite(
                value
            )
        )
        or not np.allclose(
            value,
            value.T,
            rtol=1e-12,
            atol=1e-14,
        )
        or np.min(
            np.linalg.eigvalsh(
                value
            )
        ) <= 0.0
    ):
        raise ValueError(
            "thermal_conductivity_tensor must be a finite symmetric "
            "positive-definite 3x3 matrix"
        )
    return value.copy()


@runtime_checkable
class PassiveIsotropicMaterial(Protocol):
    relative_permeability: float
    conductivity: float

    @property
    def permeability(
        self,
    ) -> float:
        ...

    def complex_permittivity(
        self,
        frequency_hz: float,
    ) -> complex:
        ...

    def relative_permittivity_at(
        self,
        frequency_hz: float,
    ) -> complex:
        ...

    def loss_conductivity(
        self,
        frequency_hz: float,
    ) -> float:
        ...


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
    thermal_conductivity_tensor: np.ndarray | None = None

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
        scalar_thermal = (
            self.thermal_conductivity
        )
        tensor_thermal = (
            self.thermal_conductivity_tensor
        )
        if (
            scalar_thermal is not None
            and tensor_thermal is not None
        ):
            raise ValueError(
                "thermal_conductivity and thermal_conductivity_tensor are "
                "mutually exclusive"
            )
        has_thermal = bool(
            scalar_thermal is not None
            or tensor_thermal is not None
            or self.density is not None
            or self.heat_capacity is not None
        )
        if has_thermal:
            if (
                self.density is None
                or self.heat_capacity is None
                or not np.isfinite(
                    self.density
                )
                or self.density <= 0.0
                or not np.isfinite(
                    self.heat_capacity
                )
                or self.heat_capacity <= 0.0
                or (
                    scalar_thermal is None
                    and tensor_thermal is None
                )
            ):
                raise ValueError(
                    "thermal conductivity, density, and heat_capacity must be "
                    "supplied together as positive finite properties"
                )
            if scalar_thermal is not None:
                if (
                    not np.isfinite(
                        scalar_thermal
                    )
                    or scalar_thermal <= 0.0
                ):
                    raise ValueError(
                        "thermal_conductivity must be positive and finite"
                    )
            else:
                tensor = np.asarray(
                    tensor_thermal,
                    dtype=float,
                )
                if (
                    tensor.shape != (
                        3,
                        3,
                    )
                    or np.any(
                        ~np.isfinite(
                            tensor
                        )
                    )
                    or not np.allclose(
                        tensor,
                        tensor.T,
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    or np.min(
                        np.linalg.eigvalsh(
                            tensor
                        )
                    ) <= 0.0
                ):
                    raise ValueError(
                        "thermal_conductivity_tensor must be a finite "
                        "symmetric positive-definite 3x3 matrix"
                    )
                object.__setattr__(
                    self,
                    "thermal_conductivity_tensor",
                    tensor.copy(),
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
    thermal_conductivity_tensor: np.ndarray | None = None

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
        tensor = _validated_optional_thermal_tensor(
            self.thermal_conductivity,
            self.thermal_conductivity_tensor,
            self.density,
            self.heat_capacity,
        )
        if tensor is not None:
            object.__setattr__(
                self,
                "thermal_conductivity_tensor",
                tensor,
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
class MultiDebyeMaterial:
    """Passive isotropic multi-pole Debye dielectric.

    Positive relaxation strengths approximate a broad causal dielectric
    spectrum while preserving passivity under the e^{+j omega t} convention.
    """

    relative_permittivity_infinite: float
    relaxation_strengths: tuple[float, ...]
    relaxation_times: tuple[float, ...]
    relative_permeability: float = 1.0
    conductivity: float = 0.0
    thermal_conductivity: float | None = None
    density: float | None = None
    heat_capacity: float | None = None
    thermal_conductivity_tensor: np.ndarray | None = None

    def __post_init__(
        self,
    ):
        strengths = tuple(
            float(
                value
            )
            for value
            in self.relaxation_strengths
        )
        times = tuple(
            float(
                value
            )
            for value
            in self.relaxation_times
        )
        if (
            not np.isfinite(
                self.relative_permittivity_infinite
            )
            or self.relative_permittivity_infinite
            <= 0.0
            or not strengths
            or len(
                strengths
            )
            != len(
                times
            )
            or any(
                not np.isfinite(
                    value
                )
                or value < 0.0
                for value
                in strengths
            )
            or any(
                not np.isfinite(
                    value
                )
                or value <= 0.0
                for value
                in times
            )
            or not np.isfinite(
                self.relative_permeability
            )
            or self.relative_permeability
            <= 0.0
            or not np.isfinite(
                self.conductivity
            )
            or self.conductivity < 0.0
        ):
            raise ValueError(
                "invalid passive multi-Debye material parameters"
            )
        tensor = _validated_optional_thermal_tensor(
            self.thermal_conductivity,
            self.thermal_conductivity_tensor,
            self.density,
            self.heat_capacity,
        )
        if tensor is not None:
            object.__setattr__(
                self,
                "thermal_conductivity_tensor",
                tensor,
            )
        object.__setattr__(
            self,
            "relaxation_strengths",
            strengths,
        )
        object.__setattr__(
            self,
            "relaxation_times",
            times,
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
        return float(
            self.relative_permittivity_infinite
            + sum(
                self.relaxation_strengths
            )
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
                    "conductive multi-Debye material has no finite "
                    "complex-permittivity representation at DC"
                )
            return complex(
                EPS0
                * self.relative_permittivity
            )
        omega = (
            2.0
            * np.pi
            * float(
                frequency_hz
            )
        )
        relative = complex(
            self.relative_permittivity_infinite
        )
        for strength, tau in zip(
            self.relaxation_strengths,
            self.relaxation_times,
        ):
            relative += (
                strength
                / (
                    1.0
                    + 1j
                    * omega
                    * tau
                )
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
        epsilon = self.complex_permittivity(
            frequency_hz
        )
        return float(
            max(
                -omega
                * float(
                    np.imag(
                        epsilon
                    )
                ),
                0.0,
            )
        )


@dataclass(frozen=True)
class TabulatedMaterial:
    """Passive isotropic material interpolated on a positive frequency table.

    The table stores Re(epsilon_r) and additional effective AC loss
    conductivity. Interpolation is linear in log-frequency and extrapolation is
    intentionally rejected.
    """

    frequencies_hz: tuple[float, ...]
    relative_permittivity_real: tuple[float, ...]
    loss_conductivity_values: tuple[float, ...]
    relative_permeability: float = 1.0
    conductivity: float = 0.0
    thermal_conductivity: float | None = None
    density: float | None = None
    heat_capacity: float | None = None
    thermal_conductivity_tensor: np.ndarray | None = None

    def __post_init__(
        self,
    ):
        frequency = np.asarray(
            self.frequencies_hz,
            dtype=float,
        )
        epsilon = np.asarray(
            self.relative_permittivity_real,
            dtype=float,
        )
        loss = np.asarray(
            self.loss_conductivity_values,
            dtype=float,
        )
        if (
            frequency.ndim
            != 1
            or len(
                frequency
            ) < 2
            or epsilon.shape
            != frequency.shape
            or loss.shape
            != frequency.shape
            or np.any(
                ~np.isfinite(
                    frequency
                )
            )
            or np.any(
                frequency <= 0.0
            )
            or np.any(
                np.diff(
                    frequency
                ) <= 0.0
            )
            or np.any(
                ~np.isfinite(
                    epsilon
                )
            )
            or np.any(
                epsilon <= 0.0
            )
            or np.any(
                ~np.isfinite(
                    loss
                )
            )
            or np.any(
                loss < 0.0
            )
            or not np.isfinite(
                self.relative_permeability
            )
            or self.relative_permeability <= 0.0
            or not np.isfinite(
                self.conductivity
            )
            or self.conductivity < 0.0
        ):
            raise ValueError(
                "invalid passive tabulated material"
            )
        tensor = _validated_optional_thermal_tensor(
            self.thermal_conductivity,
            self.thermal_conductivity_tensor,
            self.density,
            self.heat_capacity,
        )
        if tensor is not None:
            object.__setattr__(
                self,
                "thermal_conductivity_tensor",
                tensor,
            )
        object.__setattr__(
            self,
            "frequencies_hz",
            tuple(
                float(
                    value
                )
                for value in frequency
            ),
        )
        object.__setattr__(
            self,
            "relative_permittivity_real",
            tuple(
                float(
                    value
                )
                for value in epsilon
            ),
        )
        object.__setattr__(
            self,
            "loss_conductivity_values",
            tuple(
                float(
                    value
                )
                for value in loss
            ),
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
        return float(
            self.relative_permittivity_real[
                0
            ]
        )

    def _interpolate(
        self,
        values,
        frequency_hz: float,
    ) -> float:
        frequency = float(
            frequency_hz
        )
        if (
            not np.isfinite(
                frequency
            )
            or frequency <= 0.0
        ):
            raise ValueError(
                "tabulated material requires a positive finite frequency"
            )
        table = np.asarray(
            self.frequencies_hz,
            dtype=float,
        )
        tolerance = (
            1e-12
            * max(
                float(
                    table[
                        -1
                    ]
                ),
                1.0,
            )
        )
        if (
            frequency
            < table[
                0
            ] - tolerance
            or frequency
            > table[
                -1
            ] + tolerance
        ):
            raise ValueError(
                "frequency is outside the tabulated material domain"
            )
        frequency = min(
            max(
                frequency,
                float(
                    table[
                        0
                    ]
                ),
            ),
            float(
                table[
                    -1
                ]
            ),
        )
        return float(
            np.interp(
                np.log(
                    frequency
                ),
                np.log(
                    table
                ),
                np.asarray(
                    values,
                    dtype=float,
                ),
            )
        )

    def relative_permittivity_at(
        self,
        frequency_hz: float,
    ) -> complex:
        frequency = float(
            frequency_hz
        )
        epsilon_real = self._interpolate(
            self.relative_permittivity_real,
            frequency,
        )
        loss = self.loss_conductivity(
            frequency
        )
        omega = (
            2.0
            * np.pi
            * frequency
        )
        return complex(
            epsilon_real
            - 1j
            * loss
            / (
                omega
                * EPS0
            )
        )

    def complex_permittivity(
        self,
        frequency_hz: float,
    ) -> complex:
        return (
            EPS0
            * self.relative_permittivity_at(
                frequency_hz
            )
        )

    def loss_conductivity(
        self,
        frequency_hz: float,
    ) -> float:
        frequency = float(
            frequency_hz
        )
        if (
            not np.isfinite(
                frequency
            )
            or frequency < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        if frequency == 0.0:
            return float(
                self.conductivity
            )
        dispersive = self._interpolate(
            self.loss_conductivity_values,
            frequency,
        )
        return float(
            self.conductivity + dispersive
        )


@dataclass(frozen=True)
class TensorElectricMaterial:
    """Passive tensor electric material for package or homogeneous regions.

    Electromagnetic anisotropy is represented by a symmetric positive-definite
    relative-permittivity tensor and a symmetric positive-semidefinite static
    conductivity tensor. Tensor axes follow the containing region frame: a
    package uses its local pose while a Scene background uses world axes.
    Magnetic permeability remains isotropic in this stage.
    """

    relative_permittivity_tensor: np.ndarray
    conductivity_tensor: np.ndarray | None = None
    relative_permeability: float = 1.0
    thermal_conductivity: float | None = None
    density: float | None = None
    heat_capacity: float | None = None
    thermal_conductivity_tensor: np.ndarray | None = None

    def __post_init__(
        self,
    ):
        epsilon = np.asarray(
            self.relative_permittivity_tensor,
            dtype=float,
        )
        if (
            epsilon.shape != (
                3,
                3,
            )
            or np.any(
                ~np.isfinite(
                    epsilon
                )
            )
            or not np.allclose(
                epsilon,
                epsilon.T,
                rtol=1e-12,
                atol=1e-14,
            )
            or np.min(
                np.linalg.eigvalsh(
                    epsilon
                )
            ) <= 0.0
        ):
            raise ValueError(
                "relative_permittivity_tensor must be a finite symmetric "
                "positive-definite 3x3 matrix"
            )
        sigma = (
            np.zeros(
                (
                    3,
                    3,
                ),
                dtype=float,
            )
            if self.conductivity_tensor is None
            else np.asarray(
                self.conductivity_tensor,
                dtype=float,
            )
        )
        if (
            sigma.shape != (
                3,
                3,
            )
            or np.any(
                ~np.isfinite(
                    sigma
                )
            )
            or not np.allclose(
                sigma,
                sigma.T,
                rtol=1e-12,
                atol=1e-14,
            )
            or np.min(
                np.linalg.eigvalsh(
                    sigma
                )
            ) < -1e-14
            or not np.isfinite(
                self.relative_permeability
            )
            or self.relative_permeability <= 0.0
        ):
            raise ValueError(
                "conductivity_tensor must be finite symmetric positive "
                "semidefinite and relative_permeability must be positive"
            )
        commutator = epsilon @ sigma - sigma @ epsilon
        if (
            np.linalg.norm(
                commutator
            )
            > 1e-10
            * max(
                np.linalg.norm(
                    epsilon
                )
                * np.linalg.norm(
                    sigma
                ),
                1.0,
            )
        ):
            raise ValueError(
                "relative_permittivity_tensor and conductivity_tensor must "
                "share principal axes in the current tensor-electric backend"
            )
        tensor_thermal = _validated_optional_thermal_tensor(
            self.thermal_conductivity,
            self.thermal_conductivity_tensor,
            self.density,
            self.heat_capacity,
        )
        object.__setattr__(
            self,
            "relative_permittivity_tensor",
            epsilon.copy(),
        )
        object.__setattr__(
            self,
            "conductivity_tensor",
            sigma.copy(),
        )
        if tensor_thermal is not None:
            object.__setattr__(
                self,
                "thermal_conductivity_tensor",
                tensor_thermal,
            )

    @property
    def permeability(
        self,
    ) -> float:
        return MU0 * float(
            self.relative_permeability
        )

    @property
    def conductivity(
        self,
    ) -> float:
        return float(
            np.max(
                np.linalg.eigvalsh(
                    self.conductivity_tensor
                )
            )
        )

    @property
    def relative_permittivity(
        self,
    ) -> float:
        return float(
            np.trace(
                self.relative_permittivity_tensor
            ) / 3.0
        )

    def electric_coefficient_tensor(
        self,
        frequency_hz: float,
        *,
        conduction_dc: bool = False,
    ) -> np.ndarray:
        frequency = float(
            frequency_hz
        )
        if (
            not np.isfinite(
                frequency
            )
            or frequency < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        if conduction_dc:
            if frequency != 0.0:
                raise ValueError(
                    "conduction_dc coefficient is defined only at exact DC"
                )
            return np.asarray(
                self.conductivity_tensor,
                dtype=complex,
            )
        if frequency == 0.0:
            if self.conductivity > 0.0:
                raise ValueError(
                    "conductive tensor electric material requires the exact "
                    "DC conduction formulation"
                )
            return (
                EPS0
                * np.asarray(
                    self.relative_permittivity_tensor,
                    dtype=complex,
                )
            )
        omega = 2.0 * np.pi * frequency
        return (
            EPS0
            * np.asarray(
                self.relative_permittivity_tensor,
                dtype=complex,
            )
            - 1j
            * np.asarray(
                self.conductivity_tensor,
                dtype=complex,
            ) / omega
        )

    def loss_conductivity_tensor(
        self,
        frequency_hz: float,
    ) -> np.ndarray:
        frequency = float(
            frequency_hz
        )
        if (
            not np.isfinite(
                frequency
            )
            or frequency < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        return self.conductivity_tensor.copy()

    def loss_conductivity(
        self,
        frequency_hz: float,
    ) -> float:
        self.loss_conductivity_tensor(
            frequency_hz
        )
        return self.conductivity

    def relative_permittivity_at(
        self,
        frequency_hz: float,
    ) -> complex:
        raise NotImplementedError(
            "TensorElectricMaterial has no scalar relative permittivity; "
            "use the tensor-aware REFERENCE electric transmission backend"
        )

    def complex_permittivity(
        self,
        frequency_hz: float,
    ) -> complex:
        raise NotImplementedError(
            "TensorElectricMaterial has no scalar complex permittivity; "
            "use the tensor-aware REFERENCE electric transmission backend"
        )


@dataclass(frozen=True)
class PackageObject:
    geometry: SuperquadricPackageGeometry
    material: PassiveIsotropicMaterial
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
            PassiveIsotropicMaterial,
        ):
            raise TypeError(
                "package material must implement the passive electric "
                "frequency-response interface"
            )


@dataclass(frozen=True)
class CoilObject:
    geometry: SuperellipseSpiral
    material: ConductorMaterial
    name: str = "coil"


@dataclass(frozen=True)
class Scene:
    coils: Tuple[CoilObject, ...]
    medium: PassiveIsotropicMaterial = HomogeneousMedium()
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
            PassiveIsotropicMaterial,
        ):
            raise TypeError(
                "scene medium must implement the passive electric "
                "frequency-response interface"
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
        """Validate only the historical conductor-only MVP subset.

        This legacy guard is intentionally narrower than the current vNext
        system. Package-aware SIE, lossy/dispersive media, magnetic contrast,
        exact-DC conduction, and nested material regions are available through
        the package-aware reference/FAST APIs.
        """
        if not np.isclose(
            self.medium.loss_conductivity(
                0.0
            ),
            0.0,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "the legacy conductor-only MVP subset requires a lossless "
                "background; use the package-aware vNext system for lossy or "
                "conductive media"
            )
        if self.packages:
            raise ValueError(
                "the legacy conductor-only MVP subset excludes packages; "
                "use the package-aware vNext system for package coupling"
            )
