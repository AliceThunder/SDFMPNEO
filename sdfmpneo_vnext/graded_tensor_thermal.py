from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .scene import IsotropicMaterial


def _as_profile_tuple(
    values,
    *,
    name: str,
    count: int,
    positive: bool,
):
    out = tuple(
        float(
            value
        )
        for value in values
    )
    if len(
        out
    ) != count:
        raise ValueError(
            f"{name} must have the same length as normalized_radius"
        )
    if any(
        not np.isfinite(
            value
        )
        or (
            value <= 0.0
            if positive
            else value < 0.0
        )
        for value in out
    ):
        qualifier = (
            "positive"
            if positive
            else "nonnegative"
        )
        raise ValueError(
            f"{name} must contain finite {qualifier} values"
        )
    return out


def _validate_spd_tensor(
    value,
    *,
    name: str,
) -> np.ndarray:
    tensor = np.asarray(
        value,
        dtype=float,
    )
    if (
        tensor.shape
        != (
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
    ):
        raise ValueError(
            f"{name} must be a finite symmetric 3x3 matrix"
        )
    eigenvalues = np.linalg.eigvalsh(
        tensor
    )
    if np.min(
        eigenvalues
    ) <= 0.0:
        raise ValueError(
            f"{name} must be positive definite"
        )
    return tensor.copy()


def _symmetric_matrix_log(
    tensor,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        tensor
    )
    return (
        eigenvectors
        @ np.diag(
            np.log(
                eigenvalues
            )
        )
        @ eigenvectors.T
    )


def _symmetric_matrix_exp(
    matrix,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5
        * (
            matrix
            + matrix.T
        )
    )
    tensor = (
        eigenvectors
        @ np.diag(
            np.exp(
                eigenvalues
            )
        )
        @ eigenvectors.T
    )
    return 0.5 * (
        tensor
        + tensor.T
    )


@dataclass(frozen=True)
class RadialTensorThermalMaterialProfile:
    """Radial passive EM material with an SPD thermal-conductivity tensor.

    Thermal tensors are expressed in the local frame of the package compiled
    from this profile. Tensor interpolation is log-Euclidean, so every
    interpolated conductivity remains symmetric positive definite even when
    principal values and axes vary between radial knots.
    """

    normalized_radius: tuple[float, ...]
    relative_permittivity: tuple[float, ...]
    relative_permeability: tuple[float, ...]
    conductivity: tuple[float, ...]
    thermal_conductivity_tensors: tuple[np.ndarray, ...]
    density: tuple[float, ...]
    heat_capacity: tuple[float, ...]

    def __post_init__(
        self,
    ):
        radius = tuple(
            float(
                value
            )
            for value in self.normalized_radius
        )
        if (
            len(
                radius
            ) < 2
            or any(
                not np.isfinite(
                    value
                )
                for value in radius
            )
            or abs(
                radius[
                    0
                ]
            ) > 1e-12
            or abs(
                radius[
                    -1
                ]
                - 1.0
            ) > 1e-12
            or any(
                right <= left
                for left, right in zip(
                    radius[
                        :-1
                    ],
                    radius[
                        1:
                    ],
                )
            )
        ):
            raise ValueError(
                "normalized_radius must be strictly increasing from 0 to 1"
            )
        count = len(
            radius
        )
        epsilon = _as_profile_tuple(
            self.relative_permittivity,
            name="relative_permittivity",
            count=count,
            positive=True,
        )
        permeability = _as_profile_tuple(
            self.relative_permeability,
            name="relative_permeability",
            count=count,
            positive=True,
        )
        conductivity = _as_profile_tuple(
            self.conductivity,
            name="conductivity",
            count=count,
            positive=False,
        )
        density = _as_profile_tuple(
            self.density,
            name="density",
            count=count,
            positive=True,
        )
        heat_capacity = _as_profile_tuple(
            self.heat_capacity,
            name="heat_capacity",
            count=count,
            positive=True,
        )
        tensors = tuple(
            _validate_spd_tensor(
                value,
                name=(
                    "thermal_conductivity_tensors"
                    f"[{index}]"
                ),
            )
            for index, value in enumerate(
                self.thermal_conductivity_tensors
            )
        )
        if len(
            tensors
        ) != count:
            raise ValueError(
                "thermal_conductivity_tensors must have the same length as "
                "normalized_radius"
            )
        logarithms = tuple(
            _symmetric_matrix_log(
                tensor
            )
            for tensor in tensors
        )
        object.__setattr__(
            self,
            "normalized_radius",
            radius,
        )
        object.__setattr__(
            self,
            "relative_permittivity",
            epsilon,
        )
        object.__setattr__(
            self,
            "relative_permeability",
            permeability,
        )
        object.__setattr__(
            self,
            "conductivity",
            conductivity,
        )
        object.__setattr__(
            self,
            "thermal_conductivity_tensors",
            tensors,
        )
        object.__setattr__(
            self,
            "density",
            density,
        )
        object.__setattr__(
            self,
            "heat_capacity",
            heat_capacity,
        )
        object.__setattr__(
            self,
            "_thermal_log_tensors",
            logarithms,
        )

    def _interval(
        self,
        normalized_radius: float,
    ):
        radius = float(
            normalized_radius
        )
        if (
            not np.isfinite(
                radius
            )
            or radius < 0.0
            or radius > 1.0
        ):
            raise ValueError(
                "normalized_radius must lie in [0,1]"
            )
        knots = np.asarray(
            self.normalized_radius,
            dtype=float,
        )
        if radius <= knots[
            0
        ]:
            return (
                0,
                0.0,
            )
        if radius >= knots[
            -1
        ]:
            return (
                len(
                    knots
                )
                - 2,
                1.0,
            )
        index = int(
            np.searchsorted(
                knots,
                radius,
                side="right",
            )
            - 1
        )
        fraction = (
            radius
            - knots[
                index
            ]
        ) / (
            knots[
                index
                + 1
            ]
            - knots[
                index
            ]
        )
        return (
            index,
            float(
                fraction
            ),
        )

    def _scalar_at(
        self,
        values,
        index: int,
        fraction: float,
    ) -> float:
        return float(
            (
                1.0
                - fraction
            )
            * values[
                index
            ]
            + fraction
            * values[
                index
                + 1
            ]
        )

    def thermal_conductivity_tensor_at(
        self,
        normalized_radius: float,
    ) -> np.ndarray:
        index, fraction = self._interval(
            normalized_radius
        )
        logarithm = (
            (
                1.0
                - fraction
            )
            * self._thermal_log_tensors[
                index
            ]
            + fraction
            * self._thermal_log_tensors[
                index
                + 1
            ]
        )
        return _symmetric_matrix_exp(
            logarithm
        )

    def material_at(
        self,
        normalized_radius: float,
    ) -> IsotropicMaterial:
        index, fraction = self._interval(
            normalized_radius
        )
        return IsotropicMaterial(
            relative_permittivity=self._scalar_at(
                self.relative_permittivity,
                index,
                fraction,
            ),
            relative_permeability=self._scalar_at(
                self.relative_permeability,
                index,
                fraction,
            ),
            conductivity=self._scalar_at(
                self.conductivity,
                index,
                fraction,
            ),
            thermal_conductivity_tensor=(
                self.thermal_conductivity_tensor_at(
                    normalized_radius
                )
            ),
            density=self._scalar_at(
                self.density,
                index,
                fraction,
            ),
            heat_capacity=self._scalar_at(
                self.heat_capacity,
                index,
                fraction,
            ),
        )

    def __call__(
        self,
        normalized_radius: float,
    ) -> IsotropicMaterial:
        return self.material_at(
            normalized_radius
        )
