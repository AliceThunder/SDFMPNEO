from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np

from .package_geometry import SuperquadricPackageGeometry
from .scene import (
    CoilObject,
    IsotropicMaterial,
    PackageObject,
    PassiveIsotropicMaterial,
)


@dataclass(frozen=True)
class RadialIsotropicMaterialProfile:
    """Passive isotropic material profile over normalized superquadric radius.

    The profile is continuous and piecewise-linear in the normalized homothetic
    radius r in [0, 1].  It is compiled into nested homogeneous shells for the
    existing mesh-free interface solvers.
    """

    normalized_radius: tuple[float, ...]
    relative_permittivity: tuple[float, ...]
    relative_permeability: tuple[float, ...]
    conductivity: tuple[float, ...]
    thermal_conductivity: tuple[float, ...] | None = None
    density: tuple[float, ...] | None = None
    heat_capacity: tuple[float, ...] | None = None

    def __post_init__(
        self,
    ):
        radius = np.asarray(
            self.normalized_radius,
            dtype=float,
        )
        epsilon = np.asarray(
            self.relative_permittivity,
            dtype=float,
        )
        mu = np.asarray(
            self.relative_permeability,
            dtype=float,
        )
        sigma = np.asarray(
            self.conductivity,
            dtype=float,
        )
        if (
            radius.ndim != 1
            or len(
                radius
            ) < 2
            or epsilon.shape != radius.shape
            or mu.shape != radius.shape
            or sigma.shape != radius.shape
            or np.any(
                ~np.isfinite(
                    radius
                )
            )
            or np.any(
                ~np.isfinite(
                    epsilon
                )
            )
            or np.any(
                ~np.isfinite(
                    mu
                )
            )
            or np.any(
                ~np.isfinite(
                    sigma
                )
            )
            or not np.isclose(
                radius[
                    0
                ],
                0.0,
                rtol=0.0,
                atol=1e-14,
            )
            or not np.isclose(
                radius[
                    -1
                ],
                1.0,
                rtol=0.0,
                atol=1e-14,
            )
            or np.any(
                np.diff(
                    radius
                )
                <= 0.0
            )
            or np.any(
                epsilon
                <= 0.0
            )
            or np.any(
                mu
                <= 0.0
            )
            or np.any(
                sigma
                < 0.0
            )
        ):
            raise ValueError(
                "invalid passive radial isotropic material profile"
            )

        thermal_fields = (
            self.thermal_conductivity,
            self.density,
            self.heat_capacity,
        )
        if any(
            value is not None
            for value in thermal_fields
        ):
            if not all(
                value is not None
                for value in thermal_fields
            ):
                raise ValueError(
                    "thermal_conductivity, density, and heat_capacity "
                    "profiles must be supplied together"
                )
            for name, values in (
                (
                    "thermal_conductivity",
                    self.thermal_conductivity,
                ),
                (
                    "density",
                    self.density,
                ),
                (
                    "heat_capacity",
                    self.heat_capacity,
                ),
            ):
                array = np.asarray(
                    values,
                    dtype=float,
                )
                if (
                    array.shape != radius.shape
                    or np.any(
                        ~np.isfinite(
                            array
                        )
                    )
                    or np.any(
                        array
                        <= 0.0
                    )
                ):
                    raise ValueError(
                        f"invalid {name} radial profile"
                    )

        object.__setattr__(
            self,
            "normalized_radius",
            tuple(
                float(
                    value
                )
                for value in radius
            ),
        )
        object.__setattr__(
            self,
            "relative_permittivity",
            tuple(
                float(
                    value
                )
                for value in epsilon
            ),
        )
        object.__setattr__(
            self,
            "relative_permeability",
            tuple(
                float(
                    value
                )
                for value in mu
            ),
        )
        object.__setattr__(
            self,
            "conductivity",
            tuple(
                float(
                    value
                )
                for value in sigma
            ),
        )
        for name in (
            "thermal_conductivity",
            "density",
            "heat_capacity",
        ):
            values = getattr(
                self,
                name,
            )
            if values is not None:
                object.__setattr__(
                    self,
                    name,
                    tuple(
                        float(
                            value
                        )
                        for value in values
                    ),
                )

    def _interpolate(
        self,
        values,
        normalized_radius: float,
    ) -> float:
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
        return float(
            np.interp(
                radius,
                np.asarray(
                    self.normalized_radius,
                    dtype=float,
                ),
                np.asarray(
                    values,
                    dtype=float,
                ),
            )
        )

    def material_at(
        self,
        normalized_radius: float,
    ) -> IsotropicMaterial:
        thermal = (
            None
            if self.thermal_conductivity is None
            else self._interpolate(
                self.thermal_conductivity,
                normalized_radius,
            )
        )
        density = (
            None
            if self.density is None
            else self._interpolate(
                self.density,
                normalized_radius,
            )
        )
        capacity = (
            None
            if self.heat_capacity is None
            else self._interpolate(
                self.heat_capacity,
                normalized_radius,
            )
        )
        return IsotropicMaterial(
            relative_permittivity=(
                self._interpolate(
                    self.relative_permittivity,
                    normalized_radius,
                )
            ),
            relative_permeability=(
                self._interpolate(
                    self.relative_permeability,
                    normalized_radius,
                )
            ),
            conductivity=(
                self._interpolate(
                    self.conductivity,
                    normalized_radius,
                )
            ),
            thermal_conductivity=thermal,
            density=density,
            heat_capacity=capacity,
        )


def _minimum_enclosing_scale(
    geometry: SuperquadricPackageGeometry,
    coils,
    *,
    clearance_fraction: float,
    longitudinal_segments: int,
    section_points: int,
) -> float:
    coils = tuple(
        coils
    )
    if not coils:
        return 0.0
    if (
        not np.isfinite(
            clearance_fraction
        )
        or clearance_fraction < 0.0
    ):
        raise ValueError(
            "clearance_fraction must be finite and nonnegative"
        )
    if (
        longitudinal_segments < 8
        or section_points < 8
    ):
        raise ValueError(
            "conductor enclosure sampling orders are too small"
        )
    if not all(
        isinstance(
            coil,
            CoilObject,
        )
        for coil in coils
    ):
        raise TypeError(
            "enclosed_coils must contain CoilObject instances"
        )

    points = np.concatenate(
        [
            coil.geometry.surface_samples(
                longitudinal_segments=(
                    longitudinal_segments
                ),
                section_points=(
                    section_points
                ),
            )
            for coil in coils
        ],
        axis=0,
    )
    if not np.all(
        geometry.contains(
            points,
            tolerance=1e-11,
        )
    ):
        raise ValueError(
            "outer graded package does not fully enclose enclosed_coils"
        )

    lower = 1e-4
    upper = 1.0
    for _ in range(
        56
    ):
        midpoint = 0.5 * (
            lower
            + upper
        )
        candidate = SuperquadricPackageGeometry(
            np.asarray(
                geometry.half_extents,
                dtype=float,
            )
            * midpoint,
            exponent_xy=(
                geometry.exponent_xy
            ),
            exponent_z=(
                geometry.exponent_z
            ),
            pose=geometry.pose,
        )
        if np.all(
            candidate.contains(
                points,
                tolerance=1e-11,
            )
        ):
            upper = midpoint
        else:
            lower = midpoint
    return float(
        min(
            upper
            * (
                1.0
                + clearance_fraction
            ),
            1.0,
        )
    )


def compile_graded_superquadric_regions(
    geometry: SuperquadricPackageGeometry,
    profile: (
        RadialIsotropicMaterialProfile
        | Callable[
            [float],
            PassiveIsotropicMaterial,
        ]
    ),
    *,
    shell_count: int = 8,
    name_prefix: str = "graded",
    enclosed_coils=(),
    minimum_inner_scale: float | None = None,
    clearance_fraction: float = 0.03,
    longitudinal_segments: int = 96,
    section_points: int = 24,
):
    """Compile a continuous radial profile into nested homogeneous regions.

    Each shell uses the material value at its normalized radial midpoint.
    The profile may be the built-in piecewise-linear radial profile or any
    callable returning a PassiveIsotropicMaterial, including dispersive or
    tabulated responses.
    Increasing shell_count refines the graded-medium approximation while
    preserving the same local surface-integral solver and unbounded world.

    If enclosed_coils are supplied, the innermost material interface is moved
    outward as needed so no shell boundary cuts a finite-section conductor.
    This keeps the compiled profile inside the strict nested/disjoint material
    topology required by the electromagnetic and thermal interface solvers.
    """
    if not isinstance(
        geometry,
        SuperquadricPackageGeometry,
    ):
        raise TypeError(
            "geometry must be SuperquadricPackageGeometry"
        )
    if not (
        isinstance(
            profile,
            RadialIsotropicMaterialProfile,
        )
        or callable(
            profile
        )
    ):
        raise TypeError(
            "profile must be RadialIsotropicMaterialProfile or a callable "
            "returning passive isotropic materials"
        )
    if (
        not isinstance(
            shell_count,
            (int, np.integer),
        )
        or shell_count < 2
    ):
        raise ValueError(
            "shell_count must be an integer >= 2"
        )
    prefix = str(
        name_prefix
    )
    if not prefix:
        raise ValueError(
            "name_prefix must be nonempty"
        )

    required_inner_scale = _minimum_enclosing_scale(
        geometry,
        enclosed_coils,
        clearance_fraction=(
            clearance_fraction
        ),
        longitudinal_segments=(
            longitudinal_segments
        ),
        section_points=(
            section_points
        ),
    )
    if minimum_inner_scale is not None:
        minimum_inner_scale = float(
            minimum_inner_scale
        )
        if (
            not np.isfinite(
                minimum_inner_scale
            )
            or minimum_inner_scale <= 0.0
            or minimum_inner_scale > 1.0
        ):
            raise ValueError(
                "minimum_inner_scale must lie in (0,1]"
            )
        required_inner_scale = max(
            required_inner_scale,
            minimum_inner_scale,
        )

    if required_inner_scale <= 0.0:
        boundaries = (
            np.arange(
                1,
                shell_count
                + 1,
                dtype=float,
            )
            / float(
                shell_count
            )
        )
    else:
        if (
            shell_count > 1
            and required_inner_scale
            >= 1.0
            - 1e-8
        ):
            raise ValueError(
                "there is no room for a strict graded shell outside the "
                "enclosed conductor; enlarge the outer package"
            )
        inner_volume = (
            required_inner_scale
            ** 3
        )
        volume_boundaries = np.linspace(
            inner_volume,
            1.0,
            shell_count,
        )
        boundaries = (
            volume_boundaries
            ** (
                1.0
                / 3.0
            )
        )

    inner_boundaries = np.concatenate(
        (
            np.asarray(
                [
                    0.0
                ]
            ),
            boundaries[
                :-1
            ],
        )
    )
    midpoints = (
        0.5
        * (
            boundaries**3
            + inner_boundaries**3
        )
    ) ** (
        1.0
        / 3.0
    )
    packages = []
    for index, (
        boundary,
        midpoint,
    ) in enumerate(
        zip(
            boundaries,
            midpoints,
        )
    ):
        shell_geometry = (
            SuperquadricPackageGeometry(
                np.asarray(
                    geometry.half_extents,
                    dtype=float,
                )
                * float(
                    boundary
                ),
                exponent_xy=(
                    geometry.exponent_xy
                ),
                exponent_z=(
                    geometry.exponent_z
                ),
                pose=geometry.pose,
            )
        )
        material = (
            profile.material_at(
                float(
                    midpoint
                )
            )
            if isinstance(
                profile,
                RadialIsotropicMaterialProfile,
            )
            else profile(
                float(
                    midpoint
                )
            )
        )
        if not isinstance(
            material,
            PassiveIsotropicMaterial,
        ):
            raise TypeError(
                "graded material callable must return a passive isotropic "
                "material response"
            )
        packages.append(
            PackageObject(
                shell_geometry,
                material,
                f"{prefix}:{index:03d}",
            )
        )
    return tuple(
        packages
    )
