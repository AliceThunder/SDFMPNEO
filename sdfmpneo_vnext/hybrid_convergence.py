from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .convergence import (
    _cross_section_refinement,
    _longitudinal_refinement,
    _quadrature_refinement,
    _relative_observable_change,
)
from .em import MQSConfig
from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
)


@dataclass(frozen=True)
class HybridReferenceConvergenceDirection:
    name: str
    config: MQSConfig
    surface_vertical_order: int
    surface_azimuthal_order: int
    impedance_relative_change: float
    channel_relative_change: float
    surface_residual: float
    maximum_relative_change: float
    magnetic_surface_residual: float = 0.0
    magnetic_volume_axial_order: int | None = None
    magnetic_volume_radial_order: int | None = None
    magnetic_volume_azimuthal_order: int | None = None


@dataclass(frozen=True)
class HybridReferenceConvergenceReport:
    base_config: MQSConfig
    surface_vertical_order: int
    surface_azimuthal_order: int
    directions: tuple[
        HybridReferenceConvergenceDirection,
        ...,
    ]
    tolerance: float
    surface_residual_tolerance: float
    maximum_relative_change: float
    maximum_surface_residual: float
    converged: bool
    magnetic_surface_residual_tolerance: float = 1e-9
    maximum_magnetic_surface_residual: float = 0.0
    magnetic_volume_axial_order: int | None = None
    magnetic_volume_radial_order: int | None = None
    magnetic_volume_azimuthal_order: int | None = None


def _surface_refinement(
    vertical_order: int,
    azimuthal_order: int,
):
    vertical = max(
        vertical_order + 2,
        int(
            np.ceil(
                1.5
                * vertical_order
            )
        ),
    )
    if vertical % 2:
        vertical += 1
    azimuthal = max(
        azimuthal_order + 8,
        int(
            np.ceil(
                1.5
                * azimuthal_order
            )
        ),
    )
    return (
        vertical,
        azimuthal,
    )


def _magnetic_volume_refinement(
    axial_order: int,
    radial_order: int,
    azimuthal_order: int,
):
    axial = max(
        axial_order + 2,
        int(
            np.ceil(
                1.5
                * axial_order
            )
        ),
    )
    radial = max(
        radial_order + 2,
        int(
            np.ceil(
                1.5
                * radial_order
            )
        ),
    )
    azimuthal = max(
        azimuthal_order + 8,
        int(
            np.ceil(
                1.5
                * azimuthal_order
            )
        ),
    )
    return (
        axial,
        radial,
        azimuthal,
    )


def _hybrid_observables(
    scene,
    frequency_hz: float,
    config: MQSConfig,
    surface_vertical_order: int,
    surface_azimuthal_order: int,
    magnetic_volume_axial_order: int,
    magnetic_volume_radial_order: int,
    magnetic_volume_azimuthal_order: int,
):
    result = (
        DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            config,
            surface_vertical_order=(
                surface_vertical_order
            ),
            surface_azimuthal_order=(
                surface_azimuthal_order
            ),
            magnetic_volume_axial_order=(
                magnetic_volume_axial_order
            ),
            magnetic_volume_radial_order=(
                magnetic_volume_radial_order
            ),
            magnetic_volume_azimuthal_order=(
                magnetic_volume_azimuthal_order
            ),
        ).solve()
    )
    return (
        result.impedance,
        result.prediction.dissipation_channels,
        float(
            result.surface_residual
        ),
        float(
            result.magnetic_surface_residual
        ),
    )


def hybrid_reference_convergence(
    scene,
    frequency_hz: float,
    base_config: MQSConfig,
    *,
    surface_vertical_order: int = 16,
    surface_azimuthal_order: int = 32,
    magnetic_volume_axial_order: int = 8,
    magnetic_volume_radial_order: int = 6,
    magnetic_volume_azimuthal_order: int = 24,
    tolerance: float = 2e-3,
    surface_residual_tolerance: float = 1e-9,
    magnetic_surface_residual_tolerance: float = 1e-9,
) -> HybridReferenceConvergenceReport:
    """Independent conductor, interface-surface, and magnetic-volume refinement."""
    if not scene.packages:
        raise ValueError(
            "hybrid reference convergence requires at least one package"
        )
    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive"
        )
    if surface_residual_tolerance <= 0.0:
        raise ValueError(
            "surface_residual_tolerance must be positive"
        )
    if magnetic_surface_residual_tolerance <= 0.0:
        raise ValueError(
            "magnetic_surface_residual_tolerance must be positive"
        )
    if (
        magnetic_volume_axial_order < 2
        or magnetic_volume_radial_order < 2
        or magnetic_volume_azimuthal_order < 8
    ):
        raise ValueError(
            "invalid base magnetic volume quadrature order"
        )
    if (
        surface_vertical_order < 4
        or surface_vertical_order % 2
        or surface_azimuthal_order < 8
    ):
        raise ValueError(
            "invalid base dielectric surface quadrature order"
        )

    (
        base_impedance,
        base_channels,
        base_surface_residual,
        base_magnetic_surface_residual,
    ) = _hybrid_observables(
        scene,
        frequency_hz,
        base_config,
        surface_vertical_order,
        surface_azimuthal_order,
        magnetic_volume_axial_order,
        magnetic_volume_radial_order,
        magnetic_volume_azimuthal_order,
    )

    surface_refined = (
        _surface_refinement(
            surface_vertical_order,
            surface_azimuthal_order,
        )
    )
    magnetic_volume_refined = (
        _magnetic_volume_refinement(
            magnetic_volume_axial_order,
            magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order,
        )
    )
    refinements = (
        (
            "longitudinal",
            _longitudinal_refinement(
                base_config
            ),
            surface_vertical_order,
            surface_azimuthal_order,
            magnetic_volume_axial_order,
            magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order,
        ),
        (
            "cross_section",
            _cross_section_refinement(
                base_config
            ),
            surface_vertical_order,
            surface_azimuthal_order,
            magnetic_volume_axial_order,
            magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order,
        ),
        (
            "quadrature",
            _quadrature_refinement(
                base_config
            ),
            surface_vertical_order,
            surface_azimuthal_order,
            magnetic_volume_axial_order,
            magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order,
        ),
        (
            "dielectric_surface",
            base_config,
            surface_refined[
                0
            ],
            surface_refined[
                1
            ],
            magnetic_volume_axial_order,
            magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order,
        ),
        (
            "magnetic_volume",
            base_config,
            surface_vertical_order,
            surface_azimuthal_order,
            magnetic_volume_refined[
                0
            ],
            magnetic_volume_refined[
                1
            ],
            magnetic_volume_refined[
                2
            ],
        ),
    )

    directions = []
    for (
        name,
        config,
        vertical,
        azimuthal,
        magnetic_axial,
        magnetic_radial,
        magnetic_azimuthal,
    ) in refinements:
        (
            impedance,
            channels,
            surface_residual,
            magnetic_surface_residual,
        ) = _hybrid_observables(
            scene,
            frequency_hz,
            config,
            vertical,
            azimuthal,
            magnetic_axial,
            magnetic_radial,
            magnetic_azimuthal,
        )
        impedance_error = (
            _relative_observable_change(
                impedance,
                base_impedance,
            )
        )
        channel_error = (
            _relative_observable_change(
                channels,
                base_channels,
            )
        )
        maximum = float(
            max(
                impedance_error,
                channel_error,
            )
        )
        directions.append(
            HybridReferenceConvergenceDirection(
                name=name,
                config=config,
                surface_vertical_order=int(
                    vertical
                ),
                surface_azimuthal_order=int(
                    azimuthal
                ),
                impedance_relative_change=float(
                    impedance_error
                ),
                channel_relative_change=float(
                    channel_error
                ),
                surface_residual=float(
                    surface_residual
                ),
                maximum_relative_change=(
                    maximum
                ),
                magnetic_surface_residual=float(
                    magnetic_surface_residual
                ),
                magnetic_volume_axial_order=int(
                    magnetic_axial
                ),
                magnetic_volume_radial_order=int(
                    magnetic_radial
                ),
                magnetic_volume_azimuthal_order=int(
                    magnetic_azimuthal
                ),
            )
        )

    directions = tuple(
        directions
    )
    maximum_relative_change = float(
        max(
            direction.maximum_relative_change
            for direction
            in directions
        )
    )
    maximum_surface_residual = float(
        max(
            [
                base_surface_residual
            ]
            + [
                direction.surface_residual
                for direction
                in directions
            ]
        )
    )
    maximum_magnetic_surface_residual = float(
        max(
            [
                base_magnetic_surface_residual
            ]
            + [
                direction.magnetic_surface_residual
                for direction
                in directions
            ]
        )
    )
    return HybridReferenceConvergenceReport(
        base_config=base_config,
        surface_vertical_order=int(
            surface_vertical_order
        ),
        surface_azimuthal_order=int(
            surface_azimuthal_order
        ),
        directions=directions,
        tolerance=float(
            tolerance
        ),
        surface_residual_tolerance=float(
            surface_residual_tolerance
        ),
        maximum_relative_change=(
            maximum_relative_change
        ),
        maximum_surface_residual=(
            maximum_surface_residual
        ),
        converged=bool(
            maximum_relative_change
            <= tolerance
            and maximum_surface_residual
            <= surface_residual_tolerance
            and maximum_magnetic_surface_residual
            <= magnetic_surface_residual_tolerance
        ),
        magnetic_surface_residual_tolerance=float(
            magnetic_surface_residual_tolerance
        ),
        maximum_magnetic_surface_residual=(
            maximum_magnetic_surface_residual
        ),
        magnetic_volume_axial_order=int(
            magnetic_volume_axial_order
        ),
        magnetic_volume_radial_order=int(
            magnetic_volume_radial_order
        ),
        magnetic_volume_azimuthal_order=int(
            magnetic_volume_azimuthal_order
        ),
    )
