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


def _hybrid_observables(
    scene,
    frequency_hz: float,
    config: MQSConfig,
    surface_vertical_order: int,
    surface_azimuthal_order: int,
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
        ).solve()
    )
    return (
        result.impedance,
        result.prediction.dissipation_channels,
        float(
            result.surface_residual
        ),
    )


def hybrid_reference_convergence(
    scene,
    frequency_hz: float,
    base_config: MQSConfig,
    *,
    surface_vertical_order: int = 16,
    surface_azimuthal_order: int = 32,
    tolerance: float = 2e-3,
    surface_residual_tolerance: float = 1e-9,
) -> HybridReferenceConvergenceReport:
    """Independent conductor and dielectric-SIE refinement for package scenes."""
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
    ) = _hybrid_observables(
        scene,
        frequency_hz,
        base_config,
        surface_vertical_order,
        surface_azimuthal_order,
    )

    surface_refined = (
        _surface_refinement(
            surface_vertical_order,
            surface_azimuthal_order,
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
        ),
        (
            "cross_section",
            _cross_section_refinement(
                base_config
            ),
            surface_vertical_order,
            surface_azimuthal_order,
        ),
        (
            "quadrature",
            _quadrature_refinement(
                base_config
            ),
            surface_vertical_order,
            surface_azimuthal_order,
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
        ),
    )

    directions = []
    for (
        name,
        config,
        vertical,
        azimuthal,
    ) in refinements:
        (
            impedance,
            channels,
            surface_residual,
        ) = _hybrid_observables(
            scene,
            frequency_hz,
            config,
            vertical,
            azimuthal,
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
        ),
    )
