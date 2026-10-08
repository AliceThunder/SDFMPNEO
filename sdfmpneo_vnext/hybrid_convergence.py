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
from .hybrid_dielectric import DielectricCoupledMixedTeacher
from .scene import TensorElectricMaterial
from .tensor_spatial_reference import prepare_tensor_spatial_reference_adaptive


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
    energy_volume_axial_order: int | None = None
    energy_volume_radial_order: int | None = None
    energy_volume_azimuthal_order: int | None = None
    energy_background_radial_order: int | None = None
    energy_background_angular_order: int | None = None


@dataclass(frozen=True)
class HybridReferenceConvergenceReport:
    base_config: MQSConfig
    surface_vertical_order: int
    surface_azimuthal_order: int
    directions: tuple[HybridReferenceConvergenceDirection, ...]
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
    energy_volume_axial_order: int | None = None
    energy_volume_radial_order: int | None = None
    energy_volume_azimuthal_order: int | None = None
    energy_background_radial_order: int | None = None
    energy_background_angular_order: int | None = None


def _surface_refinement(vertical_order: int, azimuthal_order: int):
    vertical = max(vertical_order + 2, int(np.ceil(1.5 * vertical_order)))
    if vertical % 2:
        vertical += 1
    azimuthal = max(
        azimuthal_order + 8,
        int(np.ceil(1.5 * azimuthal_order)),
    )
    return vertical, azimuthal


def _volume_refinement(axial_order: int, radial_order: int, azimuthal_order: int):
    return (
        max(axial_order + 2, int(np.ceil(1.5 * axial_order))),
        max(radial_order + 2, int(np.ceil(1.5 * radial_order))),
        max(azimuthal_order + 8, int(np.ceil(1.5 * azimuthal_order))),
    )


def _background_refinement(radial_order: int, angular_order: int):
    return (
        max(radial_order + 2, int(np.ceil(1.5 * radial_order))),
        max(angular_order + 8, int(np.ceil(1.5 * angular_order))),
    )


def _uses_tensor_electric(scene) -> bool:
    return bool(
        isinstance(scene.medium, TensorElectricMaterial)
        or any(
            isinstance(package.material, TensorElectricMaterial)
            for package in scene.packages
        )
    )


def _uses_magnetic_contrast(scene) -> bool:
    background = float(scene.medium.permeability)
    return any(
        not np.isclose(
            float(package.material.permeability),
            background,
            rtol=1e-12,
            atol=1e-18,
        )
        for package in scene.packages
    )


def _uses_package_electric_loss(scene, frequency_hz: float) -> bool:
    return any(
        float(package.material.loss_conductivity(frequency_hz)) > 0.0
        for package in scene.packages
    )


def _uses_background_electric_loss(scene, frequency_hz: float) -> bool:
    return float(scene.medium.loss_conductivity(frequency_hz)) > 0.0


def _hybrid_observables(
    scene,
    frequency_hz: float,
    config: MQSConfig,
    surface_vertical_order: int,
    surface_azimuthal_order: int,
    magnetic_volume_axial_order: int,
    magnetic_volume_radial_order: int,
    magnetic_volume_azimuthal_order: int,
    energy_volume_axial_order: int,
    energy_volume_radial_order: int,
    energy_volume_azimuthal_order: int,
    energy_background_radial_order: int,
    energy_background_angular_order: int,
):
    teacher = DielectricCoupledMixedTeacher(
        scene,
        frequency_hz,
        config,
        surface_vertical_order=surface_vertical_order,
        surface_azimuthal_order=surface_azimuthal_order,
        magnetic_volume_axial_order=magnetic_volume_axial_order,
        magnetic_volume_radial_order=magnetic_volume_radial_order,
        magnetic_volume_azimuthal_order=magnetic_volume_azimuthal_order,
    )
    result = teacher.solve()
    if result.tensor_electric_transmission is not None:
        result = prepare_tensor_spatial_reference_adaptive(
            teacher,
            result,
            volume_axial_order=energy_volume_axial_order,
            volume_radial_order=energy_volume_radial_order,
            volume_azimuthal_order=energy_volume_azimuthal_order,
            background_radial_order=energy_background_radial_order,
            background_angular_order=energy_background_angular_order,
            maximum_raw_closure_error=0.25,
            maximum_quadrature_refinements=0,
        ).prepared.result
    return (
        np.asarray(result.impedance, dtype=complex),
        np.asarray(result.prediction.dissipation_channels, dtype=complex),
        float(result.surface_residual),
        float(result.magnetic_surface_residual),
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
    energy_volume_axial_order: int = 8,
    energy_volume_radial_order: int = 6,
    energy_volume_azimuthal_order: int = 24,
    energy_background_radial_order: int = 12,
    energy_background_angular_order: int = 48,
    tolerance: float = 2e-3,
    surface_residual_tolerance: float = 1e-9,
    magnetic_surface_residual_tolerance: float = 1e-9,
) -> HybridReferenceConvergenceReport:
    """Refine every independent numerical axis affecting reported observables."""
    tensor_electric = _uses_tensor_electric(scene)
    if not scene.packages and not tensor_electric:
        raise ValueError(
            "hybrid reference convergence requires packages or a tensor background"
        )
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if surface_residual_tolerance <= 0.0:
        raise ValueError("surface_residual_tolerance must be positive")
    if magnetic_surface_residual_tolerance <= 0.0:
        raise ValueError(
            "magnetic_surface_residual_tolerance must be positive"
        )
    if (
        magnetic_volume_axial_order < 2
        or magnetic_volume_radial_order < 2
        or magnetic_volume_azimuthal_order < 8
    ):
        raise ValueError("invalid base magnetic volume quadrature order")
    if (
        energy_volume_axial_order < 2
        or energy_volume_radial_order < 2
        or energy_volume_azimuthal_order < 8
        or energy_background_radial_order < 3
        or energy_background_angular_order < 8
    ):
        raise ValueError("invalid base tensor energy quadrature order")
    if (
        surface_vertical_order < 4
        or surface_vertical_order % 2
        or surface_azimuthal_order < 8
    ):
        raise ValueError("invalid base dielectric surface quadrature order")

    base_energy = (
        int(energy_volume_axial_order),
        int(energy_volume_radial_order),
        int(energy_volume_azimuthal_order),
        int(energy_background_radial_order),
        int(energy_background_angular_order),
    )
    base_magnetic = (
        int(magnetic_volume_axial_order),
        int(magnetic_volume_radial_order),
        int(magnetic_volume_azimuthal_order),
    )

    base = _hybrid_observables(
        scene,
        frequency_hz,
        base_config,
        surface_vertical_order,
        surface_azimuthal_order,
        *base_magnetic,
        *base_energy,
    )
    base_impedance, base_channels, base_surface, base_magnetic_surface = base

    refinements = [
        (
            "longitudinal",
            _longitudinal_refinement(base_config),
            surface_vertical_order,
            surface_azimuthal_order,
            base_magnetic,
            base_energy,
        ),
        (
            "cross_section",
            _cross_section_refinement(base_config),
            surface_vertical_order,
            surface_azimuthal_order,
            base_magnetic,
            base_energy,
        ),
        (
            "quadrature",
            _quadrature_refinement(base_config),
            surface_vertical_order,
            surface_azimuthal_order,
            base_magnetic,
            base_energy,
        ),
    ]

    if scene.packages:
        surface_refined = _surface_refinement(
            surface_vertical_order,
            surface_azimuthal_order,
        )
        refinements.append(
            (
                "dielectric_surface",
                base_config,
                surface_refined[0],
                surface_refined[1],
                base_magnetic,
                base_energy,
            )
        )

    if _uses_magnetic_contrast(scene):
        refinements.append(
            (
                "magnetic_volume",
                base_config,
                surface_vertical_order,
                surface_azimuthal_order,
                _volume_refinement(*base_magnetic),
                base_energy,
            )
        )

    if tensor_electric and _uses_package_electric_loss(scene, frequency_hz):
        refined_volume = _volume_refinement(*base_energy[:3])
        refinements.append(
            (
                "electric_package_energy",
                base_config,
                surface_vertical_order,
                surface_azimuthal_order,
                base_magnetic,
                refined_volume + base_energy[3:],
            )
        )

    if tensor_electric and _uses_background_electric_loss(scene, frequency_hz):
        refined_background = _background_refinement(*base_energy[3:])
        refinements.append(
            (
                "electric_background_energy",
                base_config,
                surface_vertical_order,
                surface_azimuthal_order,
                base_magnetic,
                base_energy[:3] + refined_background,
            )
        )

    directions = []
    for name, config, vertical, azimuthal, magnetic, energy in refinements:
        impedance, channels, surface_residual, magnetic_surface_residual = (
            _hybrid_observables(
                scene,
                frequency_hz,
                config,
                vertical,
                azimuthal,
                *magnetic,
                *energy,
            )
        )
        impedance_error = _relative_observable_change(
            impedance,
            base_impedance,
        )
        channel_error = _relative_observable_change(
            channels,
            base_channels,
        )
        maximum = float(max(impedance_error, channel_error))
        directions.append(
            HybridReferenceConvergenceDirection(
                name=name,
                config=config,
                surface_vertical_order=int(vertical),
                surface_azimuthal_order=int(azimuthal),
                impedance_relative_change=float(impedance_error),
                channel_relative_change=float(channel_error),
                surface_residual=float(surface_residual),
                maximum_relative_change=maximum,
                magnetic_surface_residual=float(magnetic_surface_residual),
                magnetic_volume_axial_order=int(magnetic[0]),
                magnetic_volume_radial_order=int(magnetic[1]),
                magnetic_volume_azimuthal_order=int(magnetic[2]),
                energy_volume_axial_order=int(energy[0]),
                energy_volume_radial_order=int(energy[1]),
                energy_volume_azimuthal_order=int(energy[2]),
                energy_background_radial_order=int(energy[3]),
                energy_background_angular_order=int(energy[4]),
            )
        )

    directions = tuple(directions)
    maximum_relative_change = float(
        max(direction.maximum_relative_change for direction in directions)
    )
    maximum_surface_residual = float(
        max(
            [base_surface]
            + [direction.surface_residual for direction in directions]
        )
    )
    maximum_magnetic_surface_residual = float(
        max(
            [base_magnetic_surface]
            + [
                direction.magnetic_surface_residual
                for direction in directions
            ]
        )
    )
    return HybridReferenceConvergenceReport(
        base_config=base_config,
        surface_vertical_order=int(surface_vertical_order),
        surface_azimuthal_order=int(surface_azimuthal_order),
        directions=directions,
        tolerance=float(tolerance),
        surface_residual_tolerance=float(surface_residual_tolerance),
        maximum_relative_change=maximum_relative_change,
        maximum_surface_residual=maximum_surface_residual,
        converged=bool(
            maximum_relative_change <= tolerance
            and maximum_surface_residual <= surface_residual_tolerance
            and maximum_magnetic_surface_residual
            <= magnetic_surface_residual_tolerance
        ),
        magnetic_surface_residual_tolerance=float(
            magnetic_surface_residual_tolerance
        ),
        maximum_magnetic_surface_residual=(
            maximum_magnetic_surface_residual
        ),
        magnetic_volume_axial_order=base_magnetic[0],
        magnetic_volume_radial_order=base_magnetic[1],
        magnetic_volume_azimuthal_order=base_magnetic[2],
        energy_volume_axial_order=base_energy[0],
        energy_volume_radial_order=base_energy[1],
        energy_volume_azimuthal_order=base_energy[2],
        energy_background_radial_order=base_energy[3],
        energy_background_angular_order=base_energy[4],
    )
