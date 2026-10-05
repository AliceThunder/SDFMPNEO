from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .em import MQSConfig
from .graded_material import compile_graded_superquadric_regions
from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
    DielectricCoupledReferenceArtifact,
)
from .scene import Scene
from .thermal_field import HomogeneousThermalMedium
from .thermal_interface import PiecewiseThermalInterfaceArtifact


def _relative_change(
    refined,
    previous,
) -> float:
    refined = np.asarray(
        refined
    )
    previous = np.asarray(
        previous
    )
    return float(
        np.linalg.norm(
            refined
            - previous
        )
        / max(
            np.linalg.norm(
                refined
            ),
            1e-30,
        )
    )


@dataclass(frozen=True)
class GradedMaterialConvergenceStep:
    shell_count: int
    impedance: np.ndarray
    dissipation_channels: np.ndarray
    impedance_relative_change: float | None
    channel_relative_change: float | None
    maximum_relative_change: float | None
    surface_residual: float
    magnetic_surface_residual: float


@dataclass(frozen=True)
class GradedMaterialConvergenceReport:
    steps: tuple[
        GradedMaterialConvergenceStep,
        ...,
    ]
    tolerance: float
    converged: bool
    final_packages: tuple
    final_result: object

    @property
    def maximum_relative_change(
        self,
    ) -> float:
        value = self.steps[
            -1
        ].maximum_relative_change
        return (
            float(
                "inf"
            )
            if value is None
            else float(
                value
            )
        )


def graded_material_convergence(
    base_scene: Scene,
    frequency_hz: float,
    outer_geometry,
    profile,
    *,
    shell_counts=(
        4,
        8,
        16,
    ),
    config: MQSConfig | None = None,
    tolerance: float = 2e-3,
    enclosed_coils=None,
    minimum_inner_scale: float | None = None,
    clearance_fraction: float = 0.03,
    name_prefix: str = "graded",
    surface_vertical_order: int = 16,
    surface_azimuthal_order: int = 32,
    magnetic_volume_axial_order: int = 8,
    magnetic_volume_radial_order: int = 6,
    magnetic_volume_azimuthal_order: int = 24,
    maximum_raw_reciprocity_defect: float = 0.15,
    maximum_raw_magnetic_reciprocity_defect: float = 0.08,
) -> GradedMaterialConvergenceReport:
    """Refine only the radial material-profile discretization.

    base_scene may already contain unrelated disjoint or enclosing material
    regions. The compiled graded hierarchy is appended to those regions for
    every shell count. Conductor/SIE quadrature orders remain fixed so the
    reported change isolates graded-material discretization error.
    """
    if not isinstance(
        base_scene,
        Scene,
    ):
        raise TypeError(
            "base_scene must be Scene"
        )
    counts = tuple(
        int(
            value
        )
        for value in shell_counts
    )
    if (
        len(
            counts
        )
        < 2
        or any(
            value < 2
            for value in counts
        )
        or any(
            right <= left
            for left, right in zip(
                counts[
                    :-1
                ],
                counts[
                    1:
                ],
            )
        )
    ):
        raise ValueError(
            "shell_counts must contain at least two strictly increasing "
            "integers >= 2"
        )
    if (
        not np.isfinite(
            tolerance
        )
        or tolerance <= 0.0
    ):
        raise ValueError(
            "tolerance must be positive and finite"
        )

    resolved_config = (
        config
        or MQSConfig()
    )
    resolved_coils = (
        base_scene.coils
        if enclosed_coils is None
        else tuple(
            enclosed_coils
        )
    )

    steps = []
    previous_impedance = None
    previous_channels = None
    final_packages = ()
    final_result = None

    for shell_count in counts:
        graded_packages = (
            compile_graded_superquadric_regions(
                outer_geometry,
                profile,
                shell_count=(
                    shell_count
                ),
                name_prefix=(
                    name_prefix
                ),
                enclosed_coils=(
                    resolved_coils
                ),
                minimum_inner_scale=(
                    minimum_inner_scale
                ),
                clearance_fraction=(
                    clearance_fraction
                ),
            )
        )
        scene = Scene(
            base_scene.coils,
            base_scene.medium,
            tuple(
                base_scene.packages
            )
            + tuple(
                graded_packages
            ),
        )
        result = (
            DielectricCoupledMixedTeacher(
                scene,
                frequency_hz,
                resolved_config,
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
                maximum_raw_reciprocity_defect=(
                    maximum_raw_reciprocity_defect
                ),
                maximum_raw_magnetic_reciprocity_defect=(
                    maximum_raw_magnetic_reciprocity_defect
                ),
            ).solve()
        )
        impedance = np.asarray(
            result.impedance,
            dtype=complex,
        )
        channels = np.asarray(
            result.prediction.dissipation_channels,
            dtype=complex,
        )
        if previous_impedance is None:
            impedance_change = None
            channel_change = None
            maximum_change = None
        else:
            impedance_change = _relative_change(
                impedance,
                previous_impedance,
            )
            channel_change = _relative_change(
                channels,
                previous_channels,
            )
            maximum_change = max(
                impedance_change,
                channel_change,
            )
        steps.append(
            GradedMaterialConvergenceStep(
                shell_count=(
                    shell_count
                ),
                impedance=impedance,
                dissipation_channels=channels,
                impedance_relative_change=(
                    impedance_change
                ),
                channel_relative_change=(
                    channel_change
                ),
                maximum_relative_change=(
                    maximum_change
                ),
                surface_residual=float(
                    result.surface_residual
                ),
                magnetic_surface_residual=float(
                    result.magnetic_surface_residual
                ),
            )
        )
        previous_impedance = impedance
        previous_channels = channels
        final_packages = tuple(
            graded_packages
        )
        final_result = result

    last_change = steps[
        -1
    ].maximum_relative_change
    return GradedMaterialConvergenceReport(
        steps=tuple(
            steps
        ),
        tolerance=float(
            tolerance
        ),
        converged=bool(
            last_change is not None
            and last_change <= tolerance
        ),
        final_packages=final_packages,
        final_result=final_result,
    )


@dataclass(frozen=True)
class GradedElectrothermalConvergenceStep:
    shell_count: int
    impedance: np.ndarray
    dissipation_channels: np.ndarray
    temperature_rise: np.ndarray
    impedance_relative_change: float | None
    channel_relative_change: float | None
    temperature_relative_change: float | None
    maximum_relative_change: float | None


@dataclass(frozen=True)
class GradedElectrothermalConvergenceReport:
    steps: tuple[
        GradedElectrothermalConvergenceStep,
        ...,
    ]
    tolerance: float
    converged: bool
    final_packages: tuple
    final_port_result: object
    final_thermal_field: object

    @property
    def maximum_relative_change(
        self,
    ) -> float:
        value = self.steps[
            -1
        ].maximum_relative_change
        return (
            float(
                "inf"
            )
            if value is None
            else float(
                value
            )
        )


def graded_electrothermal_convergence(
    base_scene: Scene,
    frequency_hz: float,
    outer_geometry,
    profile,
    background_thermal_medium: HomogeneousThermalMedium,
    probe_points,
    observation_times,
    currents,
    *,
    shell_counts=(
        4,
        8,
        16,
    ),
    config: MQSConfig | None = None,
    tolerance: float = 2e-3,
    enclosed_coils=None,
    minimum_inner_scale: float | None = None,
    clearance_fraction: float = 0.03,
    name_prefix: str = "graded",
    surface_vertical_order: int = 16,
    surface_azimuthal_order: int = 32,
    magnetic_volume_axial_order: int = 8,
    magnetic_volume_radial_order: int = 6,
    magnetic_volume_azimuthal_order: int = 24,
    maximum_raw_magnetic_reciprocity_defect: float = 0.08,
    thermal_longitudinal_segments: int = 24,
    thermal_radial_order: int = 4,
    thermal_angular_order: int = 24,
    thermal_package_axial_order: int | None = None,
    thermal_package_radial_order: int | None = None,
    thermal_package_azimuthal_order: int | None = None,
    thermal_background_radial_order: int | None = None,
    thermal_background_angular_order: int | None = None,
    thermal_interface_vertical_order: int = 6,
    thermal_interface_azimuthal_order: int = 16,
    thermal_stehfest_order: int = 10,
    thermal_interface_residual_tolerance: float = 5e-5,
) -> GradedElectrothermalConvergenceReport:
    """Refine a radial material profile against port and transient thermal truth."""
    if not isinstance(
        base_scene,
        Scene,
    ):
        raise TypeError(
            "base_scene must be Scene"
        )
    if not isinstance(
        background_thermal_medium,
        HomogeneousThermalMedium,
    ):
        raise TypeError(
            "background_thermal_medium must be HomogeneousThermalMedium"
        )
    counts = tuple(
        int(
            value
        )
        for value in shell_counts
    )
    if (
        len(
            counts
        ) < 2
        or any(
            value < 2
            for value in counts
        )
        or any(
            right <= left
            for left, right in zip(
                counts[
                    :-1
                ],
                counts[
                    1:
                ],
            )
        )
    ):
        raise ValueError(
            "shell_counts must contain at least two strictly increasing "
            "integers >= 2"
        )
    if (
        not np.isfinite(
            tolerance
        )
        or tolerance <= 0.0
    ):
        raise ValueError(
            "tolerance must be positive and finite"
        )

    points = np.asarray(
        probe_points,
        dtype=float,
    )
    points = np.atleast_2d(
        points
    )
    if (
        points.ndim != 2
        or points.shape[
            1
        ] != 3
        or np.any(
            ~np.isfinite(
                points
            )
        )
    ):
        raise ValueError(
            "probe_points must have shape (3,) or (n,3)"
        )
    times = np.asarray(
        observation_times,
        dtype=float,
    )
    if (
        times.ndim != 1
        or len(
            times
        ) == 0
        or np.any(
            ~np.isfinite(
                times
            )
        )
        or np.any(
            times < 0.0
        )
        or not np.any(
            times > 0.0
        )
    ):
        raise ValueError(
            "observation_times must be finite nonnegative values with at "
            "least one positive time"
        )
    currents = np.asarray(
        currents,
        dtype=complex,
    )
    if currents.shape != (
        len(
            base_scene.coils
        ),
    ):
        raise ValueError(
            "currents must have one complex entry per scene coil"
        )
    if np.linalg.norm(
        currents
    ) <= 0.0:
        raise ValueError(
            "currents must not be identically zero"
        )

    resolved_config = (
        config
        or MQSConfig()
    )
    resolved_coils = (
        base_scene.coils
        if enclosed_coils is None
        else tuple(
            enclosed_coils
        )
    )
    spatial_artifact = DielectricCoupledReferenceArtifact(
        config=(
            resolved_config
        ),
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
        maximum_raw_magnetic_reciprocity_defect=(
            maximum_raw_magnetic_reciprocity_defect
        ),
    )
    thermal_artifact = PiecewiseThermalInterfaceArtifact(
        spatial_artifact,
        background_thermal_medium,
        longitudinal_segments=(
            thermal_longitudinal_segments
        ),
        radial_order=(
            thermal_radial_order
        ),
        angular_order=(
            thermal_angular_order
        ),
        package_axial_order=(
            thermal_package_axial_order
        ),
        package_radial_order=(
            thermal_package_radial_order
        ),
        package_azimuthal_order=(
            thermal_package_azimuthal_order
        ),
        background_radial_order=(
            thermal_background_radial_order
        ),
        background_angular_order=(
            thermal_background_angular_order
        ),
        interface_vertical_order=(
            thermal_interface_vertical_order
        ),
        interface_azimuthal_order=(
            thermal_interface_azimuthal_order
        ),
        stehfest_order=(
            thermal_stehfest_order
        ),
        interface_residual_tolerance=(
            thermal_interface_residual_tolerance
        ),
    )

    steps = []
    previous_impedance = None
    previous_channels = None
    previous_temperature = None
    final_packages = ()
    final_port_result = None
    final_thermal_field = None

    for shell_count in counts:
        graded_packages = compile_graded_superquadric_regions(
            outer_geometry,
            profile,
            shell_count=(
                shell_count
            ),
            name_prefix=(
                name_prefix
            ),
            enclosed_coils=(
                resolved_coils
            ),
            minimum_inner_scale=(
                minimum_inner_scale
            ),
            clearance_fraction=(
                clearance_fraction
            ),
        )
        scene = Scene(
            base_scene.coils,
            base_scene.medium,
            tuple(
                base_scene.packages
            )
            + tuple(
                graded_packages
            ),
        )
        port_result = spatial_artifact.solve(
            scene,
            frequency_hz,
        )
        thermal_field = thermal_artifact.prepare(
            scene,
            frequency_hz,
        )
        temperature = np.asarray(
            [
                np.asarray(
                    thermal_field.temperature_step(
                        points,
                        float(
                            time
                        ),
                        currents,
                    ),
                    dtype=float,
                ).reshape(
                    -1
                )
                for time in times
            ],
            dtype=float,
        )
        temperature_rise = (
            temperature
            - float(
                background_thermal_medium.ambient_temperature
            )
        )
        impedance = np.asarray(
            port_result.impedance,
            dtype=complex,
        )
        channels = np.asarray(
            port_result.prediction.dissipation_channels,
            dtype=complex,
        )

        if previous_impedance is None:
            impedance_change = None
            channel_change = None
            temperature_change = None
            maximum_change = None
        else:
            impedance_change = _relative_change(
                impedance,
                previous_impedance,
            )
            channel_change = _relative_change(
                channels,
                previous_channels,
            )
            temperature_change = _relative_change(
                temperature_rise,
                previous_temperature,
            )
            maximum_change = max(
                impedance_change,
                channel_change,
                temperature_change,
            )

        steps.append(
            GradedElectrothermalConvergenceStep(
                shell_count=(
                    shell_count
                ),
                impedance=impedance,
                dissipation_channels=channels,
                temperature_rise=temperature_rise,
                impedance_relative_change=(
                    impedance_change
                ),
                channel_relative_change=(
                    channel_change
                ),
                temperature_relative_change=(
                    temperature_change
                ),
                maximum_relative_change=(
                    maximum_change
                ),
            )
        )
        previous_impedance = impedance
        previous_channels = channels
        previous_temperature = temperature_rise
        final_packages = tuple(
            graded_packages
        )
        final_port_result = port_result
        final_thermal_field = thermal_field

    last_change = steps[
        -1
    ].maximum_relative_change
    return GradedElectrothermalConvergenceReport(
        steps=tuple(
            steps
        ),
        tolerance=float(
            tolerance
        ),
        converged=bool(
            last_change is not None
            and last_change <= tolerance
        ),
        final_packages=final_packages,
        final_port_result=final_port_result,
        final_thermal_field=final_thermal_field,
    )
