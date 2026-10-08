from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .analytic_baseline import analytic_port_baseline
from .em import MQSConfig
from .hybrid_dielectric import DielectricCoupledMixedTeacher
from .prediction import StructuredPortPrediction
from .scene import Scene
from .tensor_features import (
    EncodedTensorHybridScene,
    encode_tensor_hybrid_scene_invariant,
)
from .tensor_spatial_reference import prepare_tensor_spatial_reference_adaptive


TENSOR_HYBRID_REFERENCE_BACKEND = "tensor_electric_mixed_mfs_energy_v3"


@dataclass(frozen=True)
class TensorHybridTeacherSample:
    """Port-level tensor-electric teacher sample for FAST residual training.

    ``package_volume_*`` and ``background_*`` are the quadrature orders used
    to define the canonical port energy truth. Spatial-training samples keep
    their own, potentially lower, point-cloud quadrature metadata.
    """

    scene: Scene
    frequency_hz: float
    encoded: EncodedTensorHybridScene
    baseline_resistance: np.ndarray
    baseline_reactance: np.ndarray
    target_impedance: np.ndarray
    target_dissipation_channels: np.ndarray
    baseline_segments: int
    surface_vertical_order: int
    surface_azimuthal_order: int
    surface_residual: float
    raw_potential_reciprocity_defect: float
    power_closure_error: float
    magnetic_surface_residual: float = 0.0
    raw_magnetic_reciprocity_defect: float = 0.0
    teacher_config: MQSConfig = field(default_factory=MQSConfig)
    magnetic_volume_axial_order: int = 8
    magnetic_volume_radial_order: int = 6
    magnetic_volume_azimuthal_order: int = 24
    maximum_raw_magnetic_reciprocity_defect: float = 0.08
    package_volume_axial_order: int = 8
    package_volume_radial_order: int = 6
    package_volume_azimuthal_order: int = 24
    background_radial_order: int = 12
    background_angular_order: int = 48
    maximum_raw_spatial_closure_error: float = 0.35
    reference_backend: str = TENSOR_HYBRID_REFERENCE_BACKEND

    def __post_init__(self):
        frequency = float(self.frequency_hz)
        if not np.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency_hz must be finite and nonnegative")
        if not isinstance(self.teacher_config, MQSConfig):
            raise TypeError("teacher_config must be MQSConfig")

        impedance = np.asarray(self.target_impedance, dtype=complex)
        channels = np.asarray(self.target_dissipation_channels, dtype=complex)
        resistance = np.asarray(self.baseline_resistance, dtype=float)
        reactance = np.asarray(self.baseline_reactance, dtype=float)
        n = len(self.scene.coils)
        if (
            impedance.shape != (n, n)
            or resistance.shape != (n, n)
            or reactance.shape != (n, n)
            or channels.shape != (n + 1, n, n)
        ):
            raise ValueError(
                "tensor hybrid teacher matrices have incompatible shapes"
            )
        if (
            np.any(~np.isfinite(impedance))
            or np.any(~np.isfinite(channels))
            or np.any(~np.isfinite(resistance))
            or np.any(~np.isfinite(reactance))
        ):
            raise ValueError("tensor hybrid teacher matrices must be finite")
        if self.reference_backend != TENSOR_HYBRID_REFERENCE_BACKEND:
            raise ValueError(
                "tensor teacher sample uses an incompatible reference backend"
            )
        if self.baseline_segments < 8:
            raise ValueError("baseline_segments must be >= 8")
        if self.surface_vertical_order < 2 or self.surface_azimuthal_order < 4:
            raise ValueError("invalid tensor surface quadrature metadata")
        if (
            self.magnetic_volume_axial_order < 2
            or self.magnetic_volume_radial_order < 2
            or self.magnetic_volume_azimuthal_order < 8
        ):
            raise ValueError("invalid tensor magnetic quadrature metadata")
        if self.maximum_raw_magnetic_reciprocity_defect < 0.0:
            raise ValueError(
                "maximum_raw_magnetic_reciprocity_defect must be nonnegative"
            )
        if (
            self.package_volume_axial_order < 2
            or self.package_volume_radial_order < 2
            or self.package_volume_azimuthal_order < 8
            or self.background_radial_order < 3
            or self.background_angular_order < 8
        ):
            raise ValueError("invalid tensor energy quadrature metadata")
        if self.maximum_raw_spatial_closure_error <= 0.0:
            raise ValueError(
                "maximum_raw_spatial_closure_error must be positive"
            )

        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "target_impedance", impedance)
        object.__setattr__(self, "target_dissipation_channels", channels)
        object.__setattr__(self, "baseline_resistance", resistance)
        object.__setattr__(self, "baseline_reactance", reactance)

    @property
    def target_prediction(self) -> StructuredPortPrediction:
        n = len(self.scene.coils)
        return StructuredPortPrediction(
            self.target_impedance,
            self.target_dissipation_channels,
            tuple(f"coil:{index}" for index in range(n))
            + ("electric_environment:aggregate",),
        )

    @staticmethod
    def generate(
        scene: Scene,
        frequency_hz: float,
        *,
        teacher_config: MQSConfig | None = None,
        baseline_segments: int = 96,
        surface_vertical_order: int = 16,
        surface_azimuthal_order: int = 32,
        magnetic_volume_axial_order: int = 8,
        magnetic_volume_radial_order: int = 6,
        magnetic_volume_azimuthal_order: int = 24,
        maximum_raw_magnetic_reciprocity_defect: float = 0.08,
        package_volume_axial_order: int = 8,
        package_volume_radial_order: int = 6,
        package_volume_azimuthal_order: int = 24,
        background_radial_order: int = 12,
        background_angular_order: int = 48,
        maximum_raw_spatial_closure_error: float = 0.35,
        maximum_spatial_quadrature_refinements: int = 4,
    ) -> "TensorHybridTeacherSample":
        if not scene.packages:
            raise ValueError(
                "tensor hybrid teacher samples require at least one package"
            )
        resolved_config = teacher_config or MQSConfig()
        encoded = encode_tensor_hybrid_scene_invariant(scene, frequency_hz)
        conductor_scene = Scene(scene.coils, scene.medium, ())
        baseline = analytic_port_baseline(
            conductor_scene,
            frequency_hz,
            segments_per_coil=baseline_segments,
        )
        teacher = DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            resolved_config,
            surface_vertical_order=surface_vertical_order,
            surface_azimuthal_order=surface_azimuthal_order,
            magnetic_volume_axial_order=magnetic_volume_axial_order,
            magnetic_volume_radial_order=magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order=magnetic_volume_azimuthal_order,
            maximum_raw_magnetic_reciprocity_defect=(
                maximum_raw_magnetic_reciprocity_defect
            ),
        )
        result = teacher.solve()
        calibration = prepare_tensor_spatial_reference_adaptive(
            teacher,
            result,
            volume_axial_order=int(package_volume_axial_order),
            volume_radial_order=int(package_volume_radial_order),
            volume_azimuthal_order=int(package_volume_azimuthal_order),
            background_radial_order=int(background_radial_order),
            background_angular_order=int(background_angular_order),
            maximum_raw_closure_error=float(
                maximum_raw_spatial_closure_error
            ),
            maximum_quadrature_refinements=int(
                maximum_spatial_quadrature_refinements
            ),
        )
        return TensorHybridTeacherSample(
            scene=scene,
            frequency_hz=float(frequency_hz),
            encoded=encoded,
            baseline_resistance=baseline.resistance,
            baseline_reactance=(
                2.0 * np.pi * float(frequency_hz) * baseline.inductance
            ),
            target_impedance=calibration.target_impedance,
            target_dissipation_channels=(
                calibration.target_dissipation_channels
            ),
            baseline_segments=int(baseline_segments),
            surface_vertical_order=int(surface_vertical_order),
            surface_azimuthal_order=int(surface_azimuthal_order),
            surface_residual=float(result.surface_residual),
            raw_potential_reciprocity_defect=float(
                result.raw_potential_reciprocity_defect
            ),
            power_closure_error=float(calibration.power_closure_error),
            magnetic_surface_residual=float(result.magnetic_surface_residual),
            raw_magnetic_reciprocity_defect=float(
                result.raw_magnetic_reciprocity_defect
            ),
            teacher_config=resolved_config,
            magnetic_volume_axial_order=int(magnetic_volume_axial_order),
            magnetic_volume_radial_order=int(magnetic_volume_radial_order),
            magnetic_volume_azimuthal_order=int(
                magnetic_volume_azimuthal_order
            ),
            maximum_raw_magnetic_reciprocity_defect=float(
                maximum_raw_magnetic_reciprocity_defect
            ),
            package_volume_axial_order=int(package_volume_axial_order),
            package_volume_radial_order=int(package_volume_radial_order),
            package_volume_azimuthal_order=int(package_volume_azimuthal_order),
            background_radial_order=int(background_radial_order),
            background_angular_order=int(background_angular_order),
            maximum_raw_spatial_closure_error=float(
                maximum_raw_spatial_closure_error
            ),
        )
