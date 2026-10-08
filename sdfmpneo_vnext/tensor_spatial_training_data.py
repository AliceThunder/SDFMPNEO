from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .em import MQSConfig
from .exterior_quadrature import conductor_volume_mask
from .hybrid_dielectric import DielectricCoupledMixedTeacher
from .hybrid_domain import package_domain_topology
from .hybrid_training_data import (
    BackgroundSpatialLossSamples,
    PackageSpatialLossSamples,
)
from .tensor_spatial_reference import prepare_tensor_spatial_reference_adaptive
from .tensor_training_data import (
    TENSOR_HYBRID_REFERENCE_BACKEND,
    TensorHybridTeacherSample,
)
from .training_data import SpatialLossSamples


@dataclass(frozen=True)
class TensorHybridSpatialTeacherSample:
    port: TensorHybridTeacherSample
    conductor_spatial_loss: SpatialLossSamples
    package_spatial_loss: PackageSpatialLossSamples
    background_spatial_loss: BackgroundSpatialLossSamples | None
    package_volume_axial_order: int
    package_volume_radial_order: int
    package_volume_azimuthal_order: int
    background_radial_order: int
    background_angular_order: int

    @property
    def scene(self):
        return self.port.scene

    @property
    def frequency_hz(self) -> float:
        return self.port.frequency_hz

    @property
    def encoded(self):
        return self.port.encoded

    @property
    def target_dissipation_channels(self):
        return self.port.target_dissipation_channels

    @staticmethod
    def generate(
        port_sample: TensorHybridTeacherSample,
        *,
        teacher_config: MQSConfig | None = None,
        package_volume_axial_order: int | None = None,
        package_volume_radial_order: int | None = None,
        package_volume_azimuthal_order: int | None = None,
        background_radial_order: int | None = None,
        background_angular_order: int | None = None,
        maximum_raw_spatial_closure_error: float | None = None,
        maximum_spatial_quadrature_refinements: int = 4,
    ) -> "TensorHybridSpatialTeacherSample":
        # Spatial point-cloud density is intentionally independent of the
        # higher-order quadrature stored on the port sample. Callers can still
        # override these sampling resolutions explicitly.
        package_volume_axial_order = int(
            6 if package_volume_axial_order is None else package_volume_axial_order
        )
        package_volume_radial_order = int(
            4 if package_volume_radial_order is None else package_volume_radial_order
        )
        package_volume_azimuthal_order = int(
            16 if package_volume_azimuthal_order is None else package_volume_azimuthal_order
        )
        background_radial_order = int(
            10 if background_radial_order is None else background_radial_order
        )
        background_angular_order = int(
            32 if background_angular_order is None else background_angular_order
        )
        maximum_raw_spatial_closure_error = float(
            port_sample.maximum_raw_spatial_closure_error
            if maximum_raw_spatial_closure_error is None
            else maximum_raw_spatial_closure_error
        )

        if (
            package_volume_axial_order < 2
            or package_volume_radial_order < 2
            or package_volume_azimuthal_order < 8
            or background_radial_order < 3
            or background_angular_order < 8
        ):
            raise ValueError("invalid tensor spatial sampling quadrature resolution")
        if int(maximum_spatial_quadrature_refinements) < 0:
            raise ValueError(
                "maximum_spatial_quadrature_refinements must be nonnegative"
            )
        if maximum_raw_spatial_closure_error <= 0.0:
            raise ValueError(
                "maximum_raw_spatial_closure_error must be positive"
            )

        scene = port_sample.scene
        frequency_hz = port_sample.frequency_hz
        resolved_config = (
            port_sample.teacher_config
            if teacher_config is None
            else teacher_config
        )
        teacher = DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            resolved_config,
            surface_vertical_order=port_sample.surface_vertical_order,
            surface_azimuthal_order=port_sample.surface_azimuthal_order,
            magnetic_volume_axial_order=(
                port_sample.magnetic_volume_axial_order
            ),
            magnetic_volume_radial_order=(
                port_sample.magnetic_volume_radial_order
            ),
            magnetic_volume_azimuthal_order=(
                port_sample.magnetic_volume_azimuthal_order
            ),
            maximum_raw_magnetic_reciprocity_defect=(
                port_sample.maximum_raw_magnetic_reciprocity_defect
            ),
        )
        result = teacher.solve()
        calibration = prepare_tensor_spatial_reference_adaptive(
            teacher,
            result,
            volume_axial_order=port_sample.package_volume_axial_order,
            volume_radial_order=port_sample.package_volume_radial_order,
            volume_azimuthal_order=port_sample.package_volume_azimuthal_order,
            background_radial_order=port_sample.background_radial_order,
            background_angular_order=port_sample.background_angular_order,
            maximum_raw_closure_error=maximum_raw_spatial_closure_error,
            maximum_quadrature_refinements=(
                maximum_spatial_quadrature_refinements
            ),
        )
        prepared = calibration.prepared
        corrected_port = replace(
            port_sample,
            target_impedance=calibration.target_impedance,
            target_dissipation_channels=calibration.target_dissipation_channels,
            power_closure_error=calibration.power_closure_error,
            surface_residual=float(result.surface_residual),
            raw_potential_reciprocity_defect=float(
                result.raw_potential_reciprocity_defect
            ),
            magnetic_surface_residual=float(result.magnetic_surface_residual),
            raw_magnetic_reciprocity_defect=float(
                result.raw_magnetic_reciprocity_defect
            ),
            teacher_config=resolved_config,
            maximum_raw_spatial_closure_error=(
                maximum_raw_spatial_closure_error
            ),
            reference_backend=TENSOR_HYBRID_REFERENCE_BACKEND,
        )

        coil_segments = {}
        segments = teacher.conductor_teacher._mqs._segments
        for index, segment in enumerate(segments):
            coil_segments.setdefault(int(segment.coil), []).append(index)
        local_position = {
            coil: {
                segment_index: position
                for position, segment_index in enumerate(indices)
            }
            for coil, indices in coil_segments.items()
        }

        conductor_coil = []
        conductor_arc = []
        conductor_xy = []
        conductor_weights = []
        conductor_matrix = []
        for segment_index, segment in enumerate(segments):
            coil = int(segment.coil)
            position = local_position[coil][segment_index]
            n_segments = len(coil_segments[coil])
            arc = (position + 0.5) / n_segments
            quadrature = segment.basis.quadrature
            transfer = (
                segment.basis.values
                @ result.mixed_result.current_coefficients[segment.mode_slice]
            )
            conductivity = scene.coils[coil].material.conductivity
            matrices = np.einsum(
                "qi,qj->qij",
                transfer.conj(),
                transfer,
            ) / conductivity
            matrices = prepared.transform_dissipation_matrices(matrices)
            count = len(quadrature.weights)
            conductor_coil.append(np.full(count, coil, dtype=int))
            conductor_arc.append(np.full(count, arc, dtype=float))
            conductor_xy.append(quadrature.xy)
            conductor_weights.append(quadrature.weights * segment.length)
            conductor_matrix.append(matrices)
        conductor_spatial = SpatialLossSamples(
            np.concatenate(conductor_coil),
            np.concatenate(conductor_arc),
            np.concatenate(conductor_xy, axis=0),
            np.concatenate(conductor_weights),
            np.concatenate(conductor_matrix, axis=0),
        )

        topology = package_domain_topology(scene.packages)
        package_index = []
        package_local = []
        package_weights = []
        package_matrices = []
        for index, package in enumerate(scene.packages):
            quadrature = package.geometry.volume_quadrature(
                axial_order=package_volume_axial_order,
                radial_order=package_volume_radial_order,
                azimuthal_order=package_volume_azimuthal_order,
            )
            region = np.asarray(
                topology.deepest_containing(
                    scene.packages,
                    quadrature.positions,
                    tolerance=2e-12,
                ),
                dtype=int,
            )
            keep = region == index
            if np.any(keep):
                keep &= ~np.asarray(
                    conductor_volume_mask(
                        scene,
                        quadrature.positions,
                        segments=segments,
                    ),
                    dtype=bool,
                )
            count = int(np.count_nonzero(keep))
            if count == 0:
                continue
            package_index.append(np.full(count, index, dtype=int))
            package_local.append(quadrature.local_positions[keep])
            package_weights.append(quadrature.weights[keep])
            package_matrices.append(
                prepared.package_dissipation_matrices(
                    index,
                    quadrature.positions[keep],
                )
            )

        n_ports = len(scene.coils)
        if package_index:
            package_spatial = PackageSpatialLossSamples(
                np.concatenate(package_index),
                np.concatenate(package_local, axis=0),
                np.concatenate(package_weights),
                np.concatenate(package_matrices, axis=0),
            )
        else:
            package_spatial = PackageSpatialLossSamples(
                np.zeros(0, dtype=int),
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                np.zeros((0, n_ports, n_ports), dtype=complex),
            )

        background_spatial = None
        if scene.medium.loss_conductivity(frequency_hz) > 0.0:
            points, weights = prepared.background_quadrature(
                radial_order=background_radial_order,
                angular_order=background_angular_order,
            )
            root_pose = scene.coils[0].geometry.pose
            root_local = (
                points - root_pose.translation[None, :]
            ) @ root_pose.rotation
            background_spatial = BackgroundSpatialLossSamples(
                root_local,
                weights,
                prepared.background_dissipation_matrices(points),
            )

        return TensorHybridSpatialTeacherSample(
            port=corrected_port,
            conductor_spatial_loss=conductor_spatial,
            package_spatial_loss=package_spatial,
            background_spatial_loss=background_spatial,
            package_volume_axial_order=package_volume_axial_order,
            package_volume_radial_order=package_volume_radial_order,
            package_volume_azimuthal_order=package_volume_azimuthal_order,
            background_radial_order=background_radial_order,
            background_angular_order=background_angular_order,
        )
