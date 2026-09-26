from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .analytic_baseline import analytic_port_baseline
from .em import MQSConfig
from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
)
from .hybrid_field import (
    prepare_hybrid_reference_loss_field,
)
from .hybrid_features import (
    EncodedHybridScene,
    encode_hybrid_scene_invariant,
)
from .scene import Scene
from .training_data import SpatialLossSamples


HYBRID_REFERENCE_BACKEND = (
    "dielectric_mixed_sie"
)


@dataclass(frozen=True)
class PackageSpatialLossSamples:
    package_index: np.ndarray
    local_position: np.ndarray
    weights: np.ndarray
    dissipation_matrix: np.ndarray

    def __post_init__(self):
        package_index = np.asarray(
            self.package_index,
            dtype=int,
        )
        local_position = np.asarray(
            self.local_position,
            dtype=float,
        )
        weights = np.asarray(
            self.weights,
            dtype=float,
        )
        matrix = np.asarray(
            self.dissipation_matrix,
            dtype=complex,
        )
        n = len(
            package_index
        )
        if (
            local_position.shape
            != (
                n,
                3,
            )
            or weights.shape
            != (
                n,
            )
            or matrix.ndim
            != 3
            or matrix.shape[
                0
            ]
            != n
            or matrix.shape[
                1
            ]
            != matrix.shape[
                2
            ]
        ):
            raise ValueError(
                "package spatial loss arrays have incompatible shapes"
            )
        if (
            np.any(
                package_index
                < 0
            )
            or np.any(
                ~np.isfinite(
                    local_position
                )
            )
            or np.any(
                ~np.isfinite(
                    weights
                )
            )
            or np.any(
                weights
                <= 0.0
            )
        ):
            raise ValueError(
                "package spatial loss coordinates/weights are invalid"
            )
        object.__setattr__(
            self,
            "package_index",
            package_index,
        )
        object.__setattr__(
            self,
            "local_position",
            local_position,
        )
        object.__setattr__(
            self,
            "weights",
            weights,
        )
        object.__setattr__(
            self,
            "dissipation_matrix",
            matrix,
        )

    def integrated_packages(
        self,
        n_packages: int,
    ) -> np.ndarray:
        n_ports = int(
            self.dissipation_matrix.shape[
                1
            ]
        )
        out = np.zeros(
            (
                n_packages,
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )
        for index in range(
            len(
                self.package_index
            )
        ):
            package = int(
                self.package_index[
                    index
                ]
            )
            if not (
                0
                <= package
                < n_packages
            ):
                raise IndexError(
                    "package spatial sample index is out of range"
                )
            out[
                package
            ] += (
                self.weights[
                    index
                ]
                * self.dissipation_matrix[
                    index
                ]
            )
        return out


@dataclass(frozen=True)
class HybridTeacherSample:
    scene: Scene
    frequency_hz: float
    encoded: EncodedHybridScene
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
    conductor_spatial_loss: SpatialLossSamples | None = None
    package_spatial_loss: PackageSpatialLossSamples | None = None
    package_volume_axial_order: int = 0
    package_volume_radial_order: int = 0
    package_volume_azimuthal_order: int = 0
    reference_backend: str = (
        HYBRID_REFERENCE_BACKEND
    )

    @property
    def has_spatial_truth(
        self,
    ) -> bool:
        return bool(
            self.conductor_spatial_loss
            is not None
            and self.package_spatial_loss
            is not None
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
        include_spatial_truth: bool = False,
        package_volume_axial_order: int = 8,
        package_volume_radial_order: int = 6,
        package_volume_azimuthal_order: int = 24,
        maximum_raw_spatial_closure_error: float = 0.25,
    ) -> "HybridTeacherSample":
        if not scene.packages:
            raise ValueError(
                "hybrid teacher samples require at least one package"
            )
        if (
            include_spatial_truth
            and scene.medium.conductivity
            > 0.0
        ):
            raise NotImplementedError(
                "hybrid port truth supports lossy homogeneous backgrounds, "
                "but the current hybrid spatial-training schema stores only "
                "conductor/package fields and has no background spatial labels"
            )
        if include_spatial_truth and (
            package_volume_axial_order
            < 2
            or package_volume_radial_order
            < 2
            or package_volume_azimuthal_order
            < 8
        ):
            raise ValueError(
                "invalid package spatial truth quadrature order"
            )

        encoded = (
            encode_hybrid_scene_invariant(
                scene,
                frequency_hz,
            )
        )
        conductor_scene = Scene(
            scene.coils,
            scene.medium,
            (),
        )
        baseline = (
            analytic_port_baseline(
                conductor_scene,
                frequency_hz,
                segments_per_coil=(
                    baseline_segments
                ),
            )
        )
        teacher = (
            DielectricCoupledMixedTeacher(
                scene,
                frequency_hz,
                teacher_config
                or MQSConfig(),
                surface_vertical_order=(
                    surface_vertical_order
                ),
                surface_azimuthal_order=(
                    surface_azimuthal_order
                ),
            )
        )
        result = teacher.solve()

        conductor_spatial = None
        package_spatial = None
        if include_spatial_truth:
            coil_segments = {}
            for index, segment in enumerate(
                teacher.conductor_teacher._mqs._segments
            ):
                coil_segments.setdefault(
                    int(
                        segment.coil
                    ),
                    [],
                ).append(
                    index
                )
            local_position = {
                coil: {
                    segment_index: position
                    for position, segment_index
                    in enumerate(
                        indices
                    )
                }
                for coil, indices
                in coil_segments.items()
            }

            conductor_coil = []
            conductor_arc = []
            conductor_xy = []
            conductor_weights = []
            conductor_matrix = []
            for (
                segment_index,
                segment,
            ) in enumerate(
                teacher.conductor_teacher._mqs._segments
            ):
                coil = int(
                    segment.coil
                )
                position = (
                    local_position[
                        coil
                    ][
                        segment_index
                    ]
                )
                n_segments = len(
                    coil_segments[
                        coil
                    ]
                )
                arc = (
                    position
                    + 0.5
                ) / n_segments
                quadrature = (
                    segment.basis.quadrature
                )
                transfer = (
                    segment.basis.values
                    @ result.mixed_result.current_coefficients[
                        segment.mode_slice
                    ]
                )
                sigma = (
                    scene.coils[
                        coil
                    ].material.conductivity
                )
                matrices = (
                    np.einsum(
                        "qi,qj->qij",
                        transfer.conj(),
                        transfer,
                    )
                    / sigma
                )
                matrices = 0.5 * (
                    matrices
                    + matrices.conj().transpose(
                        0,
                        2,
                        1,
                    )
                )
                count = len(
                    quadrature.weights
                )
                conductor_coil.append(
                    np.full(
                        count,
                        coil,
                        dtype=int,
                    )
                )
                conductor_arc.append(
                    np.full(
                        count,
                        arc,
                        dtype=float,
                    )
                )
                conductor_xy.append(
                    quadrature.xy
                )
                conductor_weights.append(
                    quadrature.weights
                    * segment.length
                )
                conductor_matrix.append(
                    matrices
                )

            conductor_spatial = (
                SpatialLossSamples(
                    np.concatenate(
                        conductor_coil
                    ),
                    np.concatenate(
                        conductor_arc
                    ),
                    np.concatenate(
                        conductor_xy,
                        axis=0,
                    ),
                    np.concatenate(
                        conductor_weights
                    ),
                    np.concatenate(
                        conductor_matrix,
                        axis=0,
                    ),
                )
            )

            prepared = (
                prepare_hybrid_reference_loss_field(
                    teacher,
                    result,
                    volume_axial_order=(
                        package_volume_axial_order
                    ),
                    volume_radial_order=(
                        package_volume_radial_order
                    ),
                    volume_azimuthal_order=(
                        package_volume_azimuthal_order
                    ),
                    maximum_raw_closure_error=(
                        maximum_raw_spatial_closure_error
                    ),
                )
            )
            package_index = []
            package_local = []
            package_weights = []
            package_matrices = []
            for index, package in enumerate(
                scene.packages
            ):
                quadrature = (
                    package.geometry.volume_quadrature(
                        axial_order=(
                            package_volume_axial_order
                        ),
                        radial_order=(
                            package_volume_radial_order
                        ),
                        azimuthal_order=(
                            package_volume_azimuthal_order
                        ),
                    )
                )
                count = len(
                    quadrature.weights
                )
                package_index.append(
                    np.full(
                        count,
                        index,
                        dtype=int,
                    )
                )
                package_local.append(
                    quadrature.local_positions
                )
                package_weights.append(
                    quadrature.weights
                )
                package_matrices.append(
                    prepared.package_dissipation_matrices(
                        index,
                        quadrature.positions,
                    )
                )
            package_spatial = (
                PackageSpatialLossSamples(
                    np.concatenate(
                        package_index
                    ),
                    np.concatenate(
                        package_local,
                        axis=0,
                    ),
                    np.concatenate(
                        package_weights
                    ),
                    np.concatenate(
                        package_matrices,
                        axis=0,
                    ),
                )
            )

        return HybridTeacherSample(
            scene=scene,
            frequency_hz=float(
                frequency_hz
            ),
            encoded=encoded,
            baseline_resistance=(
                baseline.resistance
            ),
            baseline_reactance=(
                2.0
                * np.pi
                * float(
                    frequency_hz
                )
                * baseline.inductance
            ),
            target_impedance=(
                result.impedance
            ),
            target_dissipation_channels=(
                result.prediction.dissipation_channels
            ),
            baseline_segments=int(
                baseline_segments
            ),
            surface_vertical_order=int(
                surface_vertical_order
            ),
            surface_azimuthal_order=int(
                surface_azimuthal_order
            ),
            surface_residual=float(
                result.surface_residual
            ),
            raw_potential_reciprocity_defect=float(
                result.raw_potential_reciprocity_defect
            ),
            power_closure_error=float(
                result.power_closure_error
            ),
            conductor_spatial_loss=(
                conductor_spatial
            ),
            package_spatial_loss=(
                package_spatial
            ),
            package_volume_axial_order=(
                int(
                    package_volume_axial_order
                )
                if include_spatial_truth
                else 0
            ),
            package_volume_radial_order=(
                int(
                    package_volume_radial_order
                )
                if include_spatial_truth
                else 0
            ),
            package_volume_azimuthal_order=(
                int(
                    package_volume_azimuthal_order
                )
                if include_spatial_truth
                else 0
            ),
        )
