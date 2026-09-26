from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .analytic_baseline import analytic_port_baseline
from .em import DenseMQSTeacher, MQSConfig
from .mixed import DenseMixedConductorTeacher
from .features import EncodedScene, encode_scene_invariant
from .scene import Scene


CANONICAL_REFERENCE_BACKEND = "mixed"


@dataclass(frozen=True)
class SpatialLossSamples:
    coil_index: np.ndarray
    arc_fraction: np.ndarray
    xy: np.ndarray
    weights: np.ndarray
    dissipation_matrix: np.ndarray

    def __post_init__(self):
        coil_index = np.asarray(
            self.coil_index,
            dtype=int,
        )
        arc_fraction = np.asarray(
            self.arc_fraction,
            dtype=float,
        )
        xy = np.asarray(
            self.xy,
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
        n = len(coil_index)
        if arc_fraction.shape != (n,):
            raise ValueError(
                "arc_fraction has wrong shape"
            )
        if xy.shape != (n, 2):
            raise ValueError(
                "xy has wrong shape"
            )
        if weights.shape != (n,):
            raise ValueError(
                "weights has wrong shape"
            )
        if (
            matrix.ndim != 3
            or matrix.shape[0] != n
            or matrix.shape[1]
            != matrix.shape[2]
        ):
            raise ValueError(
                "dissipation_matrix has wrong shape"
            )
        object.__setattr__(
            self,
            "coil_index",
            coil_index,
        )
        object.__setattr__(
            self,
            "arc_fraction",
            arc_fraction,
        )
        object.__setattr__(
            self,
            "xy",
            xy,
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

    def integrated_channels(
        self,
        n_coils: int,
    ) -> np.ndarray:
        n_ports = (
            self.dissipation_matrix.shape[1]
        )
        out = np.zeros(
            (
                n_coils,
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )
        for index in range(
            len(self.coil_index)
        ):
            out[
                self.coil_index[index]
            ] += (
                self.weights[index]
                * self.dissipation_matrix[
                    index
                ]
            )
        return out


@dataclass(frozen=True)
class TeacherSample:
    scene: Scene
    frequency_hz: float
    encoded: EncodedScene
    baseline_resistance: np.ndarray
    baseline_reactance: np.ndarray
    target_impedance: np.ndarray
    baseline_segments: int
    target_dissipation_channels: np.ndarray | None = None
    spatial_loss: SpatialLossSamples | None = None
    reference_backend: str | None = None

    @staticmethod
    def generate(
        scene: Scene,
        frequency_hz: float,
        *,
        teacher_config: MQSConfig | None = None,
        baseline_segments: int = 96,
        reference_backend: str = CANONICAL_REFERENCE_BACKEND,
    ) -> "TeacherSample":
        encoded = encode_scene_invariant(
            scene,
            frequency_hz,
        )
        baseline = analytic_port_baseline(
            scene,
            frequency_hz,
            segments_per_coil=baseline_segments,
        )
        resolved_config = (
            teacher_config
            or MQSConfig()
        )
        reference_backend = str(
            reference_backend
        ).lower()
        if reference_backend == "mqs":
            teacher = DenseMQSTeacher(
                scene,
                frequency_hz,
                resolved_config,
            )
            segments = teacher._segments
        elif reference_backend == "mixed":
            teacher = DenseMixedConductorTeacher(
                scene,
                frequency_hz,
                resolved_config,
            )
            segments = teacher._mqs._segments
        else:
            raise ValueError(
                "reference_backend must be 'mqs' or 'mixed'"
            )
        truth_result = teacher.solve()
        truth = truth_result.impedance
        if hasattr(
            truth_result,
            "dissipation_channels",
        ):
            channels = (
                truth_result.dissipation_channels()
            )
        else:
            channels = (
                truth_result.coil_dissipation_matrices()
            )

        if (
            reference_backend == "mixed"
            and scene.medium.conductivity > 0.0
        ):
            # Port/background dissipation is valid, but the current spatial
            # teacher describes conductor volume only.  Do not label it as a
            # complete environmental heat-source field.
            spatial = None
        else:
            coil_segments = {}
            for index, segment in enumerate(
                segments
            ):
                coil_segments.setdefault(
                    int(segment.coil),
                    [],
                ).append(index)

            spatial_coil = []
            spatial_arc = []
            spatial_xy = []
            spatial_weights = []
            spatial_matrix = []

            local_position = {
                coil: {
                    segment_index: position
                    for position, segment_index
                    in enumerate(indices)
                }
                for coil, indices
                in coil_segments.items()
            }

            for segment_index, segment in enumerate(
                segments
            ):
                coil = int(
                    segment.coil
                )
                local_index = (
                    local_position[
                        coil
                    ][segment_index]
                )
                n_segments = len(
                    coil_segments[
                        coil
                    ]
                )
                arc = (
                    local_index + 0.5
                ) / n_segments
                quadrature = (
                    segment.basis.quadrature
                )
                transfer = (
                    segment.basis.values
                    @ truth_result.mode_coefficients[
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
                spatial_coil.append(
                    np.full(
                        count,
                        coil,
                        dtype=int,
                    )
                )
                spatial_arc.append(
                    np.full(
                        count,
                        arc,
                        dtype=float,
                    )
                )
                spatial_xy.append(
                    quadrature.xy
                )
                spatial_weights.append(
                    quadrature.weights
                    * segment.length
                )
                spatial_matrix.append(
                    matrices
                )

            spatial = SpatialLossSamples(
                np.concatenate(
                    spatial_coil
                ),
                np.concatenate(
                    spatial_arc
                ),
                np.concatenate(
                    spatial_xy,
                    axis=0,
                ),
                np.concatenate(
                    spatial_weights
                ),
                np.concatenate(
                    spatial_matrix,
                    axis=0,
                ),
            )


        return TeacherSample(
            scene,
            float(frequency_hz),
            encoded,
            baseline.resistance,
            baseline.inductance
            * (
                2.0
                * np.pi
                * float(frequency_hz)
            ),
            truth,
            int(baseline_segments),
            channels,
            spatial,
            reference_backend,
        )
