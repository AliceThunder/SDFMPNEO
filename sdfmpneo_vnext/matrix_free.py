from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator

from .em import DenseMQSTeacher, MQSConfig
from .scene import Scene


@dataclass(frozen=True)
class MatrixFreeMetadata:
    n_modes: int
    n_constraints: int
    n_ports: int
    n_support_points: int


class MatrixFreeMQSOperator:
    """Mesh-free MQS Galerkin operator without a global inductance matrix.

    The current backend applies the free-space Green kernel in target chunks,
    so arithmetic is still O(N_q^2) but persistent memory is O(N_q). Replacing
    the Green-kernel application by FMM/H2 leaves the modal/KKT interface
    unchanged.
    """

    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        config: MQSConfig | None = None,
        *,
        chunk_size: int = 512,
    ):
        if chunk_size < 1:
            raise ValueError(
                "chunk_size must be >= 1"
            )
        self.scene = scene
        self.frequency_hz = float(
            frequency_hz
        )
        if (
            not np.isfinite(
                self.frequency_hz
            )
            or self.frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        self.omega = (
            2.0
            * np.pi
            * self.frequency_hz
        )
        self.config = (
            config
            or MQSConfig()
        )
        self.chunk_size = int(
            chunk_size
        )
        self._teacher = DenseMQSTeacher(
            scene,
            frequency_hz,
            self.config,
        )
        self._segments = (
            self._teacher._segments
        )
        self._n_modes = int(
            self._teacher._n_modes
        )
        self._build_algebra()
        self._build_supports()

    def _build_algebra(
        self,
    ):
        n_segments = len(
            self._segments
        )
        n_ports = len(
            self.scene.coils
        )
        resistance = np.empty(
            self._n_modes,
            dtype=float,
        )
        constraint = np.zeros(
            (
                n_segments,
                self._n_modes,
            ),
            dtype=float,
        )
        port_map = np.zeros(
            (
                n_segments,
                n_ports,
            ),
            dtype=float,
        )
        for segment_index, segment in enumerate(
            self._segments
        ):
            sl = segment.mode_slice
            sigma = (
                self.scene.coils[
                    segment.coil
                ].material.conductivity
            )
            resistance[
                sl
            ] = (
                segment.length
                / sigma
            )
            constraint[
                segment_index,
                sl,
            ] = (
                segment.basis.moments
            )
            port_map[
                segment_index,
                segment.coil,
            ] = 1.0
        self.resistance_diagonal = (
            resistance
        )
        self.constraint_matrix = (
            constraint
        )
        self.port_map = (
            port_map
        )

    def _build_supports(
        self,
    ):
        points = []
        weights = []
        tangents = []
        segment_ids = []
        cell_scale = []
        basis_values = []
        support_slices = []
        cursor = 0

        for segment_index, segment in enumerate(
            self._segments
        ):
            (
                local_points,
                local_weights,
                local_values,
            ) = (
                self._teacher._support_quadrature(
                    segment
                )
            )
            count = len(
                local_weights
            )
            points.append(
                local_points
            )
            weights.append(
                local_weights
            )
            tangents.append(
                np.repeat(
                    segment.tangent[
                        None,
                        :,
                    ],
                    count,
                    axis=0,
                )
            )
            segment_ids.append(
                np.full(
                    count,
                    segment_index,
                    dtype=int,
                )
            )
            cell_scale.append(
                np.cbrt(
                    np.maximum(
                        local_weights,
                        1e-300,
                    )
                )
            )
            basis_values.append(
                local_values
            )
            support_slices.append(
                slice(
                    cursor,
                    cursor
                    + count,
                )
            )
            cursor += count

        self.support_points = np.concatenate(
            points,
            axis=0,
        )
        self.support_weights = np.concatenate(
            weights,
        )
        self.support_tangents = np.concatenate(
            tangents,
            axis=0,
        )
        self.support_segment_ids = np.concatenate(
            segment_ids,
        )
        self.support_cell_scale = np.concatenate(
            cell_scale,
        )
        self._support_basis_values = tuple(
            basis_values
        )
        self._support_slices = tuple(
            support_slices
        )

    @property
    def metadata(
        self,
    ) -> MatrixFreeMetadata:
        return MatrixFreeMetadata(
            n_modes=(
                self._n_modes
            ),
            n_constraints=(
                self.constraint_matrix.shape[
                    0
                ]
            ),
            n_ports=(
                self.port_map.shape[
                    1
                ]
            ),
            n_support_points=(
                len(
                    self.support_weights
                )
            ),
        )

    def _support_amplitude(
        self,
        coefficients,
    ):
        coefficients = np.asarray(
            coefficients,
            dtype=complex,
        )
        if coefficients.shape != (
            self._n_modes,
        ):
            raise ValueError(
                "coefficients have wrong shape"
            )
        amplitude = np.empty(
            len(
                self.support_weights
            ),
            dtype=complex,
        )
        for segment, support_slice, values in zip(
            self._segments,
            self._support_slices,
            self._support_basis_values,
        ):
            amplitude[
                support_slice
            ] = (
                values
                @ coefficients[
                    segment.mode_slice
                ]
            )
        return amplitude

    def _green_potential(
        self,
        weighted_vector_source,
    ):
        weighted_vector_source = np.asarray(
            weighted_vector_source,
            dtype=complex,
        )
        if weighted_vector_source.shape != (
            len(
                self.support_weights
            ),
            3,
        ):
            raise ValueError(
                "weighted_vector_source has wrong shape"
            )

        n_support = len(
            self.support_weights
        )
        potential = np.empty(
            (
                n_support,
                3,
            ),
            dtype=complex,
        )
        prefactor = (
            self.scene.medium.permeability
            / (
                4.0
                * np.pi
            )
        )

        for start in range(
            0,
            n_support,
            self.chunk_size,
        ):
            stop = min(
                start
                + self.chunk_size,
                n_support,
            )
            target_points = (
                self.support_points[
                    start:stop
                ]
            )
            difference = (
                target_points[
                    :,
                    None,
                    :,
                ]
                - self.support_points[
                    None,
                    :,
                    :
                ]
            )
            distance_squared = np.sum(
                difference
                * difference,
                axis=2,
            )
            same_segment = (
                self.support_segment_ids[
                    start:stop,
                    None,
                ]
                == self.support_segment_ids[
                    None,
                    :,
                ]
            )
            softening = (
                self.config.self_softening_factor
                * (
                    self.support_cell_scale[
                        start:stop,
                        None,
                    ]
                    + self.support_cell_scale[
                        None,
                        :,
                    ]
                )
            )
            effective_distance_squared = np.where(
                same_segment,
                distance_squared
                + softening
                * softening,
                np.maximum(
                    distance_squared,
                    1e-30,
                ),
            )
            inverse_distance = (
                1.0
                / np.sqrt(
                    effective_distance_squared
                )
            )
            potential[
                start:stop
            ] = (
                prefactor
                * (
                    inverse_distance
                    @ weighted_vector_source
                )
            )
        return potential

    def apply_inductance(
        self,
        coefficients,
    ):
        amplitude = (
            self._support_amplitude(
                coefficients
            )
        )
        weighted_source = (
            self.support_weights[
                :,
                None,
            ]
            * amplitude[
                :,
                None,
            ]
            * self.support_tangents
        )
        vector_potential = (
            self._green_potential(
                weighted_source
            )
        )
        tangential_potential = np.sum(
            vector_potential
            * self.support_tangents,
            axis=1,
        )
        result = np.zeros(
            self._n_modes,
            dtype=complex,
        )
        for segment, support_slice, values in zip(
            self._segments,
            self._support_slices,
            self._support_basis_values,
        ):
            result[
                segment.mode_slice
            ] += (
                values.T
                @ (
                    self.support_weights[
                        support_slice
                    ]
                    * tangential_potential[
                        support_slice
                    ]
                )
            )
        return result

    def apply_current_operator(
        self,
        coefficients,
    ):
        coefficients = np.asarray(
            coefficients,
            dtype=complex,
        )
        if coefficients.shape != (
            self._n_modes,
        ):
            raise ValueError(
                "coefficients have wrong shape"
            )
        return (
            self.resistance_diagonal
            * coefficients
            + 1j
            * self.omega
            * self.apply_inductance(
                coefficients
            )
        )

    def apply_kkt(
        self,
        vector,
    ):
        vector = np.asarray(
            vector,
            dtype=complex,
        )
        n_constraints = (
            self.constraint_matrix.shape[
                0
            ]
        )
        expected = (
            self._n_modes
            + n_constraints
        )
        if vector.shape != (
            expected,
        ):
            raise ValueError(
                "KKT vector has wrong shape"
            )
        current = vector[
            : self._n_modes
        ]
        multiplier = vector[
            self._n_modes :
        ]
        return np.concatenate(
            (
                self.apply_current_operator(
                    current
                )
                - self.constraint_matrix.T
                @ multiplier,
                self.constraint_matrix
                @ current,
            )
        )

    def current_linear_operator(
        self,
    ) -> LinearOperator:
        return LinearOperator(
            (
                self._n_modes,
                self._n_modes,
            ),
            matvec=(
                self.apply_current_operator
            ),
            dtype=complex,
        )

    def kkt_linear_operator(
        self,
    ) -> LinearOperator:
        n_constraints = (
            self.constraint_matrix.shape[
                0
            ]
        )
        size = (
            self._n_modes
            + n_constraints
        )
        return LinearOperator(
            (
                size,
                size,
            ),
            matvec=(
                self.apply_kkt
            ),
            dtype=complex,
        )

    def port_rhs(
        self,
        port_index: int,
    ):
        if not (
            0
            <= port_index
            < self.port_map.shape[
                1
            ]
        ):
            raise IndexError(
                "port_index out of range"
            )
        return np.concatenate(
            (
                np.zeros(
                    self._n_modes,
                    dtype=complex,
                ),
                self.port_map[
                    :,
                    port_index,
                ].astype(
                    complex
                ),
            )
        )
