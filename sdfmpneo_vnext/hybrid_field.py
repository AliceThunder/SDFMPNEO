from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .prediction import StructuredPortPrediction
from .scene import Scene


def _hermitian_psd_sqrt(
    matrix,
    *,
    inverse: bool,
):
    matrix = np.asarray(
        matrix,
        dtype=complex,
    )
    matrix = 0.5 * (
        matrix
        + matrix.conj().T
    )
    values, vectors = np.linalg.eigh(
        matrix
    )
    scale = max(
        float(
            np.max(
                np.abs(
                    values
                )
            )
        ),
        1e-30,
    )
    floor = (
        1e-12
        * scale
    )
    clipped = np.clip(
        values.real,
        floor,
        None,
    )
    diagonal = (
        1.0
        / np.sqrt(
            clipped
        )
        if inverse
        else np.sqrt(
            clipped
        )
    )
    return (
        vectors
        @ np.diag(
            diagonal
        )
        @ vectors.conj().T
    )


def _fibonacci_directions(
    count: int,
) -> np.ndarray:
    if count < 8:
        raise ValueError(
            "background angular order must be >= 8"
        )
    index = np.arange(
        count,
        dtype=float,
    )
    z = (
        1.0
        - 2.0
        * (
            index
            + 0.5
        )
        / count
    )
    radius = np.sqrt(
        np.maximum(
            1.0
            - z * z,
            0.0,
        )
    )
    golden = (
        np.pi
        * (
            3.0
            - np.sqrt(
                5.0
            )
        )
    )
    angle = (
        golden
        * index
    )
    return np.column_stack(
        (
            radius
            * np.cos(
                angle
            ),
            radius
            * np.sin(
                angle
            ),
            z,
        )
    )


@dataclass(frozen=True)
class PreparedHybridReferenceLossField:
    scene: Scene
    frequency_hz: float
    teacher: object
    result: object
    port_prediction: StructuredPortPrediction
    package_transform: np.ndarray
    raw_dielectric_closure_error: float
    normalized_dielectric_closure_error: float
    package_integrated_channels: np.ndarray
    background_integrated_channel: np.ndarray | None = None

    @property
    def normalization_closure_error(
        self,
    ) -> float:
        return float(
            max(
                self.port_prediction.power_closure_error(),
                self.normalized_dielectric_closure_error,
            )
        )

    @property
    def n_conductor_channels(
        self,
    ) -> int:
        return len(
            self.scene.coils
        )

    @property
    def dielectric_channel_index(
        self,
    ) -> int:
        # Backward-compatible name for the aggregate electric-environment
        # channel. With a lossy background this channel contains both package
        # and exterior-medium electric loss.
        return self.n_conductor_channels

    @property
    def environment_channel_index(
        self,
    ) -> int:
        return self.n_conductor_channels

    @property
    def background_channel_index(
        self,
    ) -> int | None:
        if self.scene.medium.conductivity <= 0.0:
            return None
        return self.environment_channel_index

    @property
    def environment_transform(
        self,
    ) -> np.ndarray:
        return np.asarray(
            self.package_transform,
            dtype=complex,
        )

    def conductor_local_dissipation_matrix(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return (
            self.teacher.conductor_teacher.local_dissipation_matrix(
                self.result.mixed_result,
                coil_index,
                arc_fraction,
                xy,
            )
        )

    def local_dissipation_matrix(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        """Compatibility conductor query used by existing spatial consumers."""
        return (
            self.conductor_local_dissipation_matrix(
                coil_index,
                arc_fraction,
                xy,
            )
        )

    def conductor_local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        matrix = (
            self.conductor_local_dissipation_matrix(
                coil_index,
                arc_fraction,
                xy,
            )
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        return float(
            0.5
            * np.real(
                np.vdot(
                    currents,
                    matrix
                    @ currents,
                )
            )
        )

    def _charge_geometry(
        self,
    ):
        (
            _,
            _,
            current_constraint,
            _,
        ) = (
            self.teacher.conductor_teacher._mqs.assemble()
        )
        (
            _,
            _,
            _,
            node_positions,
            node_radii,
        ) = (
            self.teacher.conductor_teacher._topology(
                current_constraint
            )
        )
        (
            _,
            source_permittivity,
        ) = self.teacher._source_regions(
            node_positions
        )
        return (
            node_positions,
            node_radii,
            source_permittivity,
        )

    def electric_field_transfer(
        self,
        points,
    ) -> np.ndarray:
        """Return E(points)=transfer @ port-current as (n,3,n_ports)."""
        points = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            points.ndim == 1
        )
        points = np.atleast_2d(
            points
        )
        if (
            points.ndim != 2
            or points.shape[1] != 3
        ):
            raise ValueError(
                "points must have shape (3,) or (n,3)"
            )

        (
            node_positions,
            node_radii,
            source_permittivity,
        ) = self._charge_geometry()

        diff = (
            points[
                :,
                None,
                :,
            ]
            - node_positions[
                None,
                :,
                :,
            ]
        )
        soft = (
            self.teacher.charge_self_radius_factor
            * node_radii[
                None,
                :,
            ]
        )
        distance2 = (
            np.sum(
                diff
                * diff,
                axis=2,
            )
            + soft**2
        )
        direct_kernel = (
            diff
            / (
                4.0
                * np.pi
                * distance2[
                    :,
                    :,
                    None,
                ] ** 1.5
                * source_permittivity[
                    None,
                    :,
                    None,
                ]
            )
        )
        direct = np.einsum(
            "qjd,jp->qdp",
            direct_kernel,
            self.result.mixed_result.node_charge,
        )

        surface_positions = (
            self.teacher.surface_solver.positions
        )
        surface_weights = (
            self.teacher.surface_solver.weights
        )
        surface_diff = (
            points[
                :,
                None,
                :,
            ]
            - surface_positions[
                None,
                :,
                :,
            ]
        )
        surface_distance = np.linalg.norm(
            surface_diff,
            axis=2,
        )
        surface_center = np.mean(
            surface_positions,
            axis=0,
        )
        geometry_scale = max(
            float(
                np.max(
                    np.linalg.norm(
                        surface_positions
                        - surface_center[
                            None,
                            :
                        ],
                        axis=1,
                    )
                )
            ),
            1.0,
        )
        if np.any(
            surface_distance
            <= 1e-13
            * geometry_scale
        ):
            raise ValueError(
                "dielectric electric-field query lies on a surface quadrature node"
            )
        induced_kernel = (
            surface_weights[
                None,
                :,
                None,
            ]
            * surface_diff
            / (
                4.0
                * np.pi
                * surface_distance[
                    :,
                    :,
                    None,
                ] ** 3
            )
        )
        induced = np.einsum(
            "qsd,sp->qdp",
            induced_kernel,
            self.result.surface_density_transfer,
        )
        transfer = (
            direct
            + induced
        )
        return (
            transfer[0]
            if scalar
            else transfer
        )

    def _conductor_exterior_mask(
        self,
        points,
    ) -> np.ndarray:
        points = np.asarray(
            points,
            dtype=float,
        )
        if (
            points.ndim != 2
            or points.shape[1] != 3
        ):
            raise ValueError(
                "points must have shape (n,3)"
            )
        exterior = np.ones(
            len(
                points
            ),
            dtype=bool,
        )
        for segment in self.teacher.conductor_teacher._mqs._segments:
            active = np.flatnonzero(
                exterior
            )
            if active.size == 0:
                break
            delta = (
                points[
                    active
                ]
                - segment.midpoint[
                    None,
                    :
                ]
            )
            longitudinal = (
                delta
                @ segment.tangent
            )
            near = (
                np.abs(
                    longitudinal
                )
                <= (
                    0.5
                    * segment.length
                    + 1e-12
                )
            )
            if not np.any(
                near
            ):
                continue
            candidate = active[
                near
            ]
            transverse = (
                delta[
                    near
                ]
                - longitudinal[
                    near,
                    None,
                ]
                * segment.tangent[
                    None,
                    :
                ]
            )
            geometry = self.scene.coils[
                segment.coil
            ].geometry
            u = (
                transverse
                @ segment.n1
            ) / (
                0.5
                * geometry.conductor_width
            )
            v = (
                transverse
                @ segment.n2
            ) / (
                0.5
                * geometry.conductor_thickness
            )
            exponent = float(
                geometry.cross_section_exponent
            )
            inside = (
                np.abs(
                    u
                ) ** exponent
                + np.abs(
                    v
                ) ** exponent
                <= (
                    1.0
                    + 1e-10
                )
            )
            exterior[
                candidate[
                    inside
                ]
            ] = False
        return exterior

    def _background_domain_mask(
        self,
        points,
    ) -> np.ndarray:
        points = np.asarray(
            points,
            dtype=float,
        )
        exterior = (
            self._conductor_exterior_mask(
                points
            )
        )
        for package in self.scene.packages:
            if not np.any(
                exterior
            ):
                break
            active = np.flatnonzero(
                exterior
            )
            inside = np.asarray(
                package.geometry.contains(
                    points[
                        active
                    ],
                    tolerance=2e-12,
                ),
                dtype=bool,
            )
            exterior[
                active[
                    inside
                ]
            ] = False
        return exterior

    def background_quadrature(
        self,
        *,
        radial_order: int = 12,
        angular_order: int = 48,
    ):
        """Positive quadrature on the unbounded background excluding objects."""
        if radial_order < 3:
            raise ValueError(
                "background radial order must be >= 3"
            )
        directions = _fibonacci_directions(
            int(
                angular_order
            )
        )
        rotation = np.asarray(
            self.scene.coils[
                0
            ].geometry.pose.rotation,
            dtype=float,
        )
        directions = (
            directions
            @ rotation.T
        )

        (
            node_positions,
            node_radii,
            _,
        ) = self._charge_geometry()
        surface_positions = np.asarray(
            self.teacher.surface_solver.positions,
            dtype=float,
        )
        geometry_points = np.concatenate(
            (
                np.asarray(
                    node_positions,
                    dtype=float,
                ),
                surface_positions,
            ),
            axis=0,
        )
        center = np.mean(
            geometry_points,
            axis=0,
        )
        scale = max(
            float(
                np.max(
                    np.linalg.norm(
                        geometry_points
                        - center[
                            None,
                            :
                        ],
                        axis=1,
                    )
                )
            ),
            float(
                np.max(
                    np.asarray(
                        node_radii,
                        dtype=float,
                    )
                )
            ),
            1e-6,
        )

        nodes, weights = (
            np.polynomial.legendre.leggauss(
                int(
                    radial_order
                )
            )
        )
        unit = 0.5 * (
            nodes
            + 1.0
        )
        unit_weights = (
            0.5
            * weights
        )
        radius = (
            scale
            * unit
            / (
                1.0
                - unit
            )
        )
        derivative = (
            scale
            / (
                1.0
                - unit
            ) ** 2
        )
        points = (
            center[
                None,
                None,
                :
            ]
            + radius[
                :,
                None,
                None,
            ]
            * directions[
                None,
                :,
                :
            ]
        ).reshape(
            -1,
            3,
        )
        volume_weights = (
            unit_weights[
                :,
                None
            ]
            * radius[
                :,
                None
            ] ** 2
            * derivative[
                :,
                None
            ]
            * (
                4.0
                * np.pi
                / int(
                    angular_order
                )
            )
            * np.ones(
                (
                    1,
                    int(
                        angular_order
                    ),
                ),
                dtype=float,
            )
        ).reshape(
            -1
        )
        exterior = (
            self._background_domain_mask(
                points
            )
        )
        return (
            points[
                exterior
            ],
            volume_weights[
                exterior
            ],
        )

    def raw_background_dissipation_matrices(
        self,
        points,
    ) -> np.ndarray:
        points = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            points.ndim == 1
        )
        points = np.atleast_2d(
            points
        )
        n_ports = int(
            self.port_prediction.impedance.shape[
                0
            ]
        )
        out = np.zeros(
            (
                len(
                    points
                ),
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )
        conductivity = (
            self.scene.medium.loss_conductivity(
                self.frequency_hz
            )
        )
        if conductivity <= 0.0:
            return (
                out[
                    0
                ]
                if scalar
                else out
            )
        exterior = (
            self._background_domain_mask(
                points
            )
        )
        if np.any(
            exterior
        ):
            transfer = (
                self.electric_field_transfer(
                    points[
                        exterior
                    ]
                )
            )
            if transfer.ndim == 2:
                transfer = transfer[
                    None,
                    :,
                    :
                ]
            matrices = (
                conductivity
                * np.einsum(
                    "qdi,qdj->qij",
                    transfer.conj(),
                    transfer,
                )
            )
            out[
                exterior
            ] = 0.5 * (
                matrices
                + matrices.conj().transpose(
                    0,
                    2,
                    1,
                )
            )
        return (
            out[
                0
            ]
            if scalar
            else out
        )

    def background_dissipation_matrices(
        self,
        points,
    ) -> np.ndarray:
        raw = (
            self.raw_background_dissipation_matrices(
                points
            )
        )
        scalar = (
            raw.ndim == 2
        )
        if scalar:
            raw = raw[
                None,
                :,
                :
            ]
        transform = (
            self.environment_transform
        )
        corrected = (
            transform[
                None,
                :,
                :
            ]
            @ raw
            @ transform.conj().T[
                None,
                :,
                :
            ]
        )
        corrected = 0.5 * (
            corrected
            + corrected.conj().transpose(
                0,
                2,
                1,
            )
        )
        return (
            corrected[
                0
            ]
            if scalar
            else corrected
        )

    def background_joule_density(
        self,
        points,
        currents,
    ):
        matrices = (
            self.background_dissipation_matrices(
                points
            )
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if currents.shape != (
            self.port_prediction.impedance.shape[
                0
            ],
        ):
            raise ValueError(
                "currents have wrong shape"
            )
        return 0.5 * np.real(
            np.einsum(
                "i,...ij,j->...",
                currents.conj(),
                matrices,
                currents,
            )
        )

    def raw_package_dissipation_matrices(
        self,
        package_index: int,
        points,
    ) -> np.ndarray:
        if not (
            0
            <= package_index
            < len(
                self.scene.packages
            )
        ):
            raise IndexError(
                "package_index out of range"
            )
        package = (
            self.scene.packages[
                package_index
            ]
        )
        points = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            points.ndim == 1
        )
        points = np.atleast_2d(
            points
        )
        inside = np.asarray(
            package.geometry.contains(
                points,
                tolerance=2e-12,
            ),
            dtype=bool,
        )
        n_ports = (
            self.port_prediction.impedance.shape[
                0
            ]
        )
        out = np.zeros(
            (
                len(points),
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )
        loss_conductivity = (
            package.material.loss_conductivity(
                self.frequency_hz
            )
        )
        if (
            loss_conductivity
            > 0.0
            and np.any(
                inside
            )
        ):
            transfer = (
                self.electric_field_transfer(
                    points[
                        inside
                    ]
                )
            )
            matrices = (
                loss_conductivity
                * np.einsum(
                    "qdi,qdj->qij",
                    transfer.conj(),
                    transfer,
                )
            )
            out[
                inside
            ] = 0.5 * (
                matrices
                + matrices.conj().transpose(
                    0,
                    2,
                    1,
                )
            )
        return (
            out[0]
            if scalar
            else out
        )

    def package_dissipation_matrices(
        self,
        package_index: int,
        points,
    ) -> np.ndarray:
        raw = (
            self.raw_package_dissipation_matrices(
                package_index,
                points,
            )
        )
        scalar = (
            raw.ndim == 2
        )
        if scalar:
            raw = raw[
                None,
                :,
                :,
            ]
        transform = (
            self.package_transform
        )
        corrected = (
            transform[
                None,
                :,
                :,
            ]
            @ raw
            @ transform.conj().T[
                None,
                :,
                :,
            ]
        )
        corrected = 0.5 * (
            corrected
            + corrected.conj().transpose(
                0,
                2,
                1,
            )
        )
        return (
            corrected[0]
            if scalar
            else corrected
        )

    def package_local_dissipation_matrix(
        self,
        package_index: int,
        local_position,
    ) -> np.ndarray:
        package = (
            self.scene.packages[
                package_index
            ]
        )
        local = np.asarray(
            local_position,
            dtype=float,
        )
        if local.shape != (3,):
            raise ValueError(
                "local_position must have shape (3,)"
            )
        world = (
            package.geometry.local_to_world(
                local
            )
        )
        return (
            self.package_dissipation_matrices(
                package_index,
                world,
            )
        )

    def package_local_joule_density(
        self,
        package_index: int,
        local_position,
        currents,
    ) -> float:
        matrix = (
            self.package_local_dissipation_matrix(
                package_index,
                local_position,
            )
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        return float(
            0.5
            * np.real(
                np.vdot(
                    currents,
                    matrix
                    @ currents,
                )
            )
        )


def prepare_hybrid_reference_loss_field(
    teacher,
    result,
    *,
    volume_axial_order: int = 8,
    volume_radial_order: int = 6,
    volume_azimuthal_order: int = 24,
    background_radial_order: int = 12,
    background_angular_order: int = 48,
    maximum_raw_closure_error: float = 0.25,
    normalized_closure_tolerance: float = 1e-6,
) -> PreparedHybridReferenceLossField:
    if maximum_raw_closure_error <= 0.0:
        raise ValueError(
            "maximum_raw_closure_error must be positive"
        )
    if normalized_closure_tolerance <= 0.0:
        raise ValueError(
            "normalized_closure_tolerance must be positive"
        )
    if (
        background_radial_order < 3
        or background_angular_order < 8
    ):
        raise ValueError(
            "invalid background spatial quadrature order"
        )

    scene = teacher.scene
    n_ports = (
        result.impedance.shape[
            0
        ]
    )
    identity = np.eye(
        n_ports,
        dtype=complex,
    )

    temporary = PreparedHybridReferenceLossField(
        scene=scene,
        frequency_hz=float(
            teacher.frequency_hz
        ),
        teacher=teacher,
        result=result,
        port_prediction=result.prediction,
        package_transform=identity,
        raw_dielectric_closure_error=0.0,
        normalized_dielectric_closure_error=0.0,
        package_integrated_channels=np.zeros(
            (
                len(
                    scene.packages
                ),
                n_ports,
                n_ports,
            ),
            dtype=complex,
        ),
        background_integrated_channel=np.zeros(
            (
                n_ports,
                n_ports,
            ),
            dtype=complex,
        ),
    )

    raw_package_channels = []
    package_quadratures = []
    for package_index, package in enumerate(
        scene.packages
    ):
        quadrature = (
            package.geometry.volume_quadrature(
                axial_order=(
                    volume_axial_order
                ),
                radial_order=(
                    volume_radial_order
                ),
                azimuthal_order=(
                    volume_azimuthal_order
                ),
            )
        )
        raw = (
            temporary.raw_package_dissipation_matrices(
                package_index,
                quadrature.positions,
            )
        )
        integrated = np.sum(
            quadrature.weights[
                :,
                None,
                None,
            ]
            * raw,
            axis=0,
        )
        raw_package_channels.append(
            0.5
            * (
                integrated
                + integrated.conj().T
            )
        )
        package_quadratures.append(
            quadrature
        )

    raw_package_channels = np.asarray(
        raw_package_channels,
        dtype=complex,
    )
    raw_background_channel = np.zeros(
        (
            n_ports,
            n_ports,
        ),
        dtype=complex,
    )
    if scene.medium.conductivity > 0.0:
        (
            background_points,
            background_weights,
        ) = temporary.background_quadrature(
            radial_order=(
                background_radial_order
            ),
            angular_order=(
                background_angular_order
            ),
        )
        raw_background = (
            temporary.raw_background_dissipation_matrices(
                background_points
            )
        )
        raw_background_channel = np.sum(
            background_weights[
                :,
                None,
                None,
            ]
            * raw_background,
            axis=0,
        )
        raw_background_channel = 0.5 * (
            raw_background_channel
            + raw_background_channel.conj().T
        )
    raw_total = (
        np.sum(
            raw_package_channels,
            axis=0,
        )
        + raw_background_channel
    )
    target = 0.5 * (
        result.dielectric_dissipation_matrix
        + result.dielectric_dissipation_matrix.conj().T
    )
    target_norm = max(
        float(
            np.linalg.norm(
                target
            )
        ),
        1e-30,
    )
    raw_error = float(
        np.linalg.norm(
            raw_total
            - target
        )
        / target_norm
    )

    if (
        np.linalg.norm(
            target
        )
        <= 1e-18
    ):
        if (
            np.linalg.norm(
                raw_total
            )
            > 1e-12
        ):
            raise RuntimeError(
                "spatial field reconstruction predicts electric-environment loss "
                "for a lossless port-level dielectric channel"
            )
        transform = identity
        corrected_package_channels = np.zeros_like(
            raw_package_channels
        )
        corrected_background_channel = np.zeros_like(
            raw_background_channel
        )
        normalized_error = 0.0
    else:
        if raw_error > maximum_raw_closure_error:
            raise RuntimeError(
                "raw electric-environment field integration does not close the port-level "
                f"dielectric loss channel: relative error={raw_error:.3e}"
            )
        scale = max(
            float(
                np.trace(
                    target
                ).real
                / max(
                    n_ports,
                    1,
                )
            ),
            target_norm
            / max(
                n_ports,
                1,
            ),
            1e-30,
        )
        regularization = (
            1e-12
            * scale
        )
        raw_regularized = (
            raw_total
            + regularization
            * identity
        )
        target_regularized = (
            target
            + regularization
            * identity
        )
        transform = (
            _hermitian_psd_sqrt(
                target_regularized,
                inverse=False,
            )
            @ _hermitian_psd_sqrt(
                raw_regularized,
                inverse=True,
            )
        )
        corrected_package_channels = np.asarray(
            [
                0.5
                * (
                    transform
                    @ channel
                    @ transform.conj().T
                    + (
                        transform
                        @ channel
                        @ transform.conj().T
                    ).conj().T
                )
                for channel
                in raw_package_channels
            ],
            dtype=complex,
        )
        corrected_background_channel = 0.5 * (
            transform
            @ raw_background_channel
            @ transform.conj().T
            + (
                transform
                @ raw_background_channel
                @ transform.conj().T
            ).conj().T
        )
        corrected_total = (
            np.sum(
                corrected_package_channels,
                axis=0,
            )
            + corrected_background_channel
        )
        normalized_error = float(
            np.linalg.norm(
                corrected_total
                - target
            )
            / target_norm
        )
        if (
            normalized_error
            > normalized_closure_tolerance
        ):
            raise RuntimeError(
                "normalized electric-environment spatial field failed power closure: "
                f"relative error={normalized_error:.3e}"
            )

    return PreparedHybridReferenceLossField(
        scene=scene,
        frequency_hz=float(
            teacher.frequency_hz
        ),
        teacher=teacher,
        result=result,
        port_prediction=result.prediction,
        package_transform=transform,
        raw_dielectric_closure_error=(
            raw_error
        ),
        normalized_dielectric_closure_error=(
            normalized_error
        ),
        package_integrated_channels=(
            corrected_package_channels
        ),
        background_integrated_channel=(
            corrected_background_channel
        ),
    )
