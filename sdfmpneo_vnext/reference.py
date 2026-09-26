from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .em import MQSConfig
from .mixed import DenseMixedConductorTeacher
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
    if inverse:
        diagonal = 1.0 / np.sqrt(
            np.maximum(
                values.real,
                1e-14 * scale,
            )
        )
    else:
        diagonal = np.sqrt(
            np.maximum(
                values.real,
                0.0,
            )
        )
    return (
        vectors
        @ np.diag(
            diagonal
        )
        @ vectors.conj().T
    )


def _channel_congruence_transform(
    raw_integral,
    target,
):
    raw_integral = np.asarray(
        raw_integral,
        dtype=complex,
    )
    raw_integral = 0.5 * (
        raw_integral
        + raw_integral.conj().T
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    target = 0.5 * (
        target
        + target.conj().T
    )
    target_norm = float(
        np.linalg.norm(
            target
        )
    )
    if target_norm <= 1e-18:
        return np.zeros_like(
            target,
            dtype=complex,
        )
    n = target.shape[
        0
    ]
    scale = max(
        float(
            np.trace(
                target
            ).real
            / max(
                n,
                1,
            )
        ),
        target_norm
        / max(
            n,
            1,
        ),
        1e-30,
    )
    regularization = (
        1e-12
        * scale
    )
    identity = np.eye(
        n,
        dtype=complex,
    )
    return (
        _hermitian_psd_sqrt(
            target
            + regularization
            * identity,
            inverse=False,
        )
        @ _hermitian_psd_sqrt(
            raw_integral
            + regularization
            * identity,
            inverse=True,
        )
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
class PreparedReferenceLossField:
    scene: Scene
    frequency_hz: float
    teacher: DenseMixedConductorTeacher
    result: object
    port_prediction: StructuredPortPrediction
    background_transform: np.ndarray | None = None
    raw_background_closure_error: float = 0.0
    normalized_background_closure_error: float = 0.0

    @property
    def normalization_closure_error(
        self,
    ) -> float:
        return float(
            max(
                self.port_prediction.power_closure_error(),
                self.normalized_background_closure_error,
            )
        )

    @property
    def background_channel_index(
        self,
    ) -> int | None:
        if (
            self.result.background_dissipation_matrix
            is None
        ):
            return None
        return len(
            self.scene.coils
        )

    def local_dissipation_matrix(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return (
            self.teacher.local_dissipation_matrix(
                self.result,
                coil_index,
                arc_fraction,
                xy,
            )
        )

    def local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        matrix = (
            self.local_dissipation_matrix(
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
        ) = self.teacher._mqs.assemble()
        (
            _,
            _,
            _,
            positions,
            radii,
        ) = self.teacher._topology(
            current_constraint
        )
        return (
            np.asarray(
                positions,
                dtype=float,
            ),
            np.asarray(
                radii,
                dtype=float,
            ),
        )

    def _background_domain_mask(
        self,
        points,
    ) -> np.ndarray:
        """Return points that belong to the homogeneous exterior medium.

        The exterior loss integral must exclude conductor volume.  We use the
        same longitudinal segment/Bishop-frame representation as the mixed
        teacher, so the exclusion follows arbitrary pose and finite
        superelliptic conductor cross-sections without introducing a world
        voxel grid.
        """
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
        for segment in self.teacher._mqs._segments:
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
            coil = self.scene.coils[
                segment.coil
            ]
            geometry = coil.geometry
            half_width = (
                0.5
                * geometry.conductor_width
            )
            half_thickness = (
                0.5
                * geometry.conductor_thickness
            )
            exponent = float(
                geometry.cross_section_exponent
            )
            u = (
                transverse
                @ segment.n1
            ) / half_width
            v = (
                transverse
                @ segment.n2
            ) / half_thickness
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

    def background_quadrature(
        self,
        *,
        radial_order: int = 12,
        angular_order: int = 48,
    ):
        """Positive quadrature over the unbounded homogeneous background.

        The radial map r=s*x/(1-x) integrates [0,infinity) without a world
        truncation box. Angular directions are transported by the first coil
        pose so a common rigid transform rotates/translates the quadrature
        instead of changing an arbitrary world-grid orientation.
        """
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

        positions, radii = (
            self._charge_geometry()
        )
        # The integration origin must itself be SE(3)-equivariant. An
        # axis-aligned bounding-box centre is not rotation equivariant; the
        # charge-node centroid is.
        center = np.mean(
            positions,
            axis=0,
        )
        scale = max(
            float(
                np.max(
                    np.linalg.norm(
                        positions
                        - center[
                            None,
                            :
                        ],
                        axis=1,
                    )
                    + radii
                )
            ),
            4.0
            * float(
                np.max(
                    radii
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
        )
        points = points.reshape(
            -1,
            3,
        )
        volume_weights = (
            volume_weights.reshape(
                -1
            )
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

    def electric_field_transfer(
        self,
        points,
    ) -> np.ndarray:
        """Return scalar-potential background E transfer as (n,3,n_ports)."""
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
            or points.shape[
                1
            ] != 3
        ):
            raise ValueError(
                "points must have shape (3,) or (n,3)"
            )
        positions, radii = (
            self._charge_geometry()
        )
        epsilon = (
            self.scene.medium.complex_permittivity(
                self.frequency_hz
            )
        )
        difference = (
            points[
                :,
                None,
                :
            ]
            - positions[
                None,
                :,
                :
            ]
        )
        soft = (
            self.teacher.charge_self_radius_factor
            * radii[
                None,
                :
            ]
        )
        distance2 = (
            np.sum(
                difference
                * difference,
                axis=2,
            )
            + soft**2
        )
        kernel = (
            difference
            / (
                4.0
                * np.pi
                * epsilon
                * distance2[
                    :,
                    :,
                    None,
                ] ** 1.5
            )
        )
        transfer = np.einsum(
            "qjd,jp->qdp",
            kernel,
            self.result.node_charge,
        )
        return (
            transfer[
                0
            ]
            if scalar
            else transfer
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
        if (
            self.result.background_dissipation_matrix
            is None
        ):
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
            return (
                out[
                    0
                ]
                if scalar
                else out
            )
        transfer = (
            self.electric_field_transfer(
                points
            )
        )
        if transfer.ndim == 2:
            transfer = (
                transfer[
                    None,
                    :,
                    :
                ]
            )
        conductivity = (
            self.scene.medium.loss_conductivity(
                self.frequency_hz
            )
        )
        matrices = (
            conductivity
            * np.einsum(
                "qdi,qdj->qij",
                transfer.conj(),
                transfer,
            )
        )
        matrices = 0.5 * (
            matrices
            + matrices.conj().transpose(
                0,
                2,
                1,
            )
        )
        return (
            matrices[
                0
            ]
            if scalar
            else matrices
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
        if (
            self.result.background_dissipation_matrix
            is None
        ):
            corrected = raw
        else:
            if self.background_transform is None:
                raise RuntimeError(
                    "lossy-background spatial field is missing its "
                    "power-normalization transform"
                )
            transform = np.asarray(
                self.background_transform,
                dtype=complex,
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


class MixedReferenceArtifact:
    """Canonical mesh-free current-potential-charge REFERENCE artifact."""

    def __init__(
        self,
        *,
        config: MQSConfig | None = None,
        background_radial_order: int = 12,
        background_angular_order: int = 48,
    ):
        if (
            background_radial_order < 3
            or background_angular_order < 8
        ):
            raise ValueError(
                "invalid lossy-background spatial quadrature order"
            )
        self.config = (
            config
            or MQSConfig()
        )
        self.background_radial_order = int(
            background_radial_order
        )
        self.background_angular_order = int(
            background_angular_order
        )

    def solve(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        teacher = (
            DenseMixedConductorTeacher(
                scene,
                frequency_hz,
                self.config,
            )
        )
        return (
            teacher,
            teacher.solve(),
        )

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> StructuredPortPrediction:
        _, result = self.solve(
            scene,
            frequency_hz,
        )
        return StructuredPortPrediction(
            result.impedance,
            result.dissipation_channels(),
        )

    def predict(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> np.ndarray:
        return (
            self.predict_structured(
                scene,
                frequency_hz,
            ).impedance
        )

    def prepare_spatial(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedReferenceLossField:
        teacher, result = self.solve(
            scene,
            frequency_hz,
        )
        prediction = (
            StructuredPortPrediction(
                result.impedance,
                result.dissipation_channels(),
            )
        )
        provisional = (
            PreparedReferenceLossField(
                scene,
                float(
                    frequency_hz
                ),
                teacher,
                result,
                prediction,
            )
        )
        target = (
            result.background_dissipation_matrix
        )
        if target is None:
            return provisional

        points, weights = (
            provisional.background_quadrature(
                radial_order=(
                    self.background_radial_order
                ),
                angular_order=(
                    self.background_angular_order
                ),
            )
        )
        raw = (
            provisional.raw_background_dissipation_matrices(
                points
            )
        )
        raw_integral = np.sum(
            weights[
                :,
                None,
                None,
            ]
            * raw,
            axis=0,
        )
        target = np.asarray(
            target,
            dtype=complex,
        )
        raw_error = float(
            np.linalg.norm(
                raw_integral
                - target
            )
            / max(
                np.linalg.norm(
                    target
                ),
                1e-30,
            )
        )
        transform = (
            _channel_congruence_transform(
                raw_integral,
                target,
            )
        )
        normalized = (
            transform
            @ raw_integral
            @ transform.conj().T
        )
        normalized = 0.5 * (
            normalized
            + normalized.conj().T
        )
        normalized_error = float(
            np.linalg.norm(
                normalized
                - target
            )
            / max(
                np.linalg.norm(
                    target
                ),
                1e-30,
            )
        )
        return PreparedReferenceLossField(
            scene,
            float(
                frequency_hz
            ),
            teacher,
            result,
            prediction,
            transform,
            raw_error,
            normalized_error,
        )
