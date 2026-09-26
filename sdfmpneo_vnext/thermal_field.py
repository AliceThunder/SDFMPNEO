from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import erfc

from .basis import superellipse_section_quadrature
from .scene import Scene


@dataclass(frozen=True)
class HomogeneousThermalMedium:
    conductivity: float
    density: float
    heat_capacity: float
    ambient_temperature: float = 293.15

    def __post_init__(self):
        if (
            not np.isfinite(self.conductivity)
            or self.conductivity <= 0.0
            or not np.isfinite(self.density)
            or self.density <= 0.0
            or not np.isfinite(self.heat_capacity)
            or self.heat_capacity <= 0.0
            or not np.isfinite(self.ambient_temperature)
        ):
            raise ValueError(
                "thermal-medium properties must be finite and positive"
            )

    @property
    def volumetric_heat_capacity(
        self,
    ) -> float:
        return float(
            self.density
            * self.heat_capacity
        )

    @property
    def diffusivity(
        self,
    ) -> float:
        return float(
            self.conductivity
            / self.volumetric_heat_capacity
        )


@dataclass(frozen=True)
class ThermalSourceQuadrature:
    positions: np.ndarray
    volume_weights: np.ndarray
    coil_index: np.ndarray
    arc_fraction: np.ndarray
    xy: np.ndarray
    dissipation_matrices: np.ndarray
    effective_radius: np.ndarray
    normalization_closure_error: float
    normalization_correction: float

    def __post_init__(self):
        positions = np.asarray(
            self.positions,
            dtype=float,
        )
        weights = np.asarray(
            self.volume_weights,
            dtype=float,
        )
        coil_index = np.asarray(
            self.coil_index,
            dtype=int,
        )
        arc = np.asarray(
            self.arc_fraction,
            dtype=float,
        )
        xy = np.asarray(
            self.xy,
            dtype=float,
        )
        matrices = np.asarray(
            self.dissipation_matrices,
            dtype=complex,
        )
        radius = np.asarray(
            self.effective_radius,
            dtype=float,
        )
        n = len(weights)
        if (
            positions.shape != (n, 3)
            or coil_index.shape != (n,)
            or arc.shape != (n,)
            or xy.shape != (n, 2)
            or radius.shape != (n,)
            or matrices.ndim != 3
            or matrices.shape[0] != n
            or matrices.shape[1]
            != matrices.shape[2]
        ):
            raise ValueError(
                "thermal source quadrature has incompatible shapes"
            )
        if (
            np.any(weights <= 0.0)
            or np.any(radius <= 0.0)
        ):
            raise ValueError(
                "thermal quadrature weights/radii must be positive"
            )
        object.__setattr__(
            self,
            "positions",
            positions,
        )
        object.__setattr__(
            self,
            "volume_weights",
            weights,
        )
        object.__setattr__(
            self,
            "coil_index",
            coil_index,
        )
        object.__setattr__(
            self,
            "arc_fraction",
            arc,
        )
        object.__setattr__(
            self,
            "xy",
            xy,
        )
        object.__setattr__(
            self,
            "dissipation_matrices",
            matrices,
        )
        object.__setattr__(
            self,
            "effective_radius",
            radius,
        )

    @property
    def n_ports(
        self,
    ) -> int:
        return int(
            self.dissipation_matrices.shape[
                1
            ]
        )

    @property
    def channel_index(
        self,
    ) -> np.ndarray:
        return self.coil_index

    @property
    def n_channels(
        self,
    ) -> int:
        return int(
            np.max(
                self.coil_index
            )
            + 1
        )

    @property
    def n_coils(
        self,
    ) -> int:
        # Backward-compatible alias; source ids are generic loss-channel ids.
        return self.n_channels

    def integrated_channels(
        self,
    ) -> np.ndarray:
        out = np.zeros(
            (
                self.n_coils,
                self.n_ports,
                self.n_ports,
            ),
            dtype=complex,
        )
        for coil in range(
            self.n_coils
        ):
            mask = (
                self.coil_index
                == coil
            )
            out[
                coil
            ] = np.sum(
                self.volume_weights[
                    mask,
                    None,
                    None,
                ]
                * self.dissipation_matrices[
                    mask
                ],
                axis=0,
            )
        return out

    def joule_density(
        self,
        currents,
    ) -> np.ndarray:
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if currents.shape != (
            self.n_ports,
        ):
            raise ValueError(
                "currents have wrong shape"
            )
        return 0.5 * np.real(
            np.einsum(
                "i,qij,j->q",
                currents.conj(),
                self.dissipation_matrices,
                currents,
            )
        )

    def channel_power(
        self,
        currents,
    ) -> np.ndarray:
        density = self.joule_density(
            currents
        )
        out = np.zeros(
            self.n_channels,
            dtype=float,
        )
        np.add.at(
            out,
            self.coil_index,
            self.volume_weights
            * density,
        )
        return out

    def coil_power(
        self,
        currents,
    ) -> np.ndarray:
        # Backward-compatible alias for conductor-only callers.
        return self.channel_power(
            currents
        )



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
    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            matrix
        )
    )
    scale = max(
        float(
            np.max(
                np.abs(
                    eigenvalues
                )
            )
        ),
        1e-30,
    )
    if inverse:
        values = 1.0 / np.sqrt(
            np.maximum(
                eigenvalues.real,
                1e-14 * scale,
            )
        )
    else:
        values = np.sqrt(
            np.maximum(
                eigenvalues.real,
                0.0,
            )
        )
    return (
        eigenvectors
        @ np.diag(
            values
        )
        @ eigenvectors.conj().T
    )


def _normalize_source_matrices(
    matrices,
    weights,
    coil_index,
    target_channels,
):
    matrices = np.asarray(
        matrices,
        dtype=complex,
    ).copy()
    weights = np.asarray(
        weights,
        dtype=float,
    )
    coil_index = np.asarray(
        coil_index,
        dtype=int,
    )
    target_channels = np.asarray(
        target_channels,
        dtype=complex,
    )
    raw_copy = matrices.copy()
    n_ports = matrices.shape[1]
    identity = np.eye(
        n_ports,
        dtype=complex,
    )

    for coil in range(
        target_channels.shape[0]
    ):
        mask = (
            coil_index == coil
        )
        if not np.any(mask):
            raise ValueError(
                "thermal quadrature is missing a coil"
            )
        target = 0.5 * (
            target_channels[
                coil
            ]
            + target_channels[
                coil
            ].conj().T
        )
        if (
            np.linalg.norm(
                target
            )
            <= 1e-18
        ):
            matrices[
                mask
            ] = 0.0
            continue
        volume = float(
            np.sum(
                weights[
                    mask
                ]
            )
        )
        raw_integral = np.sum(
            weights[
                mask,
                None,
                None,
            ]
            * matrices[
                mask
            ],
            axis=0,
        )
        target_scale = max(
            float(
                np.trace(
                    target
                ).real
                / max(
                    n_ports,
                    1,
                )
            ),
            float(
                np.linalg.norm(
                    target
                )
                / max(
                    n_ports,
                    1,
                )
            ),
            1e-30,
        )
        regularization = (
            1e-12
            * target_scale
        )
        matrices[
            mask
        ] += (
            regularization
            / max(
                volume,
                1e-30,
            )
        ) * identity[
            None,
            :,
            :,
        ]
        raw_integral = (
            raw_integral
            + regularization
            * identity
        )
        transform = (
            _hermitian_psd_sqrt(
                target,
                inverse=False,
            )
            @ _hermitian_psd_sqrt(
                raw_integral,
                inverse=True,
            )
        )
        values = (
            transform[
                None,
                :,
                :,
            ]
            @ matrices[
                mask
            ]
            @ transform.conj().T[
                None,
                :,
                :,
            ]
        )
        matrices[
            mask
        ] = 0.5 * (
            values
            + values.conj().transpose(
                0,
                2,
                1,
            )
        )

    normalized = np.zeros_like(
        target_channels,
        dtype=complex,
    )
    for coil in range(
        target_channels.shape[0]
    ):
        mask = (
            coil_index == coil
        )
        normalized[
            coil
        ] = np.sum(
            weights[
                mask,
                None,
                None,
            ]
            * matrices[
                mask
            ],
            axis=0,
        )
    closure = float(
        np.linalg.norm(
            normalized
            - target_channels
        )
        / max(
            np.linalg.norm(
                target_channels
            ),
            1e-30,
        )
    )
    correction = float(
        np.linalg.norm(
            matrices
            - raw_copy
        )
        / max(
            np.linalg.norm(
                matrices
            ),
            1e-30,
        )
    )
    return (
        matrices,
        closure,
        correction,
    )


def build_thermal_source_quadrature(
    scene: Scene,
    prepared_spatial,
    *,
    longitudinal_segments: int = 24,
    radial_order: int = 4,
    angular_order: int = 24,
    package_axial_order: int | None = None,
    package_radial_order: int | None = None,
    package_azimuthal_order: int | None = None,
    background_radial_order: int | None = None,
    background_angular_order: int | None = None,
) -> ThermalSourceQuadrature:
    if (
        longitudinal_segments < 2
        or radial_order < 2
        or angular_order < 8
    ):
        raise ValueError(
            "invalid thermal source quadrature order"
        )
    if (
        background_radial_order is not None
        and int(
            background_radial_order
        ) < 3
    ):
        raise ValueError(
            "background radial order must be >= 3"
        )
    if (
        background_angular_order is not None
        and int(
            background_angular_order
        ) < 8
    ):
        raise ValueError(
            "background angular order must be >= 8"
        )
    if not hasattr(
        prepared_spatial,
        "local_dissipation_matrix",
    ):
        raise TypeError(
            "prepared_spatial must expose local dissipation queries"
        )
    if not hasattr(
        prepared_spatial,
        "port_prediction",
    ):
        raise TypeError(
            "prepared_spatial must expose port_prediction"
        )

    conductor_positions = []
    conductor_weights = []
    conductor_ids = []
    conductor_arcs = []
    conductor_xy = []

    for coil_index, coil in enumerate(
        scene.coils
    ):
        geometry = (
            coil.geometry
        )
        poly = geometry.polyline(
            longitudinal_segments
        )
        section = (
            superellipse_section_quadrature(
                geometry.conductor_width,
                geometry.conductor_thickness,
                geometry.cross_section_exponent,
                radial_order=radial_order,
                angular_order=angular_order,
            )
        )
        n_section = len(
            section.weights
        )
        for segment in range(
            longitudinal_segments
        ):
            center = poly.midpoints[
                segment
            ]
            points = (
                center[
                    None,
                    :
                ]
                + section.xy[
                    :,
                    0,
                    None,
                ]
                * poly.normal1[
                    segment
                ][
                    None,
                    :
                ]
                + section.xy[
                    :,
                    1,
                    None,
                ]
                * poly.normal2[
                    segment
                ][
                    None,
                    :
                ]
            )
            conductor_positions.append(
                points
            )
            conductor_weights.append(
                section.weights
                * poly.lengths[
                    segment
                ]
            )
            conductor_ids.append(
                np.full(
                    n_section,
                    coil_index,
                    dtype=int,
                )
            )
            conductor_arcs.append(
                np.full(
                    n_section,
                    (
                        segment
                        + 0.5
                    )
                    / longitudinal_segments,
                    dtype=float,
                )
            )
            conductor_xy.append(
                section.xy
            )

    positions = np.concatenate(
        conductor_positions,
        axis=0,
    )
    weights = np.concatenate(
        conductor_weights
    )
    channel_ids = np.concatenate(
        conductor_ids
    )
    arcs = np.concatenate(
        conductor_arcs
    )
    section_xy = np.concatenate(
        conductor_xy,
        axis=0,
    )

    if hasattr(
        prepared_spatial,
        "local_dissipation_matrices",
    ):
        matrices = (
            prepared_spatial.local_dissipation_matrices(
                channel_ids,
                arcs,
                section_xy,
            )
        )
    else:
        matrices = np.asarray(
            [
                prepared_spatial.local_dissipation_matrix(
                    int(coil),
                    float(arc),
                    xy,
                )
                for coil, arc, xy
                in zip(
                    channel_ids,
                    arcs,
                    section_xy,
                )
            ],
            dtype=complex,
        )

    # Package-aware REFERENCE fields append dielectric volume sources to the
    # aggregate dielectric loss channel. Their actual world positions are
    # retained, so subsequent thermal Green evaluation resolves package heat
    # spatially even though the port model exposes one aggregate dielectric
    # channel.
    if scene.packages:
        if not hasattr(
            prepared_spatial,
            "package_dissipation_matrices",
        ):
            raise TypeError(
                "package scenes require prepared_spatial package dissipation queries"
            )
        dielectric_channel = int(
            getattr(
                prepared_spatial,
                "dielectric_channel_index",
                len(
                    scene.coils
                ),
            )
        )
        package_axial = (
            max(
                4,
                longitudinal_segments
                // 2,
            )
            if package_axial_order
            is None
            else int(
                package_axial_order
            )
        )
        package_radial = (
            radial_order
            if package_radial_order
            is None
            else int(
                package_radial_order
            )
        )
        package_azimuthal = (
            angular_order
            if package_azimuthal_order
            is None
            else int(
                package_azimuthal_order
            )
        )
        if (
            package_axial < 2
            or package_radial < 2
            or package_azimuthal < 8
        ):
            raise ValueError(
                "invalid package thermal source quadrature order"
            )

        extra_positions = []
        extra_weights = []
        extra_ids = []
        extra_arcs = []
        extra_xy = []
        extra_matrices = []
        for package_index, package in enumerate(
            scene.packages
        ):
            quadrature = (
                package.geometry.volume_quadrature(
                    axial_order=(
                        package_axial
                    ),
                    radial_order=(
                        package_radial
                    ),
                    azimuthal_order=(
                        package_azimuthal
                    ),
                )
            )
            count = len(
                quadrature.weights
            )
            extra_positions.append(
                quadrature.positions
            )
            extra_weights.append(
                quadrature.weights
            )
            extra_ids.append(
                np.full(
                    count,
                    dielectric_channel,
                    dtype=int,
                )
            )
            # These fields are retained only for backward-compatible source
            # metadata; package heat queries use world positions directly.
            extra_arcs.append(
                np.zeros(
                    count,
                    dtype=float,
                )
            )
            extra_xy.append(
                np.zeros(
                    (
                        count,
                        2,
                    ),
                    dtype=float,
                )
            )
            extra_matrices.append(
                prepared_spatial.package_dissipation_matrices(
                    package_index,
                    quadrature.positions,
                )
            )

        positions = np.concatenate(
            (
                positions,
                np.concatenate(
                    extra_positions,
                    axis=0,
                ),
            ),
            axis=0,
        )
        weights = np.concatenate(
            (
                weights,
                np.concatenate(
                    extra_weights
                ),
            )
        )
        channel_ids = np.concatenate(
            (
                channel_ids,
                np.concatenate(
                    extra_ids
                ),
            )
        )
        arcs = np.concatenate(
            (
                arcs,
                np.concatenate(
                    extra_arcs
                ),
            )
        )
        section_xy = np.concatenate(
            (
                section_xy,
                np.concatenate(
                    extra_xy,
                    axis=0,
                ),
            ),
            axis=0,
        )
        matrices = np.concatenate(
            (
                np.asarray(
                    matrices,
                    dtype=complex,
                ),
                np.concatenate(
                    extra_matrices,
                    axis=0,
                ),
            ),
            axis=0,
        )

    if (
        scene.medium.loss_conductivity(
            prepared_spatial.frequency_hz
        )
        > 0.0
    ):
        required = (
            "background_quadrature",
            "background_dissipation_matrices",
            "background_channel_index",
        )
        missing = tuple(
            name
            for name in required
            if not hasattr(
                prepared_spatial,
                name,
            )
        )
        if missing:
            raise NotImplementedError(
                "lossy homogeneous-background thermal coupling requires "
                "a spatial artifact with an explicit continuous background "
                "loss field; missing "
                + ", ".join(
                    missing
                )
            )
        background_channel = (
            prepared_spatial.background_channel_index
        )
        if background_channel is None:
            raise RuntimeError(
                "lossy background did not expose its dissipation channel"
            )
        background_radial = (
            12
            if background_radial_order
            is None
            else int(
                background_radial_order
            )
        )
        background_angular = (
            48
            if background_angular_order
            is None
            else int(
                background_angular_order
            )
        )
        (
            background_positions,
            background_weights,
        ) = (
            prepared_spatial.background_quadrature(
                radial_order=(
                    background_radial
                ),
                angular_order=(
                    background_angular
                ),
            )
        )
        background_matrices = (
            prepared_spatial.background_dissipation_matrices(
                background_positions
            )
        )
        background_count = len(
            background_weights
        )
        positions = np.concatenate(
            (
                positions,
                np.asarray(
                    background_positions,
                    dtype=float,
                ),
            ),
            axis=0,
        )
        weights = np.concatenate(
            (
                weights,
                np.asarray(
                    background_weights,
                    dtype=float,
                ),
            )
        )
        channel_ids = np.concatenate(
            (
                channel_ids,
                np.full(
                    background_count,
                    int(
                        background_channel
                    ),
                    dtype=int,
                ),
            )
        )
        arcs = np.concatenate(
            (
                arcs,
                np.zeros(
                    background_count,
                    dtype=float,
                ),
            )
        )
        section_xy = np.concatenate(
            (
                section_xy,
                np.zeros(
                    (
                        background_count,
                        2,
                    ),
                    dtype=float,
                ),
            ),
            axis=0,
        )
        matrices = np.concatenate(
            (
                np.asarray(
                    matrices,
                    dtype=complex,
                ),
                np.asarray(
                    background_matrices,
                    dtype=complex,
                ),
            ),
            axis=0,
        )

    target_channels = np.asarray(
        prepared_spatial.port_prediction.dissipation_channels,
        dtype=complex,
    )
    expected_channels = int(
        np.max(
            channel_ids
        )
        + 1
    )
    if (
        target_channels.shape[
            0
        ]
        != expected_channels
    ):
        raise ValueError(
            "thermal source channel ids do not match structured port channels"
        )
    (
        matrices,
        closure,
        correction,
    ) = _normalize_source_matrices(
        matrices,
        weights,
        channel_ids,
        target_channels,
    )
    effective_radius = (
        3.0
        * weights
        / (
            4.0
            * np.pi
        )
    ) ** (
        1.0
        / 3.0
    )
    return ThermalSourceQuadrature(
        positions,
        weights,
        channel_ids,
        arcs,
        section_xy,
        matrices,
        effective_radius,
        closure,
        correction,
    )


@dataclass(frozen=True)
class PreparedThermalGreenField:
    source: ThermalSourceQuadrature
    medium: HomogeneousThermalMedium

    def _query_points(
        self,
        points,
    ):
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
        if points.shape[1] != 3:
            raise ValueError(
                "query points must have shape (3,) or (n,3)"
            )
        return (
            points,
            scalar,
        )

    def _distance(
        self,
        points,
    ):
        difference = (
            points[
                :,
                None,
                :
            ]
            - self.source.positions[
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
        return np.sqrt(
            distance_squared
            + self.source.effective_radius[
                None,
                :
            ] ** 2
        )

    def step_response_matrix(
        self,
        points,
        time: float,
    ) -> np.ndarray:
        points, scalar = (
            self._query_points(
                points
            )
        )
        n_ports = (
            self.source.n_ports
        )
        if time < 0.0:
            raise ValueError(
                "time must be nonnegative"
            )
        if time == 0.0:
            out = np.zeros(
                (
                    len(points),
                    n_ports,
                    n_ports,
                ),
                dtype=complex,
            )
            return (
                out[0]
                if scalar
                else out
            )

        distance = self._distance(
            points
        )
        alpha = (
            self.medium.diffusivity
        )
        coefficient = (
            erfc(
                distance
                / (
                    2.0
                    * np.sqrt(
                        alpha
                        * time
                    )
                )
            )
            / (
                4.0
                * np.pi
                * self.medium.conductivity
                * distance
            )
        )
        weighted = (
            coefficient
            * self.source.volume_weights[
                None,
                :
            ]
        )
        out = np.einsum(
            "pq,qij->pij",
            weighted,
            self.source.dissipation_matrices,
        )
        out = 0.5 * (
            out
            + out.conj().transpose(
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

    def steady_response_matrix(
        self,
        points,
    ) -> np.ndarray:
        points, scalar = (
            self._query_points(
                points
            )
        )
        distance = self._distance(
            points
        )
        coefficient = (
            1.0
            / (
                4.0
                * np.pi
                * self.medium.conductivity
                * distance
            )
        )
        weighted = (
            coefficient
            * self.source.volume_weights[
                None,
                :
            ]
        )
        out = np.einsum(
            "pq,qij->pij",
            weighted,
            self.source.dissipation_matrices,
        )
        out = 0.5 * (
            out
            + out.conj().transpose(
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

    @staticmethod
    def _contract(
        matrices,
        currents,
    ):
        matrices = np.asarray(
            matrices,
            dtype=complex,
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        return 0.5 * np.real(
            np.einsum(
                "i,...ij,j->...",
                currents.conj(),
                matrices,
                currents,
            )
        )

    def temperature_step(
        self,
        points,
        time: float,
        currents,
    ):
        rise = self._contract(
            self.step_response_matrix(
                points,
                time,
            ),
            currents,
        )
        return (
            self.medium.ambient_temperature
            + rise
        )

    def steady_temperature(
        self,
        points,
        currents,
    ):
        rise = self._contract(
            self.steady_response_matrix(
                points
            ),
            currents,
        )
        return (
            self.medium.ambient_temperature
            + rise
        )

    def temperature_history(
        self,
        points,
        interval_edges,
        interval_currents,
        observation_times,
    ) -> np.ndarray:
        edges = np.asarray(
            interval_edges,
            dtype=float,
        )
        currents = np.asarray(
            interval_currents,
            dtype=complex,
        )
        times = np.asarray(
            observation_times,
            dtype=float,
        )
        if (
            edges.ndim != 1
            or len(edges) < 2
            or np.any(
                np.diff(
                    edges
                )
                <= 0.0
            )
        ):
            raise ValueError(
                "interval_edges must be strictly increasing"
            )
        if currents.shape != (
            len(edges) - 1,
            self.source.n_ports,
        ):
            raise ValueError(
                "interval_currents have wrong shape"
            )
        if (
            times.ndim != 1
            or np.any(
                ~np.isfinite(
                    times
                )
            )
        ):
            raise ValueError(
                "observation_times must be a finite vector"
            )

        query, scalar = (
            self._query_points(
                points
            )
        )
        result = np.full(
            (
                len(times),
                len(query),
            ),
            self.medium.ambient_temperature,
            dtype=float,
        )
        for time_index, time in enumerate(
            times
        ):
            for interval in range(
                len(edges) - 1
            ):
                start = edges[
                    interval
                ]
                end = edges[
                    interval + 1
                ]
                if time <= start:
                    continue
                age_start = (
                    time - start
                )
                age_end = max(
                    time - end,
                    0.0,
                )
                response = (
                    self.step_response_matrix(
                        query,
                        age_start,
                    )
                )
                if age_end > 0.0:
                    response = (
                        response
                        - self.step_response_matrix(
                            query,
                            age_end,
                        )
                    )
                result[
                    time_index
                ] += self._contract(
                    response,
                    currents[
                        interval
                    ],
                )
        if scalar:
            return result[
                :,
                0
            ]
        return result


class ContinuousThermalGreenArtifact:
    """Mesh-free continuous thermal transfer for a homogeneous infinite medium."""

    def __init__(
        self,
        spatial_artifact,
        medium: HomogeneousThermalMedium,
        *,
        longitudinal_segments: int = 24,
        radial_order: int = 4,
        angular_order: int = 24,
        background_radial_order: int | None = None,
        background_angular_order: int | None = None,
        spatial_prepare_options=None,
    ):
        if not (
            hasattr(
                spatial_artifact,
                "prepare",
            )
            or hasattr(
                spatial_artifact,
                "prepare_spatial",
            )
        ):
            raise TypeError(
                "spatial_artifact must expose prepare or prepare_spatial"
            )
        self.spatial_artifact = (
            spatial_artifact
        )
        self.medium = medium
        self.longitudinal_segments = int(
            longitudinal_segments
        )
        self.radial_order = int(
            radial_order
        )
        self.angular_order = int(
            angular_order
        )
        self.background_radial_order = (
            None
            if background_radial_order
            is None
            else int(
                background_radial_order
            )
        )
        self.background_angular_order = (
            None
            if background_angular_order
            is None
            else int(
                background_angular_order
            )
        )
        self.spatial_prepare_options = (
            {}
            if spatial_prepare_options is None
            else dict(
                spatial_prepare_options
            )
        )

    def prepare(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedThermalGreenField:
        if hasattr(
            self.spatial_artifact,
            "prepare",
        ):
            spatial = (
                self.spatial_artifact.prepare(
                    scene,
                    frequency_hz,
                    **self.spatial_prepare_options,
                )
            )
        else:
            spatial = (
                self.spatial_artifact.prepare_spatial(
                    scene,
                    frequency_hz,
                    **self.spatial_prepare_options,
                )
            )
        source = (
            build_thermal_source_quadrature(
                scene,
                spatial,
                longitudinal_segments=(
                    self.longitudinal_segments
                ),
                radial_order=(
                    self.radial_order
                ),
                angular_order=(
                    self.angular_order
                ),
                background_radial_order=(
                    self.background_radial_order
                ),
                background_angular_order=(
                    self.background_angular_order
                ),
            )
        )
        return PreparedThermalGreenField(
            source,
            self.medium,
        )
