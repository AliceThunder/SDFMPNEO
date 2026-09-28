from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log
import numpy as np


@dataclass(frozen=True)
class _ThermalRegion:
    package_index: int
    geometry: object
    conductivity: float
    volumetric_heat_capacity: float
    boundary_positions: np.ndarray
    boundary_normals: np.ndarray
    interior_sources: np.ndarray
    exterior_sources: np.ndarray
    source_mask: np.ndarray
    boundary_slice: slice


def _material_thermal_properties(
    material,
):
    values = (
        getattr(
            material,
            "thermal_conductivity",
            None,
        ),
        getattr(
            material,
            "density",
            None,
        ),
        getattr(
            material,
            "heat_capacity",
            None,
        ),
    )
    if all(
        value is None
        for value in values
    ):
        return None
    if any(
        value is None
        for value in values
    ):
        raise ValueError(
            "package thermal_conductivity, density, and heat_capacity "
            "must be supplied together"
        )
    conductivity, density, heat_capacity = (
        float(
            value
        )
        for value in values
    )
    if (
        conductivity <= 0.0
        or density <= 0.0
        or heat_capacity <= 0.0
        or not np.all(
            np.isfinite(
                (
                    conductivity,
                    density,
                    heat_capacity,
                )
            )
        )
    ):
        raise ValueError(
            "package thermal properties must be positive and finite"
        )
    return (
        conductivity,
        density
        * heat_capacity,
    )


def _has_contrast(
    conductivity,
    capacity,
    background,
) -> bool:
    return bool(
        not np.isclose(
            conductivity,
            background.conductivity,
            rtol=1e-12,
            atol=0.0,
        )
        or not np.isclose(
            capacity,
            background.volumetric_heat_capacity,
            rtol=1e-12,
            atol=0.0,
        )
    )


def _kernel(
    observations,
    sources,
    *,
    conductivity: float,
    capacity: float,
    laplace_s: float,
    source_radius=None,
):
    observations = np.asarray(
        observations,
        dtype=float,
    )
    sources = np.asarray(
        sources,
        dtype=float,
    )
    difference = (
        observations[
            :,
            None,
            :
        ]
        - sources[
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
    if source_radius is None:
        distance = np.sqrt(
            np.maximum(
                distance_squared,
                1e-30,
            )
        )
    else:
        radius = np.asarray(
            source_radius,
            dtype=float,
        )
        distance = np.sqrt(
            distance_squared
            + radius[
                None,
                :
            ] ** 2
        )
    if laplace_s < 0.0:
        raise ValueError(
            "laplace_s must be nonnegative"
        )
    decay = (
        0.0
        if laplace_s == 0.0
        else np.sqrt(
            laplace_s
            * capacity
            / conductivity
        )
    )
    green = (
        np.exp(
            -decay
            * distance
        )
        / (
            4.0
            * np.pi
            * conductivity
            * distance
        )
    )
    return (
        green,
        difference,
        distance,
        decay,
    )


def _normal_derivative(
    observations,
    normals,
    sources,
    *,
    conductivity: float,
    capacity: float,
    laplace_s: float,
    source_radius=None,
):
    (
        green,
        difference,
        distance,
        decay,
    ) = _kernel(
        observations,
        sources,
        conductivity=conductivity,
        capacity=capacity,
        laplace_s=laplace_s,
        source_radius=source_radius,
    )
    projection = np.einsum(
        "qsd,qd->qs",
        difference,
        np.asarray(
            normals,
            dtype=float,
        ),
    )
    return (
        -green
        * (
            decay
            + 1.0
            / distance
        )
        * projection
        / distance
    )


def _stehfest_coefficients(
    order: int,
) -> np.ndarray:
    if (
        order < 6
        or order % 2
        != 0
    ):
        raise ValueError(
            "stehfest_order must be an even integer >= 6"
        )
    half = (
        order // 2
    )
    coefficients = np.empty(
        order,
        dtype=float,
    )
    for k in range(
        1,
        order + 1,
    ):
        value = 0.0
        lower = (
            k + 1
        ) // 2
        upper = min(
            k,
            half,
        )
        for j in range(
            lower,
            upper + 1,
        ):
            numerator = (
                j**half
                * factorial(
                    2
                    * j
                )
            )
            denominator = (
                factorial(
                    half
                    - j
                )
                * factorial(
                    j
                )
                * factorial(
                    j
                    - 1
                )
                * factorial(
                    k
                    - j
                )
                * factorial(
                    2
                    * j
                    - k
                )
            )
            value += (
                numerator
                / denominator
            )
        coefficients[
            k - 1
        ] = (
            (-1.0) ** (
                k
                + half
            )
            * value
        )
    return coefficients


class PreparedPackageThermalContrastField:
    """Mesh-free multi-region transient thermal transmission field.

    The unbounded background remains analytic. Package interfaces are enforced
    with a method-of-fundamental-solutions collocation system in Laplace space,
    so no global volume mesh or finite world box is introduced.
    """

    def __init__(
        self,
        source,
        medium,
        regions,
        *,
        stehfest_order: int = 10,
    ):
        self.source = source
        self.medium = medium
        self.regions = tuple(
            regions
        )
        self.stehfest_order = int(
            stehfest_order
        )
        self._stehfest = (
            _stehfest_coefficients(
                self.stehfest_order
            )
        )
        self._log2 = log(
            2.0
        )
        self._source_power_matrices = (
            self.source.volume_weights[
                :,
                None,
                None,
            ]
            * self.source.dissipation_matrices
        )
        self._background_source_mask = (
            np.ones(
                len(
                    self.source.volume_weights
                ),
                dtype=bool,
            )
        )
        for region in self.regions:
            self._background_source_mask[
                region.source_mask
            ] = False

        self._boundary_positions = np.concatenate(
            [
                region.boundary_positions
                for region in self.regions
            ],
            axis=0,
        )
        self._boundary_normals = np.concatenate(
            [
                region.boundary_normals
                for region in self.regions
            ],
            axis=0,
        )
        self._interior_sources = np.concatenate(
            [
                region.interior_sources
                for region in self.regions
            ],
            axis=0,
        )
        self._coefficient_cache = {}

    @property
    def n_ports(
        self,
    ) -> int:
        return self.source.n_ports

    def _query_points(
        self,
        points,
    ):
        points = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            points.ndim
            == 1
        )
        points = np.atleast_2d(
            points
        )
        if (
            points.ndim != 2
            or points.shape[
                1
            ]
            != 3
            or np.any(
                ~np.isfinite(
                    points
                )
            )
        ):
            raise ValueError(
                "query points must have shape (3,) or (n,3)"
            )
        return (
            points,
            scalar,
        )

    def _direct(
        self,
        points,
        *,
        conductivity,
        capacity,
        laplace_s,
        source_mask,
        step_laplace,
    ):
        mask = np.asarray(
            source_mask,
            dtype=bool,
        )
        if not np.any(
            mask
        ):
            return np.zeros(
                (
                    len(
                        points
                    ),
                    self.n_ports,
                    self.n_ports,
                ),
                dtype=complex,
            )
        green, _, _, _ = _kernel(
            points,
            self.source.positions[
                mask
            ],
            conductivity=conductivity,
            capacity=capacity,
            laplace_s=laplace_s,
            source_radius=(
                self.source.effective_radius[
                    mask
                ]
            ),
        )
        values = np.einsum(
            "qs,sij->qij",
            green,
            self._source_power_matrices[
                mask
            ],
        )
        if step_laplace:
            values = (
                values
                / laplace_s
            )
        return values

    def _direct_flux(
        self,
        points,
        normals,
        *,
        conductivity,
        capacity,
        laplace_s,
        source_mask,
        step_laplace,
    ):
        mask = np.asarray(
            source_mask,
            dtype=bool,
        )
        if not np.any(
            mask
        ):
            return np.zeros(
                (
                    len(
                        points
                    ),
                    self.n_ports,
                    self.n_ports,
                ),
                dtype=complex,
            )
        derivative = _normal_derivative(
            points,
            normals,
            self.source.positions[
                mask
            ],
            conductivity=conductivity,
            capacity=capacity,
            laplace_s=laplace_s,
            source_radius=(
                self.source.effective_radius[
                    mask
                ]
            ),
        )
        values = np.einsum(
            "qs,sij->qij",
            derivative,
            self._source_power_matrices[
                mask
            ],
        )
        if step_laplace:
            values = (
                values
                / laplace_s
            )
        return values

    def _solve_coefficients(
        self,
        laplace_s: float,
        *,
        step_laplace: bool,
    ):
        cache_key = (
            float(
                laplace_s
            ),
            bool(
                step_laplace
            ),
        )
        cached = (
            self._coefficient_cache.get(
                cache_key
            )
        )
        if cached is not None:
            return cached

        n_boundary = len(
            self._boundary_positions
        )
        background_capacity = (
            self.medium.volumetric_heat_capacity
        )
        green_out, _, _, _ = _kernel(
            self._boundary_positions,
            self._interior_sources,
            conductivity=(
                self.medium.conductivity
            ),
            capacity=(
                background_capacity
            ),
            laplace_s=laplace_s,
        )
        derivative_out = _normal_derivative(
            self._boundary_positions,
            self._boundary_normals,
            self._interior_sources,
            conductivity=(
                self.medium.conductivity
            ),
            capacity=(
                background_capacity
            ),
            laplace_s=laplace_s,
        )

        matrix = np.zeros(
            (
                2
                * n_boundary,
                2
                * n_boundary,
            ),
            dtype=float,
        )
        matrix[
            :n_boundary,
            :n_boundary,
        ] = green_out
        matrix[
            n_boundary:,
            :n_boundary,
        ] = (
            self.medium.conductivity
            * derivative_out
        )

        direct_out = self._direct(
            self._boundary_positions,
            conductivity=(
                self.medium.conductivity
            ),
            capacity=(
                background_capacity
            ),
            laplace_s=laplace_s,
            source_mask=(
                self._background_source_mask
            ),
            step_laplace=(
                step_laplace
            ),
        )
        direct_out_flux = self._direct_flux(
            self._boundary_positions,
            self._boundary_normals,
            conductivity=(
                self.medium.conductivity
            ),
            capacity=(
                background_capacity
            ),
            laplace_s=laplace_s,
            source_mask=(
                self._background_source_mask
            ),
            step_laplace=(
                step_laplace
            ),
        )

        rhs_temperature = np.empty_like(
            direct_out
        )
        rhs_flux = np.empty_like(
            direct_out
        )

        for region in self.regions:
            sl = (
                region.boundary_slice
            )
            points = (
                region.boundary_positions
            )
            normals = (
                region.boundary_normals
            )
            green_in, _, _, _ = _kernel(
                points,
                region.exterior_sources,
                conductivity=(
                    region.conductivity
                ),
                capacity=(
                    region.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
            )
            derivative_in = _normal_derivative(
                points,
                normals,
                region.exterior_sources,
                conductivity=(
                    region.conductivity
                ),
                capacity=(
                    region.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
            )
            matrix[
                sl,
                n_boundary
                + sl.start:
                n_boundary
                + sl.stop,
            ] = (
                -green_in
            )
            matrix[
                n_boundary
                + sl.start:
                n_boundary
                + sl.stop,
                n_boundary
                + sl.start:
                n_boundary
                + sl.stop,
            ] = (
                -region.conductivity
                * derivative_in
            )

            direct_in = self._direct(
                points,
                conductivity=(
                    region.conductivity
                ),
                capacity=(
                    region.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
                source_mask=(
                    region.source_mask
                ),
                step_laplace=(
                    step_laplace
                ),
            )
            direct_in_flux = self._direct_flux(
                points,
                normals,
                conductivity=(
                    region.conductivity
                ),
                capacity=(
                    region.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
                source_mask=(
                    region.source_mask
                ),
                step_laplace=(
                    step_laplace
                ),
            )
            rhs_temperature[
                sl
            ] = (
                direct_in
                - direct_out[
                    sl
                ]
            )
            rhs_flux[
                sl
            ] = (
                region.conductivity
                * direct_in_flux
                - self.medium.conductivity
                * direct_out_flux[
                    sl
                ]
            )

        rhs = np.concatenate(
            (
                rhs_temperature,
                rhs_flux,
            ),
            axis=0,
        ).reshape(
            2
            * n_boundary,
            -1,
        )
        try:
            solved = np.linalg.solve(
                matrix,
                rhs,
            )
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(
                matrix,
                rhs,
                rcond=1e-11,
            )[
                0
            ]
        solved = solved.reshape(
            2
            * n_boundary,
            self.n_ports,
            self.n_ports,
        )
        exterior = solved[
            :n_boundary
        ]
        interior = solved[
            n_boundary:
        ]
        self._coefficient_cache[
            cache_key
        ] = (
            exterior,
            interior,
        )
        return (
            exterior,
            interior,
        )

    def _response_at_s(
        self,
        points,
        *,
        laplace_s: float,
        step_laplace: bool,
    ):
        points, scalar = (
            self._query_points(
                points
            )
        )
        exterior_coefficients, interior_coefficients = (
            self._solve_coefficients(
                laplace_s,
                step_laplace=(
                    step_laplace
                ),
            )
        )
        result = np.empty(
            (
                len(
                    points
                ),
                self.n_ports,
                self.n_ports,
            ),
            dtype=complex,
        )

        membership = np.full(
            len(
                points
            ),
            -1,
            dtype=int,
        )
        for region_index, region in enumerate(
            self.regions
        ):
            inside = np.asarray(
                region.geometry.contains(
                    points,
                    tolerance=1e-10,
                ),
                dtype=bool,
            )
            conflict = (
                inside
                & (
                    membership
                    >= 0
                )
            )
            if np.any(
                conflict
            ):
                raise ValueError(
                    "thermal query lies in overlapping package regions"
                )
            membership[
                inside
            ] = (
                region_index
            )

        outside = (
            membership
            < 0
        )
        if np.any(
            outside
        ):
            query = points[
                outside
            ]
            direct = self._direct(
                query,
                conductivity=(
                    self.medium.conductivity
                ),
                capacity=(
                    self.medium.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
                source_mask=(
                    self._background_source_mask
                ),
                step_laplace=(
                    step_laplace
                ),
            )
            green, _, _, _ = _kernel(
                query,
                self._interior_sources,
                conductivity=(
                    self.medium.conductivity
                ),
                capacity=(
                    self.medium.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
            )
            result[
                outside
            ] = (
                direct
                + np.einsum(
                    "qs,sij->qij",
                    green,
                    exterior_coefficients,
                )
            )

        for region_index, region in enumerate(
            self.regions
        ):
            inside = (
                membership
                == region_index
            )
            if not np.any(
                inside
            ):
                continue
            query = points[
                inside
            ]
            direct = self._direct(
                query,
                conductivity=(
                    region.conductivity
                ),
                capacity=(
                    region.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
                source_mask=(
                    region.source_mask
                ),
                step_laplace=(
                    step_laplace
                ),
            )
            sl = (
                region.boundary_slice
            )
            green, _, _, _ = _kernel(
                query,
                region.exterior_sources,
                conductivity=(
                    region.conductivity
                ),
                capacity=(
                    region.volumetric_heat_capacity
                ),
                laplace_s=laplace_s,
            )
            result[
                inside
            ] = (
                direct
                + np.einsum(
                    "qs,sij->qij",
                    green,
                    interior_coefficients[
                        sl
                    ],
                )
            )

        result = 0.5 * (
            result
            + result.conj().transpose(
                0,
                2,
                1,
            )
        )
        return (
            result[
                0
            ]
            if scalar
            else result
        )

    def step_response_matrix(
        self,
        points,
        time: float,
    ):
        if (
            not np.isfinite(
                time
            )
            or time < 0.0
        ):
            raise ValueError(
                "time must be finite and nonnegative"
            )
        points_array, scalar = (
            self._query_points(
                points
            )
        )
        if time == 0.0:
            result = np.zeros(
                (
                    len(
                        points_array
                    ),
                    self.n_ports,
                    self.n_ports,
                ),
                dtype=complex,
            )
            return (
                result[
                    0
                ]
                if scalar
                else result
            )

        result = np.zeros(
            (
                len(
                    points_array
                ),
                self.n_ports,
                self.n_ports,
            ),
            dtype=complex,
        )
        scale = (
            self._log2
            / time
        )
        for index, coefficient in enumerate(
            self._stehfest,
            start=1,
        ):
            laplace_s = (
                index
                * scale
            )
            result += (
                coefficient
                * self._response_at_s(
                    points_array,
                    laplace_s=(
                        laplace_s
                    ),
                    step_laplace=True,
                )
            )
        result *= scale
        result = 0.5 * (
            result
            + result.conj().transpose(
                0,
                2,
                1,
            )
        )
        return (
            result[
                0
            ]
            if scalar
            else result
        )

    def steady_response_matrix(
        self,
        points,
    ):
        return self._response_at_s(
            points,
            laplace_s=0.0,
            step_laplace=False,
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
    ):
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
            or len(
                edges
            )
            < 2
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
            len(
                edges
            )
            - 1,
            self.n_ports,
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
                len(
                    times
                ),
                len(
                    query
                ),
            ),
            self.medium.ambient_temperature,
            dtype=float,
        )
        for time_index, time in enumerate(
            times
        ):
            for interval in range(
                len(
                    edges
                )
                - 1
            ):
                start = edges[
                    interval
                ]
                end = edges[
                    interval
                    + 1
                ]
                if time <= start:
                    continue
                age_start = (
                    time
                    - start
                )
                age_end = max(
                    time
                    - end,
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
        return (
            result[
                :,
                0
            ]
            if scalar
            else result
        )


def prepare_package_thermal_contrast_field(
    scene,
    source,
    medium,
    *,
    surface_vertical_order: int = 6,
    surface_azimuthal_order: int = 12,
    fictitious_scale: float = 0.18,
    stehfest_order: int = 10,
):
    if (
        surface_vertical_order < 4
        or surface_azimuthal_order < 8
    ):
        raise ValueError(
            "thermal contrast surface orders are too small"
        )
    if (
        not np.isfinite(
            fictitious_scale
        )
        or fictitious_scale <= 0.02
        or fictitious_scale >= 0.45
    ):
        raise ValueError(
            "fictitious_scale must lie in (0.02, 0.45)"
        )

    active = []
    for package_index, package in enumerate(
        scene.packages
    ):
        properties = (
            _material_thermal_properties(
                package.material
            )
        )
        if properties is None:
            continue
        conductivity, capacity = (
            properties
        )
        if not _has_contrast(
            conductivity,
            capacity,
            medium,
        ):
            continue
        active.append(
            (
                package_index,
                package,
                conductivity,
                capacity,
            )
        )

    if not active:
        return None

    surface_data = []
    for (
        package_index,
        package,
        conductivity,
        capacity,
    ) in active:
        surface = (
            package.geometry.surface_quadrature(
                vertical_order=(
                    surface_vertical_order
                ),
                azimuthal_order=(
                    surface_azimuthal_order
                ),
            )
        )
        local = (
            package.geometry.world_to_local(
                surface.positions
            )
        )
        interior = (
            package.geometry.local_to_world(
                (
                    1.0
                    - fictitious_scale
                )
                * local
            )
        )
        exterior = (
            package.geometry.local_to_world(
                (
                    1.0
                    + fictitious_scale
                )
                * local
            )
        )
        if not np.all(
            package.geometry.contains(
                interior,
                tolerance=1e-10,
            )
        ):
            raise RuntimeError(
                "failed to place interior thermal MFS sources"
            )
        if np.any(
            package.geometry.contains(
                exterior,
                tolerance=1e-12,
            )
        ):
            raise RuntimeError(
                "failed to place exterior thermal MFS sources"
            )
        surface_data.append(
            (
                package_index,
                package,
                conductivity,
                capacity,
                surface.positions,
                surface.normals,
                interior,
                exterior,
            )
        )

    for left in range(
        len(
            surface_data
        )
    ):
        for right in range(
            left + 1,
            len(
                surface_data
            ),
        ):
            left_package = (
                surface_data[
                    left
                ][
                    1
                ]
            )
            right_package = (
                surface_data[
                    right
                ][
                    1
                ]
            )
            if (
                np.any(
                    left_package.geometry.contains(
                        surface_data[
                            right
                        ][
                            4
                        ],
                        tolerance=1e-10,
                    )
                )
                or np.any(
                    right_package.geometry.contains(
                        surface_data[
                            left
                        ][
                            4
                        ],
                        tolerance=1e-10,
                    )
                )
            ):
                raise NotImplementedError(
                    "overlapping or nested thermal-contrast packages are not "
                    "yet supported by the MFS transmission solver"
                )

    source_membership = np.full(
        len(
            source.positions
        ),
        -1,
        dtype=int,
    )
    masks = []
    for region_index, data in enumerate(
        surface_data
    ):
        package = data[
            1
        ]
        mask = np.asarray(
            package.geometry.contains(
                source.positions,
                tolerance=1e-10,
            ),
            dtype=bool,
        )
        if np.any(
            mask
            & (
                source_membership
                >= 0
            )
        ):
            raise ValueError(
                "thermal source lies in overlapping package regions"
            )
        source_membership[
            mask
        ] = region_index
        masks.append(
            mask
        )

    regions = []
    start = 0
    for data, source_mask in zip(
        surface_data,
        masks,
    ):
        count = len(
            data[
                4
            ]
        )
        sl = slice(
            start,
            start
            + count,
        )
        start += count
        regions.append(
            _ThermalRegion(
                package_index=(
                    data[
                        0
                    ]
                ),
                geometry=(
                    data[
                        1
                    ].geometry
                ),
                conductivity=(
                    data[
                        2
                    ]
                ),
                volumetric_heat_capacity=(
                    data[
                        3
                    ]
                ),
                boundary_positions=(
                    data[
                        4
                    ]
                ),
                boundary_normals=(
                    data[
                        5
                    ]
                ),
                interior_sources=(
                    data[
                        6
                    ]
                ),
                exterior_sources=(
                    data[
                        7
                    ]
                ),
                source_mask=(
                    source_mask
                ),
                boundary_slice=sl,
            )
        )

    return PreparedPackageThermalContrastField(
        source,
        medium,
        regions,
        stehfest_order=(
            stehfest_order
        ),
    )
