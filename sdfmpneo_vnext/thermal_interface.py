from __future__ import annotations

from math import factorial
import numpy as np
from scipy.linalg import lstsq

from .hybrid_domain import (
    package_domain_topology,
    validate_package_conductor_topology,
)
from .scene import Scene
from .thermal_field import (
    AnisotropicThermalMedium,
    ContinuousThermalGreenArtifact,
    HomogeneousThermalMedium,
    PreparedThermalGreenField,
    build_thermal_source_quadrature,
)


def _thermal_properties(
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
            "thermal_conductivity, density, and heat_capacity must be "
            "declared together"
        )
    return tuple(
        float(
            value
        )
        for value in values
    )


def _same_thermal_medium(
    first: HomogeneousThermalMedium,
    second: HomogeneousThermalMedium,
) -> bool:
    return bool(
        np.isclose(
            first.conductivity,
            second.conductivity,
            rtol=1e-12,
            atol=0.0,
        )
        and np.isclose(
            first.density,
            second.density,
            rtol=1e-12,
            atol=0.0,
        )
        and np.isclose(
            first.heat_capacity,
            second.heat_capacity,
            rtol=1e-12,
            atol=0.0,
        )
    )


def scene_thermal_package_media(
    scene: Scene,
    background,
):
    """Return thermal interfaces whose material differs from the parent region."""
    if not isinstance(
        background,
        HomogeneousThermalMedium,
    ):
        has_package_thermal_contrast = any(
            _thermal_properties(
                package.material
            )
            is not None
            for package in scene.packages
        )
        if has_package_thermal_contrast:
            raise NotImplementedError(
                "thermally distinct package interfaces with an anisotropic "
                "background require the tensor thermal-interface solver"
            )
        return ()
    topology = package_domain_topology(
        scene.packages
    )
    resolved = [
        None
        for _ in scene.packages
    ]
    out = []
    order = sorted(
        range(
            len(
                scene.packages
            )
        ),
        key=lambda index: (
            topology.depth[
                index
            ]
        ),
    )
    for index in order:
        package = scene.packages[
            index
        ]
        parent = topology.parent[
            index
        ]
        parent_medium = (
            background
            if parent is None
            else resolved[
                parent
            ]
        )
        values = _thermal_properties(
            package.material
        )
        if values is None:
            medium = parent_medium
        else:
            medium = HomogeneousThermalMedium(
                values[
                    0
                ],
                values[
                    1
                ],
                values[
                    2
                ],
                background.ambient_temperature,
            )
        resolved[
            index
        ] = medium
        if not _same_thermal_medium(
            medium,
            parent_medium,
        ):
            out.append(
                (
                    index,
                    medium,
                )
            )
    return tuple(
        out
    )


def _medium_metric_data(
    medium,
):
    if isinstance(
        medium,
        AnisotropicThermalMedium,
    ):
        inverse = (
            medium.inverse_conductivity_tensor
        )
        determinant = (
            medium.conductivity_determinant
        )
        geometric_mean = (
            medium.geometric_mean_conductivity
        )
    elif isinstance(
        medium,
        HomogeneousThermalMedium,
    ):
        conductivity = float(
            medium.conductivity
        )
        inverse = (
            np.eye(
                3,
                dtype=float,
            )
            / conductivity
        )
        determinant = (
            conductivity**3
        )
        geometric_mean = conductivity
    else:
        raise TypeError(
            "thermal interface medium must be homogeneous isotropic or "
            "homogeneous anisotropic"
        )
    return (
        inverse,
        float(
            determinant
        ),
        float(
            geometric_mean
        ),
    )


def _yukawa_kernel(
    targets,
    sources,
    medium,
    laplace_s: float,
    *,
    source_radius=None,
):
    targets = np.asarray(
        targets,
        dtype=float,
    )
    sources = np.asarray(
        sources,
        dtype=float,
    )
    difference = (
        targets[
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
    (
        inverse_conductivity,
        conductivity_determinant,
        geometric_mean_conductivity,
    ) = _medium_metric_data(
        medium
    )
    metric_squared = np.einsum(
        "qsi,ij,qsj->qs",
        difference,
        inverse_conductivity,
        difference,
    )
    if source_radius is not None:
        radius = np.asarray(
            source_radius,
            dtype=float,
        )
        if radius.shape != (
            len(
                sources
            ),
        ):
            raise ValueError(
                "source_radius has incompatible shape"
            )
        metric_squared = (
            metric_squared
            + radius[
                None,
                :
            ] ** 2
            / geometric_mean_conductivity
        )
    distance = np.sqrt(
        np.maximum(
            metric_squared,
            1e-30,
        )
    )
    if laplace_s < 0.0:
        raise ValueError(
            "laplace_s must be nonnegative"
        )
    decay = (
        0.0
        if laplace_s == 0.0
        else np.sqrt(
            float(
                laplace_s
            )
            * medium.volumetric_heat_capacity
        )
    )
    exponential = np.exp(
        -decay
        * distance
    )
    value = (
        exponential
        / (
            4.0
            * np.pi
            * np.sqrt(
                conductivity_determinant
            )
            * distance
        )
    )
    return (
        value,
        difference,
        distance,
        exponential,
        decay,
    )


def _normal_flux_kernel(
    targets,
    normals,
    sources,
    medium,
    laplace_s: float,
    *,
    source_radius=None,
):
    (
        value,
        difference,
        distance,
        _,
        decay,
    ) = _yukawa_kernel(
        targets,
        sources,
        medium,
        laplace_s,
        source_radius=(
            source_radius
        ),
    )
    normals = np.asarray(
        normals,
        dtype=float,
    )
    if normals.shape != (
        len(
            targets
        ),
        3,
    ):
        raise ValueError(
            "normals have incompatible shape"
        )
    projection = np.einsum(
        "qsd,qd->qs",
        difference,
        normals,
    )
    return (
        -value
        * (
            decay
            + 1.0
            / distance
        )
        * projection
        / distance
    )

def _equilibrated_lstsq(
    matrix,
    rhs,
    *,
    rcond: float,
):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )
    rhs = np.asarray(
        rhs,
        dtype=complex,
    )
    row_norm = np.max(
        np.abs(
            matrix
        ),
        axis=1,
    )
    row_scale = (
        1.0
        / np.maximum(
            row_norm,
            1e-30,
        )
    )
    scaled = (
        row_scale[
            :,
            None,
        ]
        * matrix
    )
    scaled_rhs = (
        row_scale[
            :,
            None,
        ]
        * rhs
    )
    column_norm = np.max(
        np.abs(
            scaled
        ),
        axis=0,
    )
    column_scale = (
        1.0
        / np.maximum(
            column_norm,
            1e-30,
        )
    )
    equilibrated = (
        scaled
        * column_scale[
            None,
            :
        ]
    )
    solution, _, rank, singular = lstsq(
        equilibrated,
        scaled_rhs,
        cond=rcond,
        lapack_driver="gelsd",
    )
    solution = (
        column_scale[
            :,
            None,
        ]
        * solution
    )
    residual = (
        matrix
        @ solution
        - rhs
    )
    scaled_residual = (
        row_scale[
            :,
            None,
        ]
        * residual
    )
    denominator = max(
        float(
            np.linalg.norm(
                scaled_rhs
            )
        ),
        1e-30,
    )
    relative_residual = float(
        np.linalg.norm(
            scaled_residual
        )
        / denominator
    )
    if singular is None or len(
        singular
    ) == 0:
        condition = float(
            "inf"
        )
    else:
        positive = singular[
            singular
            > 0.0
        ]
        condition = (
            float(
                "inf"
            )
            if len(
                positive
            ) == 0
            else float(
                np.max(
                    positive
                )
                / np.min(
                    positive
                )
            )
        )
    return (
        solution,
        relative_residual,
        int(
            rank
        ),
        condition,
    )


def _stehfest_weights(
    order: int,
) -> np.ndarray:
    if (
        order < 4
        or order % 2
        != 0
    ):
        raise ValueError(
            "Stehfest order must be an even integer >= 4"
        )
    half = (
        order
        // 2
    )
    out = np.empty(
        order,
        dtype=float,
    )
    for k in range(
        1,
        order + 1,
    ):
        total = 0.0
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
                    j - 1
                )
                * factorial(
                    k - j
                )
                * factorial(
                    2
                    * j
                    - k
                )
            )
            total += (
                numerator
                / denominator
            )
        out[
            k - 1
        ] = (
            (-1) ** (
                k
                + half
            )
            * total
        )
    return out


class PreparedThermalInterfaceField:
    """Mesh-free transient thermal transmission through one package interface."""

    def __init__(
        self,
        source,
        background_medium: HomogeneousThermalMedium,
        package_geometry,
        package_medium: HomogeneousThermalMedium,
        *,
        surface_vertical_order: int,
        surface_azimuthal_order: int,
        mfs_offset_fraction: float,
        stehfest_order: int,
        interface_residual_tolerance: float,
        svd_rcond: float,
    ):
        self.source = source
        self.medium = (
            background_medium
        )
        self.background_medium = (
            background_medium
        )
        self.package_geometry = (
            package_geometry
        )
        self.package_medium = (
            package_medium
        )
        self.stehfest_order = int(
            stehfest_order
        )
        self.stehfest_weights = (
            _stehfest_weights(
                self.stehfest_order
            )
        )
        self.interface_residual_tolerance = float(
            interface_residual_tolerance
        )
        self.svd_rcond = float(
            svd_rcond
        )
        if (
            self.interface_residual_tolerance
            <= 0.0
            or self.svd_rcond
            <= 0.0
            or not (
                0.0
                < mfs_offset_fraction
                < 0.5
            )
        ):
            raise ValueError(
                "invalid thermal interface solver configuration"
            )

        surface = (
            package_geometry.surface_quadrature(
                vertical_order=(
                    surface_vertical_order
                ),
                azimuthal_order=(
                    surface_azimuthal_order
                ),
            )
        )
        self.surface_positions = (
            surface.positions
        )
        self.surface_normals = (
            surface.normals
        )

        offset = (
            float(
                mfs_offset_fraction
            )
            * float(
                np.min(
                    package_geometry.half_extents
                )
            )
        )
        for _ in range(
            12
        ):
            interior = (
                self.surface_positions
                - offset
                * self.surface_normals
            )
            exterior = (
                self.surface_positions
                + offset
                * self.surface_normals
            )
            interior_ok = np.all(
                package_geometry.contains(
                    interior,
                    tolerance=1e-12,
                )
            )
            exterior_ok = not np.any(
                package_geometry.contains(
                    exterior,
                    tolerance=1e-12,
                )
            )
            if (
                interior_ok
                and exterior_ok
            ):
                break
            offset *= 0.5
        else:
            raise RuntimeError(
                "failed to place thermal MFS sources on opposite sides "
                "of the package interface"
            )
        self.mfs_offset = float(
            offset
        )
        self.exterior_mfs_sources = (
            interior
        )
        self.interior_mfs_sources = (
            exterior
        )

        self.source_inside_package = np.asarray(
            package_geometry.contains(
                source.positions,
                tolerance=1e-12,
            ),
            dtype=bool,
        )
        self.source_strength = (
            source.volume_weights[
                :,
                None,
                None,
            ]
            * source.dissipation_matrices
        )
        self.source_strength_flat = (
            self.source_strength.reshape(
                len(
                    source.volume_weights
                ),
                -1,
            )
        )
        self._solve_cache = {}
        self._residual_cache = {}
        self._condition_cache = {}

    @property
    def maximum_interface_residual(
        self,
    ) -> float:
        if not self._residual_cache:
            return 0.0
        return float(
            max(
                self._residual_cache.values()
            )
        )

    @property
    def maximum_interface_condition(
        self,
    ) -> float:
        if not self._condition_cache:
            return 0.0
        return float(
            max(
                self._condition_cache.values()
            )
        )

    def _particular_boundary(
        self,
        laplace_s: float,
        *,
        inside: bool,
    ):
        mask = (
            self.source_inside_package
            if inside
            else ~self.source_inside_package
        )
        medium = (
            self.package_medium
            if inside
            else self.background_medium
        )
        if not np.any(
            mask
        ):
            shape = (
                len(
                    self.surface_positions
                ),
                self.source.n_ports
                * self.source.n_ports,
            )
            return (
                np.zeros(
                    shape,
                    dtype=complex,
                ),
                np.zeros(
                    shape,
                    dtype=complex,
                ),
            )
        source_positions = (
            self.source.positions[
                mask
            ]
        )
        radius = (
            self.source.effective_radius[
                mask
            ]
        )
        strength = (
            self.source_strength_flat[
                mask
            ]
        )
        values = _yukawa_kernel(
            self.surface_positions,
            source_positions,
            medium,
            laplace_s,
            source_radius=(
                radius
            ),
        )[
            0
        ]
        flux = (
            _normal_flux_kernel(
                self.surface_positions,
                self.surface_normals,
                source_positions,
                medium,
                laplace_s,
                source_radius=(
                    radius
                ),
            )
        )
        return (
            values
            @ strength,
            flux
            @ strength,
        )

    def _interface_solution(
        self,
        laplace_s: float,
    ):
        key = float(
            laplace_s
        )
        cached = self._solve_cache.get(
            key
        )
        if cached is not None:
            return cached

        exterior_value = _yukawa_kernel(
            self.surface_positions,
            self.exterior_mfs_sources,
            self.background_medium,
            laplace_s,
        )[
            0
        ]
        interior_value = _yukawa_kernel(
            self.surface_positions,
            self.interior_mfs_sources,
            self.package_medium,
            laplace_s,
        )[
            0
        ]
        exterior_flux = (
            _normal_flux_kernel(
                self.surface_positions,
                self.surface_normals,
                self.exterior_mfs_sources,
                self.background_medium,
                laplace_s,
            )
        )
        interior_flux = (
            _normal_flux_kernel(
                self.surface_positions,
                self.surface_normals,
                self.interior_mfs_sources,
                self.package_medium,
                laplace_s,
            )
        )
        matrix = np.block(
            [
                [
                    exterior_value,
                    -interior_value,
                ],
                [
                    exterior_flux,
                    -interior_flux,
                ],
            ]
        )

        (
            background_particular,
            background_flux,
        ) = self._particular_boundary(
            laplace_s,
            inside=False,
        )
        (
            package_particular,
            package_flux,
        ) = self._particular_boundary(
            laplace_s,
            inside=True,
        )
        rhs = np.vstack(
            (
                package_particular
                - background_particular,
                package_flux
                - background_flux,
            )
        )
        (
            solution,
            residual,
            rank,
            condition,
        ) = _equilibrated_lstsq(
            matrix,
            rhs,
            rcond=(
                self.svd_rcond
            ),
        )
        if (
            residual
            > self.interface_residual_tolerance
        ):
            raise RuntimeError(
                "thermal interface solve did not meet the declared residual "
                f"tolerance: {residual:.3e} > "
                f"{self.interface_residual_tolerance:.3e}"
            )
        if rank < min(
            matrix.shape
        ):
            raise RuntimeError(
                "thermal interface MFS system is rank deficient"
            )

        n = len(
            self.surface_positions
        )
        coefficients = (
            solution[
                :n
            ],
            solution[
                n:
            ],
        )
        self._solve_cache[
            key
        ] = coefficients
        self._residual_cache[
            key
        ] = residual
        self._condition_cache[
            key
        ] = condition
        return coefficients

    def _particular_query(
        self,
        points,
        laplace_s: float,
        *,
        inside: bool,
    ):
        mask = (
            self.source_inside_package
            if inside
            else ~self.source_inside_package
        )
        if not np.any(
            mask
        ):
            return np.zeros(
                (
                    len(
                        points
                    ),
                    self.source.n_ports
                    * self.source.n_ports,
                ),
                dtype=complex,
            )
        medium = (
            self.package_medium
            if inside
            else self.background_medium
        )
        kernel = _yukawa_kernel(
            points,
            self.source.positions[
                mask
            ],
            medium,
            laplace_s,
            source_radius=(
                self.source.effective_radius[
                    mask
                ]
            ),
        )[
            0
        ]
        return (
            kernel
            @ self.source_strength_flat[
                mask
            ]
        )

    def laplace_response_matrix(
        self,
        points,
        laplace_s: float,
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
            points.ndim
            != 2
            or points.shape[
                1
            ]
            != 3
        ):
            raise ValueError(
                "query points must have shape (3,) or (n,3)"
            )
        if (
            not np.isfinite(
                laplace_s
            )
            or laplace_s < 0.0
        ):
            raise ValueError(
                "laplace_s must be finite and nonnegative"
            )

        (
            exterior_coefficients,
            interior_coefficients,
        ) = self._interface_solution(
            float(
                laplace_s
            )
        )
        inside = np.asarray(
            self.package_geometry.contains(
                points,
                tolerance=1e-12,
            ),
            dtype=bool,
        )
        flat = np.zeros(
            (
                len(
                    points
                ),
                self.source.n_ports
                * self.source.n_ports,
            ),
            dtype=complex,
        )

        outside = ~inside
        if np.any(
            outside
        ):
            query = points[
                outside
            ]
            flat[
                outside
            ] = (
                self._particular_query(
                    query,
                    laplace_s,
                    inside=False,
                )
                + _yukawa_kernel(
                    query,
                    self.exterior_mfs_sources,
                    self.background_medium,
                    laplace_s,
                )[
                    0
                ]
                @ exterior_coefficients
            )
        if np.any(
            inside
        ):
            query = points[
                inside
            ]
            flat[
                inside
            ] = (
                self._particular_query(
                    query,
                    laplace_s,
                    inside=True,
                )
                + _yukawa_kernel(
                    query,
                    self.interior_mfs_sources,
                    self.package_medium,
                    laplace_s,
                )[
                    0
                ]
                @ interior_coefficients
            )
        matrix = flat.reshape(
            len(
                points
            ),
            self.source.n_ports,
            self.source.n_ports,
        )
        matrix = 0.5 * (
            matrix
            + matrix.conj().transpose(
                0,
                2,
                1,
            )
        )
        return (
            matrix[
                0
            ]
            if scalar
            else matrix
        )

    def steady_response_matrix(
        self,
        points,
    ):
        return self.laplace_response_matrix(
            points,
            0.0,
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
        points_array = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            points_array.ndim
            == 1
        )
        points_array = np.atleast_2d(
            points_array
        )
        if time == 0.0:
            result = np.zeros(
                (
                    len(
                        points_array
                    ),
                    self.source.n_ports,
                    self.source.n_ports,
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

        logarithm = np.log(
            2.0
        )
        result = np.zeros(
            (
                len(
                    points_array
                ),
                self.source.n_ports,
                self.source.n_ports,
            ),
            dtype=complex,
        )
        for k, weight in enumerate(
            self.stehfest_weights,
            start=1,
        ):
            laplace_s = (
                k
                * logarithm
                / float(
                    time
                )
            )
            result += (
                weight
                / k
                * np.asarray(
                    self.laplace_response_matrix(
                        points_array,
                        laplace_s,
                    ),
                    dtype=complex,
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

    @staticmethod
    def _contract(
        matrices,
        currents,
    ):
        return PreparedThermalGreenField._contract(
            matrices,
            currents,
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
            edges.ndim
            != 1
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
            self.source.n_ports,
        ):
            raise ValueError(
                "interval_currents have wrong shape"
            )
        query = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            query.ndim
            == 1
        )
        query = np.atleast_2d(
            query
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
                response = self.step_response_matrix(
                    query,
                    age_start,
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


class PreparedMultiThermalInterfaceField:
    """Mesh-free transient transmission through disjoint or nested interfaces."""

    def __init__(
        self,
        source,
        background_medium: HomogeneousThermalMedium,
        package_regions,
        *,
        surface_vertical_order: int,
        surface_azimuthal_order: int,
        mfs_offset_fraction: float,
        stehfest_order: int,
        interface_residual_tolerance: float,
        svd_rcond: float,
    ):
        self.source = source
        self.medium = background_medium
        self.background_medium = background_medium
        self.stehfest_order = int(
            stehfest_order
        )
        self.stehfest_weights = (
            _stehfest_weights(
                self.stehfest_order
            )
        )
        self.interface_residual_tolerance = float(
            interface_residual_tolerance
        )
        self.svd_rcond = float(
            svd_rcond
        )
        if (
            self.interface_residual_tolerance
            <= 0.0
            or self.svd_rcond
            <= 0.0
            or not (
                0.0
                < mfs_offset_fraction
                < 0.5
            )
        ):
            raise ValueError(
                "invalid multi-interface thermal solver configuration"
            )

        regions = []
        start = 0
        for (
            package_index,
            geometry,
            package_medium,
        ) in package_regions:
            surface = geometry.surface_quadrature(
                vertical_order=(
                    surface_vertical_order
                ),
                azimuthal_order=(
                    surface_azimuthal_order
                ),
            )
            offset = (
                float(
                    mfs_offset_fraction
                )
                * float(
                    np.min(
                        geometry.half_extents
                    )
                )
            )
            for _ in range(
                12
            ):
                exterior_mfs = (
                    surface.positions
                    - offset
                    * surface.normals
                )
                interior_mfs = (
                    surface.positions
                    + offset
                    * surface.normals
                )
                if (
                    np.all(
                        geometry.contains(
                            exterior_mfs,
                            tolerance=1e-12,
                        )
                    )
                    and not np.any(
                        geometry.contains(
                            interior_mfs,
                            tolerance=1e-12,
                        )
                    )
                ):
                    break
                offset *= 0.5
            else:
                raise RuntimeError(
                    "failed to place multi-package thermal MFS sources"
                )
            count = len(
                surface.positions
            )
            regions.append(
                {
                    "package_index": int(
                        package_index
                    ),
                    "geometry": geometry,
                    "medium": package_medium,
                    "positions": surface.positions,
                    "normals": surface.normals,
                    "exterior_mfs": exterior_mfs,
                    "interior_mfs": interior_mfs,
                    "slice": slice(
                        start,
                        start
                        + count,
                    ),
                    "offset": float(
                        offset
                    ),
                }
            )
            start += count

        geometries = tuple(
            region[
                "geometry"
            ]
            for region in regions
        )
        self.topology = package_domain_topology(
            geometries
        )
        children = [
            []
            for _ in regions
        ]
        for index, parent in enumerate(
            self.topology.parent
        ):
            if parent is not None:
                children[
                    parent
                ].append(
                    index
                )
        for index, region in enumerate(
            regions
        ):
            parent = self.topology.parent[
                index
            ]
            region[
                "parent"
            ] = parent
            region[
                "children"
            ] = tuple(
                children[
                    index
                ]
            )
            region[
                "outside_medium"
            ] = (
                self.background_medium
                if parent is None
                else regions[
                    parent
                ][
                    "medium"
                ]
            )
            if parent is not None:
                parent_geometry = regions[
                    parent
                ][
                    "geometry"
                ]
                offset = float(
                    region[
                        "offset"
                    ]
                )
                for _ in range(
                    12
                ):
                    exterior_mfs = (
                        region[
                            "positions"
                        ]
                        - offset
                        * region[
                            "normals"
                        ]
                    )
                    interior_mfs = (
                        region[
                            "positions"
                        ]
                        + offset
                        * region[
                            "normals"
                        ]
                    )
                    if (
                        np.all(
                            region[
                                "geometry"
                            ].contains(
                                exterior_mfs,
                                tolerance=1e-12,
                            )
                        )
                        and not np.any(
                            region[
                                "geometry"
                            ].contains(
                                interior_mfs,
                                tolerance=1e-12,
                            )
                        )
                        and np.all(
                            parent_geometry.contains(
                                interior_mfs,
                                tolerance=1e-12,
                            )
                        )
                    ):
                        region[
                            "exterior_mfs"
                        ] = exterior_mfs
                        region[
                            "interior_mfs"
                        ] = interior_mfs
                        region[
                            "offset"
                        ] = offset
                        break
                    offset *= 0.5
                else:
                    raise RuntimeError(
                        "failed to place nested thermal MFS sources inside "
                        "the parent material shell"
                    )

        self.regions = tuple(
            regions
        )
        self.surface_positions = np.concatenate(
            [
                region[
                    "positions"
                ]
                for region in self.regions
            ],
            axis=0,
        )
        self.surface_normals = np.concatenate(
            [
                region[
                    "normals"
                ]
                for region in self.regions
            ],
            axis=0,
        )
        self.exterior_mfs_sources = np.concatenate(
            [
                region[
                    "exterior_mfs"
                ]
                for region in self.regions
            ],
            axis=0,
        )
        self.source_region = np.asarray(
            self.topology.deepest_containing(
                tuple(
                    region[
                        "geometry"
                    ]
                    for region in self.regions
                ),
                source.positions,
                tolerance=1e-12,
            ),
            dtype=int,
        )

        self.source_strength = (
            source.volume_weights[
                :,
                None,
                None,
            ]
            * source.dissipation_matrices
        )
        self.source_strength_flat = (
            self.source_strength.reshape(
                len(
                    source.volume_weights
                ),
                -1,
            )
        )
        self._solve_cache = {}
        self._residual_cache = {}
        self._condition_cache = {}

    @property
    def maximum_interface_residual(
        self,
    ) -> float:
        if not self._residual_cache:
            return 0.0
        return float(
            max(
                self._residual_cache.values()
            )
        )

    @property
    def maximum_interface_condition(
        self,
    ) -> float:
        if not self._condition_cache:
            return 0.0
        return float(
            max(
                self._condition_cache.values()
            )
        )

    def _particular(
        self,
        points,
        laplace_s: float,
        *,
        region_index: int,
        normals=None,
    ):
        mask = (
            self.source_region
            == region_index
        )
        medium = (
            self.background_medium
            if region_index
            < 0
            else self.regions[
                region_index
            ][
                "medium"
            ]
        )
        width = (
            self.source.n_ports
            * self.source.n_ports
        )
        if not np.any(
            mask
        ):
            zeros = np.zeros(
                (
                    len(
                        points
                    ),
                    width,
                ),
                dtype=complex,
            )
            return (
                zeros,
                zeros
                if normals is not None
                else None,
            )
        kernel = _yukawa_kernel(
            points,
            self.source.positions[
                mask
            ],
            medium,
            laplace_s,
            source_radius=(
                self.source.effective_radius[
                    mask
                ]
            ),
        )[
            0
        ]
        values = (
            kernel
            @ self.source_strength_flat[
                mask
            ]
        )
        if normals is None:
            return (
                values,
                None,
            )
        flux = (
            _normal_flux_kernel(
                points,
                normals,
                self.source.positions[
                    mask
                ],
                medium,
                laplace_s,
                source_radius=(
                    self.source.effective_radius[
                        mask
                    ]
                ),
            )
            @ self.source_strength_flat[
                mask
            ]
        )
        return (
            values,
            flux,
        )

    def _region_basis(
        self,
        points,
        laplace_s: float,
        *,
        region_index: int,
        normals=None,
    ):
        points = np.asarray(
            points,
            dtype=float,
        )
        n = len(
            self.surface_positions
        )
        value = np.zeros(
            (
                len(
                    points
                ),
                2
                * n,
            ),
            dtype=float,
        )
        flux = (
            None
            if normals is None
            else np.zeros_like(
                value
            )
        )
        medium = (
            self.background_medium
            if region_index < 0
            else self.regions[
                region_index
            ][
                "medium"
            ]
        )
        blocks = []
        if region_index < 0:
            for index, region in enumerate(
                self.regions
            ):
                if region[
                    "parent"
                ] is None:
                    blocks.append(
                        (
                            region[
                                "slice"
                            ],
                            region[
                                "exterior_mfs"
                            ],
                        )
                    )
        else:
            region = self.regions[
                region_index
            ]
            sl = region[
                "slice"
            ]
            blocks.append(
                (
                    slice(
                        n
                        + sl.start,
                        n
                        + sl.stop,
                    ),
                    region[
                        "interior_mfs"
                    ],
                )
            )
            for child_index in region[
                "children"
            ]:
                child = self.regions[
                    child_index
                ]
                blocks.append(
                    (
                        child[
                            "slice"
                        ],
                        child[
                            "exterior_mfs"
                        ],
                    )
                )

        for columns, sources in blocks:
            value[
                :,
                columns,
            ] = _yukawa_kernel(
                points,
                sources,
                medium,
                laplace_s,
            )[
                0
            ]
            if normals is not None:
                flux[
                    :,
                    columns,
                ] = _normal_flux_kernel(
                    points,
                    normals,
                    sources,
                    medium,
                    laplace_s,
                )
        return (
            value,
            flux,
            medium,
        )

    def _interface_solution(
        self,
        laplace_s: float,
    ):
        key = float(
            laplace_s
        )
        cached = self._solve_cache.get(
            key
        )
        if cached is not None:
            return cached

        n = len(
            self.surface_positions
        )
        width = (
            self.source.n_ports
            * self.source.n_ports
        )
        matrix = np.zeros(
            (
                2
                * n,
                2
                * n,
            ),
            dtype=float,
        )
        rhs_value = np.zeros(
            (
                n,
                width,
            ),
            dtype=complex,
        )
        rhs_flux = np.zeros_like(
            rhs_value
        )

        for region_index, region in enumerate(
            self.regions
        ):
            sl = region[
                "slice"
            ]
            points = region[
                "positions"
            ]
            normals = region[
                "normals"
            ]
            parent = region[
                "parent"
            ]
            outside_index = (
                -1
                if parent is None
                else parent
            )
            (
                outside_value,
                outside_flux,
                outside_medium,
            ) = self._region_basis(
                points,
                laplace_s,
                region_index=(
                    outside_index
                ),
                normals=normals,
            )
            (
                inside_value,
                inside_flux,
                inside_medium,
            ) = self._region_basis(
                points,
                laplace_s,
                region_index=(
                    region_index
                ),
                normals=normals,
            )
            matrix[
                sl,
                :,
            ] = (
                outside_value
                - inside_value
            )
            matrix[
                n
                + sl.start:
                n
                + sl.stop,
                :,
            ] = (
                outside_flux
                - inside_flux
            )

            (
                outside_particular,
                outside_particular_flux,
            ) = self._particular(
                points,
                laplace_s,
                region_index=(
                    outside_index
                ),
                normals=normals,
            )
            (
                inside_particular,
                inside_particular_flux,
            ) = self._particular(
                points,
                laplace_s,
                region_index=(
                    region_index
                ),
                normals=normals,
            )
            rhs_value[
                sl
            ] = (
                inside_particular
                - outside_particular
            )
            rhs_flux[
                sl
            ] = (
                inside_particular_flux
                - outside_particular_flux
            )

        rhs = np.vstack(
            (
                rhs_value,
                rhs_flux,
            )
        )
        (
            solution,
            residual,
            rank,
            condition,
        ) = _equilibrated_lstsq(
            matrix,
            rhs,
            rcond=(
                self.svd_rcond
            ),
        )
        if (
            residual
            > self.interface_residual_tolerance
        ):
            raise RuntimeError(
                "multi-package thermal interface solve did not meet the "
                "declared residual tolerance: "
                f"{residual:.3e} > "
                f"{self.interface_residual_tolerance:.3e}"
            )
        if rank < min(
            matrix.shape
        ):
            raise RuntimeError(
                "multi-package thermal interface MFS system is rank deficient"
            )

        coefficients = (
            solution[
                :n
            ],
            solution[
                n:
            ],
        )
        self._solve_cache[
            key
        ] = coefficients
        self._residual_cache[
            key
        ] = residual
        self._condition_cache[
            key
        ] = condition
        return coefficients

    def laplace_response_matrix(
        self,
        points,
        laplace_s: float,
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
            points.ndim
            != 2
            or points.shape[
                1
            ]
            != 3
        ):
            raise ValueError(
                "query points must have shape (3,) or (n,3)"
            )
        if (
            not np.isfinite(
                laplace_s
            )
            or laplace_s < 0.0
        ):
            raise ValueError(
                "laplace_s must be finite and nonnegative"
            )

        (
            exterior_coefficients,
            interior_coefficients,
        ) = self._interface_solution(
            float(
                laplace_s
            )
        )
        membership = np.asarray(
            self.topology.deepest_containing(
                tuple(
                    region[
                        "geometry"
                    ]
                    for region in self.regions
                ),
                points,
                tolerance=1e-12,
            ),
            dtype=int,
        )
        coefficients = np.vstack(
            (
                exterior_coefficients,
                interior_coefficients,
            )
        )
        flat = np.zeros(
            (
                len(
                    points
                ),
                self.source.n_ports
                * self.source.n_ports,
            ),
            dtype=complex,
        )
        for region_index in (
            -1,
            *range(
                len(
                    self.regions
                )
            ),
        ):
            select = (
                membership
                == region_index
            )
            if not np.any(
                select
            ):
                continue
            query = points[
                select
            ]
            direct = self._particular(
                query,
                laplace_s,
                region_index=(
                    region_index
                ),
            )[
                0
            ]
            basis = self._region_basis(
                query,
                laplace_s,
                region_index=(
                    region_index
                ),
            )[
                0
            ]
            flat[
                select
            ] = (
                direct
                + basis
                @ coefficients
            )

        matrix = flat.reshape(
            len(
                points
            ),
            self.source.n_ports,
            self.source.n_ports,
        )
        matrix = 0.5 * (
            matrix
            + matrix.conj().transpose(
                0,
                2,
                1,
            )
        )
        return (
            matrix[
                0
            ]
            if scalar
            else matrix
        )

    def steady_response_matrix(
        self,
        points,
    ):
        return self.laplace_response_matrix(
            points,
            0.0,
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
        query = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            query.ndim
            == 1
        )
        query = np.atleast_2d(
            query
        )
        if time == 0.0:
            zeros = np.zeros(
                (
                    len(
                        query
                    ),
                    self.source.n_ports,
                    self.source.n_ports,
                ),
                dtype=complex,
            )
            return (
                zeros[
                    0
                ]
                if scalar
                else zeros
            )

        logarithm = np.log(
            2.0
        )
        result = np.zeros(
            (
                len(
                    query
                ),
                self.source.n_ports,
                self.source.n_ports,
            ),
            dtype=complex,
        )
        for k, weight in enumerate(
            self.stehfest_weights,
            start=1,
        ):
            laplace_s = (
                k
                * logarithm
                / float(
                    time
                )
            )
            result += (
                weight
                / k
                * np.asarray(
                    self.laplace_response_matrix(
                        query,
                        laplace_s,
                    ),
                    dtype=complex,
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

    @staticmethod
    def _contract(
        matrices,
        currents,
    ):
        return PreparedThermalGreenField._contract(
            matrices,
            currents,
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
            edges.ndim
            != 1
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
            self.source.n_ports,
        ):
            raise ValueError(
                "interval_currents have wrong shape"
            )
        if (
            times.ndim
            != 1
            or np.any(
                ~np.isfinite(
                    times
                )
            )
        ):
            raise ValueError(
                "observation_times must be finite"
            )
        query = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            query.ndim
            == 1
        )
        query = np.atleast_2d(
            query
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
                response = self.step_response_matrix(
                    query,
                    age_start,
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


class PiecewiseThermalInterfaceArtifact:
    """REFERENCE thermal transfer for one or more disjoint package interfaces."""

    def __init__(
        self,
        spatial_artifact,
        background_medium: HomogeneousThermalMedium,
        *,
        longitudinal_segments: int = 24,
        radial_order: int = 4,
        angular_order: int = 24,
        package_axial_order: int | None = None,
        package_radial_order: int | None = None,
        package_azimuthal_order: int | None = None,
        background_radial_order: int | None = None,
        background_angular_order: int | None = None,
        interface_vertical_order: int = 6,
        interface_azimuthal_order: int = 16,
        mfs_offset_fraction: float = 0.15,
        stehfest_order: int = 10,
        interface_residual_tolerance: float = 5e-5,
        svd_rcond: float = 1e-11,
        spatial_prepare_options=None,
    ):
        if not isinstance(
            background_medium,
            HomogeneousThermalMedium,
        ):
            raise NotImplementedError(
                "piecewise package thermal interfaces currently require an "
                "isotropic homogeneous thermal background"
            )
        self.spatial_artifact = (
            spatial_artifact
        )
        self.background_medium = (
            background_medium
        )
        self.longitudinal_segments = int(
            longitudinal_segments
        )
        self.radial_order = int(
            radial_order
        )
        self.angular_order = int(
            angular_order
        )
        self.package_axial_order = (
            package_axial_order
        )
        self.package_radial_order = (
            package_radial_order
        )
        self.package_azimuthal_order = (
            package_azimuthal_order
        )
        self.background_radial_order = (
            background_radial_order
        )
        self.background_angular_order = (
            background_angular_order
        )
        self.interface_vertical_order = int(
            interface_vertical_order
        )
        self.interface_azimuthal_order = int(
            interface_azimuthal_order
        )
        self.mfs_offset_fraction = float(
            mfs_offset_fraction
        )
        self.stehfest_order = int(
            stehfest_order
        )
        self.interface_residual_tolerance = float(
            interface_residual_tolerance
        )
        self.svd_rcond = float(
            svd_rcond
        )
        self.spatial_prepare_options = (
            {}
            if spatial_prepare_options
            is None
            else dict(
                spatial_prepare_options
            )
        )

    def prepare(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        thermal_packages = (
            scene_thermal_package_media(
                scene,
                self.background_medium,
            )
        )
        if not thermal_packages:
            return ContinuousThermalGreenArtifact(
                self.spatial_artifact,
                self.background_medium,
                longitudinal_segments=(
                    self.longitudinal_segments
                ),
                radial_order=(
                    self.radial_order
                ),
                angular_order=(
                    self.angular_order
                ),
                package_axial_order=(
                    self.package_axial_order
                ),
                package_radial_order=(
                    self.package_radial_order
                ),
                package_azimuthal_order=(
                    self.package_azimuthal_order
                ),
                background_radial_order=(
                    self.background_radial_order
                ),
                background_angular_order=(
                    self.background_angular_order
                ),
                spatial_prepare_options=(
                    self.spatial_prepare_options
                ),
            ).prepare(
                scene,
                frequency_hz,
            )
        validate_package_conductor_topology(
            scene
        )
        if hasattr(
            self.spatial_artifact,
            "prepare",
        ):
            spatial = self.spatial_artifact.prepare(
                scene,
                frequency_hz,
                **self.spatial_prepare_options,
            )
        elif hasattr(
            self.spatial_artifact,
            "prepare_spatial",
        ):
            spatial = (
                self.spatial_artifact.prepare_spatial(
                    scene,
                    frequency_hz,
                    **self.spatial_prepare_options,
                )
            )
        else:
            raise TypeError(
                "spatial_artifact must expose prepare or prepare_spatial"
            )

        source = build_thermal_source_quadrature(
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
            package_axial_order=(
                self.package_axial_order
            ),
            package_radial_order=(
                self.package_radial_order
            ),
            package_azimuthal_order=(
                self.package_azimuthal_order
            ),
            background_radial_order=(
                self.background_radial_order
            ),
            background_angular_order=(
                self.background_angular_order
            ),
        )
        if len(
            thermal_packages
        ) == 1:
            package_index, package_medium = (
                thermal_packages[
                    0
                ]
            )
            return PreparedThermalInterfaceField(
                source,
                self.background_medium,
                scene.packages[
                    package_index
                ].geometry,
                package_medium,
                surface_vertical_order=(
                    self.interface_vertical_order
                ),
                surface_azimuthal_order=(
                    self.interface_azimuthal_order
                ),
                mfs_offset_fraction=(
                    self.mfs_offset_fraction
                ),
                stehfest_order=(
                    self.stehfest_order
                ),
                interface_residual_tolerance=(
                    self.interface_residual_tolerance
                ),
                svd_rcond=(
                    self.svd_rcond
                ),
            )

        package_regions = tuple(
            (
                package_index,
                scene.packages[
                    package_index
                ].geometry,
                package_medium,
            )
            for (
                package_index,
                package_medium,
            )
            in thermal_packages
        )
        return PreparedMultiThermalInterfaceField(
            source,
            self.background_medium,
            package_regions,
            surface_vertical_order=(
                self.interface_vertical_order
            ),
            surface_azimuthal_order=(
                self.interface_azimuthal_order
            ),
            mfs_offset_fraction=(
                self.mfs_offset_fraction
            ),
            stehfest_order=(
                self.stehfest_order
            ),
            interface_residual_tolerance=(
                self.interface_residual_tolerance
            ),
            svd_rcond=(
                self.svd_rcond
            ),
        )
