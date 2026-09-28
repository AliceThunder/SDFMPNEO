from __future__ import annotations

import numpy as np
from scipy.linalg import solve

from .scene import (
    PackageObject,
    PassiveIsotropicMaterial,
)


class MagneticSurfaceSolver:
    """Magnetostatic permeability-transmission SIE on package surfaces.

    The unknown is an equivalent magnetic scalar single-layer density.  The
    incident H field is supplied by the exact same conductor current modes used
    by the MQS partial-inductance operator.
    """

    def __init__(
        self,
        packages,
        background: PassiveIsotropicMaterial,
        *,
        conductors=(),
        vertical_order: int = 16,
        azimuthal_order: int = 32,
        conductor_longitudinal_segments: int = 64,
        conductor_section_points: int = 16,
        conductor_end_radial_order: int = 3,
    ):
        self.packages = tuple(
            packages
        )
        self.conductors = tuple(
            conductors
        )
        if not self.packages:
            raise ValueError(
                "at least one magnetic package is required"
            )
        if not all(
            isinstance(
                package,
                PackageObject,
            )
            for package in self.packages
        ):
            raise TypeError(
                "packages must contain PackageObject instances"
            )
        if not isinstance(
            background,
            PassiveIsotropicMaterial,
        ):
            raise TypeError(
                "background must implement the passive isotropic material "
                "interface"
            )
        self.background = background
        self.background_permeability = float(
            background.permeability
        )
        self.vertical_order = int(
            vertical_order
        )
        self.azimuthal_order = int(
            azimuthal_order
        )
        self.conductor_longitudinal_segments = int(
            conductor_longitudinal_segments
        )
        self.conductor_section_points = int(
            conductor_section_points
        )
        self.conductor_end_radial_order = int(
            conductor_end_radial_order
        )
        if (
            self.vertical_order < 4
            or self.azimuthal_order < 8
            or self.conductor_longitudinal_segments < 2
            or self.conductor_section_points < 8
            or self.conductor_end_radial_order < 2
        ):
            raise ValueError(
                "magnetic surface quadrature orders are too small"
            )
        self._build_geometry()

    def _build_geometry(
        self,
    ):
        positions = []
        normals = []
        weights = []
        package_index = []
        interface_slices = []
        interface_inside_permeability = []
        interface_outside_permeability = []
        interface_kind = []
        package_slices = []
        cursor = 0

        # Package exterior interfaces. Normals point from package material to
        # the unbounded background.
        for index, package in enumerate(
            self.packages
        ):
            quadrature = (
                package.geometry.surface_quadrature(
                    self.vertical_order,
                    self.azimuthal_order,
                )
            )
            count = len(
                quadrature.weights
            )
            sl = slice(
                cursor,
                cursor
                + count,
            )
            cursor += count
            positions.append(
                quadrature.positions
            )
            normals.append(
                quadrature.normals
            )
            weights.append(
                quadrature.weights
            )
            package_index.append(
                np.full(
                    count,
                    index,
                    dtype=int,
                )
            )
            package_slices.append(
                sl
            )
            interface_slices.append(
                sl
            )
            interface_inside_permeability.append(
                float(
                    package.material.permeability
                )
            )
            interface_outside_permeability.append(
                self.background_permeability
            )
            interface_kind.append(
                (
                    "package",
                    index,
                )
            )

        # A conductor enclosed by a magnetic package is a nonmagnetic cavity
        # in the package material under the current conductor model. Its
        # closed boundary is therefore another permeability interface. The
        # conductor-surface normal points from the cavity into the package.
        conductor_cavity_slices = []
        for conductor_index, conductor in enumerate(
            self.conductors
        ):
            containing = []
            for package_index_value, package in enumerate(
                self.packages
            ):
                classification = (
                    package.geometry.classify_conductor(
                        conductor.geometry,
                        longitudinal_segments=64,
                        section_points=16,
                        tolerance=1e-10,
                    )
                )
                if classification == "inside":
                    containing.append(
                        package_index_value
                    )
            if len(
                containing
            ) > 1:
                raise NotImplementedError(
                    "a conductor enclosed by multiple package material regions "
                    "requires nested magnetic-domain topology"
                )
            if not containing:
                continue
            package_index_value = (
                containing[
                    0
                ]
            )
            package = self.packages[
                package_index_value
            ]
            package_mu = float(
                package.material.permeability
            )
            if np.isclose(
                package_mu,
                self.background_permeability,
                rtol=1e-12,
                atol=0.0,
            ):
                continue
            quadrature = (
                conductor.geometry.surface_quadrature(
                    longitudinal_segments=(
                        self.conductor_longitudinal_segments
                    ),
                    section_points=(
                        self.conductor_section_points
                    ),
                    end_radial_order=(
                        self.conductor_end_radial_order
                    ),
                )
            )
            count = len(
                quadrature.weights
            )
            sl = slice(
                cursor,
                cursor
                + count,
            )
            cursor += count
            positions.append(
                quadrature.positions
            )
            normals.append(
                quadrature.normals
            )
            weights.append(
                quadrature.weights
            )
            package_index.append(
                np.full(
                    count,
                    package_index_value,
                    dtype=int,
                )
            )
            conductor_cavity_slices.append(
                (
                    conductor_index,
                    sl,
                )
            )
            interface_slices.append(
                sl
            )
            interface_inside_permeability.append(
                self.background_permeability
            )
            interface_outside_permeability.append(
                package_mu
            )
            interface_kind.append(
                (
                    "conductor_cavity",
                    conductor_index,
                )
            )

        self.positions = np.concatenate(
            positions,
            axis=0,
        )
        self.normals = np.concatenate(
            normals,
            axis=0,
        )
        self.weights = np.concatenate(
            weights,
            axis=0,
        )
        self.package_index = np.concatenate(
            package_index,
            axis=0,
        )
        self.package_slices = tuple(
            package_slices
        )
        self.conductor_cavity_slices = tuple(
            conductor_cavity_slices
        )
        self.interface_slices = tuple(
            interface_slices
        )
        self.interface_inside_permeability = tuple(
            interface_inside_permeability
        )
        self.interface_outside_permeability = tuple(
            interface_outside_permeability
        )
        self.interface_kind = tuple(
            interface_kind
        )

    @property
    def has_contrast(
        self,
    ) -> bool:
        return any(
            not np.isclose(
                permeability_inside,
                permeability_outside,
                rtol=1e-12,
                atol=0.0,
            )
            for (
                permeability_inside,
                permeability_outside,
            )
            in zip(
                self.interface_inside_permeability,
                self.interface_outside_permeability,
            )
        )

    def _adjoint_double_layer(
        self,
    ) -> np.ndarray:
        difference = (
            self.positions[
                :,
                None,
                :
            ]
            - self.positions[
                None,
                :,
                :
            ]
        )
        distance = np.linalg.norm(
            difference,
            axis=2,
        )
        np.fill_diagonal(
            distance,
            np.inf,
        )
        normal_dot = np.einsum(
            "ik,ijk->ij",
            self.normals,
            difference,
        )
        kernel = (
            -normal_dot
            / (
                4.0
                * np.pi
                * distance**3
            )
        )
        kernel *= self.weights[
            None,
            :
        ]
        np.fill_diagonal(
            kernel,
            0.0,
        )
        return kernel

    def operator_matrix(
        self,
    ) -> np.ndarray:
        kstar = self._adjoint_double_layer()
        matrix = (
            -kstar
        )
        for (
            interface_slice,
            permeability_inside,
            permeability_outside,
        ) in zip(
            self.interface_slices,
            self.interface_inside_permeability,
            self.interface_outside_permeability,
        ):
            contrast = (
                permeability_inside
                - permeability_outside
            )
            indices = np.arange(
                interface_slice.start,
                interface_slice.stop,
            )
            if abs(
                contrast
            ) <= (
                1e-13
                * max(
                    abs(
                        permeability_inside
                    ),
                    abs(
                        permeability_outside
                    ),
                    1e-30,
                )
            ):
                matrix[
                    indices,
                    :
                ] = 0.0
                matrix[
                    indices,
                    indices,
                ] = 1.0
                continue
            coefficient = (
                (
                    permeability_outside
                    + permeability_inside
                )
                / (
                    2.0
                    * (
                        permeability_outside
                        - permeability_inside
                    )
                )
            )
            matrix[
                indices,
                indices,
            ] += coefficient
        return matrix

    def solve_density_matrix(
        self,
        incident_normal_potential_derivative,
    ):
        rhs = np.asarray(
            incident_normal_potential_derivative,
            dtype=float,
        )
        vector = (
            rhs.ndim == 1
        )
        if vector:
            rhs = rhs[
                :,
                None
            ]
        if (
            rhs.ndim != 2
            or rhs.shape[
                0
            ] != len(
                self.weights
            )
        ):
            raise ValueError(
                "incident normal derivative has incompatible shape"
            )
        effective_rhs = rhs.copy()
        for (
            interface_slice,
            permeability_inside,
            permeability_outside,
        ) in zip(
            self.interface_slices,
            self.interface_inside_permeability,
            self.interface_outside_permeability,
        ):
            if np.isclose(
                permeability_inside,
                permeability_outside,
                rtol=1e-12,
                atol=0.0,
            ):
                effective_rhs[
                    interface_slice,
                    :
                ] = 0.0
        matrix = self.operator_matrix()
        density = solve(
            matrix,
            effective_rhs,
            assume_a="gen",
            check_finite=True,
        )
        residual = (
            matrix
            @ density
            - effective_rhs
        )
        denominator = np.maximum(
            np.linalg.norm(
                effective_rhs,
                axis=0,
            ),
            1.0,
        )
        normalized = (
            np.linalg.norm(
                residual,
                axis=0,
            )
            / denominator
        )
        if vector:
            return (
                density[
                    :,
                    0
                ],
                float(
                    normalized[
                        0
                    ]
                ),
            )
        return (
            density,
            normalized,
        )

    def solve_incident_mode_fields(
        self,
        incident_surface_fields,
    ):
        fields = np.asarray(
            incident_surface_fields,
            dtype=float,
        )
        if (
            fields.ndim != 3
            or fields.shape[
                :2
            ] != (
                len(
                    self.weights
                ),
                3,
            )
        ):
            raise ValueError(
                "incident_surface_fields must have shape "
                "(n_surface,3,n_modes)"
            )
        normal_derivative = (
            -np.einsum(
                "sd,sdm->sm",
                self.normals,
                fields,
            )
        )
        return self.solve_density_matrix(
            normal_derivative
        )

    def induced_field(
        self,
        points,
        density_matrix,
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
        density = np.asarray(
            density_matrix,
            dtype=float,
        )
        vector = (
            density.ndim == 1
        )
        if vector:
            density = density[
                :,
                None
            ]
        if (
            points.ndim != 2
            or points.shape[
                1
            ] != 3
            or density.ndim != 2
            or density.shape[
                0
            ] != len(
                self.weights
            )
        ):
            raise ValueError(
                "points/density have incompatible shapes"
            )
        difference = (
            points[
                :,
                None,
                :
            ]
            - self.positions[
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
        scale = max(
            float(
                np.max(
                    np.linalg.norm(
                        self.positions
                        - np.mean(
                            self.positions,
                            axis=0,
                        )[
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
            distance_squared
            <= (
                1e-13
                * scale
            ) ** 2
        ):
            raise ValueError(
                "induced magnetic field is defined only away from surface "
                "quadrature nodes"
            )
        kernel = (
            difference
            / (
                4.0
                * np.pi
                * distance_squared[
                    :,
                    :,
                    None
                ] ** 1.5
            )
        )
        weighted_density = (
            self.weights[
                :,
                None
            ]
            * density
        )
        field = np.einsum(
            "qsd,sm->qdm",
            kernel,
            weighted_density,
        )
        if vector:
            field = field[
                :,
                :,
                0
            ]
        if scalar:
            return field[
                0
            ]
        return field

    def inductance_correction(
        self,
        mqs_teacher,
        *,
        volume_axial_order: int = 8,
        volume_radial_order: int = 6,
        volume_azimuthal_order: int = 24,
        maximum_raw_reciprocity_defect: float = 0.08,
    ):
        if maximum_raw_reciprocity_defect <= 0.0:
            raise ValueError(
                "maximum_raw_reciprocity_defect must be positive"
            )
        if not self.has_contrast:
            zeros = np.zeros(
                (
                    mqs_teacher._n_modes,
                    mqs_teacher._n_modes,
                ),
                dtype=float,
            )
            return (
                zeros,
                0.0,
                0.0,
            )

        incident_surface = (
            mqs_teacher.magnetic_field_mode_transfer(
                self.positions
            )
        )
        density, residuals = (
            self.solve_incident_mode_fields(
                incident_surface
            )
        )
        raw = np.zeros(
            (
                mqs_teacher._n_modes,
                mqs_teacher._n_modes,
            ),
            dtype=float,
        )

        for package_index, package in enumerate(
            self.packages
        ):
            delta_mu = float(
                package.material.permeability
                - self.background_permeability
            )
            if np.isclose(
                delta_mu,
                0.0,
                rtol=0.0,
                atol=(
                    1e-13
                    * max(
                        abs(
                            package.material.permeability
                        ),
                        abs(
                            self.background_permeability
                        ),
                        1e-30,
                    )
                ),
            ):
                continue
            quadrature = package.geometry.volume_quadrature(
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
            positions = quadrature.positions
            weights = quadrature.weights
            conductor = np.asarray(
                mqs_teacher.points_in_conductors(
                    positions
                ),
                dtype=bool,
            )
            positions = positions[
                ~conductor
            ]
            weights = weights[
                ~conductor
            ]
            if len(
                weights
            ) == 0:
                raise RuntimeError(
                    "magnetic package volume quadrature is fully occupied by "
                    "conductor volume"
                )

            for other_index, other in enumerate(
                self.packages
            ):
                if other_index == package_index:
                    continue
                if np.any(
                    other.geometry.contains(
                        positions,
                        tolerance=1e-12,
                    )
                ):
                    raise NotImplementedError(
                        "overlapping/nested magnetic package volumes require "
                        "a hierarchical material-domain formulation"
                    )

            incident = (
                mqs_teacher.magnetic_field_mode_transfer(
                    positions
                )
            )
            induced = self.induced_field(
                positions,
                density,
            )
            total = (
                incident
                + induced
            )
            raw += (
                delta_mu
                * np.einsum(
                    "qda,qdb,q->ab",
                    total,
                    incident,
                    weights,
                )
            )

        denominator = max(
            float(
                np.linalg.norm(
                    raw
                )
            ),
            1e-30,
        )
        reciprocity_defect = float(
            np.linalg.norm(
                raw
                - raw.T
            )
            / denominator
        )
        if (
            reciprocity_defect
            > maximum_raw_reciprocity_defect
        ):
            raise RuntimeError(
                "magnetic permeability correction failed the raw reciprocity "
                "diagnostic: "
                f"{reciprocity_defect:.3e} > "
                f"{maximum_raw_reciprocity_defect:.3e}"
            )
        correction = 0.5 * (
            raw
            + raw.T
        )
        return (
            correction,
            float(
                np.max(
                    residuals
                )
            ),
            reciprocity_defect,
        )
