from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

from .scene import EPS0, HomogeneousMedium, PackageObject


def _complex_permittivity(
    medium,
    frequency_hz: float,
) -> complex:
    if (
        not np.isfinite(frequency_hz)
        or frequency_hz < 0.0
    ):
        raise ValueError(
            "frequency_hz must be finite and nonnegative"
        )
    epsilon = (
        EPS0
        * float(
            medium.relative_permittivity
        )
    )
    conductivity = float(
        medium.conductivity
    )
    if conductivity == 0.0:
        return complex(
            epsilon
        )
    if frequency_hz == 0.0:
        raise ValueError(
            "conductive media require nonzero frequency in the "
            "complex-permittivity dielectric SIE"
        )
    omega = (
        2.0
        * np.pi
        * frequency_hz
    )
    return complex(
        epsilon
        - 1j
        * conductivity
        / omega
    )


@dataclass(frozen=True)
class DielectricSurfaceResult:
    positions: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    package_index: np.ndarray
    equivalent_density: np.ndarray
    background_permittivity: complex
    normalized_residual: float

    def __post_init__(self):
        positions = np.asarray(
            self.positions,
            dtype=float,
        )
        normals = np.asarray(
            self.normals,
            dtype=float,
        )
        weights = np.asarray(
            self.weights,
            dtype=float,
        )
        package_index = np.asarray(
            self.package_index,
            dtype=int,
        )
        density = np.asarray(
            self.equivalent_density,
            dtype=complex,
        )
        n = len(
            weights
        )
        if (
            positions.shape != (n, 3)
            or normals.shape != (n, 3)
            or package_index.shape != (n,)
            or density.shape != (n,)
        ):
            raise ValueError(
                "dielectric surface result arrays have incompatible shapes"
            )
        object.__setattr__(
            self,
            "positions",
            positions,
        )
        object.__setattr__(
            self,
            "normals",
            normals,
        )
        object.__setattr__(
            self,
            "weights",
            weights,
        )
        object.__setattr__(
            self,
            "package_index",
            package_index,
        )
        object.__setattr__(
            self,
            "equivalent_density",
            density,
        )

    @property
    def physical_surface_charge_density(
        self,
    ) -> np.ndarray:
        return (
            self.background_permittivity
            * self.equivalent_density
        )

    @property
    def net_equivalent_charge(
        self,
    ) -> complex:
        return complex(
            np.sum(
                self.physical_surface_charge_density
                * self.weights
            )
        )

    def induced_dipole(
        self,
        origin=(0.0, 0.0, 0.0),
    ) -> np.ndarray:
        origin = np.asarray(
            origin,
            dtype=float,
        )
        if origin.shape != (3,):
            raise ValueError(
                "origin must have shape (3,)"
            )
        return np.sum(
            (
                self.positions
                - origin
            )
            * (
                self.physical_surface_charge_density
                * self.weights
            )[
                :,
                None,
            ],
            axis=0,
        )

    def induced_potential(
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
        if points.shape[1] != 3:
            raise ValueError(
                "points must have shape (3,) or (n,3)"
            )
        diff = (
            points[
                :,
                None,
                :,
            ]
            - self.positions[
                None,
                :,
                :,
            ]
        )
        distance = np.linalg.norm(
            diff,
            axis=2,
        )
        scale = max(
            float(
                np.max(
                    np.linalg.norm(
                        self.positions,
                        axis=1,
                    )
                )
            ),
            1.0,
        )
        if np.any(
            distance
            <= 1e-13
            * scale
        ):
            raise ValueError(
                "induced_potential is defined only away from the "
                "surface quadrature nodes"
            )
        potential = np.sum(
            (
                self.equivalent_density
                * self.weights
            )[
                None,
                :,
            ]
            / (
                4.0
                * np.pi
                * distance
            ),
            axis=1,
        )
        return (
            potential[0]
            if scalar
            else potential
        )


class DielectricSurfaceSolver:
    """Dense quasi-static dielectric SIE on intrinsic superquadric surfaces.

    The unknown is an equivalent single-layer density whose potential uses
    G=1/(4*pi*r). The principal-value K* operator is evaluated by Nyström
    quadrature with the singular self sample omitted; convergence is assessed
    by independent surface-order refinement.
    """

    def __init__(
        self,
        packages,
        background: HomogeneousMedium,
        frequency_hz: float,
        *,
        vertical_order: int = 16,
        azimuthal_order: int = 32,
    ):
        self.packages = tuple(
            packages
        )
        if not self.packages:
            raise ValueError(
                "at least one dielectric package is required"
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
            HomogeneousMedium,
        ):
            raise TypeError(
                "background must be HomogeneousMedium"
            )
        self.background = (
            background
        )
        self.frequency_hz = float(
            frequency_hz
        )
        self.vertical_order = int(
            vertical_order
        )
        self.azimuthal_order = int(
            azimuthal_order
        )
        self.background_permittivity = (
            _complex_permittivity(
                background,
                self.frequency_hz,
            )
        )
        self._build_geometry()

    def _build_geometry(
        self,
    ):
        positions = []
        normals = []
        weights = []
        package_index = []
        package_slices = []
        cursor = 0
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
                slice(
                    cursor,
                    cursor + count,
                )
            )
            cursor += count
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

    def _adjoint_double_layer(
        self,
    ) -> np.ndarray:
        diff = (
            self.positions[
                :,
                None,
                :,
            ]
            - self.positions[
                None,
                :,
                :,
            ]
        )
        distance = np.linalg.norm(
            diff,
            axis=2,
        )
        np.fill_diagonal(
            distance,
            np.inf,
        )
        normal_dot = np.einsum(
            "ik,ijk->ij",
            self.normals,
            diff,
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
            :,
        ]
        np.fill_diagonal(
            kernel,
            0.0,
        )
        return kernel

    def operator_matrix(
        self,
    ) -> np.ndarray:
        Kstar = (
            self._adjoint_double_layer()
        ).astype(
            complex
        )
        matrix = (
            -Kstar
        )
        for package_index, package_slice in enumerate(
            self.package_slices
        ):
            epsilon_inside = (
                package.material.complex_permittivity(
                    self.frequency_hz
                )
            )
            epsilon_outside = (
                self.background_permittivity
            )
            contrast = (
                epsilon_inside
                - epsilon_outside
            )
            indices = np.arange(
                package_slice.start,
                package_slice.stop,
            )
            if abs(
                contrast
            ) <= (
                1e-13
                * max(
                    abs(
                        epsilon_inside
                    ),
                    abs(
                        epsilon_outside
                    ),
                    1e-30,
                )
            ):
                matrix[
                    indices,
                    :,
                ] = 0.0
                matrix[
                    indices,
                    indices,
                ] = 1.0
                continue
            coefficient = (
                (
                    epsilon_outside
                    + epsilon_inside
                )
                / (
                    2.0
                    * (
                        epsilon_outside
                        - epsilon_inside
                    )
                )
            )
            matrix[
                indices,
                indices,
            ] += coefficient
        return matrix

    def solve_normal_potential_derivative(
        self,
        normal_derivative,
    ) -> DielectricSurfaceResult:
        rhs = np.asarray(
            normal_derivative,
            dtype=complex,
        )
        if rhs.shape != (
            len(
                self.weights
            ),
        ):
            raise ValueError(
                "normal_derivative has wrong shape"
            )
        matrix = (
            self.operator_matrix()
        )
        effective_rhs = (
            rhs.copy()
        )
        for package_index, package_slice in enumerate(
            self.package_slices
        ):
            epsilon_inside = (
                self.packages[
                    package_index
                ].material.complex_permittivity(
                    self.frequency_hz
                )
            )
            contrast = (
                epsilon_inside
                - self.background_permittivity
            )
            if abs(
                contrast
            ) <= (
                1e-13
                * max(
                    abs(
                        epsilon_inside
                    ),
                    abs(
                        self.background_permittivity
                    ),
                    1e-30,
                )
            ):
                effective_rhs[
                    package_slice
                ] = 0.0

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
        eta = float(
            np.linalg.norm(
                residual
            )
            / max(
                np.linalg.norm(
                    effective_rhs
                ),
                1.0,
            )
        )
        return DielectricSurfaceResult(
            self.positions,
            self.normals,
            self.weights,
            self.package_index,
            density,
            self.background_permittivity,
            eta,
        )

    def solve_uniform_field(
        self,
        electric_field,
    ) -> DielectricSurfaceResult:
        field = np.asarray(
            electric_field,
            dtype=complex,
        )
        if field.shape != (3,):
            raise ValueError(
                "electric_field must have shape (3,)"
            )
        normal_derivative = (
            -self.normals
            @ field
        )
        return (
            self.solve_normal_potential_derivative(
                normal_derivative
            )
        )
