from __future__ import annotations

from dataclasses import dataclass, field
from math import gamma
import numpy as np

from .geometry import RigidPose


def _positive_vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(
        value,
        dtype=float,
    )
    if (
        arr.shape != (3,)
        or not np.all(
            np.isfinite(arr)
        )
        or np.any(
            arr <= 0.0
        )
    ):
        raise ValueError(
            f"{name} must be a finite positive length-3 vector"
        )
    return arr


def _signed_power(
    value,
    exponent: float,
):
    value = np.asarray(
        value,
        dtype=float,
    )
    return (
        np.sign(value)
        * np.abs(value) ** exponent
    )


@dataclass(frozen=True)
class SuperquadricVolumeQuadrature:
    positions: np.ndarray
    weights: np.ndarray
    local_positions: np.ndarray

    def __post_init__(self):
        positions = np.asarray(
            self.positions,
            dtype=float,
        )
        weights = np.asarray(
            self.weights,
            dtype=float,
        )
        local_positions = np.asarray(
            self.local_positions,
            dtype=float,
        )
        n = len(weights)
        if (
            positions.shape != (n, 3)
            or local_positions.shape != (n, 3)
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(local_positions))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "volume quadrature arrays must be finite and shape-compatible"
            )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "local_positions",
            local_positions,
        )

    @property
    def volume(self) -> float:
        return float(
            np.sum(self.weights)
        )


@dataclass(frozen=True)
class SuperquadricSurfaceQuadrature:
    positions: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    eta: np.ndarray
    azimuth: np.ndarray

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
        eta = np.asarray(
            self.eta,
            dtype=float,
        )
        azimuth = np.asarray(
            self.azimuth,
            dtype=float,
        )
        n = len(
            weights
        )
        if (
            positions.shape != (n, 3)
            or normals.shape != (n, 3)
            or eta.shape != (n,)
            or azimuth.shape != (n,)
        ):
            raise ValueError(
                "surface quadrature arrays have incompatible shapes"
            )
        if (
            np.any(
                ~np.isfinite(
                    positions
                )
            )
            or np.any(
                ~np.isfinite(
                    normals
                )
            )
            or np.any(
                ~np.isfinite(
                    weights
                )
            )
            or np.any(
                weights <= 0.0
            )
        ):
            raise ValueError(
                "surface quadrature must be finite with positive weights"
            )
        norm = np.linalg.norm(
            normals,
            axis=1,
        )
        if not np.allclose(
            norm,
            1.0,
            rtol=0.0,
            atol=2e-12,
        ):
            raise ValueError(
                "surface quadrature normals must be unit length"
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
            "eta",
            eta,
        )
        object.__setattr__(
            self,
            "azimuth",
            azimuth,
        )

    @property
    def area(
        self,
    ) -> float:
        return float(
            np.sum(
                self.weights
            )
        )


@dataclass(frozen=True)
class SuperquadricPackageGeometry:
    """Continuous superellipsoid package primitive.

    The implicit local-coordinate surface is

        (|x/a|^p + |y/b|^p)^(q/p) + |z/c|^q = 1.

    p=q=2 is an ellipsoid; larger exponents approach rounded boxes.
    """

    half_extents: np.ndarray
    exponent_xy: float = 4.0
    exponent_z: float = 4.0
    pose: RigidPose = field(
        default_factory=RigidPose.identity
    )

    def __post_init__(self):
        object.__setattr__(
            self,
            "half_extents",
            _positive_vec3(
                self.half_extents,
                "half_extents",
            ),
        )
        for name, value in (
            (
                "exponent_xy",
                self.exponent_xy,
            ),
            (
                "exponent_z",
                self.exponent_z,
            ),
        ):
            if (
                not np.isfinite(value)
                or float(value) < 2.0
            ):
                raise ValueError(
                    f"{name} must be finite and >= 2"
                )

    @property
    def characteristic_length(
        self,
    ) -> float:
        return float(
            np.cbrt(
                np.prod(
                    self.half_extents
                )
            )
        )

    @property
    def volume(
        self,
    ) -> float:
        a, b, c = (
            self.half_extents
        )
        p = float(
            self.exponent_xy
        )
        q = float(
            self.exponent_z
        )
        area_xy = (
            4.0
            * a
            * b
            * gamma(
                1.0
                + 1.0 / p
            ) ** 2
            / gamma(
                1.0
                + 2.0 / p
            )
        )
        vertical_factor = (
            2.0
            * c
            * gamma(
                1.0
                + 1.0 / q
            )
            * gamma(
                1.0
                + 2.0 / q
            )
            / gamma(
                1.0
                + 3.0 / q
            )
        )
        return float(
            area_xy
            * vertical_factor
        )

    def world_to_local(
        self,
        points,
    ) -> np.ndarray:
        points = np.asarray(
            points,
            dtype=float,
        )
        if (
            points.shape[-1:]
            != (3,)
        ):
            raise ValueError(
                "points must end in dimension 3"
            )
        return (
            (
                points
                - self.pose.translation
            )
            @ self.pose.rotation
        )

    def local_to_world(
        self,
        points,
    ) -> np.ndarray:
        return self.pose.apply(
            points
        )

    def implicit_local(
        self,
        points,
    ) -> np.ndarray:
        local = np.asarray(
            points,
            dtype=float,
        )
        if (
            local.shape[-1:]
            != (3,)
        ):
            raise ValueError(
                "points must end in dimension 3"
            )
        a, b, c = (
            self.half_extents
        )
        p = float(
            self.exponent_xy
        )
        q = float(
            self.exponent_z
        )
        xy = (
            np.abs(
                local[..., 0]
                / a
            ) ** p
            + np.abs(
                local[..., 1]
                / b
            ) ** p
        )
        return (
            xy ** (
                q / p
            )
            + np.abs(
                local[..., 2]
                / c
            ) ** q
            - 1.0
        )

    def implicit(
        self,
        points,
    ) -> np.ndarray:
        return self.implicit_local(
            self.world_to_local(
                points
            )
        )

    def contains(
        self,
        points,
        *,
        tolerance: float = 1e-12,
    ) -> np.ndarray:
        if tolerance < 0.0:
            raise ValueError(
                "tolerance must be nonnegative"
            )
        return (
            self.implicit(
                points
            )
            <= tolerance
        )

    def surface_points(
        self,
        vertical_order: int = 17,
        azimuthal_order: int = 48,
    ) -> np.ndarray:
        if (
            vertical_order < 3
            or azimuthal_order < 8
        ):
            raise ValueError(
                "surface sampling orders are too small"
            )
        eta = np.linspace(
            -0.5 * np.pi,
            0.5 * np.pi,
            vertical_order,
        )
        omega = np.linspace(
            -np.pi,
            np.pi,
            azimuthal_order,
            endpoint=False,
        )
        ee, ww = np.meshgrid(
            eta,
            omega,
            indexing="ij",
        )
        a, b, c = (
            self.half_extents
        )
        vertical_power = (
            2.0
            / float(
                self.exponent_z
            )
        )
        horizontal_power = (
            2.0
            / float(
                self.exponent_xy
            )
        )
        radial = _signed_power(
            np.cos(ee),
            vertical_power,
        )
        x = (
            a
            * radial
            * _signed_power(
                np.cos(ww),
                horizontal_power,
            )
        )
        y = (
            b
            * radial
            * _signed_power(
                np.sin(ww),
                horizontal_power,
            )
        )
        z = (
            c
            * _signed_power(
                np.sin(ee),
                vertical_power,
            )
        )
        local = np.stack(
            (
                x,
                y,
                z,
            ),
            axis=-1,
        )
        return self.local_to_world(
            local.reshape(
                -1,
                3,
            )
        )

    def surface_quadrature(
        self,
        vertical_order: int = 24,
        azimuthal_order: int = 48,
    ) -> SuperquadricSurfaceQuadrature:
        if (
            vertical_order < 4
            or vertical_order % 2 != 0
            or azimuthal_order < 8
        ):
            raise ValueError(
                "surface quadrature requires an even vertical_order >= 4 "
                "and azimuthal_order >= 8"
            )

        eta = (
            -0.5
            * np.pi
            + (
                np.arange(
                    vertical_order
                )
                + 0.5
            )
            * np.pi
            / vertical_order
        )
        azimuth = (
            -np.pi
            + (
                np.arange(
                    azimuthal_order
                )
                + 0.5
            )
            * 2.0
            * np.pi
            / azimuthal_order
        )
        ee, ww = np.meshgrid(
            eta,
            azimuth,
            indexing="ij",
        )

        a, b, c = (
            self.half_extents
        )
        ev = (
            2.0
            / float(
                self.exponent_z
            )
        )
        eh = (
            2.0
            / float(
                self.exponent_xy
            )
        )

        cos_eta = np.cos(
            ee
        )
        sin_eta = np.sin(
            ee
        )
        cos_az = np.cos(
            ww
        )
        sin_az = np.sin(
            ww
        )

        cv = _signed_power(
            cos_eta,
            ev,
        )
        sv = _signed_power(
            sin_eta,
            ev,
        )
        cp = _signed_power(
            cos_az,
            eh,
        )
        sp = _signed_power(
            sin_az,
            eh,
        )

        dcv = (
            -ev
            * np.abs(
                cos_eta
            ) ** (
                ev - 1.0
            )
            * sin_eta
        )
        dsv = (
            ev
            * np.abs(
                sin_eta
            ) ** (
                ev - 1.0
            )
            * cos_eta
        )
        dcp = (
            -eh
            * np.abs(
                cos_az
            ) ** (
                eh - 1.0
            )
            * sin_az
        )
        dsp = (
            eh
            * np.abs(
                sin_az
            ) ** (
                eh - 1.0
            )
            * cos_az
        )

        local = np.stack(
            (
                a
                * cv
                * cp,
                b
                * cv
                * sp,
                c
                * sv,
            ),
            axis=-1,
        )
        derivative_eta = (
            np.stack(
                (
                    a
                    * dcv
                    * cp,
                    b
                    * dcv
                    * sp,
                    c
                    * dsv,
                ),
                axis=-1,
            )
        )
        derivative_azimuth = (
            np.stack(
                (
                    a
                    * cv
                    * dcp,
                    b
                    * cv
                    * dsp,
                    np.zeros_like(
                        cv
                    ),
                ),
                axis=-1,
            )
        )

        outward = np.cross(
            derivative_azimuth,
            derivative_eta,
        )
        jacobian = np.linalg.norm(
            outward,
            axis=-1,
        )
        if np.any(
            ~np.isfinite(
                jacobian
            )
        ) or np.any(
            jacobian <= 0.0
        ):
            raise RuntimeError(
                "superquadric parameterization produced a singular "
                "surface quadrature point"
            )

        local_normals = (
            outward
            / jacobian[
                ...,
                None,
            ]
        )
        world_positions = (
            self.local_to_world(
                local.reshape(
                    -1,
                    3,
                )
            )
        )
        world_normals = (
            local_normals.reshape(
                -1,
                3,
            )
            @ self.pose.rotation.T
        )
        world_normals /= np.linalg.norm(
            world_normals,
            axis=1,
        )[
            :,
            None,
        ]
        parameter_weight = (
            np.pi
            / vertical_order
            * 2.0
            * np.pi
            / azimuthal_order
        )
        weights = (
            jacobian.reshape(
                -1
            )
            * parameter_weight
        )
        return SuperquadricSurfaceQuadrature(
            world_positions,
            world_normals,
            weights,
            ee.reshape(
                -1
            ),
            ww.reshape(
                -1
            ),
        )

    def volume_quadrature(
        self,
        axial_order: int = 12,
        radial_order: int = 8,
        azimuthal_order: int = 32,
    ) -> SuperquadricVolumeQuadrature:
        if (
            axial_order < 2
            or radial_order < 2
            or azimuthal_order < 8
        ):
            raise ValueError(
                "volume quadrature orders are too small"
            )

        uz, wz = np.polynomial.legendre.leggauss(
            axial_order
        )
        rr_raw, wr_raw = np.polynomial.legendre.leggauss(
            radial_order
        )
        rr = 0.5 * (
            rr_raw + 1.0
        )
        wr = 0.5 * wr_raw
        theta = (
            np.arange(
                azimuthal_order
            )
            + 0.5
        ) * (
            2.0
            * np.pi
            / azimuthal_order
        )
        wt = (
            2.0
            * np.pi
            / azimuthal_order
        )

        a, b, c_axis = self.half_extents
        p = float(
            self.exponent_xy
        )
        q = float(
            self.exponent_z
        )

        ct = np.cos(theta)
        st = np.sin(theta)
        radial_boundary = (
            (
                np.abs(ct) / a
            ) ** p
            + (
                np.abs(st) / b
            ) ** p
        ) ** (
            -1.0 / p
        )

        local = []
        weights = []
        for u, wu in zip(
            uz,
            wz,
        ):
            scale_xy = (
                max(
                    1.0
                    - abs(float(u)) ** q,
                    0.0,
                )
                ** (
                    1.0 / q
                )
            )
            if scale_xy <= 0.0:
                continue
            rho, angle = np.meshgrid(
                rr,
                theta,
                indexing="ij",
            )
            wr_grid, rb_grid = np.meshgrid(
                wr,
                radial_boundary,
                indexing="ij",
            )
            x = (
                scale_xy
                * rho
                * rb_grid
                * np.cos(angle)
            )
            y = (
                scale_xy
                * rho
                * rb_grid
                * np.sin(angle)
            )
            z = np.full_like(
                x,
                c_axis * u,
            )
            local.append(
                np.stack(
                    (
                        x.ravel(),
                        y.ravel(),
                        z.ravel(),
                    ),
                    axis=1,
                )
            )
            jacobian_xy = (
                scale_xy**2
                * rho
                * rb_grid**2
            )
            weights.append(
                (
                    c_axis
                    * wu
                    * wt
                    * wr_grid
                    * jacobian_xy
                ).ravel()
            )

        local_positions = np.concatenate(
            local,
            axis=0,
        )
        volume_weights = np.concatenate(
            weights
        )
        return SuperquadricVolumeQuadrature(
            self.local_to_world(
                local_positions
            ),
            volume_weights,
            local_positions,
        )

    def transformed(
        self,
        pose: RigidPose,
    ) -> "SuperquadricPackageGeometry":
        return SuperquadricPackageGeometry(
            self.half_extents.copy(),
            self.exponent_xy,
            self.exponent_z,
            pose.compose(
                self.pose
            ),
        )
