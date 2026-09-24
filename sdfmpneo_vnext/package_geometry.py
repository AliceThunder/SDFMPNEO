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
