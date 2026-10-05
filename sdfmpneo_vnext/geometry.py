from __future__ import annotations

from dataclasses import dataclass, field
from math import gamma, pi
import numpy as np


def _vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return arr


def _rot3(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(arr.T @ arr, np.eye(3), rtol=0.0, atol=2e-10):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(arr), 1.0, rtol=0.0, atol=2e-10):
        raise ValueError("rotation must have determinant +1")
    return arr


@dataclass(frozen=True)
class RigidPose:
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        object.__setattr__(self, "rotation", _rot3(self.rotation))
        object.__setattr__(self, "translation", _vec3(self.translation, "translation"))

    @staticmethod
    def identity() -> "RigidPose":
        return RigidPose()

    @staticmethod
    def from_axis_angle(axis, angle: float, translation=(0.0, 0.0, 0.0)) -> "RigidPose":
        axis = _vec3(axis, "axis")
        norm = np.linalg.norm(axis)
        if norm <= 0.0:
            raise ValueError("axis must be nonzero")
        axis = axis / norm
        x, y, z = axis
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        c, s = np.cos(float(angle)), np.sin(float(angle))
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        return RigidPose(R, np.asarray(translation, dtype=float))

    def apply(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return p @ self.rotation.T + self.translation

    def compose(self, other: "RigidPose") -> "RigidPose":
        return RigidPose(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
        )


@dataclass(frozen=True)
class ConductorSurfaceQuadrature:
    positions: np.ndarray
    normals: np.ndarray
    weights: np.ndarray

    def __post_init__(
        self,
    ):
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
        n = len(
            weights
        )
        if (
            positions.shape != (
                n,
                3,
            )
            or normals.shape != (
                n,
                3,
            )
            or np.any(
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
                weights
                <= 0.0
            )
        ):
            raise ValueError(
                "conductor surface quadrature arrays are invalid"
            )
        norm = np.linalg.norm(
            normals,
            axis=1,
        )
        if np.any(
            norm
            <= 1e-14
        ):
            raise ValueError(
                "conductor surface normals must be nonzero"
            )
        object.__setattr__(
            self,
            "positions",
            positions,
        )
        object.__setattr__(
            self,
            "normals",
            normals
            / norm[
                :,
                None,
            ],
        )
        object.__setattr__(
            self,
            "weights",
            weights,
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
class SuperellipseSpiral:
    outer_a: float
    outer_b: float
    turns: float
    pitch_a: float
    pitch_b: float
    exponent: float = 2.0
    conductor_width: float = 1e-3
    conductor_thickness: float = 1e-3
    cross_section_exponent: float = 2.0
    pose: RigidPose = field(default_factory=RigidPose.identity)

    def __post_init__(self):
        positive = {
            "outer_a": self.outer_a,
            "outer_b": self.outer_b,
            "turns": self.turns,
            "conductor_width": self.conductor_width,
            "conductor_thickness": self.conductor_thickness,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.exponent < 2.0 or self.cross_section_exponent < 2.0:
            raise ValueError("superellipse exponents must be >= 2")
        if self.pitch_a < 0.0 or self.pitch_b < 0.0:
            raise ValueError("pitch must be nonnegative")
        inner_a = self.outer_a - self.pitch_a * self.turns
        inner_b = self.outer_b - self.pitch_b * self.turns
        if inner_a <= 0.5 * self.conductor_width or inner_b <= 0.5 * self.conductor_width:
            raise ValueError("spiral shrinks through the conductor footprint")

    @property
    def cross_section_area(self) -> float:
        a = 0.5 * self.conductor_width
        b = 0.5 * self.conductor_thickness
        m = self.cross_section_exponent
        return 4.0 * a * b * gamma(1.0 + 1.0 / m) ** 2 / gamma(1.0 + 2.0 / m)

    @property
    def equivalent_radius(self) -> float:
        return float(np.sqrt(self.cross_section_area / pi))

    def _radius(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        turns = phi / (2.0 * np.pi)
        return self.outer_a - self.pitch_a * turns, self.outer_b - self.pitch_b * turns

    def local_centerline(self, phi) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        a, b = self._radius(phi)
        p = 2.0 / self.exponent
        c, s = np.cos(phi), np.sin(phi)
        x = a * np.sign(c) * np.abs(c) ** p
        y = b * np.sign(s) * np.abs(s) ** p
        return np.stack((x, y, np.zeros_like(x)), axis=-1)

    def centerline(self, phi) -> np.ndarray:
        return self.pose.apply(self.local_centerline(phi))

    def sample_centerline(self, n_points: int = 257, *, equal_arclength: bool = True) -> np.ndarray:
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        if not equal_arclength:
            phi = np.linspace(0.0, 2.0 * np.pi * self.turns, n_points)
            return self.centerline(phi)
        dense_n = max(4097, 16 * n_points)
        phi_dense = np.linspace(0.0, 2.0 * np.pi * self.turns, dense_n)
        pts = self.centerline(phi_dense)
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        targets = np.linspace(0.0, s[-1], n_points)
        out = np.empty((n_points, 3), dtype=float)
        for k in range(3):
            out[:, k] = np.interp(targets, s, pts[:, k])
        return out

    def polyline(self, n_segments: int) -> "PolylineConductor":
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")
        points = self.sample_centerline(n_segments + 1, equal_arclength=True)
        return PolylineConductor.from_points(points, self)

    def surface_samples(
        self,
        longitudinal_segments: int = 96,
        section_points: int = 20,
    ) -> np.ndarray:
        """Sample the finite conductor boundary without a global volume mesh."""
        if longitudinal_segments < 2:
            raise ValueError(
                "longitudinal_segments must be >= 2"
            )
        if section_points < 4:
            raise ValueError(
                "section_points must be >= 4"
            )
        segments = max(
            int(
                longitudinal_segments
            ),
            int(
                np.ceil(
                    48.0
                    * self.turns
                )
            ),
            24,
        )
        polyline = self.polyline(
            segments
        )
        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            int(
                section_points
            ),
            endpoint=False,
        )
        power = (
            2.0
            / float(
                self.cross_section_exponent
            )
        )
        cosine = np.cos(
            theta
        )
        sine = np.sin(
            theta
        )
        section_x = (
            0.5
            * self.conductor_width
            * np.sign(
                cosine
            )
            * np.abs(
                cosine
            ) ** power
        )
        section_y = (
            0.5
            * self.conductor_thickness
            * np.sign(
                sine
            )
            * np.abs(
                sine
            ) ** power
        )
        offsets = (
            section_x[
                None,
                :,
                None,
            ]
            * polyline.normal1[
                :,
                None,
                :,
            ]
            + section_y[
                None,
                :,
                None,
            ]
            * polyline.normal2[
                :,
                None,
                :,
            ]
        )
        middle = (
            polyline.midpoints[
                :,
                None,
                :,
            ]
            + offsets
        ).reshape(
            -1,
            3,
        )
        ends = []
        for point, n1, n2 in (
            (
                polyline.points[
                    0
                ],
                polyline.normal1[
                    0
                ],
                polyline.normal2[
                    0
                ],
            ),
            (
                polyline.points[
                    -1
                ],
                polyline.normal1[
                    -1
                ],
                polyline.normal2[
                    -1
                ],
            ),
        ):
            ends.append(
                point[
                    None,
                    :
                ]
                + section_x[
                    :,
                    None,
                ]
                * n1[
                    None,
                    :
                ]
                + section_y[
                    :,
                    None,
                ]
                * n2[
                    None,
                    :
                ]
            )
        return np.concatenate(
            (
                middle,
                *ends,
            ),
            axis=0,
        )

    def surface_quadrature(
        self,
        longitudinal_segments: int = 96,
        section_points: int = 24,
        end_radial_order: int = 4,
    ) -> ConductorSurfaceQuadrature:
        """Closed object-local quadrature for the finite conductor boundary."""
        if longitudinal_segments < 2:
            raise ValueError(
                "longitudinal_segments must be >= 2"
            )
        if section_points < 8:
            raise ValueError(
                "section_points must be >= 8"
            )
        if end_radial_order < 2:
            raise ValueError(
                "end_radial_order must be >= 2"
            )

        segments = max(
            int(
                longitudinal_segments
            ),
            int(
                np.ceil(
                    48.0
                    * self.turns
                )
            ),
            24,
        )
        polyline = self.polyline(
            segments
        )
        theta = (
            np.arange(
                int(
                    section_points
                ),
                dtype=float,
            )
            + 0.5
        ) * (
            2.0
            * np.pi
            / int(
                section_points
            )
        )
        dtheta = (
            2.0
            * np.pi
            / int(
                section_points
            )
        )
        exponent = (
            2.0
            / float(
                self.cross_section_exponent
            )
        )
        cosine = np.cos(
            theta
        )
        sine = np.sin(
            theta
        )
        a = (
            0.5
            * self.conductor_width
        )
        b = (
            0.5
            * self.conductor_thickness
        )
        section_x = (
            a
            * np.sign(
                cosine
            )
            * np.abs(
                cosine
            ) ** exponent
        )
        section_y = (
            b
            * np.sign(
                sine
            )
            * np.abs(
                sine
            ) ** exponent
        )
        derivative_x = (
            -a
            * exponent
            * np.abs(
                cosine
            ) ** (
                exponent
                - 1.0
            )
            * sine
        )
        derivative_y = (
            b
            * exponent
            * np.abs(
                sine
            ) ** (
                exponent
                - 1.0
            )
            * cosine
        )
        boundary_speed = np.sqrt(
            derivative_x**2
            + derivative_y**2
        )
        if np.any(
            ~np.isfinite(
                boundary_speed
            )
        ) or np.any(
            boundary_speed
            <= 0.0
        ):
            raise RuntimeError(
                "failed to construct finite superellipse boundary Jacobian"
            )

        side_positions = (
            polyline.midpoints[
                :,
                None,
                :
            ]
            + section_x[
                None,
                :,
                None,
            ]
            * polyline.normal1[
                :,
                None,
                :
            ]
            + section_y[
                None,
                :,
                None,
            ]
            * polyline.normal2[
                :,
                None,
                :
            ]
        )
        side_normal_raw = (
            derivative_y[
                None,
                :,
                None,
            ]
            * polyline.normal1[
                :,
                None,
                :
            ]
            - derivative_x[
                None,
                :,
                None,
            ]
            * polyline.normal2[
                :,
                None,
                :
            ]
        )
        side_normals = (
            side_normal_raw
            / boundary_speed[
                None,
                :,
                None,
            ]
        )
        side_weights = (
            polyline.lengths[
                :,
                None,
            ]
            * boundary_speed[
                None,
                :
            ]
            * dtheta
        )

        radial_raw, radial_weight_raw = (
            np.polynomial.legendre.leggauss(
                int(
                    end_radial_order
                )
            )
        )
        radial = (
            0.5
            * (
                radial_raw
                + 1.0
            )
        )
        radial_weights = (
            0.5
            * radial_weight_raw
        )
        boundary_jacobian = np.abs(
            section_x
            * derivative_y
            - section_y
            * derivative_x
        )

        cap_positions = []
        cap_normals = []
        cap_weights = []
        for (
            endpoint,
            n1,
            n2,
            tangent,
            sign,
        ) in (
            (
                polyline.points[
                    0
                ],
                polyline.normal1[
                    0
                ],
                polyline.normal2[
                    0
                ],
                polyline.tangents[
                    0
                ],
                -1.0,
            ),
            (
                polyline.points[
                    -1
                ],
                polyline.normal1[
                    -1
                ],
                polyline.normal2[
                    -1
                ],
                polyline.tangents[
                    -1
                ],
                1.0,
            ),
        ):
            rr, tt = np.meshgrid(
                radial,
                np.arange(
                    int(
                        section_points
                    )
                ),
                indexing="ij",
            )
            x = (
                rr
                * section_x[
                    tt
                ]
            )
            y = (
                rr
                * section_y[
                    tt
                ]
            )
            position = (
                endpoint[
                    None,
                    None,
                    :
                ]
                + x[
                    :,
                    :,
                    None,
                ]
                * n1[
                    None,
                    None,
                    :
                ]
                + y[
                    :,
                    :,
                    None,
                ]
                * n2[
                    None,
                    None,
                    :
                ]
            )
            weight = (
                radial_weights[
                    :,
                    None,
                ]
                * rr
                * boundary_jacobian[
                    None,
                    :
                ]
                * dtheta
            )
            cap_positions.append(
                position.reshape(
                    -1,
                    3,
                )
            )
            cap_normals.append(
                np.repeat(
                    (
                        sign
                        * tangent
                    )[
                        None,
                        :
                    ],
                    position.shape[
                        0
                    ]
                    * position.shape[
                        1
                    ],
                    axis=0,
                )
            )
            cap_weights.append(
                weight.reshape(
                    -1
                )
            )

        return ConductorSurfaceQuadrature(
            np.concatenate(
                (
                    side_positions.reshape(
                        -1,
                        3,
                    ),
                    *cap_positions,
                ),
                axis=0,
            ),
            np.concatenate(
                (
                    side_normals.reshape(
                        -1,
                        3,
                    ),
                    *cap_normals,
                ),
                axis=0,
            ),
            np.concatenate(
                (
                    side_weights.reshape(
                        -1
                    ),
                    *cap_weights,
                )
            ),
        )

    def transformed(self, pose: RigidPose) -> "SuperellipseSpiral":
        return SuperellipseSpiral(
            self.outer_a, self.outer_b, self.turns, self.pitch_a, self.pitch_b,
            self.exponent, self.conductor_width, self.conductor_thickness,
            self.cross_section_exponent, pose.compose(self.pose),
        )


@dataclass(frozen=True)
class PolylineConductor:
    points: np.ndarray
    midpoints: np.ndarray
    tangents: np.ndarray
    lengths: np.ndarray
    normal1: np.ndarray
    normal2: np.ndarray
    source: SuperellipseSpiral

    @staticmethod
    def from_points(points, source: SuperellipseSpiral) -> "PolylineConductor":
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
            raise ValueError("points must have shape (n>=3, 3)")
        delta = np.diff(p, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths <= 1e-14):
            raise ValueError("polyline contains zero-length segments")
        tangents = delta / lengths[:, None]
        mid = 0.5 * (p[:-1] + p[1:])
        n1, n2 = bishop_segment_frames(tangents)
        return PolylineConductor(p, mid, tangents, lengths, n1, n2, source)

    @property
    def total_length(self) -> float:
        return float(np.sum(self.lengths))


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1.0 - np.cos(angle))
    )


def bishop_segment_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tangents, dtype=float)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("tangents must have shape (n,3)")
    t = t / np.linalg.norm(t, axis=1)[:, None]
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t[0]))]
    n = seed - np.dot(seed, t[0]) * t[0]
    n /= np.linalg.norm(n)
    n1 = np.empty_like(t)
    n2 = np.empty_like(t)
    n1[0] = n
    n2[0] = np.cross(t[0], n1[0])
    for k in range(1, len(t)):
        cross = np.cross(t[k - 1], t[k])
        sn = np.linalg.norm(cross)
        cs = float(np.clip(np.dot(t[k - 1], t[k]), -1.0, 1.0))
        if sn > 1e-12:
            transported = _rodrigues(n1[k - 1], cross / sn, np.arctan2(sn, cs))
        else:
            transported = n1[k - 1]
        transported -= np.dot(transported, t[k]) * t[k]
        norm = np.linalg.norm(transported)
        if norm <= 1e-12:
            seed = axes[np.argmin(np.abs(axes @ t[k]))]
            transported = seed - np.dot(seed, t[k]) * t[k]
            norm = np.linalg.norm(transported)
        n1[k] = transported / norm
        n2[k] = np.cross(t[k], n1[k])
    return n1, n2


def haar_rotation(rng: np.random.Generator) -> np.ndarray:
    q = np.asarray(rng.normal(size=4), dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
