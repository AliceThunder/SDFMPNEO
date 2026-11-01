"""BFZI assemblies and finite conducting seawater domain.

Reuses the vNext separation of geometry, material, and pose, but never
reinterprets the historical superellipse as a BFZI spiral.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .geometry import BFZICoil, RigidPose
from sdfmpneo_vnext.scene import ConductorMaterial


@dataclass(frozen=True)
class InsulatingBox:
    """Sealed nonconductive, nonmagnetic, oriented rectangular enclosure."""

    half_extents: tuple[float, float, float]

    def __post_init__(self):
        h = np.asarray(self.half_extents, dtype=float)
        if h.shape != (3,) or not np.all(np.isfinite(h)) or np.any(h <= 0):
            raise ValueError("box half_extents must be positive finite lengths")
        object.__setattr__(self, "half_extents", tuple(float(x) for x in h))

    def squared_exterior_distance(self, points, pose: RigidPose):
        """Squared distance from box and half-gradient in world coordinates."""
        x = np.asarray(points, dtype=float)
        local = (x - pose.center) @ pose.rotation
        excess = np.maximum(np.abs(local) - self.half_extents, 0.0)
        exterior = excess * np.sign(local)
        return np.sum(exterior**2, axis=-1), exterior @ pose.rotation.T

    def contains(self, points, pose: RigidPose):
        x = (np.asarray(points, dtype=float) - pose.center) @ pose.rotation
        return np.all(np.abs(x) <= np.asarray(self.half_extents), axis=-1)


@dataclass(frozen=True)
class BFZIAssembly:
    coil: BFZICoil
    pose: RigidPose
    enclosure: InsulatingBox
    return_waypoints: tuple[tuple[float, float, float], ...]
    wire_radius: float = 0.0008
    # Round finite conductor cross-section; vNext material law is reused.
    material: ConductorMaterial = field(
        default_factory=lambda: ConductorMaterial(5.8e7)
    )

    def __post_init__(self):
        radius = float(self.wire_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("wire_radius must be positive")
        if len(self.return_waypoints) < 2:
            raise ValueError("explicit return route requires at least two waypoints")
        waypoints = np.asarray(self.return_waypoints, dtype=float)
        if waypoints.shape != (len(self.return_waypoints), 3) or not np.isfinite(waypoints).all():
            raise ValueError("return waypoints must be finite local 3-vectors")
        h = np.asarray(self.enclosure.half_extents)
        outer = self.coil.outer_half_size + self.coil.beta + radius
        if min(h[:2]) <= outer or h[2] <= radius:
            raise ValueError("enclosure does not contain the BFZI spiral footprint")
        if not isinstance(self.material, ConductorMaterial):
            raise ValueError("material must be a vNext ConductorMaterial")
        object.__setattr__(self, "wire_radius", radius)
        object.__setattr__(
            self, "return_waypoints",
            tuple(tuple(float(v) for v in row) for row in waypoints),
        )


def _obb_disjoint(a: BFZIAssembly, b: BFZIAssembly) -> bool:
    """Separating-axis theorem for two oriented boxes; touching is rejected."""
    axes_a = a.pose.rotation
    axes_b = b.pose.rotation
    delta = b.pose.center - a.pose.center
    axes = [*axes_a.T, *axes_b.T]
    axes += [np.cross(aa, bb) for aa in axes_a.T for bb in axes_b.T]
    ha = np.asarray(a.enclosure.half_extents)
    hb = np.asarray(b.enclosure.half_extents)
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        unit = axis / length
        reach_a = np.sum(ha * np.abs(axes_a.T @ unit))
        reach_b = np.sum(hb * np.abs(axes_b.T @ unit))
        if abs(float(np.dot(delta, unit))) > reach_a + reach_b + 1e-9:
            return True
    return False


@dataclass(frozen=True)
class Seawater:
    conductivity: float = 5.0

    def __post_init__(self):
        if not math.isfinite(float(self.conductivity)) or self.conductivity <= 0:
            raise ValueError("seawater conductivity must be positive")


@dataclass(frozen=True)
class BfziScene:
    """Two BFZI ports in spherical finite outer-domain truncation.

    The seawater conducts only outside both insulating boxes. Outer-domain
    and basis convergence must be established before reference claims.
    """
    transmitter: BFZIAssembly
    receiver: BFZIAssembly
    seawater: Seawater = field(default_factory=Seawater)
    outer_radius: float = 0.8
    outer_center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not math.isfinite(self.outer_radius) or self.outer_radius <= 0:
            raise ValueError("outer_radius must be positive finite")
        c = np.asarray(self.outer_center, dtype=float)
        if c.shape != (3,) or not np.isfinite(c).all():
            raise ValueError("outer_center must be a finite 3-vector")
        if not _obb_disjoint(self.transmitter, self.receiver):
            raise ValueError("two insulating enclosures overlap or touch")
        for assembly in self.assemblies:
            half = np.asarray(assembly.enclosure.half_extents)
            local_corners = np.array(np.meshgrid(
                [-half[0], half[0]], [-half[1], half[1]],
                [-half[2], half[2]], indexing="ij",
            )).reshape(3, -1).T
            world = assembly.pose.position(local_corners)
            if np.max(np.linalg.norm(world - c, axis=1)) >= 0.98 * self.outer_radius:
                raise ValueError("enclosure touches artificial outer domain")
        object.__setattr__(self, "outer_center", tuple(float(v) for v in c))

    @property
    def assemblies(self) -> tuple[BFZIAssembly, BFZIAssembly]:
        return self.transmitter, self.receiver

    def in_seawater(self, points):
        x = np.asarray(points, dtype=float)
        c = np.asarray(self.outer_center)
        inside_outer = np.sum((x - c) ** 2, axis=-1) < self.outer_radius**2
        return inside_outer & ~self.transmitter.enclosure.contains(
            x, self.transmitter.pose
        ) & ~self.receiver.enclosure.contains(x, self.receiver.pose)

    def quadrature(
        self, radial_order: int = 7, polar_order: int = 10,
        azimuthal_order: int = 16, *, wall_layers: int = 2,
        wall_scale: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Positive meshless cubature with exact box cuts and graded wall layers.

        Each radial ray is clipped against the oriented insulation boxes.
        The wet intervals are then locally subdivided near *actual* box
        intersections, not near the artificial outer boundary. Subdivision
        depths are multiples of the physical wall scale, resolving the
        strongly localized curl-basis gradients without globally increasing
        radial order. Angular convergence is independent and still required.
        """
        if min(radial_order, polar_order, azimuthal_order) < 2:
            raise ValueError("quadrature orders must be >= 2")
        if not isinstance(wall_layers, int) or not 0 <= wall_layers <= 6:
            raise ValueError("wall_layers must be an integer from 0 to 6")
        scale = (
            0.075 * min(min(a.enclosure.half_extents) for a in self.assemblies)
            if wall_scale is None else float(wall_scale)
        )
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("wall_scale must be positive and finite")
        rr, rw = np.polynomial.legendre.leggauss(radial_order)
        mu, mw = np.polynomial.legendre.leggauss(polar_order)
        theta = (np.arange(azimuthal_order) + 0.5) * (
            2 * math.pi / azimuthal_order
        )
        transverse = np.sqrt(np.maximum(1.0 - mu**2, 0.0))
        directions = np.stack((
            transverse[:, None] * np.cos(theta),
            transverse[:, None] * np.sin(theta),
            np.broadcast_to(mu[:, None], (polar_order, azimuthal_order)),
        ), axis=-1).reshape(-1, 3)
        angular_weights = np.broadcast_to(
            mw[:, None] * (2 * math.pi / azimuthal_order),
            (polar_order, azimuthal_order),
        ).reshape(-1)
        n_rays = len(directions)
        outer_center = np.asarray(self.outer_center)
        radius = self.outer_radius
        cuts = [np.zeros(n_rays), np.full(n_rays, radius)]
        wall_cuts = [np.zeros(n_rays, dtype=bool) for _ in range(2)]
        for assembly in self.assemblies:
            origin = (outer_center - assembly.pose.center) @ assembly.pose.rotation
            rays = directions @ assembly.pose.rotation
            half = np.asarray(assembly.enclosure.half_extents)
            entering = np.full(n_rays, -np.inf)
            exiting = np.full(n_rays, np.inf)
            intersects = np.ones(n_rays, dtype=bool)
            for axis in range(3):
                component = rays[:, axis]
                parallel = np.abs(component) <= 1e-14
                intersects &= ~parallel | (abs(origin[axis]) <= half[axis])
                near = np.divide(
                    -half[axis] - origin[axis], component,
                    out=np.full(n_rays, -np.inf), where=~parallel,
                )
                far = np.divide(
                    half[axis] - origin[axis], component,
                    out=np.full(n_rays, np.inf), where=~parallel,
                )
                entering = np.maximum(entering, np.minimum(near, far))
                exiting = np.minimum(exiting, np.maximum(near, far))
            start = np.clip(entering, 0.0, radius)
            stop = np.clip(exiting, 0.0, radius)
            hits = intersects & (start < stop)
            cuts.extend((np.where(hits, start, radius),
                         np.where(hits, stop, radius)))
            wall_cuts.extend((hits, hits))
        values = np.stack(cuts, axis=1)
        flags = np.stack(wall_cuts, axis=1)
        order = np.argsort(values, axis=1, kind="stable")
        boundaries = np.take_along_axis(values, order, axis=1)
        boundary_is_wall = np.take_along_axis(flags, order, axis=1)
        left, right = boundaries[:, :-1], boundaries[:, 1:]
        mid = outer_center + (
            (left + right)[:, :, None] * 0.5 * directions[:, None, :]
        )
        wet = (right > left) & self.in_seawater(mid)
        if not np.any(wet):
            return np.empty((0, 3)), np.empty(0)

        # Each wet interval stays wet after subdivision. The Gauss weights
        # remain positive, and the sum of radial cubature weights integrates
        # r^2 exactly on every subinterval for radial_order >= 2.
        knots = [left, right]
        for layer in range(wall_layers):
            distance = scale * 4**layer
            knots.append(np.where(
                boundary_is_wall[:, :-1],
                np.minimum(left + distance, right), left,
            ))
            knots.append(np.where(
                boundary_is_wall[:, 1:],
                np.maximum(right - distance, left), right,
            ))
        bounds = np.sort(np.stack(knots, axis=-1), axis=-1)
        a, b = bounds[..., :-1], bounds[..., 1:]
        valid = wet[..., None] & (b > a)
        r = a[..., None] + 0.5 * (b - a)[..., None] * (
            rr[None, None, None, :] + 1.0
        )
        weights = (
            0.5 * (b - a)[..., None] * rw[None, None, None, :]
            * r**2 * angular_weights[:, None, None, None]
        )
        points = outer_center + r[..., None] * directions[:, None, None, None, :]
        return points[valid].reshape(-1, 3), weights[valid].reshape(-1)
