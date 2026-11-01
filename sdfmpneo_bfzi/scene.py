"""BFZI assemblies and finite conducting seawater domain.

Reuses the vNext separation of geometry, material, and pose, but never
reinterprets the historical superellipse as a BFZI spiral.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .geometry import BFZICoil, RigidPose


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
        azimuthal_order: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Positive spherical point cubature (not a world FEM mesh)."""
        if min(radial_order, polar_order, azimuthal_order) < 2:
            raise ValueError("quadrature orders must be >= 2")
        rr, rw = np.polynomial.legendre.leggauss(radial_order)
        mu, mw = np.polynomial.legendre.leggauss(polar_order)
        r = 0.5 * (rr + 1) * self.outer_radius
        wr = 0.5 * rw * self.outer_radius * r**2
        theta = (np.arange(azimuthal_order) + 0.5) * (2 * math.pi / azimuthal_order)
        c, s = np.cos(theta), np.sin(theta)
        uu = np.sqrt(np.maximum(1 - mu**2, 0))
        directions = np.stack(
            (uu[:, None] * c, uu[:, None] * s,
             np.broadcast_to(mu[:, None], (polar_order, azimuthal_order))), axis=-1,
        )
        points = (np.asarray(self.outer_center)
                  + r[:, None, None, None] * directions[None, :, :, :])
        weights = (wr[:, None, None] * mw[None, :, None]
                   * (2 * math.pi / azimuthal_order)
                   * np.ones((1, 1, azimuthal_order)))
        flat = points.reshape(-1, 3)
        weight = weights.reshape(-1)
        mask = self.in_seawater(flat)
        return flat[mask], weight[mask]
