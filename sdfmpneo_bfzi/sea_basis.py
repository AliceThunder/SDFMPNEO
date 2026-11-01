"""Geometry-aware, weakly divergence-free seawater current modes.

Each current mode is curl(phi*a), with a constant real direction a. The
scalar potential phi vanishes on insulating box surfaces and the finite outer
sphere. Thus current is divergence-free and has zero normal trace. The modes
are deliberately deterministic; neural subspace selection comes only after
the electromagnetic field reference path has converged.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .scene import BfziScene


@dataclass(frozen=True)
class CurrentMode:
    center: np.ndarray
    direction: np.ndarray
    width: float


@dataclass(frozen=True)
class SeaCurrentBasis:
    scene: BfziScene
    modes: tuple[CurrentMode, ...]
    wall_scale: float

    @property
    def size(self) -> int:
        return len(self.modes)

    def evaluate(self, xyz) -> np.ndarray:
        """Vector current modes at points, shaped (n_points,n_modes,3).

        The sea-side wall trace is tangential. At the exact piecewise box
        surface the gradient of unsigned distance is not unique; queries
        there are classified as outside the open seawater region.
        """
        points = np.asarray(xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz must have shape (n,3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("basis query positions must be finite")
        if np.any(~self.scene.in_seawater(points)):
            raise ValueError("sea current basis can only be evaluated in seawater")
        sphere = points - np.asarray(self.scene.outer_center)
        radius2 = self.scene.outer_radius**2
        out_factor = 1.0 - np.sum(sphere**2, axis=1) / radius2
        out_grad = -2.0 * sphere / radius2

        boxes = np.ones(len(points), dtype=float)
        boxes_grad = np.zeros_like(points)
        for assembly in self.scene.assemblies:
            d2, halfgrad = assembly.enclosure.squared_exterior_distance(
                points, assembly.pose
            )
            d = np.sqrt(d2)
            factor = d / (d + self.wall_scale)
            normal = halfgrad / np.maximum(d[:, None], 1e-30)
            grad_factor = self.wall_scale * normal / (d + self.wall_scale)[:, None] ** 2
            boxes_grad = boxes_grad * factor[:, None] + boxes[:, None] * grad_factor
            boxes *= factor
        envelope = out_factor * boxes
        envelope_grad = out_grad * boxes[:, None] + out_factor[:, None] * boxes_grad

        out = np.empty((len(points), self.size, 3), dtype=float)
        for k, mode in enumerate(self.modes):
            offset = points - mode.center
            width2 = mode.width**2
            gauss = np.exp(-np.sum(offset**2, axis=1) / width2)
            gradient = gauss[:, None] * (
                envelope_grad - 2 * envelope[:, None] * offset / width2
            )
            out[:, k, :] = np.cross(gradient, mode.direction)
        return out


def build_sea_basis(
    scene: BfziScene, *,
    wall_scale: float | None = None,
    modes_per_face: int = 2,
    include_global: bool = True,
) -> SeaCurrentBasis:
    """Generate box-following curl modes and optional long-range modes.

    The face-localized modes are tangential and update with 6-DOF box pose;
    the global modes resolve long-range circulating current. This spans only
    a finite trial subspace; it is not claimed to approximate arbitrary
    seawater eddies without basis/cubature convergence.
    """
    if modes_per_face not in (1, 2):
        raise ValueError("modes_per_face must be 1 or 2")
    scale = (
        0.075 * min(scene.transmitter.enclosure.half_extents)
        if wall_scale is None else float(wall_scale)
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("wall_scale must be positive")
    modes = []
    for assembly in scene.assemblies:
        half = np.asarray(assembly.enclosure.half_extents)
        for axis in range(3):
            tangents = [j for j in range(3) if j != axis]
            for sign in (-1, 1):
                local = np.zeros(3)
                local[axis] = sign * (half[axis] + max(scale, 0.15 * half[axis]))
                center = assembly.pose.position(local)
                width = max(half.max() * 0.9, 3 * scale)
                for tangent_axis in tangents[:modes_per_face]:
                    direction = assembly.pose.rotation[:, tangent_axis]
                    modes.append(CurrentMode(center, direction, width))
    if include_global:
        r = scene.outer_radius
        center = np.asarray(scene.outer_center)
        for axis in range(3):
            modes.append(CurrentMode(center, np.eye(3)[axis], 0.8 * r))
    return SeaCurrentBasis(scene, tuple(modes), scale)
