"""Compatibility adapter that reuses the existing vNext conductor discretization.

The BFZI path is evaluated from its *own* eight original analytic groups.
Only its polyline/cross-section representation is adapted to the vNext
PolylineConductor, Bishop frames and conductor MQS teacher. The original
SuperellipseSpiral definition is intentionally not modified or renamed.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gamma, pi

import numpy as np

from .geometry import BFZICoil, RigidPose


@dataclass(frozen=True)
class BFZIVNextGeometry:
    coil: BFZICoil
    pose: RigidPose
    conductor_width: float
    conductor_thickness: float
    cross_section_exponent: float = 2.0

    def __post_init__(self):
        if self.conductor_width <= 0 or self.conductor_thickness <= 0:
            raise ValueError("conductor cross section must be positive")
        if self.cross_section_exponent < 2:
            raise ValueError("cross section exponent must be >= 2")

    @property
    def turns(self) -> int:
        return self.coil.turns

    @property
    def cross_section_area(self) -> float:
        m = self.cross_section_exponent
        return (
            self.conductor_width * self.conductor_thickness
            * gamma(1 + 1 / m)**2 / gamma(1 + 2 / m)
        )

    @property
    def equivalent_radius(self) -> float:
        return float(np.sqrt(self.cross_section_area / pi))

    def sample_centerline(self, n_points: int = 257, *, equal_arclength: bool = True):
        if n_points < 3:
            raise ValueError("n_points must be >= 3")
        # Sample analytic BFZI groups in their *actual path order*. Duplicate
        # zero-length sides in the circular limit are explicitly removed.
        per_group = max(9, int(np.ceil(2.5 * n_points / (8 * self.turns))))
        points = []
        for k in range(self.coil.turns):
            for side in range(4):
                for kind, low, high in (("side", -1.0, 1.0),
                                        ("corner", 0.0, pi / 2)):
                    t = np.linspace(low, high, per_group)
                    p, _ = self.coil.segment(k, kind, side, t, self.pose)
                    for v in p:
                        if not points or np.linalg.norm(v - points[-1]) > 1e-13:
                            points.append(v)
        dense = np.asarray(points)
        if len(dense) < 3:
            raise ValueError("degenerate BFZI path")
        ds = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        arclength = np.concatenate(([0.0], np.cumsum(ds)))
        if equal_arclength:
            distance = np.linspace(0.0, arclength[-1], n_points)
        else:
            distance = np.interp(
                np.linspace(0.0, len(dense)-1, n_points),
                np.arange(len(dense)), arclength,
            )
        return np.stack(
            [np.interp(distance, arclength, dense[:, k]) for k in range(3)],
            axis=-1,
        )

    def polyline(self, n_segments: int):
        # Reuse existing vNext geometry and parallel-transport frame code.
        from sdfmpneo_vnext.geometry import PolylineConductor
        return PolylineConductor.from_points(
            self.sample_centerline(n_segments + 1), self,
        )


def as_vnext_coil_object(
    coil: BFZICoil, pose: RigidPose, *,
    width: float, thickness: float, conductivity: float,
    cross_section_exponent: float = 2.0,
):
    """Build the existing vNext conductor-only teacher geometry.

    This DOES NOT port the old package/electrothermal solver to BFZI,
    and it does NOT model the externally specified BFZI return route.
    """
    from sdfmpneo_vnext.scene import CoilObject, ConductorMaterial
    geometry = BFZIVNextGeometry(
        coil, pose, width, thickness, cross_section_exponent
    )
    return CoilObject(geometry, ConductorMaterial(conductivity))
