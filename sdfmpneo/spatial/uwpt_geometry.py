from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tetra3d import TetrahedralComplex3D


@dataclass(frozen=True)
class RigidPose:
    translation: np.ndarray
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        t = np.asarray(self.translation, dtype=float)
        if t.shape != (3,) or np.any(~np.isfinite(t)):
            raise ValueError("translation must be a finite 3-vector")
        object.__setattr__(self, "translation", t)

    @property
    def rotation(self) -> np.ndarray:
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ Ry @ Rx
        if not np.allclose(R.T @ R, np.eye(3), rtol=0.0, atol=64 * np.finfo(float).eps):
            raise FloatingPointError("rigid-pose rotation lost orthogonality")
        return R

    def apply(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return p @ self.rotation.T + self.translation


@dataclass(frozen=True)
class SpiralCoilGeometry:
    shape: str
    turns: float
    outer_half_size: float
    pitch: float
    conductor_width: float
    conductor_thickness: float
    corner_radius: float | None = None
    pose: RigidPose = field(default_factory=lambda: RigidPose(np.zeros(3)))

    def __post_init__(self) -> None:
        if self.shape not in {"circle", "rounded_square"}:
            raise ValueError("shape must be 'circle' or 'rounded_square'")
        values = np.array([
            self.turns, self.outer_half_size, self.pitch,
            self.conductor_width, self.conductor_thickness,
        ], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("coil geometric dimensions must be finite and positive")
        inward = self.pitch * self.turns
        clearance = 0.5 * self.conductor_width
        if self.outer_half_size - inward <= clearance:
            raise ValueError("spiral collapses before the requested number of turns")
        if self.shape == "rounded_square":
            if self.corner_radius is None or self.corner_radius <= inward + clearance:
                raise ValueError(
                    "rounded-square corner_radius must exceed total inward offset plus conductor half-width"
                )
            if self.corner_radius >= self.outer_half_size:
                raise ValueError("corner_radius must be smaller than outer_half_size")

    @property
    def theta_end(self) -> float:
        return 2.0 * np.pi * self.turns

    def _offset(self, theta: np.ndarray) -> np.ndarray:
        return self.pitch * theta / (2.0 * np.pi)

    @staticmethod
    def _rounded_square_radial(angle: np.ndarray, half: np.ndarray, radius: np.ndarray) -> np.ndarray:
        c = np.abs(np.cos(angle))
        s = np.abs(np.sin(angle))
        center = half - radius
        tiny = np.finfo(float).tiny
        vertical_t = half / np.maximum(c, tiny)
        vertical_y = vertical_t * s
        horizontal_t = half / np.maximum(s, tiny)
        horizontal_x = horizontal_t * c
        straight_vertical = vertical_y <= center
        straight_horizontal = horizontal_x <= center
        b = center * (c + s)
        discriminant = b * b - (2.0 * center * center - radius * radius)
        discriminant = np.maximum(discriminant, 0.0)
        corner_t = b + np.sqrt(discriminant)
        return np.where(straight_vertical, vertical_t, np.where(straight_horizontal, horizontal_t, corner_t))

    def local_centerline(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        if np.any(th < 0.0) or np.any(th > self.theta_end):
            raise ValueError("theta lies outside the spiral chart")
        offset = self._offset(th)
        half = self.outer_half_size - offset
        radial = half if self.shape == "circle" else self._rounded_square_radial(
            th, half, float(self.corner_radius) - offset
        )
        return np.column_stack([radial * np.cos(th), radial * np.sin(th), np.zeros_like(th)])

    def centerline(self, theta: np.ndarray) -> np.ndarray:
        return self.pose.apply(self.local_centerline(theta))

    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        points = self.centerline(np.array([0.0, self.theta_end]))
        return points[0], points[1]

    def sample_for_geometric_tolerance(self, chord_error: float) -> np.ndarray:
        error = float(chord_error)
        if not 0.0 < error < self.outer_half_size:
            raise ValueError("chord_error must lie in (0, outer_half_size)")
        ratio = max(-1.0, min(1.0, 1.0 - error / self.outer_half_size))
        dtheta = 2.0 * np.arccos(ratio)
        count = max(2, int(np.ceil(self.theta_end / dtheta)) + 1)
        return self.centerline(np.linspace(0.0, self.theta_end, count))


@dataclass(frozen=True)
class UnderwaterWPTGeometry:
    transmitter: SpiralCoilGeometry
    receiver: SpiralCoilGeometry
    package_half_extent: np.ndarray
    seawater_padding: float

    def __post_init__(self) -> None:
        extent = np.asarray(self.package_half_extent, dtype=float)
        if extent.shape != (3,) or np.any(extent <= 0.0):
            raise ValueError("package_half_extent must be a positive 3-vector")
        if self.seawater_padding <= 0.0:
            raise ValueError("seawater_padding must be positive")
        object.__setattr__(self, "package_half_extent", extent)

    def chart_non_degeneracy(self) -> dict[str, float]:
        return {
            "transmitter_rotation_det": float(np.linalg.det(self.transmitter.pose.rotation)),
            "receiver_rotation_det": float(np.linalg.det(self.receiver.pose.rotation)),
            "transmitter_inner_half_size": float(
                self.transmitter.outer_half_size - self.transmitter.pitch * self.transmitter.turns
            ),
            "receiver_inner_half_size": float(
                self.receiver.outer_half_size - self.receiver.pitch * self.receiver.turns
            ),
            "seawater_padding": float(self.seawater_padding),
        }


@dataclass(frozen=True)
class TaggedTetrahedralMesh:
    mesh: TetrahedralComplex3D
    tetra_physical_tags: np.ndarray
    boundary_triangles: np.ndarray
    boundary_physical_tags: np.ndarray

    def tetra_mask(self, physical_tag: int) -> np.ndarray:
        return np.asarray(self.tetra_physical_tags == int(physical_tag), dtype=bool)

    def boundary_mask(self, physical_tag: int) -> np.ndarray:
        return np.asarray(self.boundary_physical_tags == int(physical_tag), dtype=bool)

    def boundary_nodes(self, physical_tag: int) -> np.ndarray:
        triangles = self.boundary_triangles[self.boundary_mask(physical_tag)]
        return np.unique(triangles.ravel()) if triangles.size else np.empty(0, dtype=int)


def read_gmsh_v22_ascii(path: str | Path) -> TaggedTetrahedralMesh:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        ni = lines.index("$Nodes")
        ei = lines.index("$Elements")
    except ValueError as exc:
        raise ValueError("Gmsh file must contain $Nodes and $Elements") from exc

    n_nodes = int(lines[ni + 1])
    ids, vertices = [], []
    for row in lines[ni + 2 : ni + 2 + n_nodes]:
        fields = row.split()
        ids.append(int(fields[0]))
        vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    index = {node_id: k for k, node_id in enumerate(ids)}

    n_elements = int(lines[ei + 1])
    tetra, tetra_tags, triangles, triangle_tags = [], [], [], []
    for row in lines[ei + 2 : ei + 2 + n_elements]:
        fields = row.split()
        element_type = int(fields[1])
        n_tags = int(fields[2])
        tags = [int(x) for x in fields[3 : 3 + n_tags]]
        physical = tags[0] if tags else 0
        nodes = [index[int(x)] for x in fields[3 + n_tags :]]
        if element_type == 4:
            if len(nodes) != 4:
                raise ValueError("linear tetrahedron must have four nodes")
            tetra.append(nodes)
            tetra_tags.append(physical)
        elif element_type == 2:
            if len(nodes) != 3:
                raise ValueError("linear triangle must have three nodes")
            triangles.append(nodes)
            triangle_tags.append(physical)

    if not tetra:
        raise ValueError("Gmsh mesh contains no linear tetrahedra")
    mesh = TetrahedralComplex3D.build(np.asarray(vertices, dtype=float), np.asarray(tetra, dtype=int))
    return TaggedTetrahedralMesh(
        mesh=mesh,
        tetra_physical_tags=np.asarray(tetra_tags, dtype=int),
        boundary_triangles=np.asarray(triangles, dtype=int).reshape((-1, 3)),
        boundary_physical_tags=np.asarray(triangle_tags, dtype=int),
    )
