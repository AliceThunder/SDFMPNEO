"""BFZI complete-port line sources, air inductance and field kernels.

Adapts the vNext vectorized Neumann/Green quadrature approach, but retains
the exact BFZI path construction. A user-supplied return conductor explicitly
closes each current path; no open-wire self-inductance is called a port result.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .scene import BFZIAssembly, BfziScene

MU0 = 4e-7 * math.pi


@dataclass(frozen=True)
class LineSource:
    """World-space source nodes and weighted oriented d-ell vectors."""
    points: np.ndarray
    line_elements: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.points, dtype=float)
        h = np.asarray(self.line_elements, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or h.shape != p.shape:
            raise ValueError("line source arrays must have shape (n,3)")
        if not np.isfinite(p).all() or not np.isfinite(h).all():
            raise ValueError("line source arrays must be finite")
        object.__setattr__(self, "points", p)
        object.__setattr__(self, "line_elements", h)

    def potential_kernel(self, positions, chunk_size: int = 128):
        """C(x) = integral dl/(4*pi*|x-r|), without the mu0 factor."""
        x = np.asarray(positions, dtype=float)
        original_shape = x.shape[:-1]
        if x.shape[-1:] != (3,):
            raise ValueError("positions must have trailing dimension 3")
        flat = x.reshape(-1, 3)
        out = np.empty_like(flat)
        for start in range(0, len(flat), chunk_size):
            query = flat[start:start + chunk_size]
            offset = query[:, None, :] - self.points[None, :, :]
            dist = np.linalg.norm(offset, axis=-1)
            if np.any(dist < 1e-12):
                raise ValueError("field evaluation too close to ideal line source")
            out[start:start + chunk_size] = (
                np.reciprocal(dist) @ self.line_elements
            ) / (4 * math.pi)
        return out.reshape(original_shape + (3,))

    def magnetic_field(self, positions, currents: complex = 1.0, chunk_size: int = 128):
        """Biot-Savart field for an ideal (zero-radius) closed source path."""
        x = np.asarray(positions, dtype=float)
        original_shape = x.shape[:-1]
        if x.shape[-1:] != (3,):
            raise ValueError("positions must have trailing dimension 3")
        flat = x.reshape(-1, 3)
        out = np.empty((len(flat), 3), dtype=complex)
        for start in range(0, len(flat), chunk_size):
            offset = flat[start:start + chunk_size, None, :] - self.points[None, :, :]
            dist = np.linalg.norm(offset, axis=-1)
            if np.any(dist < 1e-12):
                raise ValueError("magnetic field too close to ideal line source")
            out[start:start + chunk_size] = (
                MU0 * currents / (4 * math.pi)
                * np.sum(
                    np.cross(self.line_elements[None, :, :], offset)
                    / dist[..., None]**3, axis=1
                )
            )
        return out.reshape(original_shape + (3,))


def build_line_source(assembly: BFZIAssembly, order: int = 4) -> LineSource:
    """Complete BFZI spiral followed by explicit return path in world frame."""
    spiral_points, spiral_dl = assembly.coil.gauss_nodes(order, assembly.pose)
    inner, outer = assembly.coil.endpoints()
    route_local = np.vstack(
        (inner, np.asarray(assembly.return_waypoints, dtype=float), outer)
    )
    if np.any(np.linalg.norm(np.diff(route_local, axis=0), axis=1) < 1e-12):
        raise ValueError("return path has zero-length sections")
    q, w = np.polynomial.legendre.leggauss(order)
    t = 0.5 * (q + 1.0)
    source_points = [spiral_points]
    source_dl = [spiral_dl]
    for start, end in zip(route_local[:-1], route_local[1:]):
        delta = end - start
        local = start + t[:, None] * delta
        source_points.append(assembly.pose.position(local))
        source_dl.append(np.broadcast_to(0.5 * w[:, None] * delta, local.shape))
    points = np.vstack(source_points)
    dl = np.vstack([spiral_dl, *(assembly.pose.vector(v) for v in source_dl[1:])])
    return LineSource(points, dl)


def air_inductance(
    scene: BfziScene, sources: tuple[LineSource, LineSource],
    chunk_size: int = 192,
) -> np.ndarray:
    """Real symmetric regularized Neumann matrix.

    The self term uses sqrt(r^2 + a^2) smoothing with specified wire_radius;
    this is a convergence baseline, NOT an exact finite-cross-section GMD
    or a validated skin/proximity conductor model.
    """
    L = np.zeros((2, 2), dtype=float)
    for p in range(2):
        for q in range(p, 2):
            a, b = sources[p], sources[q]
            smooth = scene.assemblies[p].wire_radius if p == q else 0.0
            total = 0.0
            for start in range(0, len(a.points), chunk_size):
                offset = a.points[start:start + chunk_size, None, :] - b.points[None, :, :]
                dist2 = np.einsum("ijk,ijk->ij", offset, offset) + smooth**2
                if np.any(dist2 <= 1e-24):
                    raise ValueError("distinct port paths intersect or are unresolved")
                dot = a.line_elements[start:start + chunk_size] @ b.line_elements.T
                total += float(np.sum(dot / np.sqrt(dist2)))
            L[p, q] = MU0 / (4 * math.pi) * total
            L[q, p] = L[p, q]
    return L


@dataclass(frozen=True)
class BFZISources:
    scene: BfziScene
    frequency_hz: float
    lines: tuple[LineSource, LineSource]
    inductance_air: np.ndarray

    @property
    def omega(self) -> float:
        return 2 * math.pi * self.frequency_hz

    def potential_kernels(self, positions) -> np.ndarray:
        """Return (...,3,2) unit-port C source."""
        return np.stack(
            [line.potential_kernel(positions) for line in self.lines], axis=-1
        )

    def magnetic_field(self, positions, port_currents=(1.0, 0.0)):
        current = np.asarray(port_currents, dtype=complex)
        if current.shape != (2,):
            raise ValueError("port_currents must have shape (2,)")
        return sum(
            (line.magnetic_field(positions, current[i])
             for i, line in enumerate(self.lines)),
            start=np.zeros(np.asarray(positions).shape, dtype=complex),
        )


def build_bfzi_source(scene: BfziScene, frequency_hz: float, order: int = 4):
    if not math.isfinite(float(frequency_hz)) or frequency_hz <= 0:
        raise ValueError("frequency must be positive and finite")
    lines = tuple(build_line_source(item, order) for item in scene.assemblies)
    return BFZISources(
        scene=scene,
        frequency_hz=float(frequency_hz),
        lines=lines,
        inductance_air=air_inductance(scene, lines),
    )
