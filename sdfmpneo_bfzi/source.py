"""BFZI complete-port line sources, air inductance and field kernels.

Adapts the vNext vectorized Neumann/Green quadrature approach, but retains
the exact BFZI path construction. A user-supplied return conductor explicitly
closes each current path; no open-wire self-inductance is called a port result.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ive

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
    if not isinstance(order, int) or order < 2:
        raise ValueError("complete port source requires Gauss order >= 2")
    spiral_points, spiral_dl = assembly.coil.gauss_nodes(order, assembly.pose)
    # endpoints() returns the outer winding START and the inner winding END.
    # The explicit return must run from INNER end back to OUTER start, so
    # spiral + return is a single closed physical current path.
    outer, inner = assembly.coil.endpoints()
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
    # Gauss quadrature approximates the curved spiral integral.  Failure
    # to close at this scale indicates an incorrect endpoint/return order
    # rather than a tolerable high-order integration error.
    closure = np.linalg.norm(dl.sum(axis=0))
    size = max(float(np.linalg.norm(dl, axis=1).sum()), 1e-30)
    if closure > max(1e-8, size * 1e-4):
        raise ValueError("BFZI port source did not close; check return direction and order")
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

    def conductor_internal_impedance(self, temperatures=None) -> np.ndarray:
        """Complex two-port diagonal for finite circular conductor sections.

        Uses the exact uniform cylindrical-wire skin-effect solution (modified
        Bessel ratio) at each material conductivity and the quadrature length
        of the *complete* spiral plus explicit return. Includes DC resistance
        and internal magnetic reactance. Does not include proximity-effect
        redistribution, lead contacts or rectangular cross sections.
        """
        if temperatures is None:
            temperatures = [a.material.reference_temperature
                            for a in self.scene.assemblies]
        temp = np.asarray(temperatures, dtype=float)
        if temp.shape != (2,) or not np.isfinite(temp).all():
            raise ValueError("temperatures must be a finite 2-vector")
        result = np.empty(2, dtype=complex)
        for p, (assembly, line) in enumerate(zip(self.scene.assemblies, self.lines)):
            sigma = assembly.material.conductivity_at(float(temp[p]))
            mu = MU0 * assembly.material.relative_permeability
            radius = assembly.wire_radius
            k = np.sqrt(1j * self.omega * mu * sigma)
            argument = k * radius
            # ive scales away the common exp(Re(argument)) in the ratio.
            # The small-argument expansion avoids cancellation at low omega.
            if abs(argument) < 1e-3:
                per_length = (
                    1.0 / (math.pi * radius**2 * sigma)
                    + 1j * self.omega * mu / (8 * math.pi)
                )
            else:
                per_length = (
                    k * ive(0, argument) /
                    (2 * math.pi * radius * sigma * ive(1, argument))
                )
            length = float(np.linalg.norm(line.line_elements, axis=1).sum())
            result[p] = per_length * length
        return result

    def conductor_cross_section_loss(
        self, radius_fraction, port_currents=(1.0, 0.0), temperatures=None,
    ) -> np.ndarray:
        """Round-wire AC Joule density q(r) in W/m^3, shape (..., 2).

        radius_fraction is r/a in [0,1]. The exact isolated cylindrical
        conductor skin-current profile is used on each complete BFZI path.
        This neglects proximity-induced angular redistribution; it is not a
        world-coordinate point-in-wire query.
        """
        r = np.asarray(radius_fraction, dtype=float)
        if not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
            raise ValueError("radius_fraction must be finite and in [0,1]")
        i = np.asarray(port_currents, dtype=complex)
        if i.shape != (2,) or not np.isfinite(i).all():
            raise ValueError("port_currents must be a finite complex two-vector")
        if temperatures is None:
            temperatures = [a.material.reference_temperature
                            for a in self.scene.assemblies]
        temp = np.asarray(temperatures, dtype=float)
        if temp.shape != (2,) or not np.isfinite(temp).all():
            raise ValueError("temperatures must be a finite two-vector")
        result = np.empty(r.shape + (2,), dtype=float)
        for port, assembly in enumerate(self.scene.assemblies):
            sigma = assembly.material.conductivity_at(float(temp[port]))
            mu = MU0 * assembly.material.relative_permeability
            radius = assembly.wire_radius
            k = np.sqrt(1j * self.omega * mu * sigma)
            argument = k * radius
            if abs(argument) < 1e-3:
                unit_density = np.full(r.shape, 1.0 / (math.pi * radius**2),
                                       dtype=complex)
            else:
                # ive(0,z) and ive(1,z) contain exp(-abs(Re z));
                # restore only the nonpositive relative exponent.
                ratio = (
                    ive(0, argument * r) / ive(1, argument)
                    * np.exp(argument.real * (r - 1.0))
                )
                unit_density = k / (2 * math.pi * radius) * ratio
            result[..., port] = (
                0.5 * abs(i[port])**2 * np.abs(unit_density)**2 / sigma
            )
        return result

    def conductor_heat_nodes(
        self, port_currents=(1.0, 0.0), temperatures=None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Positive centerline Joule-power nodes (positions, watts per node).

        The line nodes span both the physical spiral and the return. Their
        summed power is exactly the conductor port channel under the same
        line quadrature. The radial cross-section distribution is available
        separately through conductor_cross_section_loss.
        """
        i = np.asarray(port_currents, dtype=complex)
        if i.shape != (2,) or not np.isfinite(i).all():
            raise ValueError("port_currents must be a finite complex two-vector")
        total_z = self.conductor_internal_impedance(temperatures).real
        results = []
        for port, line in enumerate(self.lines):
            lengths = np.linalg.norm(line.line_elements, axis=1)
            resistance_per_length = total_z[port] / lengths.sum()
            power = 0.5 * resistance_per_length * abs(i[port])**2 * lengths
            results.append((line.points.copy(), power))
        return tuple(results)

    def conductor_volume_heat(
        self, port_currents=(1.0, 0.0), temperatures=None, *,
        radial_order: int = 12, angular_order: int = 8,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Positive wire volume cubature (xyz, dV, q) for each port.

        Cylindrical cross sections are swept along the complete BFZI+return
        Gauss path using local tangent frames. The round-wire radial AC skin
        profile gives the true isolated-straight-wire q(r) at those sections.
        No FEM mesh or absolute frame orientation is needed; small-radius
        curved-wire and no-proximity approximations remain in force.
        """
        if (not isinstance(radial_order, int) or radial_order < 2
                or not isinstance(angular_order, int) or angular_order < 4):
            raise ValueError("radial_order >= 2 and angular_order >= 4 required")
        gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(radial_order)
        fraction = 0.5 * (gauss_nodes + 1.0)
        radial_weights = 0.5 * gauss_weights * fraction
        angles = 2.0 * math.pi * (
            np.arange(angular_order) + 0.5
        ) / angular_order
        radial_density = self.conductor_cross_section_loss(
            fraction, port_currents, temperatures,
        )
        result = []
        for port, (assembly, line) in enumerate(
            zip(self.scene.assemblies, self.lines)
        ):
            lengths = np.linalg.norm(line.line_elements, axis=1)
            tangents = line.line_elements / lengths[:, None]
            seed = np.tile([0.0, 0.0, 1.0], (len(tangents), 1))
            seed[np.abs(tangents[:, 2]) > 0.9] = [1.0, 0.0, 0.0]
            normal1 = np.cross(tangents, seed)
            normal1 /= np.linalg.norm(normal1, axis=1)[:, None]
            normal2 = np.cross(tangents, normal1)
            unit_offset = (
                np.cos(angles)[None, None, :, None]
                * normal1[:, None, None, :]
                + np.sin(angles)[None, None, :, None]
                * normal2[:, None, None, :]
            )
            radius = assembly.wire_radius
            points = (
                line.points[:, None, None, :]
                + radius * fraction[None, :, None, None] * unit_offset
            )
            weights = (
                lengths[:, None, None] * radius**2
                * radial_weights[None, :, None]
                * (2.0 * math.pi / angular_order)
                * np.ones((1, 1, angular_order))
            )
            q = np.broadcast_to(
                radial_density[None, :, port, None],
                weights.shape,
            )
            result.append((
                points.reshape(-1, 3),
                weights.reshape(-1),
                q.reshape(-1),
            ))
        return tuple(result)

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
