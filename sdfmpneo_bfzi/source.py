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
from scipy.spatial.distance import cdist

from .scene import BFZIAssembly, BfziScene

MU0 = 4e-7 * math.pi


@dataclass(frozen=True)
class LineSource:
    """Complete port quadrature and optional physically exact straight field.

    `points`/`line_elements` retain the full closed BFZI path for
    finite-wire length, copper heat and legacy Neumann air inductance.
    When `straight_segments` is set, C/B fields integrate the actual
    straight line segments analytically, while `field_mask` selects
    only the curved-corner Gauss elements, preventing double counting.
    """
    points: np.ndarray
    line_elements: np.ndarray
    field_mask: np.ndarray | None = None
    straight_segments: np.ndarray | None = None

    def __post_init__(self):
        p = np.asarray(self.points, dtype=float)
        h = np.asarray(self.line_elements, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or h.shape != p.shape:
            raise ValueError("line source arrays must have shape (n,3)")
        if not np.isfinite(p).all() or not np.isfinite(h).all():
            raise ValueError("line source arrays must be finite")
        mask = self.field_mask
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != (len(p),) or mask.dtype != np.bool_:
                raise ValueError("field_mask must be a boolean per line element")
        segments = self.straight_segments
        if segments is not None:
            segments = np.asarray(segments, dtype=float)
            if (segments.ndim != 3 or segments.shape[1:] != (2, 3)
                    or not np.isfinite(segments).all()
                    or np.any(np.linalg.norm(
                        segments[:, 1] - segments[:, 0], axis=1,
                    ) <= 0)):
                raise ValueError("straight_segments must be nondegenerate (n,2,3)")
            if mask is None:
                raise ValueError("analytic segments require a curved-source field_mask")
        elif mask is not None:
            raise ValueError("field_mask requires analytic straight segments")
        object.__setattr__(self, "points", p)
        object.__setattr__(self, "line_elements", h)
        object.__setattr__(self, "field_mask", mask)
        object.__setattr__(self, "straight_segments", segments)

    def potential_kernel(self, positions, chunk_size: int = 128):
        """C(x) = integral dl/(4*pi*|x-r|), without the mu0 factor."""
        x = np.asarray(positions, dtype=float)
        original_shape = x.shape[:-1]
        if x.shape[-1:] != (3,):
            raise ValueError("positions must have trailing dimension 3")
        flat = x.reshape(-1, 3)
        out = np.empty_like(flat)
        src = self.points if self.field_mask is None else self.points[self.field_mask]
        dl = self.line_elements if self.field_mask is None else self.line_elements[self.field_mask]
        if self.straight_segments is not None:
            # Lazy import: the analytic source-only reference also uses
            # LineSource for curved BFZI corners.
            from .source_reference import straight_segment_field
        for start in range(0, len(flat), chunk_size):
            query = flat[start:start + chunk_size]
            offset = query[:, None, :] - src[None, :, :]
            dist = np.linalg.norm(offset, axis=-1)
            if np.any(dist < 1e-12):
                raise ValueError("field evaluation too close to ideal line source")
            result = (np.reciprocal(dist) @ dl) / (4 * math.pi)
            if self.straight_segments is not None:
                for a, b in self.straight_segments:
                    potential, _ = straight_segment_field(a, b, query)
                    result += potential
            out[start:start + chunk_size] = result
        return out.reshape(original_shape + (3,))

    def magnetic_field(self, positions, currents: complex = 1.0, chunk_size: int = 128):
        """Biot-Savart field for an ideal (zero-radius) closed source path."""
        x = np.asarray(positions, dtype=float)
        original_shape = x.shape[:-1]
        if x.shape[-1:] != (3,):
            raise ValueError("positions must have trailing dimension 3")
        flat = x.reshape(-1, 3)
        out = np.empty((len(flat), 3), dtype=complex)
        src = self.points if self.field_mask is None else self.points[self.field_mask]
        dl = self.line_elements if self.field_mask is None else self.line_elements[self.field_mask]
        if self.straight_segments is not None:
            from .source_reference import straight_segment_field
        for start in range(0, len(flat), chunk_size):
            query = flat[start:start + chunk_size]
            offset = query[:, None, :] - src[None, :, :]
            dist = np.linalg.norm(offset, axis=-1)
            if np.any(dist < 1e-12):
                raise ValueError("magnetic field too close to ideal line source")
            result = MU0 / (4 * math.pi) * np.sum(
                np.cross(dl[None, :, :], offset)
                / dist[..., None]**3, axis=1
            )
            if self.straight_segments is not None:
                for a, b in self.straight_segments:
                    _, magnetic = straight_segment_field(a, b, query)
                    result += magnetic
            out[start:start + chunk_size] = currents * result
        return out.reshape(original_shape + (3,))

    def divergence_kernel(self, positions, chunk_size: int = 128):
        """Analytic div_x C for a line-source quadrature.

        For a truly closed continuous filament, div C=0 off the wire.
        Exact straight-segment contributions use endpoint telescoping,
        not finite-differencing C; only curved corner integration can
        leave a residual in a hybrid source. The residual is a direct
        source-continuity diagnostic, not an electromagnetic error bound.
        """
        x = np.asarray(positions, dtype=float)
        original_shape = x.shape[:-1]
        if x.shape[-1:] != (3,):
            raise ValueError("positions must have trailing dimension 3")
        flat = x.reshape(-1, 3)
        out = np.empty(len(flat), dtype=float)
        src = self.points if self.field_mask is None else self.points[self.field_mask]
        dl = self.line_elements if self.field_mask is None else self.line_elements[self.field_mask]
        for start in range(0, len(flat), chunk_size):
            query = flat[start:start + chunk_size]
            delta = query[:, None, :] - src[None, :, :]
            distance = np.linalg.norm(delta, axis=-1)
            if np.any(distance < 1e-12):
                raise ValueError("source divergence query too close to ideal wire")
            value = -np.einsum(
                "ijk,jk->i", delta / distance[..., None]**3, dl,
                optimize=True,
            ) / (4.0 * math.pi)
            if self.straight_segments is not None:
                for aa, bb in self.straight_segments:
                    ra = np.linalg.norm(query-aa, axis=1)
                    rb = np.linalg.norm(query-bb, axis=1)
                    # The exact divergence of a straight C segment is
                    # (1/|x-a|-1/|x-b|)/(4*pi).
                    value += (1.0/ra-1.0/rb)/(4.0*math.pi)
            out[start:start + chunk_size] = value
        return out.reshape(original_shape)


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
    # One physical source, two representations:
    # - full Gauss nodes remain available for length/heat/air mutual terms;
    # - C and B integrate real straight segments exactly, leaving only
    #   actual rounded-square/circular corner curves under Gauss.
    mask = np.zeros(len(points), dtype=bool)
    exact_segments = []
    for turn in range(assembly.coil.turns):
        for side in range(4):
            start_index = (4 * turn + side) * 2 * order
            mask[start_index + order:start_index + 2 * order] = True
            a, _ = assembly.coil.segment(
                turn, "side", side, -1.0, assembly.pose,
            )
            b, _ = assembly.coil.segment(
                turn, "side", side, 1.0, assembly.pose,
            )
            if np.linalg.norm(b-a) > 1e-14:
                exact_segments.append((a, b))
    world_route = assembly.pose.position(route_local)
    exact_segments.extend(
        (a, b) for a, b in zip(world_route[:-1], world_route[1:])
    )
    return LineSource(
        points, dl, field_mask=mask,
        straight_segments=np.asarray(exact_segments, dtype=float),
    )


def _segmented_complete_wire(assembly: BFZIAssembly, max_length: float):
    """Analytic BFZI groups and return subdivided into oriented line panels."""
    starts, deltas = [], []

    def add(polyline):
        differences = np.diff(polyline, axis=0)
        for point, delta in zip(polyline[:-1], differences):
            if np.linalg.norm(delta) > 1e-15:
                starts.append(point)
                deltas.append(delta)

    for turn in range(assembly.coil.turns):
        for side in range(4):
            for kind, left, right in (
                ("side", -1.0, 1.0),
                ("corner", 0.0, math.pi / 2),
            ):
                end_points, _ = assembly.coil.segment(
                    turn, kind, side, np.array([left, right]), assembly.pose,
                )
                chord = np.linalg.norm(end_points[1] - end_points[0])
                # The curved-corner arc can be longer than its chord.
                factor = 1.2 if kind == "corner" else 1.0
                count = max(1, int(math.ceil(factor * chord / max_length)))
                parameters = np.linspace(left, right, count + 1)
                positions, _ = assembly.coil.segment(
                    turn, kind, side, parameters, assembly.pose,
                )
                add(positions)

    outer, inner = assembly.coil.endpoints()
    route = assembly.pose.position(np.vstack((
        inner, np.asarray(assembly.return_waypoints), outer,
    )))
    for p, q in zip(route[:-1], route[1:]):
        count = max(1, int(math.ceil(np.linalg.norm(q - p) / max_length)))
        add(p + np.linspace(0.0, 1.0, count + 1)[:, None] * (q - p))
    return np.asarray(starts), np.asarray(deltas)


def _regularized_self_inductance(
    assembly: BFZIAssembly, *, panel_length_factor: float,
    panel_gauss_order: int, chunk_size: int,
) -> float:
    """Neumann self term: exact straight-panel diagonal, Gauss off diagonal.

    Regularized Green kernel is 1/sqrt(r^2+a^2), a=wire radius. Treating a
    curved BFZI segment as several short straight panels is an explicit
    geometric approximation, and is refineable independently of the
    external-port field quadrature. Correcting the diagonal avoids the
    O(1/a) low-order Gauss self-node bias.
    """
    radius = assembly.wire_radius
    origins, segments = _segmented_complete_wire(
        assembly, radius * panel_length_factor,
    )
    length = np.linalg.norm(segments, axis=1)
    if not len(length) or np.any(length <= 0):
        raise ValueError("degenerate finite-wire self-inductance panels")
    q, w = np.polynomial.legendre.leggauss(panel_gauss_order)
    points = (
        origins[:, None, :]
        + 0.5 * (q[None, :, None] + 1.0) * segments[:, None, :]
    ).reshape(-1, 3)
    elements = np.broadcast_to(
        0.5 * w[None, :, None] * segments[:, None, :],
        (len(length), panel_gauss_order, 3),
    ).reshape(-1, 3)
    indices = np.repeat(np.arange(len(length)), panel_gauss_order)
    off_diagonal = 0.0
    for first in range(0, len(points), chunk_size):
        stop = min(first + chunk_size, len(points))
        distance2 = cdist(points[first:stop], points, "sqeuclidean")
        kernel = np.reciprocal(np.sqrt(distance2 + radius**2))
        kernel *= (indices[first:stop, None] != indices[None, :])
        off_diagonal += float(np.sum(
            (elements[first:stop] @ elements.T) * kernel,
        ))
    # Exact pair integral along a straight panel of length h:
    # 2 [h asinh(h/a) - sqrt(h^2+a^2) + a].
    diagonal = 2.0 * np.sum(
        length * np.arcsinh(length / radius)
        - np.hypot(length, radius) + radius
    )
    return float(MU0 / (4 * math.pi) * (off_diagonal + diagonal))


def air_inductance(
    scene: BfziScene, sources: tuple[LineSource, LineSource],
    chunk_size: int = 192, *,
    panel_length_factor: float = 4.0,
    panel_gauss_order: int = 3,
) -> np.ndarray:
    """Closed-path approximate external Neumann inductance in henries.

    Mutual inductance uses the existing unit-port Gauss sources. Each self
    inductance integrates the regularized finite-radius kernel on analytic
    BFZI + return path panels; panel self-interactions are analytically
    integrated rather than approximated by coincident Gauss source nodes.
    It is still a prescribed-radius soft-kernel approximation, not a
    surface-current/GMD truth solution or proximity-effect conductor solve.
    """
    if (not math.isfinite(panel_length_factor)
            or panel_length_factor <= 0):
        raise ValueError("panel_length_factor must be finite and positive")
    if (isinstance(panel_gauss_order, bool)
            or not isinstance(panel_gauss_order, int)
            or panel_gauss_order < 2):
        raise ValueError("panel_gauss_order must be >= 2")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    L = np.zeros((2, 2), dtype=float)
    for p in range(2):
        L[p, p] = _regularized_self_inductance(
            scene.assemblies[p],
            panel_length_factor=panel_length_factor,
            panel_gauss_order=panel_gauss_order,
            chunk_size=chunk_size,
        )
    a, b = sources
    total = 0.0
    for first in range(0, len(a.points), chunk_size):
        offset = (
            a.points[first:first + chunk_size, None, :]
            - b.points[None, :, :]
        )
        distance2 = np.einsum("ijk,ijk->ij", offset, offset)
        if np.any(distance2 <= 1e-24):
            raise ValueError("distinct port paths intersect or are unresolved")
        dot = a.line_elements[first:first + chunk_size] @ b.line_elements.T
        total += float(np.sum(dot / np.sqrt(distance2)))
    L[0, 1] = L[1, 0] = MU0 / (4 * math.pi) * total
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
            active = lengths > 0.0
            lengths = lengths[active]
            elements = line.line_elements[active]
            centers = line.points[active]
            tangents = elements / lengths[:, None]
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
                centers[:, None, None, :]
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


def build_bfzi_source(
    scene: BfziScene, frequency_hz: float, order: int = 4, *,
    air_panel_length_factor: float = 4.0,
    air_panel_gauss_order: int = 3,
):
    if not math.isfinite(float(frequency_hz)) or frequency_hz <= 0:
        raise ValueError("frequency must be positive and finite")
    # The current free-space C/B and seawater Green kernels require
    # uniform mu0. A magnetic wire/core needs interface magnetization
    # coupling; merely changing the isolated-wire skin mu is not valid.
    for assembly in scene.assemblies:
        if not math.isclose(
            assembly.material.relative_permeability, 1.0,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(
                "BFZI MQS-v1 requires nonmagnetic wire (mu_r=1); "
                "magnetic conductor/core coupling is not implemented"
            )
    lines = tuple(build_line_source(item, order) for item in scene.assemblies)
    return BFZISources(
        scene=scene,
        frequency_hz=float(frequency_hz),
        lines=lines,
        inductance_air=air_inductance(
            scene, lines,
            panel_length_factor=air_panel_length_factor,
            panel_gauss_order=air_panel_gauss_order,
        ),
    )
