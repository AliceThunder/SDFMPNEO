"""BFZI assemblies and finite conducting seawater domain.

Reuses the vNext separation of geometry, material, and pose, but never
reinterprets the historical superellipse as a BFZI spiral.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import numpy as np

from .geometry import BFZICoil, RigidPose
from sdfmpneo_vnext.scene import ConductorMaterial


def _wall_normal_gauss(
    lower: float, upper: float, scale: float,
    nodes: np.ndarray, weights: np.ndarray, mapping: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive Gauss measure in true wall distance d, optionally log(1+d/s).

    The logarithmic coordinate resolves the physical curl-envelope
    derivative scale d/(d+s); dd=(s+d)dt. Exact continuum measure is
    unchanged and no softened electromagnetic kernel is introduced.
    """
    if mapping == "log":
        t0, t1 = math.log1p(lower/scale), math.log1p(upper/scale)
        t = 0.5*(t1-t0)*nodes+0.5*(t1+t0)
        d = scale*np.expm1(t)
        w = 0.5*(t1-t0)*weights*(scale+d)
    elif mapping == "linear":
        d = 0.5*(upper-lower)*nodes+0.5*(upper+lower)
        w = 0.5*(upper-lower)*weights
    else:
        raise ValueError("shell_normal_mapping must be 'linear' or 'log'")
    return d, w


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
        # An exposed return would invalidate the assumption that seawater
        # carries only induced currents outside the sealed insulating box.
        if np.any(np.abs(waypoints) + radius >= h):
            raise ValueError("return route must remain inside the insulating box")
        if self.coil.turns > 1 and 2.0 * radius >= self.coil.pitch:
            raise ValueError("wire diameter must be smaller than coil pitch")
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
        exclusion_padding: float = 0.0,
        outer_taper_width: float | None = None,
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
        if not math.isfinite(exclusion_padding) or exclusion_padding < 0:
            raise ValueError("exclusion_padding must be finite and nonnegative")
        if outer_taper_width is not None and (
            not math.isfinite(outer_taper_width)
            or not 0 < outer_taper_width < self.outer_radius
        ):
            raise ValueError("outer_taper_width must be within (0, outer_radius)")
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
            half = np.asarray(assembly.enclosure.half_extents) + exclusion_padding
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
        wet = right > left
        for assembly in self.assemblies:
            local = (mid - assembly.pose.center) @ assembly.pose.rotation
            half = np.asarray(assembly.enclosure.half_extents) + exclusion_padding
            wet &= ~np.all(np.abs(local) <= half, axis=-1)
        if not np.any(wet):
            return np.empty((0, 3)), np.empty(0)

        # Each wet interval stays wet after subdivision. The Gauss weights
        # remain positive, and the sum of radial cubature weights integrates
        # r^2 exactly on every subinterval for radial_order >= 2.
        knots = [left, right]
        if outer_taper_width is not None:
            # Split the ray cubature at the same artificial sphere transition
            # as the box-shell rule, without changing its wet domain.
            knots.append(np.clip(
                radius - outer_taper_width, left, right,
            ))
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

    def box_shell_quadrature(
        self, radial_order: int = 7, polar_order: int = 10,
        azimuthal_order: int = 16, *, wall_scale: float | None = None,
        shell_width_factor: float = 32.0, shell_normal_order: int = 3,
        shell_tangential_order: int = 4,
        shell_tangential_max_panel_width: float | None = None,
        shell_source_tangential_cuts: bool = False,
        shell_normal_mapping: str = "linear",
        outer_taper_width: float | None = None,
        max_radial_panel_width: float | None = None,
        radial_geometry_cuts: bool = False,
        polar_panel_count: int = 1,
        azimuthal_panel_order: int = 1,
        polar_geometry_order: int | None = None,
        azimuthal_geometry_order: int | None = None,
        return_sphere_count: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int]:
        """Positive, meshless partition-of-unity near-box / sphere cubature.

        Six nonoverlapping oriented rectangular slabs cover the neighborhood
        of each insulating box. The full outer sphere is integrated *without*
        cutting angular rays at the expanded box boundaries: instead a C2
        smooth step of the physical distance to each box blends the two
        integration rules. This removes a discontinuous angular wet-mask
        at each expanded box, the source of severe coarse-angle aliasing.

        The shell rule carries 1 - blend and the full-sphere rule carries
        blend. Their weights add to exactly one at every wet point in the
        continuous integral. The box interiors get zero spherical weight.
        Geometry-local slabs are not an electromagnetic volumetric mesh.
        If return_sphere_count=True, also return the number of initial
        spherical nodes. The remaining nodes are the disjoint box slabs;
        the two weighted regions exactly partition the continuous integral.
        """
        if min(radial_order, polar_order, azimuthal_order) < 2:
            raise ValueError("outer sphere Gauss orders must be >= 2")
        if (not isinstance(polar_panel_count, int)
                or isinstance(polar_panel_count, bool)
                or not 1 <= polar_panel_count <= polar_order
                or polar_order % polar_panel_count != 0):
            raise ValueError("polar_panel_count must divide polar_order")
        if (not isinstance(azimuthal_panel_order, int)
                or isinstance(azimuthal_panel_order, bool)
                or azimuthal_panel_order < 1):
            raise ValueError("azimuthal_panel_order must be a positive integer")
        if polar_geometry_order is not None and (
            not isinstance(polar_geometry_order, int)
            or isinstance(polar_geometry_order, bool)
            or polar_geometry_order < 2
            or polar_panel_count != 1
        ):
            raise ValueError(
                "polar_geometry_order must be an integer >= 2 "
                "and cannot be combined with polar_panel_count"
            )
        if (not isinstance(shell_normal_order, int)
                or shell_normal_order < 2
                or not isinstance(shell_tangential_order, int)
                or shell_tangential_order < 2):
            raise ValueError("shell Gauss orders must be integers >= 2")
        if not isinstance(shell_source_tangential_cuts, bool):
            raise ValueError("shell_source_tangential_cuts must be a boolean")
        if shell_normal_mapping not in ("linear", "log"):
            raise ValueError("shell_normal_mapping must be 'linear' or 'log'")
        if not math.isfinite(shell_width_factor) or shell_width_factor <= 0:
            raise ValueError("shell_width_factor must be finite and positive")
        if shell_tangential_max_panel_width is not None and (
            not math.isfinite(shell_tangential_max_panel_width)
            or shell_tangential_max_panel_width <= 0
        ):
            raise ValueError(
                "shell_tangential_max_panel_width must be finite and positive"
            )
        scale = (
            0.075 * min(min(a.enclosure.half_extents) for a in self.assemblies)
            if wall_scale is None else float(wall_scale)
        )
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("wall_scale must be positive and finite")
        if outer_taper_width is not None and (
            not math.isfinite(outer_taper_width)
            or not 0 < outer_taper_width < self.outer_radius
        ):
            raise ValueError("outer_taper_width must be within (0, outer_radius)")
        if max_radial_panel_width is not None and (
            not math.isfinite(max_radial_panel_width)
            or max_radial_panel_width <= 0
        ):
            raise ValueError("max_radial_panel_width must be positive and finite")
        if not isinstance(radial_geometry_cuts, bool):
            raise ValueError("radial_geometry_cuts must be a boolean")
        if azimuthal_geometry_order is not None and (
            not isinstance(azimuthal_geometry_order, int)
            or isinstance(azimuthal_geometry_order, bool)
            or azimuthal_geometry_order < 2
            or azimuthal_panel_order != 1
        ):
            raise ValueError(
                "azimuthal_geometry_order must be integer >=2 and "
                "requires azimuthal_panel_order=1"
            )
        if not isinstance(return_sphere_count, bool):
            raise ValueError("return_sphere_count must be a boolean")

        shell_width = shell_width_factor * scale
        center = np.asarray(self.outer_center)
        # Expanded shells must stay disjoint and entirely inside the sphere.
        # Only the shell *width* is reduced when geometric clearance demands it.
        for _ in range(32):
            expanded = [
                replace(
                    assembly,
                    enclosure=InsulatingBox(tuple(
                        np.asarray(assembly.enclosure.half_extents) + shell_width
                    )),
                )
                for assembly in self.assemblies
            ]
            valid = _obb_disjoint(*expanded)
            if valid:
                for assembly in expanded:
                    half = np.asarray(assembly.enclosure.half_extents)
                    corners = np.array(np.meshgrid(
                        [-half[0], half[0]], [-half[1], half[1]],
                        [-half[2], half[2]], indexing="ij",
                    )).reshape(3, -1).T
                    world = assembly.pose.position(corners)
                    if np.max(np.linalg.norm(world - center, axis=1)) >= self.outer_radius:
                        valid = False
                        break
            if valid:
                break
            shell_width *= 0.5
        else:
            raise ValueError("cannot place disjoint box shells inside outer sphere")

        def smooth_far_weight(x: np.ndarray, assembly: BFZIAssembly):
            distance2, _ = assembly.enclosure.squared_exterior_distance(
                x, assembly.pose,
            )
            t = np.clip(np.sqrt(distance2) / shell_width, 0.0, 1.0)
            return t**3 * (10.0 + t * (-15.0 + 6.0 * t))

        # Full sphere without a hard box mask. An interior-box quadrature
        # node has exactly zero weight; skip it before basis evaluation.
        r_nodes, r_weights = np.polynomial.legendre.leggauss(radial_order)
        if polar_geometry_order is not None:
            # Enclosure projections determine *fixed physical* mu-panel
            # boundaries, independent of the integration order. Both the
            # box and its partition-of-unity outer shell contribute; this
            # avoids cutting through their rapidly changing angular
            # silhouettes with a single broad Gauss interval.
            mu_cuts = [-1.0, 1.0]
            for inside, outside in zip(self.assemblies, expanded):
                for assembly in (inside, outside):
                    half = np.asarray(assembly.enclosure.half_extents)
                    corners = np.array(np.meshgrid(
                        (-half[0], half[0]),
                        (-half[1], half[1]),
                        (-half[2], half[2]),
                        indexing="ij",
                    )).reshape(3, -1).T
                    vectors = assembly.pose.position(corners) - center
                    norm = np.linalg.norm(vectors, axis=1)
                    mu_corners = vectors[:, 2] / np.maximum(norm, 1e-30)
                    mu_cuts.extend((
                        float(np.min(mu_corners)),
                        float(np.max(mu_corners)),
                    ))
            boundaries = [-1.0]
            for value in sorted(mu_cuts):
                if boundaries[-1] + 1e-5 < value < 1.0 - 1e-5:
                    boundaries.append(value)
            boundaries.append(1.0)
            gaussian_mu, gaussian_mw = np.polynomial.legendre.leggauss(
                polar_geometry_order
            )
            panel_mu, panel_mw = [], []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                panel_mu.append(
                    0.5 * (right - left) * gaussian_mu
                    + 0.5 * (right + left)
                )
                panel_mw.append(0.5 * (right - left) * gaussian_mw)
            mu = np.concatenate(panel_mu)
            mu_weights = np.concatenate(panel_mw)
        elif polar_panel_count == 1:
            mu, mu_weights = np.polynomial.legendre.leggauss(polar_order)
        else:
            # Resolve angular changes around the equatorial and other
            # latitude panel interfaces without moving *every* polar
            # sample when total order changes. Preserve exactly
            # polar_order positive Gauss points over [-1,1].
            count_per_panel = polar_order // polar_panel_count
            boundaries = np.linspace(-1.0, 1.0, polar_panel_count + 1)
            angular_mu = []
            angular_mw = []
            gauss_mu, gauss_mw = np.polynomial.legendre.leggauss(count_per_panel)
            for index in range(polar_panel_count):
                a, b = boundaries[index:index + 2]
                angular_mu.append(0.5 * (b - a) * gauss_mu
                                  + 0.5 * (b + a))
                angular_mw.append(0.5 * (b - a) * gauss_mw)
            mu = np.concatenate(angular_mu)
            mu_weights = np.concatenate(angular_mw)
        if azimuthal_geometry_order is not None:
            # Projection of both physical box corners and smooth shell
            # corners into phi. These angular references are FIXED when
            # changing the Gauss order. No hard angular wet masking:
            # the exact box-shell partition of unity stays in place.
            phi_cuts = [0.0, 2.0 * math.pi]
            for inner, outer in zip(self.assemblies, expanded):
                for assembly in (inner, outer):
                    half = np.asarray(assembly.enclosure.half_extents)
                    corners = np.array(np.meshgrid(
                        (-half[0], half[0]),
                        (-half[1], half[1]),
                        (-half[2], half[2]), indexing="ij",
                    )).reshape(3, -1).T
                    vectors = assembly.pose.position(corners)-center
                    transverse = np.linalg.norm(vectors[:, :2], axis=1)
                    good = transverse > 1e-12
                    phi_cuts.extend(np.mod(
                        np.arctan2(vectors[good, 1], vectors[good, 0]),
                        2.0*math.pi,
                    ).tolist())
            boundaries = [0.0]
            for value in sorted(phi_cuts):
                if boundaries[-1]+1e-7 < value < 2.0*math.pi-1e-7:
                    boundaries.append(value)
            boundaries.append(2.0*math.pi)
            gauss_phi, gauss_pw = np.polynomial.legendre.leggauss(
                azimuthal_geometry_order
            )
            angles, azimuthal_weights = [], []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                halfspan = 0.5*(right-left)
                midpoint = 0.5*(right+left)
                angles.append(midpoint+halfspan*gauss_phi)
                azimuthal_weights.append(halfspan*gauss_pw)
            angles = np.concatenate(angles)
            azimuthal_weights = np.concatenate(azimuthal_weights)
        elif azimuthal_panel_order == 1:
            angles = 2.0 * math.pi * (
                np.arange(azimuthal_order) + 0.5
            ) / azimuthal_order
            azimuthal_weights = np.full(
                azimuthal_order, 2.0 * math.pi / azimuthal_order,
            )
        else:
            # Periodic azimuth split into azimuthal_order fixed sectors;
            # Gauss nodes within each sector suppress angular aliasing.
            gauss_phi, gauss_pw = np.polynomial.legendre.leggauss(
                azimuthal_panel_order
            )
            delta_phi = 2.0 * math.pi / azimuthal_order
            angles = (
                delta_phi * (
                    np.arange(azimuthal_order)[:, None]
                    + 0.5 * (1.0 + gauss_phi)[None, :]
                )
            ).reshape(-1)
            azimuthal_weights = np.tile(
                0.5 * delta_phi * gauss_pw, azimuthal_order,
            )
        # Resolve radial changes of the blending field at approximately
        # the distances of the packages, independently of angular order.
        cuts = [0.0, self.outer_radius]
        if outer_taper_width is not None:
            # Resolve the artificial outer-boundary taper with its own
            # radial Gauss interval. This is essential when R grows:
            # otherwise fixed-order far-field integration may miss the
            # thin transition and falsely look like domain convergence.
            cuts.append(self.outer_radius - outer_taper_width)
        for assembly in self.assemblies:
            distance = float(np.linalg.norm(assembly.pose.center - center))
            reach = float(np.linalg.norm(assembly.enclosure.half_extents))
            cuts.extend((
                max(0.0, distance - shell_width),
                distance,
                min(self.outer_radius, distance + reach + shell_width),
            ))
        if radial_geometry_cuts:
            # Physical radial breakpoints of the actual (possibly
            # rotated) enclosures and expanded box-shell partitions.
            # The closest point of an OBB to the sphere center is not
            # generally a corner; use its exact exterior distance.
            for inner, outer in zip(self.assemblies, expanded):
                for assembly in (inner, outer):
                    closest2, _ = assembly.enclosure.squared_exterior_distance(
                        center[None, :], assembly.pose,
                    )
                    half = np.asarray(assembly.enclosure.half_extents)
                    corners = np.array(np.meshgrid(
                        (-half[0], half[0]),
                        (-half[1], half[1]),
                        (-half[2], half[2]), indexing="ij",
                    )).reshape(3, -1).T
                    farthest = max(np.linalg.norm(
                        assembly.pose.position(corners) - center, axis=1,
                    ))
                    cuts.extend((float(np.sqrt(closest2[0])), float(farthest)))
        cuts = sorted(set(
            float(np.clip(v, 0.0, self.outer_radius)) for v in cuts
        ))
        intervals = [
            (a, b) for a, b in zip(cuts[:-1], cuts[1:])
            if b - a > 1e-12
        ]
        if max_radial_panel_width is not None:
            # Fixed Gauss order on a growing outer-radius interval loses
            # physical radial resolution and masquerades as domain error.
            # Split only the smooth spherical rule, never the near-box slabs.
            radial_panels = []
            for a, b in intervals:
                count = max(1, math.ceil((b - a) / max_radial_panel_width))
                step = (b - a) / count
                radial_panels.extend(
                    (a + k * step, a + (k + 1) * step)
                    for k in range(count)
                )
            intervals = radial_panels
        rr = np.concatenate([
            0.5 * (hi - lo) * r_nodes + 0.5 * (hi + lo)
            for lo, hi in intervals
        ])
        rw = np.concatenate([
            0.5 * (hi - lo) * r_weights
            * (0.5 * (hi - lo) * r_nodes + 0.5 * (hi + lo))**2
            for lo, hi in intervals
        ])
        transverse = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
        directions = np.stack((
            transverse[:, None] * np.cos(angles),
            transverse[:, None] * np.sin(angles),
            np.broadcast_to(mu[:, None], (len(mu), len(angles))),
        ), axis=-1).reshape(-1, 3)
        angular_weights = (
            mu_weights[:, None] * azimuthal_weights[None, :]
        ).reshape(-1)
        sphere_points = (
            center + rr[:, None, None] * directions[None, :, :]
        ).reshape(-1, 3)
        sphere_weights = (rw[:, None] * angular_weights[None, :]).reshape(-1)
        for assembly in self.assemblies:
            sphere_weights *= smooth_far_weight(sphere_points, assembly)
        # Strict >0 also removes the non-conductive box interiors.
        wet = sphere_weights > 0
        point_sets = [sphere_points[wet]]
        weight_sets = [sphere_weights[wet]]

        normal_nodes, normal_weights = np.polynomial.legendre.leggauss(
            shell_normal_order
        )
        # Integrate tangential coordinates *piecewise* at the physical
        # box edges and the face center. A single Gauss rule across the
        # enlarged [-h-w,h+w] interval samples different near-wall patches
        # at even/odd orders, which can alter R and K by order-one amounts.
        # On exterior edge strips use u**2 mapping to cluster positive
        # Gauss weights within one wall_scale of the sharp edge transition.
        center_order = max(2, (shell_tangential_order + 1) // 2)
        edge_order = max(2, shell_tangential_order // 2)
        center_nodes, center_weights = np.polynomial.legendre.leggauss(
            center_order
        )
        edge_nodes, edge_weights = np.polynomial.legendre.leggauss(
            edge_order
        )
        center_t = 0.5 * (center_nodes + 1.0)
        center_w = 0.5 * center_weights
        edge_t = 0.5 * (edge_nodes + 1.0)
        edge_w = 0.5 * edge_weights

        def tangent_rule(
            half_extent: float, extended: bool, feature_knots=(),
        ):
            # The physical face center and two real edges remain panel
            # boundaries. When requested, every turn's actual planar
            # coil side coordinates and return-waypoint coordinates are
            # additional FIXED integration-panel boundaries. The knots
            # follow each box's local axes under arbitrary SE(3) poses.
            # They change neither the conductor nor the seawater domain.
            cuts = [-half_extent, 0.0, half_extent]
            cuts.extend(
                float(k) for k in feature_knots
                if np.isfinite(k) and -half_extent < k < half_extent
            )
            cuts.sort()
            boundaries = [cuts[0]]
            tolerance = max(1e-12, 1e-7*half_extent)
            for knot in cuts[1:]:
                if knot-boundaries[-1] > tolerance:
                    boundaries.append(knot)
            boundaries[-1] = half_extent
            panels = []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                count = (
                    1 if shell_tangential_max_panel_width is None
                    else max(
                        1, math.ceil(
                            (right-left)/shell_tangential_max_panel_width
                        )
                    )
                )
                segment = np.linspace(left, right, count+1)
                widths = np.diff(segment)
                panels.append((segment[:-1],widths))
            patches = [
                (start[:,None]+width[:,None]*center_t[None,:]).reshape(-1)
                for start,width in panels
            ]
            patch_weights = [
                (width[:,None]*center_w[None,:]).reshape(-1)
                for _,width in panels
            ]
            nodes = patches
            weights = patch_weights
            if extended:
                # Side and corner strips need the same control: the
                # near-edge envelope changes on the tiny wall_scale,
                # while the whole exterior strip spans shell_width.
                # Composite u=sqrt(distance/shell_width) panels retain
                # the existing positive edge clustering/Jacobian.
                edge_count = (
                    1 if shell_tangential_max_panel_width is None
                    else max(
                        1, math.ceil(
                            shell_width / shell_tangential_max_panel_width
                        ),
                    )
                )
                # Equal physical-distance subpanels in s=shell_width*u^2:
                # their actual s-width is <= the requested cap, while
                # Gauss nodes still cluster at the sharp physical edge.
                edge_boundaries = np.sqrt(
                    np.linspace(0.0, 1.0, edge_count + 1)
                )
                edge_du = np.diff(edge_boundaries)
                u = (
                    edge_boundaries[:-1, None]
                    + edge_du[:, None] * edge_t[None, :]
                ).reshape(-1)
                stretch = shell_width * u**2
                stretch_w = (
                    2.0 * shell_width * u
                    * (
                        edge_du[:, None] * edge_w[None, :]
                    ).reshape(-1)
                )
                nodes = [
                    -half_extent - stretch, *nodes,
                    half_extent + stretch,
                ]
                weights = [stretch_w, *weights, stretch_w]
            return np.concatenate(nodes), np.concatenate(weights)

        for assembly in self.assemblies:
            half = np.asarray(assembly.enclosure.half_extents)
            source_knots = [[], [], []]
            if shell_source_tangential_cuts:
                # For the continuous round-square spiral, each turn has
                # actual in-plane side scales A(eta), with four quarter-
                # turn phases. Include both signs to cover every side.
                coil = assembly.coil
                for turn in range(coil.turns):
                    for side in range(4):
                        eta = turn + side/4.0
                        a_side = coil.outer_half_size-coil.pitch*eta
                        source_knots[0].extend((-a_side,a_side))
                        source_knots[1].extend((-a_side,a_side))
                for waypoint in assembly.return_waypoints:
                    for j,coordinate in enumerate(waypoint):
                        source_knots[j].append(float(coordinate))
            # Earlier slab axes stay inside the physical face extent; later
            # axes include face, edge and corner regions. This partitions
            # the full shell exactly without geometric overlap.
            for axis in range(3):
                tangential = [
                    tangent_rule(
                        half[j],extended=(j > axis),
                        feature_knots=source_knots[j],
                    ) if j != axis else None
                    for j in range(3)
                ]
                for sign in (-1.0, 1.0):
                    # Geometric normal splitting resolves the sharp box
                    # boundary gradient and the smooth blending shoulder.
                    boundaries = [0.0]
                    distance = 1.5 * scale
                    while distance < shell_width:
                        boundaries.append(distance)
                        distance *= 3.0
                    boundaries.append(shell_width)
                    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
                        distance_nodes, distance_weights = _wall_normal_gauss(
                            lower, upper, scale,
                            normal_nodes, normal_weights,
                            shell_normal_mapping,
                        )
                        # Maintain increasing local coordinate order on
                        # negative faces; the Jacobian is the positive
                        # absolute wall-normal distance measure.
                        face_nodes = sign*(half[axis]+distance_nodes)
                        face_weights = distance_weights
                        if sign < 0:
                            face_nodes = face_nodes[::-1]
                            face_weights = face_weights[::-1]
                        axes = []
                        weights = []
                        for j in range(3):
                            if j == axis:
                                axes.append(face_nodes)
                                weights.append(face_weights)
                            else:
                                nodes, w = tangential[j]
                                axes.append(nodes)
                                weights.append(w)
                        grid = np.meshgrid(*axes, indexing="ij")
                        weight_grid = np.meshgrid(*weights, indexing="ij")
                        points = assembly.pose.position(
                            np.stack([c.ravel() for c in grid], axis=-1)
                        )
                        w = np.prod(
                            np.stack(weight_grid, axis=0), axis=0
                        ).ravel()
                        w *= 1.0 - smooth_far_weight(points, assembly)
                        active = w > 0
                        if np.any(active):
                            point_sets.append(points[active])
                            weight_sets.append(w[active])
        positions = np.concatenate(point_sets)
        weights = np.concatenate(weight_sets)
        if return_sphere_count:
            return positions, weights, len(point_sets[0])
        return positions, weights
