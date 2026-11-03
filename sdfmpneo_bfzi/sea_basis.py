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
from .source import BFZISources, LineSource, MU0


@dataclass(frozen=True)
class CurrentMode:
    center: np.ndarray
    direction: np.ndarray
    width: float
    # -1 means curl(scalar * constant vector); 0/1 means
    # curl(scalar * actual closed-port vector potential).
    source_port: int = -1
    # Optional rotated SPD Gaussian precision tensor, exp(-dx.T @ G @ dx).
    # None exactly preserves the historical isotropic width-based modes.
    gaussian_metric: np.ndarray | None = None


@dataclass(frozen=True)
class SeaCurrentBasis:
    scene: BfziScene
    modes: tuple[CurrentMode, ...]
    wall_scale: float
    source_lines: tuple[LineSource, LineSource] | None = None
    outer_envelope: str = "quadratic"
    outer_taper_width: float = 0.0

    @property
    def size(self) -> int:
        return len(self.modes)

    def potential_primitive(self, xyz) -> np.ndarray:
        """Curl potentials F for all modes, with J=curl(F).

        F is identically zero on the insulating-box and artificial outer
        boundaries. For a source vector potential C with curl(C)=B/mu0,
        integration by parts gives
            mu0 * integral curl(F).C = integral F.B
        with no surface term. The latter is a smoother way to integrate
        the physical two-port source coupling K near the insulating walls.
        """
        points = np.asarray(xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz must have shape (n,3)")
        if not np.isfinite(points).all():
            raise ValueError("basis query positions must be finite")
        if not self.scene.in_seawater(points).all():
            raise ValueError("sea current basis can only be evaluated in seawater")
        radial = np.linalg.norm(
            points - np.asarray(self.scene.outer_center), axis=1,
        )
        if self.outer_envelope == "quadratic":
            envelope = 1.0 - (radial / self.scene.outer_radius)**2
        else:
            t = np.clip(
                (self.scene.outer_radius - radial)
                / self.outer_taper_width, 0.0, 1.0,
            )
            envelope = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
        for assembly in self.scene.assemblies:
            squared, _ = assembly.enclosure.squared_exterior_distance(
                points, assembly.pose,
            )
            distance = np.sqrt(squared)
            envelope *= distance / (distance + self.wall_scale)

        guided_ports = {
            mode.source_port for mode in self.modes if mode.source_port >= 0
        }
        source_fields = {}
        if guided_ports:
            if self.source_lines is None:
                raise ValueError("source-guided basis has no complete port lines")
            source_fields = {
                port: self.source_lines[port].potential_kernel(points)
                for port in guided_ports
            }
        primitive = np.empty((len(points), self.size, 3), dtype=float)
        for index, mode in enumerate(self.modes):
            offset = points - mode.center
            scaled = (
                offset / mode.width**2 if mode.gaussian_metric is None
                else offset @ mode.gaussian_metric
            )
            factor = envelope * np.exp(
                -np.einsum("ni,ni->n", offset, scaled),
            )
            direction = (
                mode.direction if mode.source_port < 0
                else source_fields[mode.source_port]
            )
            primitive[:, index, :] = factor[:, None] * direction
        return primitive

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
        radius2 = np.einsum("ni,ni->n", sphere, sphere)
        if self.outer_envelope == "quadratic":
            # Historical global factor, kept only for explicit comparison.
            # This modifies even near-coil currents whenever R changes.
            out_factor = 1.0 - radius2 / self.scene.outer_radius**2
            out_grad = -2.0 * sphere / self.scene.outer_radius**2
        else:
            # Compact outer boundary transition. The physical interior,
            # including all enclosure corners, sees exactly f=1 and grad f=0
            # for every sufficiently large truncation radius. At the outer
            # surface f=0 and grad f=0, so curl(f*g*a) remains solenoidal
            # and has zero normal flux (the trial space is more restrictive
            # than the continuum's tangential boundary trace).
            thickness = self.outer_taper_width
            t = np.clip(
                (self.scene.outer_radius - np.sqrt(radius2)) / thickness,
                0.0, 1.0,
            )
            out_factor = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
            slope = 30.0 * t**2 * (1.0 - t)**2
            out_grad = (
                -(slope / thickness
                  / np.maximum(np.sqrt(radius2), 1e-30))[:, None] * sphere
            )

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

        guided_ports = {
            mode.source_port for mode in self.modes if mode.source_port >= 0
        }
        source_fields = {}
        if guided_ports:
            if self.source_lines is None:
                raise ValueError("source-guided basis has no complete port lines")
            for port in guided_ports:
                line = self.source_lines[port]
                # LineSource stores C=A/mu0 and B=mu0*curl(C).
                # Evaluating once per port avoids repeating Biot-Savart
                # evaluation for modes with different spatial envelopes.
                source_fields[port] = (
                    line.potential_kernel(points),
                    np.real(line.magnetic_field(points)) / MU0,
                )
        out = np.empty((len(points), self.size, 3), dtype=float)
        for k, mode in enumerate(self.modes):
            offset = points - mode.center
            scaled = (
                offset / mode.width**2 if mode.gaussian_metric is None
                else offset @ mode.gaussian_metric
            )
            gauss = np.exp(-np.einsum("ni,ni->n", offset, scaled))
            gradient = gauss[:, None] * (
                envelope_grad - 2 * envelope[:, None] * scaled
            )
            if mode.source_port < 0:
                out[:, k, :] = np.cross(gradient, mode.direction)
            else:
                # Exact curl(g C_port), real, solenoidal by construction:
                # grad(g) x C_port + g curl(C_port). Because g=0 on
                # insulating boxes, the trace has zero normal component.
                potential, curl_potential = source_fields[mode.source_port]
                out[:, k, :] = (
                    np.cross(gradient, potential)
                    + (envelope * gauss)[:, None] * curl_potential
                )
        return out


def build_sea_basis(
    scene: BfziScene, *,
    wall_scale: float | None = None,
    modes_per_face: int = 2,
    face_refinement: int = 1,
    include_global: bool = True,
    refine_thin_faces_only: bool = False,
    sources: BFZISources | None = None,
    include_source_guided: bool = False,
    include_source_face_guided: bool = False,
    include_anisotropic_face_guided: bool = False,
    include_global_multiscale: bool = False,
    global_scale_factors: tuple[float, ...] = (0.75, 2.4),
    include_corner_source_guided: bool = False,
    outer_envelope: str = "boundary_layer",
    outer_taper_width: float | None = None,
) -> SeaCurrentBasis:
    """Generate box-following curl modes and optional long-range modes.

    The face-localized modes are tangential and update with 6-DOF box
    pose. Face refinement adds four nested smaller Gaussian patches per
    selected face. Level 3 nests four narrower edge-centered patches.
    Level 4 adds eight compact patches per selected face, including four
    corner locations. Level 5 adds four narrower edge-lip patches per
    selected face (without replacing previous modes) to independently
    test whether the 167-mode parent misses near-perimeter currents.
    refine_thin_faces_only=True restricts these *extra*
    patches to the two faces normal to the box's shortest dimension
    (typically the broad top/bottom charging-pad surfaces), keeping the
    original complete 27-mode two-tangent base unchanged. Optionally,
    include_source_face_guided adds eight full-port curl potentials centered
    on the two broad package faces, with two physically scaled footprints
    per face and port; it is nested after the existing four center-guided
    full-port modes. This allows distinct near-wall induced responses
    without adding another hierarchy of arbitrary constant-vector patches.
    Optional anisotropic full-port face curls add eight independent
    geometry-following responses with physical face-tangential widths
    and narrower face-normal widths. The Gaussian metric rotates with
    each package pose, so this is a genuine normal-vs-tangential
    refinement rather than another set of face-plane translations.
    Optional global_multiscale adds six independent large/medium-length
    curl directions; optional corner_source_guided adds 16 face-corner
    full-port curls with rotated anisotropic Gaussian footprints.
    Both are experimental holdouts of the already solved 207-mode space,
    not silently selected new reference defaults.
    The global modes resolve long-range circulating currents. A finite
    trial space is not guaranteed accurate before a hierarchy converges.
    """
    if modes_per_face not in (1, 2):
        raise ValueError("modes_per_face must be 1 or 2")
    if face_refinement not in (1, 2, 3, 4, 5):
        raise ValueError("face_refinement must be 1, 2, 3, 4 or 5")
    if not isinstance(refine_thin_faces_only, bool):
        raise ValueError("refine_thin_faces_only must be a boolean")
    if outer_envelope not in ("boundary_layer", "quadratic"):
        raise ValueError("outer_envelope must be 'boundary_layer' or 'quadratic'")
    if include_anisotropic_face_guided and not include_source_face_guided:
        raise ValueError(
            "anisotropic face modes require source-face guided parent"
        )
    if include_corner_source_guided and not include_anisotropic_face_guided:
        raise ValueError(
            "corner source-guided modes require anisotropic face parent"
        )
    if include_global_multiscale and (
        not include_global or not global_scale_factors
        or any(not np.isfinite(factor) or factor <= 0
               for factor in global_scale_factors)
        or len(set(global_scale_factors)) != len(global_scale_factors)
    ):
        raise ValueError(
            "global scale factors must be distinct positive finite "
            "values and include_global must be True"
        )
    if (include_source_guided or include_source_face_guided
            or include_anisotropic_face_guided
            or include_corner_source_guided) and (
        sources is None or sources.scene is not scene
    ):
        raise ValueError(
            "source-guided trial modes require complete port sources "
            "constructed for the same seawater scene"
        )
    scale = (
        0.075 * min(
            min(assembly.enclosure.half_extents)
            for assembly in scene.assemblies
        )
        if wall_scale is None else float(wall_scale)
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("wall_scale must be positive")
    # Physical package reach is independent of the artificial outer radius.
    # Use a fixed physical taper width whenever the boundary has clearance;
    # only geometrically tight scenes require reducing that width.
    origin = np.asarray(scene.outer_center)
    package_reach = max(
        float(np.linalg.norm(assembly.pose.center - origin)
              + np.linalg.norm(assembly.enclosure.half_extents))
        for assembly in scene.assemblies
    )
    clearance = scene.outer_radius - package_reach
    if clearance <= 0:
        raise ValueError("outer sphere does not contain all package extents")
    if outer_taper_width is None:
        taper_width = min(0.25 * package_reach, 0.5 * clearance)
    else:
        taper_width = float(outer_taper_width)
        if not np.isfinite(taper_width) or not 0 < taper_width < clearance:
            raise ValueError("outer_taper_width must be finite, positive, and below package clearance")
    modes = []
    for assembly in scene.assemblies:
        half = np.asarray(assembly.enclosure.half_extents)
        thinnest = int(np.argmin(half))
        for axis in range(3):
            tangents = [j for j in range(3) if j != axis]
            for sign in (-1, 1):
                # Always retain the original face-center modes. The next
                # level adds four tangentially shifted modes per face, giving
                # a nested trial space rather than replacing the baseline.
                patches = [(0.0, 0.0, 1.0)]
                if face_refinement >= 2 and (
                    not refine_thin_faces_only or axis == thinnest
                ):
                    patches.extend(
                        (u, v, 0.70)
                        for u in (-0.45, 0.45)
                        for v in (-0.45, 0.45)
                    )
                    if face_refinement >= 3:
                        patches.extend((u, v, 0.50) for u, v in (
                            (-0.70, 0.0), (0.70, 0.0),
                            (0.0, -0.70), (0.0, 0.70),
                        ))
                    if face_refinement >= 4:
                        patches.extend(
                            (u, v, 0.34)
                            for u in (-0.72, 0.72)
                            for v in (-0.72, 0.72)
                        )
                        patches.extend((u, v, 0.34) for u, v in (
                            (-0.25, 0.0), (0.25, 0.0),
                            (0.0, -0.25), (0.0, 0.25),
                        ))
                    if face_refinement >= 5:
                        # Independent near-perimeter edge-lip responses.
                        # Stage 4 resolves corners and quarter-face
                        # gradients, but not thin current variation near
                        # the midpoints of the four outer face edges.
                        # A width ~0.03 m for 0.1 m half faces remains
                        # resolved by the refined tangential panel rule.
                        patches.extend((u, v, 0.30) for u, v in (
                            (-0.92, 0.0), (0.92, 0.0),
                            (0.0, -0.92), (0.0, 0.92),
                        ))
                for u, v, width_factor in patches:
                    local = np.zeros(3)
                    local[axis] = sign * (
                        half[axis] + max(scale, 0.15 * half[axis])
                    )
                    local[tangents] = np.array((u, v)) * half[tangents]
                    center = assembly.pose.position(local)
                    width = max(half.max() * 0.9 * width_factor, 3 * scale)
                    for tangent_axis in tangents[:modes_per_face]:
                        direction = assembly.pose.rotation[:, tangent_axis]
                        modes.append(CurrentMode(center, direction, width))
    if include_global:
        center = origin
        # Keep the physical Gaussian part of the global trial functions
        # unchanged when only the artificial outer radius is refined.
        # Previously width=0.8*outer_radius changed the entire basis at
        # the same time as the truncation boundary, confounding domain
        # convergence by an unrelated approximation-space change.
        global_width = 1.5 * package_reach
        for axis in range(3):
            modes.append(CurrentMode(center, np.eye(3)[axis], global_width))
        if include_global_multiscale:
            # The influential long-range direction may still miss other
            # radial scales. Test these separately from corner-local modes.
            # Widths depend only on the physical package reach, never on
            # the artificial outer integration radius.
            for factor in global_scale_factors:
                for axis in range(3):
                    modes.append(CurrentMode(
                        center, np.eye(3)[axis], factor*package_reach,
                    ))
    if include_source_guided:
        # Use two envelope widths per physical port. These are new
        # current *directions*, not extra translations of the generic
        # constant-vector Gaussian basis. The widths depend on the pad,
        # not on the artificial outer sphere.
        for port, assembly in enumerate(scene.assemblies):
            reach = float(np.linalg.norm(assembly.enclosure.half_extents))
            for width_factor in (1.0, 2.0):
                modes.append(CurrentMode(
                    np.asarray(assembly.pose.center).copy(),
                    np.zeros(3), reach * width_factor,
                    source_port=port,
                ))
    if include_source_face_guided:
        # Current loops induced by a complete physical port concentrate
        # around the two broad package faces, but the previous center-only
        # source modes have no independent top/bottom amplitudes. Curl of
        # each localized full-port vector potential provides that missing
        # physics-informed direction while retaining div(J)=0 and zero
        # normal current at every insulating boundary.
        for port, assembly in enumerate(scene.assemblies):
            half=np.asarray(assembly.enclosure.half_extents)
            normal=int(np.argmin(half))
            for sign in (-1.0,1.0):
                local=np.zeros(3)
                local[normal]=sign*(
                    half[normal]+max(scale,0.15*half[normal])
                )
                center=assembly.pose.position(local)
                for width_factor in (0.45,0.85):
                    modes.append(CurrentMode(
                        center,np.zeros(3),
                        max(width_factor*half.max(),3.0*scale),
                        source_port=port,
                    ))
    if include_anisotropic_face_guided:
        # Test missing normal boundary-layer resolution independently of
        # lateral edge patches: two distinct normal scales at every broad
        # face, but one geometry-sized tangential footprint. The precision
        # matrix rotates as R diag(1/w_i^2) R.T for arbitrary SE(3) poses.
        for port, assembly in enumerate(scene.assemblies):
            half = np.asarray(assembly.enclosure.half_extents,dtype=float)
            normal = int(np.argmin(half))
            lateral_widths = np.maximum(0.9*half,6.0*scale)
            for sign in (-1.,1.):
                local=np.zeros(3)
                local[normal]=sign*(
                    half[normal]+max(scale,0.15*half[normal])
                )
                center=assembly.pose.position(local)
                for normal_factor in (0.20,0.35):
                    widths=lateral_widths.copy()
                    widths[normal]=max(
                        normal_factor*half.max(),
                        (2.0 if normal_factor==0.20 else 4.0)*half[normal],
                        6.0*scale,
                    )
                    metric=(
                        assembly.pose.rotation
                        @ np.diag(1.0/widths**2)
                        @ assembly.pose.rotation.T
                    )
                    modes.append(CurrentMode(
                        center,np.zeros(3),float(np.max(widths)),
                        source_port=port,gaussian_metric=metric,
                    ))
    if include_corner_source_guided:
        # Each physical box contributes a different field shape at each
        # broad-face corner. The full-port vector potential supplies a
        # direction different from the existing constant-vector patches.
        # Keep these strictly *after* the 207-mode family.
        for port, assembly in enumerate(scene.assemblies):
            half=np.asarray(assembly.enclosure.half_extents,dtype=float)
            normal=int(np.argmin(half))
            tangents=[axis for axis in range(3) if axis!=normal]
            widths=np.maximum(0.35*half,6.0*scale)
            widths[normal]=max(
                0.22*float(half.max()),2.0*half[normal],6.0*scale,
            )
            metric=(
                assembly.pose.rotation @ np.diag(1.0/widths**2)
                @ assembly.pose.rotation.T
            )
            for sign in (-1.0,1.0):
                for u in (-0.72,0.72):
                    for v in (-0.72,0.72):
                        local=np.zeros(3)
                        local[normal]=sign*(
                            half[normal]+max(scale,0.15*half[normal])
                        )
                        local[tangents]=np.array((u,v))*half[tangents]
                        modes.append(CurrentMode(
                            assembly.pose.position(local),
                            np.zeros(3),float(np.max(widths)),
                            source_port=port,gaussian_metric=metric.copy(),
                        ))
    return SeaCurrentBasis(
        scene, tuple(modes), scale,
        sources.lines if (include_source_guided or include_source_face_guided
                          or include_anisotropic_face_guided
                          or include_corner_source_guided) else None,
        outer_envelope, taper_width,
    )
