"""Independent unsoftened Coulomb double-integral reference for small G1 spaces.

Splits 1/|x-y| with a *partition of unity*, rather than replacing cubature
nodes by overlapping volume balls. The singular local part is integrated
in physical spherical coordinates about x, where r**2/r = r; the smooth
remainder is integrated using an independently selected y cubature.

Refining both volume rules and the local sphere rule approaches the original
Coulomb bilinear form. This is deliberately a small-space reference path:
it is NOT a fast 229-mode magnetic assembly or an error certificate.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.distance import cdist

from .source import MU0


def _near_window(radius_ratio):
    """C2 near-field partition: one for s<=1/2, zero for s>=1."""
    s = np.asarray(radius_ratio)
    t = np.clip(2.0*s-1.0, 0.0, 1.0)
    return 1.0 - t**3*(10.0 + t*(-15.0+6.0*t))


def assemble_coulomb_split(
    x, wx, y, wy, evaluate, clearance, *,
    local_radius: float = 0.05,
    local_fraction: float = 0.6,
    local_radial_order: int = 5,
    local_polar_order: int = 6,
    local_azimuthal_order: int = 12,
    chunk_size: int = 32,
    radial_segments=None,
    angular_frame=None,
    return_diagnostics: bool = False,
):
    """Return real Galerkin L for a continuous vector-current evaluator.

    Parameters
    ----------
    x,wx and y,wy : two independent positive 3D volume cubatures.
    evaluate : callable(points)->(n,m,3), *same continuous* current modes.
    clearance : callable(points)->(n,) distance to the nearest domain boundary.
        Every local integration ball must be wholly inside the same conductor.
    local_radius : maximum physical sphere radius in metres.
    local_fraction : strictly less than one for the legacy interior-ball split.
    radial_segments : callable(x, directions, radius)->(low, high), optional.
        Supplies disjoint seawater segments along each ray so the near ball
        can cross insulating walls without sampling inside the solid.
    angular_frame : callable(x, radius)->(unit_axis, polar_cut) or None.
        Optional geometry-aligned polar coordinate system. Each side of
        polar_cut gets its own Gauss panel, resolving the angular kink where
        near rays first reach a solid face or the outer spherical boundary.
        This changes only the angular cubature, not the Coulomb identity.

    Formula for each outer point x:
       ∫ J_n(y)/|x-y| dy
         = ∫ [1-χ(|x-y|/a)] J_n(y)/|x-y| dy
           + ∫_{|y-x|<a} χ(|x-y|/a) J_n(y)/|x-y| dy.
    There is NO softened Coulomb kernel, substitute cloud or artificial
    self-pair. The local spherical Jacobian cancels the 1/r singularity.
    If return_diagnostics is true, also return the relative Frobenius
    norm of the UNSYMMETRIZED Galerkin matrix's antisymmetric part.
    Nonzero skew from independent x/y cubature is integration error,
    not a physical violation of Lorentz reciprocity.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    wx = np.asarray(wx, float)
    wy = np.asarray(wy, float)
    for nodes, weights in ((x, wx), (y, wy)):
        if (nodes.ndim != 2 or nodes.shape[1] != 3 or
                weights.shape != (len(nodes),) or not len(nodes) or
                not np.isfinite(nodes).all() or
                not np.isfinite(weights).all() or np.any(weights <= 0)):
            raise ValueError("expected finite 3D nodes and positive volume weights")
    if (not 0.0 < local_fraction < 1.0 or
            not np.isfinite(local_radius) or local_radius <= 0.0 or
            min(local_radial_order, local_polar_order,
                local_azimuthal_order, chunk_size) < 2):
        raise ValueError("invalid local spherical quadrature")
    distances = np.asarray(clearance(x), float)
    if (distances.shape != (len(x),) or not np.isfinite(distances).all()
            or np.any(distances <= 0.0)):
        raise ValueError("outer integration nodes must be strictly inside conductor")
    radii = (np.minimum(local_radius, local_fraction*distances)
             if radial_segments is None else np.full(len(x), local_radius))
    vx = np.asarray(evaluate(x), float)
    vy = np.asarray(evaluate(y), float)
    if (vx.ndim != 3 or vx.shape[0] != len(x) or vx.shape[2] != 3
            or vy.shape != (len(y), vx.shape[1], 3)):
        raise ValueError("current evaluator must return consistent (nodes,modes,3)")
    if not np.isfinite(vx).all() or not np.isfinite(vy).all():
        raise ValueError("invalid continuous basis values")

    m = vx.shape[1]
    accum = np.zeros((m, m), float)
    tr, tw = np.polynomial.legendre.leggauss(local_radial_order)
    mu, mw = np.polynomial.legendre.leggauss(local_polar_order)
    phi = 2.0*math.pi*(np.arange(local_azimuthal_order)+0.5)/local_azimuthal_order
    transverse = np.sqrt(np.maximum(0.0, 1.0-mu*mu))
    dirs = np.stack((
        transverse[:, None]*np.cos(phi)[None, :],
        transverse[:, None]*np.sin(phi)[None, :],
        np.broadcast_to(mu[:, None], (len(mu),len(phi))),
    ), axis=-1).reshape(-1, 3)
    domega = np.repeat(mw, len(phi))*(2.0*math.pi/len(phi))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    def angular_nodes_for(point, radius):
        if angular_frame is None:
            return dirs, domega
        frame = angular_frame(point, radius)
        if frame is None:
            return dirs, domega
        axis, cut = frame
        axis = np.asarray(axis, float)
        if (axis.shape != (3,) or not np.isfinite(axis).all()
                or not np.isfinite(cut) or abs(np.linalg.norm(axis)-1)>1e-8):
            raise ValueError("invalid local spherical angular frame")
        cut = float(np.clip(cut, -1.0, 1.0))
        # The normal-oriented ray/surface transition becomes a fixed
        # polar-panel interface. No missed angular silhouette inside
        # the wide single-interval Gauss rule near a wall.
        polar_nodes = []
        polar_weights = []
        for left, right in ((-1.0, cut), (cut, 1.0)):
            if right-left > 1e-13:
                polar_nodes.append(
                    0.5*(right-left)*mu+0.5*(left+right)
                )
                polar_weights.append(0.5*(right-left)*mw)
        panel_mu = np.concatenate(polar_nodes)
        panel_w = np.concatenate(polar_weights)
        ref = np.array([0.,0.,1.]) if abs(axis[2]) < 0.9 else np.array([0.,1.,0.])
        e1 = np.cross(ref, axis)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        st = np.sqrt(np.maximum(0.0, 1-panel_mu*panel_mu))
        new_dirs = (
            panel_mu[:,None,None]*axis
            + st[:,None,None]*(
                cos_phi[None,:,None]*e1 + sin_phi[None,:,None]*e2
            )
        ).reshape(-1, 3)
        new_weights = np.repeat(panel_w,len(phi))*(2*math.pi/len(phi))
        return new_dirs, new_weights
    # The far term vanishes identically around x; even a coinciding
    # quadrature node contributes exactly zero, not an epsilon substitute.
    for start in range(0, len(x), chunk_size):
        stop = min(len(x), start+chunk_size)
        d = cdist(x[start:stop], y)
        cutoff = radii[start:stop, None]
        numerator = 1.0-_near_window(d/cutoff)
        far_kernel = np.divide(
            numerator, d, out=np.zeros_like(d), where=d>0.0,
        )
        far_pot = np.einsum(
            "ij,jmc->imc", far_kernel*wy[None, :], vy,
            optimize=True,
        )
        accum += np.einsum(
            "i,imc,inc->mn",
            wx[start:stop], vx[start:stop], far_pot, optimize=True,
        )
        for j in range(start,stop):
            a = radii[j]
            local_dirs, local_domega = angular_nodes_for(x[j], a)
            if radial_segments is None:
                rho = 0.5*a*(tr+1.0)
                weight_r = 0.5*a*tw*rho*_near_window(rho/a)
                locations = (
                    x[j, None, None, :]
                    + rho[:, None, None]*local_dirs[None, :, :]
                ).reshape(-1,3)
                local_modes = np.asarray(evaluate(locations), float).reshape(
                    len(rho), len(local_dirs), m, 3,
                )
                local_pot = np.einsum(
                    "r,a,ramc->mc", weight_r, local_domega,
                    local_modes, optimize=True,
                )
            else:
                # Integrate wet subintervals of an x-centred *fixed*
                # physical ball. No near-wall collapse of cutoff radius.
                low, high = radial_segments(x[j], local_dirs, a)
                low, high = np.asarray(low, float), np.asarray(high, float)
                if (low.shape != high.shape or low.ndim != 2
                        or low.shape[0] != len(local_dirs)
                        or not np.isfinite(low).all()
                        or not np.isfinite(high).all()
                        or np.any(low < -1e-12)
                        or np.any(high > a+1e-12)):
                    raise ValueError("invalid clipped conducting ray intervals")
                di, si = np.nonzero(high > low+1e-14)
                if not len(di):
                    local_pot = np.zeros((m, 3))
                else:
                    begin = low[di, si]
                    finish = high[di, si]
                    # The C2 cutoff changes polynomial pieces at a/2.
                    # Split the *radial* Gauss rule there exactly instead
                    # of asking one low-order rule to cross the kink.
                    seam = np.clip(0.5*a, begin, finish)
                    start_r = np.concatenate((begin, seam))
                    end_r = np.concatenate((seam, finish))
                    ray = np.concatenate((di, di))
                    nonempty = end_r > start_r+1e-14
                    start_r, end_r, ray = (
                        start_r[nonempty], end_r[nonempty],
                        ray[nonempty],
                    )
                    half = 0.5*(end_r-start_r)
                    rho = start_r[:, None]+half[:, None]*(tr[None, :]+1)
                    locations = (
                        x[j, None, None, :]
                        + rho[:, :, None]*local_dirs[ray, None, :]
                    ).reshape(-1, 3)
                    local_modes = np.asarray(
                        evaluate(locations), float,
                    ).reshape(len(ray), len(tr), m, 3)
                    weights = (
                        half[:, None]*tw[None, :]*rho
                        *_near_window(rho/a)*local_domega[ray, None]
                    )
                    local_pot = np.einsum(
                        "sr,srmc->mc", weights, local_modes,
                        optimize=True,
                    )
            accum += wx[j]*np.einsum(
                "mc,nc->mn", vx[j], local_pot, optimize=True,
            )
    # Continuum Coulomb L is symmetric; independently chosen x/y
    # cubature is not. Monitor the raw skew *before* symmetrization so
    # that a symmetric final matrix cannot hide an under-resolved rule.
    skew = float(
        np.linalg.norm(accum-accum.T)
        /max(np.linalg.norm(accum),1e-60)
    )
    symmetric = MU0/(4.0*math.pi)*0.5*(accum+accum.T)
    return (symmetric,skew) if return_diagnostics else symmetric


def _bfzi_wet_radial_segments(scene, point, directions, radius):
    """Clip an x-centred spherical patch against real rotated boxes.

    For each ray, return all disjoint positive radial seawater intervals.
    Intersections with oriented box slabs and the outer sphere are exact;
    angular/radial Gauss integration remains independently refinable.
    """
    point = np.asarray(point, float)
    directions = np.asarray(directions, float)
    origin = point-np.asarray(scene.outer_center, float)
    dot = directions @ origin
    remaining = scene.outer_radius**2-float(origin @ origin)
    if remaining <= 0:
        raise ValueError("Coulomb outer point must be inside exterior sphere")
    outer_exit = -dot+np.sqrt(dot*dot+remaining)
    rmax = np.minimum(radius, outer_exit)
    low = np.zeros((len(directions), 1), float)
    high = rmax[:, None]

    for assembly in scene.assemblies:
        pose = assembly.pose
        local_point = (point-pose.center) @ pose.rotation
        local_directions = directions @ pose.rotation
        half = np.asarray(assembly.enclosure.half_extents, float)
        parallel = np.abs(local_directions) < 1e-14
        outside_parallel = np.any(
            parallel & (np.abs(local_point)[None, :] > half[None, :]),
            axis=1,
        )
        near_face = np.divide(
            -half-local_point, local_directions,
            out=np.zeros_like(local_directions), where=~parallel,
        )
        far_face = np.divide(
            half-local_point, local_directions,
            out=np.zeros_like(local_directions), where=~parallel,
        )
        entry = np.max(np.where(
            parallel, -np.inf, np.minimum(near_face, far_face),
        ), axis=1)
        exit = np.min(np.where(
            parallel, np.inf, np.maximum(near_face, far_face),
        ), axis=1)
        hit = (
            ~outside_parallel & (exit > np.maximum(entry, 0))
            & (entry < rmax)
        )
        entry = np.where(hit, np.maximum(entry, 0), rmax)
        exit = np.where(hit, np.minimum(exit, rmax), rmax)
        # Subtract each solid interval from all remaining wet segments.
        # The two independent BFZI solids cannot overlap, so each ray
        # produces at most three nonempty seawater intervals.
        before_end = np.where(
            hit[:, None], np.minimum(high, entry[:, None]), high,
        )
        after_start = np.where(
            hit[:, None], np.maximum(low, exit[:, None]), high,
        )
        low, high = (
            np.concatenate((low, after_start), axis=1),
            np.concatenate((before_end, high), axis=1),
        )
    return low, np.maximum(low, high)


def _bfzi_angular_face_frame(scene, point, radius):
    """Choose the nearest physical boundary for one angular polar panel cut.

    This is an integration-coordinate change ONLY. The actual conducting
    intervals are still determined by exact box and sphere ray intersection.
    A face-normal cut resolves the major tangency kink; near edges, corners
    and multiple objects the angular quadrature still needs refinement.
    """
    point = np.asarray(point, float)
    center = np.asarray(scene.outer_center, float)
    rel = point-center
    radial = np.linalg.norm(rel)
    d_outer = scene.outer_radius-radial
    best = None
    if 0 < d_outer < radius and radial > 0:
        # At rho=radius, the sphere boundary appears at this polar cosine.
        cut = (scene.outer_radius**2-radial**2-radius**2)/(
            2*radius*radial
        )
        best = (d_outer, rel/radial, cut)
    for assembly in scene.assemblies:
        pose = assembly.pose
        local = (point-pose.center) @ pose.rotation
        half = np.asarray(assembly.enclosure.half_extents)
        displacement = local-np.clip(local, -half, half)
        distance = np.linalg.norm(displacement)
        if 0 < distance < radius and (
                best is None or distance < best[0]):
            # Face normal when the nearest point is in a face interior;
            # closest-point direction elsewhere handles corners/edges.
            toward = -(displacement @ pose.rotation.T)/distance
            best = (distance, toward, distance/radius)
    return None if best is None else (best[1],best[2])


def assemble_basis_coulomb_split(
    basis, transform, x, wx, y, wy, *,
    clip_local_to_domain: bool = True,
    angular_face_panels: bool = True, **kwargs,
):
    """Unsoftened Coulomb L with domain-clipped near balls by default.

    clip_local_to_domain=False reproduces the previous clearance-shrunk
    split for controlled comparison at the same x/y quadrature nodes.
    angular_face_panels=True resolves the first box/outer-boundary ray
    crossing in a separate polar Gauss panel when near a boundary.
    """
    scene = basis.scene
    origin = np.asarray(scene.outer_center)
    def clearance(points):
        distances = [scene.outer_radius-np.linalg.norm(points-origin,axis=1)]
        for assembly in scene.assemblies:
            dist2, _ = assembly.enclosure.squared_exterior_distance(
                points, assembly.pose,
            )
            distances.append(np.sqrt(dist2))
        return np.minimum.reduce(distances)

    def evaluate(points):
        return np.einsum(
            "nmc,mr->nrc", basis.evaluate(points), transform,
            optimize=True,
        )
    return assemble_coulomb_split(
        x, wx, y, wy, evaluate, clearance,
        radial_segments=(
            (lambda point, directions, radius:
             _bfzi_wet_radial_segments(scene, point, directions, radius))
            if clip_local_to_domain else None
        ),
        angular_frame=(
            (lambda point, radius:
             _bfzi_angular_face_frame(scene, point, radius))
            if clip_local_to_domain and angular_face_panels else None
        ), **kwargs,
    )
