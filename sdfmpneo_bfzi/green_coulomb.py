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


def assemble_coulomb_subtracted(
    x, wx, y, wy, evaluate, scalar_potential, *,
    gradient=None, first_moment=None,
    chunk_size: int = 32, return_diagnostics: bool = False,
):
    """Unsoftened Coulomb L with exact continuous constant-density subtraction.

    Let S_Omega(x) = ∫_Omega 1/|x-y| dy, supplied by an INDEPENDENT,
    accurate scalar geometry-potential evaluator. Then, exactly:

      ∫_Omega W(y)/|x-y| dy
        = W(x)*S_Omega(x)
          + ∫_Omega [W(y)-W(x)]/|x-y| dy.

    Optionally, with an independently verified first geometry moment
      M_Omega(x) = ∫_Omega (y-x)/|x-y| dy
                 = ∫_{boundary Omega} n_y |x-y| dS_y,
    and a current gradient D W(x), subtract the affine Taylor term:
      W(x)*S_Omega(x) + D W(x)*M_Omega(x)
        + ∫ [W(y)-W(x)-D W(x)*(y-x)]/|x-y| dy.
    This identity is exact and the residual is O(|y-x|) for C2 W.
    Exact affine modes have IDENTICALLY ZERO residual: they test the
    scalar/first-moment mathematics, not the near-kernel cubature.
    To test actual residual integration, use a nonaffine current mode.

    For Lipschitz W the constant-subtracted residual is bounded near
    the diagonal, unlike the original 1/r kernel. This does not silently
    replace the original physical kernel with a softened/cloud kernel.
    x and y quadrature nodes must be DISJOINT, since the residual
    has an angle-dependent pointwise limit on the diagonal; a single
    zero-distance node cannot be assigned a valid quadrature value.

    This routine is useful only after the geometry-dependent scalar
    potential S_Omega has its OWN independent validation. The P1 unit
    ball supplies S_B(x) = 2π(R²-|x|²/3) analytically. In general two
    rotated box exclusions, this spherical S_B is NOT the true S_Omega.
    """
    x, y = np.asarray(x,float), np.asarray(y,float)
    wx, wy = np.asarray(wx,float), np.asarray(wy,float)
    if (x.ndim!=2 or y.ndim!=2 or x.shape[1]!=3 or y.shape[1]!=3
            or wx.shape!=(len(x),) or wy.shape!=(len(y),)
            or not len(x) or not len(y) or
            not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(wx).all() or not np.isfinite(wy).all()
            or np.any(wx<=0) or np.any(wy<=0) or chunk_size<1):
        raise ValueError("expected nonempty finite (n,3) volume rules")
    vx, vy = np.asarray(evaluate(x),float), np.asarray(evaluate(y),float)
    if (vx.ndim!=3 or vx.shape[0]!=len(x) or vx.shape[2]!=3
            or vy.shape!=(len(y),vx.shape[1],3)
            or not np.isfinite(vx).all() or not np.isfinite(vy).all()):
        raise ValueError("current modes must be finite (n,m,3)")
    scalar = np.asarray(scalar_potential(x),float)
    if scalar.shape!=(len(x),) or not np.isfinite(scalar).all():
        raise ValueError("scalar Coulomb potential must be finite (n,)")
    m=vx.shape[1]
    if (gradient is None) != (first_moment is None):
        raise ValueError("gradient and first_moment must be provided together")
    grads=moments=None
    if gradient is not None:
        grads=np.asarray(gradient(x),float)
        moments=np.asarray(first_moment(x),float)
        if (grads.shape!=(len(x),m,3,3) or
                moments.shape!=(len(x),3) or
                not np.isfinite(grads).all() or
                not np.isfinite(moments).all()):
            raise ValueError("expected gradient (n,m,3,3) and first_moment (n,3)")
    accum=np.zeros((m,m))
    for i in range(0,len(x),chunk_size):
        end=min(len(x),i+chunk_size)
        distance=cdist(x[i:end],y)
        if np.any(distance<=1e-14):
            raise ValueError(
                "singularity-subtracted Coulomb requires disjoint x/y nodes"
            )
        kernel=wy[None,:]/distance
        numerator=vy[None,:,:,:]-vx[i:end,None,:,:]
        if grads is not None:
            # Tensor convention: grads[n,mode,component,coordinate].
            taylor=np.einsum(
                "imck,ijk->ijmc",grads[i:end],
                y[None,:,:]-x[i:end,None,:],optimize=True,
            )
            numerator=numerator-taylor
        residual=np.einsum(
            "ij,ijmc->imc",kernel,numerator,optimize=True,
        )
        potential=scalar[i:end,None,None]*vx[i:end]+residual
        if grads is not None:
            potential+=np.einsum(
                "imck,ik->imc",grads[i:end],moments[i:end],
                optimize=True,
            )
        accum+=np.einsum(
            "i,imc,inc->mn",wx[i:end],vx[i:end],potential,
            optimize=True,
        )
    skew=float(np.linalg.norm(accum-accum.T)/max(np.linalg.norm(accum),1e-60))
    result=MU0/(4.0*math.pi)*0.5*(accum+accum.T)
    return (result,skew) if return_diagnostics else result



def assemble_coulomb_difference(
    x, wx, y, wy, evaluate, scalar_potential, *,
    chunk_size: int = 24, return_diagnostics: bool = False,
):
    """Symmetric-difference Galerkin form of the ORIGINAL unsoftened 1/r.

    For real vector fields W_i on Omega, exchange x and y in the double
    integral to obtain the exact continuum identity

      L_ij * (4*pi/mu0) =
          integral S_Omega(x) W_i(x).W_j(x) dx
        - (1/2) double_integral
          [W_i(x)-W_i(y)].[W_j(x)-W_j(y)]/|x-y| dx dy.

    For locally Lipschitz currents the difference integrand is O(|x-y|)
    on the diagonal; its value there is exactly zero. Thus even coincident
    x/y nodes are legal, unlike the one-sided Taylor residual. Both
    orders must still be refined independently. Analytic S_Omega MUST
    correspond to the actual conducting integration domain, not a ball
    substituted for a box-excluded sea.

    The formula is symmetric algebraically, so a tiny skew is NOT an
    accuracy certificate. A near-unity subtraction ratio indicates that
    the difference of the two large terms may be ill conditioned.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    wx, wy = np.asarray(wx, float), np.asarray(wy, float)
    if (
        x.ndim != 2 or y.ndim != 2 or x.shape[1] != 3 or y.shape[1] != 3
        or wx.shape != (len(x),) or wy.shape != (len(y),)
        or not len(x) or not len(y) or chunk_size < 1
        or not np.isfinite(x).all() or not np.isfinite(y).all()
        or not np.isfinite(wx).all() or not np.isfinite(wy).all()
        or np.any(wx <= 0) or np.any(wy <= 0)
    ):
        raise ValueError("expected positive finite seawater volume rules")
    vx = np.asarray(evaluate(x), float)
    vy = np.asarray(evaluate(y), float)
    if (
        vx.ndim != 3 or vx.shape[0] != len(x) or vx.shape[2] != 3
        or vy.shape != (len(y), vx.shape[1], 3)
        or not np.isfinite(vx).all() or not np.isfinite(vy).all()
    ):
        raise ValueError("current modes must be finite (n,m,3)")
    sx = np.asarray(scalar_potential(x), float)
    sy = np.asarray(scalar_potential(y), float)
    if (
        sx.shape != (len(x),) or sy.shape != (len(y),)
        or not np.isfinite(sx).all() or not np.isfinite(sy).all()
        or np.any(sx <= 0) or np.any(sy <= 0)
    ):
        raise ValueError("exact domain scalar moments must be positive")
    local = 0.5 * (
        np.einsum("n,n,nic,njc->ij", wx, sx, vx, vx, optimize=True)
        + np.einsum("n,n,nic,njc->ij", wy, sy, vy, vy, optimize=True)
    )
    difference = np.zeros_like(local)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        radius = cdist(x[start:stop], y)
        weight = np.divide(
            wx[start:stop, None] * wy[None, :], radius,
            out=np.zeros_like(radius), where=radius > 0,
        )
        delta = vx[start:stop, None, :, :] - vy[None, :, :, :]
        difference += np.einsum(
            "ab,abic,abjc->ij", weight, delta, delta, optimize=True,
        )
    scale = MU0 / (4.0 * math.pi)
    result = scale * (local - 0.5 * difference)
    subtraction_ratio = float(
        np.linalg.norm(0.5 * difference) / max(np.linalg.norm(local), 1e-60)
    )
    return (result, subtraction_ratio) if return_diagnostics else result


def assemble_basis_coulomb_difference(
    basis, transform, x, wx, y, wy, *,
    chunk_size: int = 24, return_diagnostics: bool = False,
):
    """Actual two-pose BFZI seawater L without Taylor Hessians or kernel softening.

    Supports every current mode evaluable by SeaCurrentBasis, including
    source-guided modes; it does not assert their completeness or G1 accuracy.
    """
    from .coulomb_geometry_moments import sea_coulomb_moments

    t = np.asarray(transform, float)
    if t.ndim != 2 or t.shape[0] != basis.size or not np.isfinite(t).all():
        raise ValueError("transform must be finite (basis.size, reduced_size)")

    def current(points):
        return np.einsum(
            "nmc,mk->nkc", basis.evaluate(points), t, optimize=True,
        )

    return assemble_coulomb_difference(
        x, wx, y, wy, current,
        lambda p: sea_coulomb_moments(basis.scene, p)[0],
        chunk_size=chunk_size,
        return_diagnostics=return_diagnostics,
    )


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


def assemble_basis_coulomb_taylor(
    basis, transform, x, wx, y, wy, *,
    chunk_size: int = 24, return_diagnostics: bool = False,
):
    """Actual BFZI seawater L with exact-domain S/M and analytic curl DW.

    Uses the ORIGINAL 1/r Coulomb bilinear form over the actual seawater
    volume (outer sphere minus TWO oriented insulating enclosures).
    Exact scalar and first moments of that FULL domain are computed from
    real rectangle face primitives, NOT the support-only ball moment in
    P1-B. The residual is integrated with independent x/y positive
    seawater volume quadratures.

    Only non-source-guided modes are supported until the spatial Hessians
    of C_port and curl C_port have an exact, validated expression.
    Convergence of this quadrature still requires physical x/y studies.
    """
    from .coulomb_geometry_moments import sea_coulomb_moments
    from .sea_basis_differential import evaluate_nonguided_curl_jacobian
    t=np.asarray(transform,float)
    if t.ndim!=2 or t.shape[0]!=basis.size or not np.isfinite(t).all():
        raise ValueError("transform must be finite (basis.size, reduced_size)")
    if any(mode.source_port>=0 for mode in basis.modes):
        raise NotImplementedError(
            "source-guided BFZI curl modes require port-field Jacobians"
        )

    def current(points):
        return np.einsum(
            "nmc,mk->nkc",basis.evaluate(points),t,optimize=True,
        )

    def current_gradient(points):
        return np.einsum(
            "nmcd,mk->nkcd",
            evaluate_nonguided_curl_jacobian(basis,points),
            t,optimize=True,
        )

    return assemble_coulomb_subtracted(
        x,wx,y,wy,current,
        lambda p:sea_coulomb_moments(basis.scene,p)[0],
        gradient=current_gradient,
        first_moment=lambda p:sea_coulomb_moments(basis.scene,p)[1],
        chunk_size=chunk_size,
        return_diagnostics=return_diagnostics,
    )




def assemble_coulomb_radial(
    x, wx, evaluate, wet_radial_segments, *,
    radial_order: int = 4,
    polar_order: int = 4,
    azimuthal_order: int = 10,
    radial_breakpoints=(),
    polar_axis=None,
    return_diagnostics: bool = False,
):
    """Direct unsoftened Green potential in x-centred polar coordinates.

    Set y = x + r*u, with |u|=1. The exact physical integral is

        U_j(x) = int_{S2} int_{wet ray intervals} r*W_j(x+r*u) dr du.

    This uses r**2 dr du / |x-y| = r dr du, so no singular node,
    artificial self term, Taylor cancellation or softened kernel exists.
    The caller supplies the EXACT wet ray intervals along each direction;
    quadrature of the radial/angle integrand still needs independent
    convergence checks, especially when box edges change the ray topology.
    radial_breakpoints are fixed physical distances that split each
    *wet* interval; they do not change the domain or the original kernel.

    The unsymmetrized matrix skew is returned as a diagnostic, NOT a
    convergence certificate. This assembly uses one outer x cubature
    and a point-dependent inner spherical cubature (no reusable y grid).
    """
    x = np.asarray(x, float)
    wx = np.asarray(wx, float)
    if (x.ndim != 2 or x.shape[1] != 3 or not len(x)
            or wx.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(wx).all()
            or np.any(wx <= 0)
            or any(not isinstance(v, int) or v < 2
                   for v in (radial_order, polar_order, azimuthal_order))):
        raise ValueError("expected finite positive x rule and orders >=2")
    cuts = np.asarray(radial_breakpoints, dtype=float)
    if cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(cuts <= 0):
        raise ValueError("radial_breakpoints must be finite positive distances")
    cuts = np.unique(cuts)
    vx = np.asarray(evaluate(x), float)
    if (vx.ndim != 3 or vx.shape[0] != len(x)
            or vx.shape[2] != 3 or not np.isfinite(vx).all()):
        raise ValueError("current evaluator must return finite (x,m,3)")
    nr, wr = np.polynomial.legendre.leggauss(radial_order)
    angular, wangular = np.polynomial.legendre.leggauss(polar_order)
    # Two exact angular hemispheres make the nearest physical wall's
    # tangent great circle a panel boundary, rather than blending across it.
    mu = np.concatenate(((angular-1)/2, (angular+1)/2))
    wmu = np.tile(wangular/2, 2)
    phi = 2*math.pi*(np.arange(azimuthal_order)+0.5)/azimuthal_order
    transversal = np.sqrt(np.maximum(0, 1-mu*mu))
    cos_az = np.cos(phi)
    sin_az = np.sin(phi)
    normal_component = np.repeat(mu, azimuthal_order)
    planar_x = np.repeat(transversal, azimuthal_order)*np.tile(
        cos_az, len(mu)
    )
    planar_y = np.repeat(transversal, azimuthal_order)*np.tile(
        sin_az, len(mu)
    )
    area_weights = (
        np.repeat(wmu, azimuthal_order)
        * np.tile(np.full(azimuthal_order, 2*math.pi/azimuthal_order),
                  len(mu))
    )
    accum = np.zeros((vx.shape[1],vx.shape[1]))
    for xi, (point, weight) in enumerate(zip(x, wx)):
        if polar_axis is None:
            axis = np.array((0.0, 0.0, 1.0))
        else:
            axis = np.asarray(polar_axis(point), float)
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError("polar_axis must return finite 3-vector")
            length = np.linalg.norm(axis)
            if length < 1e-12:
                raise ValueError("polar_axis must be nonzero")
            axis = axis/length
        seed = (
            np.array((1.0,0.0,0.0))
            if abs(axis[0]) < 0.9 else np.array((0.0,1.0,0.0))
        )
        e1 = np.cross(seed,axis)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis,e1)
        directions = (
            planar_x[:,None]*e1[None,:]
            +planar_y[:,None]*e2[None,:]
            +normal_component[:,None]*axis[None,:]
        )
        lo, hi = wet_radial_segments(point,directions)
        lo,hi = np.asarray(lo,float),np.asarray(hi,float)
        if (lo.ndim != 2 or lo.shape != hi.shape
                or lo.shape[0] != len(directions)
                or not np.isfinite(lo).all() or not np.isfinite(hi).all()
                or np.any(lo < 0) or np.any(hi < lo)):
            raise ValueError("invalid wet ray intervals")
        if len(cuts):
            clipped = np.clip(cuts[None,None,:],lo[:,:,None],hi[:,:,None])
            edges = np.concatenate(
                (lo[:,:,None],clipped,hi[:,:,None]),axis=-1
            )
        else:
            edges = np.stack((lo,hi),axis=-1)
        a, b = edges[:,:,:-1],edges[:,:,1:]
        valid = b > a
        if not np.any(valid):
            continue
        ray_ids = np.broadcast_to(
            np.arange(len(directions))[:,None,None],a.shape
        )[valid]
        lower, upper = a[valid],b[valid]
        r = (
            0.5*(lower+upper)[:,None]
            +0.5*(upper-lower)[:,None]*nr[None,:]
        )
        y = (point[None,None,:]
             +r[:,:,None]*directions[ray_ids,None,:])
        sampled = np.asarray(evaluate(y.reshape(-1,3)),float)
        if (sampled.shape != (y.size//3,vx.shape[1],3)
                or not np.isfinite(sampled).all()):
            raise ValueError("nonfinite/incorrect inner current values")
        radial_weights = (
            (0.5*(upper-lower)*area_weights[ray_ids])[:,None]
            *wr[None,:]*r
        )
        potential = np.einsum(
            "ab,abmc->mc",radial_weights,
            sampled.reshape(len(lower),radial_order,vx.shape[1],3),
            optimize=True,
        )
        accum += weight*np.einsum(
            "mc,nc->mn",vx[xi],potential,optimize=True,
        )
    skew = float(np.linalg.norm(accum-accum.T)/
                 max(np.linalg.norm(accum),1e-60))
    L = MU0/(4*math.pi)*0.5*(accum+accum.T)
    return (L,skew) if return_diagnostics else L


def assemble_basis_coulomb_radial(
    basis, transform, x, wx, *,
    radial_order: int = 4,
    polar_order: int = 4,
    azimuthal_order: int = 10,
    return_diagnostics: bool = False,
):
    """Actual box-clipped BFZI Green L using radial Jacobian cancellation.

    Geometry is the existing two independent OBBs inside the enclosing
    sphere. Supports every SeaCurrentBasis curl mode, including guided
    source modes, without differentiating C_port or imposing a false
    magnetic ball on an actual two-box seawater domain.
    """
    t = np.asarray(transform, float)
    if (t.ndim != 2 or t.shape[0] != basis.size
            or not np.isfinite(t).all()):
        raise ValueError("transform must be finite (basis.size,reduced_size)")
    scene = basis.scene
    radius = 2.0*scene.outer_radius
    # Fixed physical distances, not successive arbitrary Gaussian orders.
    # Resolve the real wall-factor length as well as mode window scales.
    widths = [mode.width for mode in basis.modes]
    breaks = tuple(sorted(set((
        basis.wall_scale,
        4*basis.wall_scale,
        16*basis.wall_scale,
        64*basis.wall_scale,
        0.5*min(widths),
        min(widths),
        max(widths),
    ))))

    def current(pts):
        return np.einsum(
            "nmc,mk->nkc",basis.evaluate(pts),t,optimize=True
        )

    def intervals(point,directions):
        return _bfzi_wet_radial_segments(
            scene,point,directions,radius
        )

    def axis(point):
        frame = _bfzi_angular_face_frame(scene,point,radius)
        return np.array((0.0,0.0,1.0)) if frame is None else frame[0]

    return assemble_coulomb_radial(
        x,wx,current,intervals,
        radial_order=radial_order,
        polar_order=polar_order,
        azimuthal_order=azimuthal_order,
        radial_breakpoints=breaks,
        polar_axis=axis,
        return_diagnostics=return_diagnostics,
    )


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
