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
    local_fraction : strictly less than one to avoid crossing a boundary.

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
    radii = np.minimum(local_radius, local_fraction*distances)
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
            rho = 0.5*a*(tr+1.0)
            weight_r = 0.5*a*tw*rho*_near_window(rho/a)
            locations = (
                x[j, None, None, :]
                + rho[:, None, None]*dirs[None, :, :]
            ).reshape(-1,3)
            local_modes = np.asarray(evaluate(locations), float).reshape(
                len(rho), len(dirs), m, 3,
            )
            local_pot = np.einsum(
                "r,a,ramc->mc", weight_r, domega,
                local_modes, optimize=True,
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


def assemble_basis_coulomb_split(
    basis, transform, x, wx, y, wy, **kwargs,
):
    """Unsoftened Coulomb L for a finite BFZI scene and fixed mode space."""
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
        x, wx, y, wy, evaluate, clearance, **kwargs,
    )
