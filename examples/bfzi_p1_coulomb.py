"""P1: genuine unsoftened Coulomb L benchmark with an exact 3D answer.

No BFZI geometry, no port line source, no neural model, no Plummer or
uniform-ball surrogate kernel. This is the necessary operator-only gate
before interpreting any two-box seawater magnetic energy.

On the BALL |x|<R, three exact solenoidal, zero-normal-current fields
    W_axis(x) = axis cross x
are curls of vector potentials with zero tangential boundary traces.
This polynomial basis is unnormalised (matrix entry is not port H).
The original free-space 1/|x-y| double integral is *analytically known*:
    L_mn = 16*pi*mu0*R**7 / 315 * delta_mn.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.distance import cdist
from sdfmpneo_bfzi.green_coulomb import (
    assemble_coulomb_split, assemble_coulomb_subtracted, _near_window,
)
from sdfmpneo_bfzi.source import MU0


def ball_rule(radius: float, radial: int, polar: int, azimuth: int):
    """Positive, independent exact-ball volume Gauss cubature."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("ball radius must be positive")
    if min(radial, polar, azimuth) < 2:
        raise ValueError("ball cubature orders must be >=2")
    r, rw = np.polynomial.legendre.leggauss(radial)
    r = radius*0.5*(r+1)
    rw = 0.5*radius*rw*r*r
    mu, mw = np.polynomial.legendre.leggauss(polar)
    az = (np.arange(azimuth)+0.5)*(2*math.pi/azimuth)
    s = np.sqrt(np.maximum(0.0,1.0-mu*mu))
    unit = np.stack((
        s[:,None]*np.cos(az)[None,:],
        s[:,None]*np.sin(az)[None,:],
        np.broadcast_to(mu[:,None],(polar,azimuth)),
    ),axis=-1).reshape(-1,3)
    directions_w = np.repeat(mw,azimuth)*(2*math.pi/azimuth)
    points = (r[:,None,None]*unit[None,:,:]).reshape(-1,3)
    weights = (rw[:,None]*directions_w[None,:]).reshape(-1)
    return points,weights


def vortex_modes(points):
    """Three exact spherical div-free, zero-normal-current test modes."""
    x = np.asarray(points,float).reshape(-1,3)
    axes = np.eye(3)
    # axis cross x: J=curl{0.5*(R^2-r^2)*axis} in the ball.
    return np.cross(axes[None,:,:],x[:,None,:])


def exact_vortex_green(radius:float):
    """Exact original Coulomb L, not a softened or cloud approximation."""
    return (16*math.pi*MU0/315)*radius**7*np.eye(3)


def vortex_green_potential(points, radius:float):
    """Exact Newtonian inner potential, ∫_ball W(y)/|x-y| dV_y.

    For |x|<R, each Cartesian linear density has potential
      U_i(x) = (2*pi*R^2/3 - 2*pi*|x|^2/5) * x_i.
    This formula follows from ΔU=-4*pi*x_i and matching the exterior
    l=1 potential, independently of numerical double integration.
    """
    x=np.asarray(points,float)
    return (
        (2*math.pi*radius**2/3-2*math.pi*np.sum(x*x,axis=-1)/5)
        [...,None,None]*vortex_modes(x)
    )


def diagnose_original_split(
    radius, x_rule, y_rule, local_radius, *,
    original_l11=None,
):
    """Locate x, y-far, and local quadrature error without fitting data.

    The C2 split window used by the production kernel has exact radial
    moment ∫_0^1 t χ(t) dt=2/7. If the local sphere is wholly inside
    the ball and W(y)=W(x)+constant-gradient*(y-x), its local Coulomb
    integral is EXACTLY (8π/7)*a(x)^2 W(x). Odd angular terms cancel.
    All quantities below have the same unsoftened physical 1/r kernel.
    """
    x, wx = ball_rule(radius, *x_rule)
    y, wy = ball_rule(radius, *y_rule)
    vx, vy = vortex_modes(x), vortex_modes(y)
    a = np.minimum(
        local_radius*radius,
        0.6*(radius-np.linalg.norm(x, axis=1)),
    )
    near_exact = (8.0*math.pi/7.0)*a[:,None,None]**2*vx
    far_l = np.zeros((3,3))
    for start in range(0,len(x),64):
        stop=min(len(x),start+64)
        d=cdist(x[start:stop],y)
        window=_near_window(d/a[start:stop,None])
        far_k=np.divide(
            1.0-window,d,out=np.zeros_like(d),where=d>0.0,
        )
        # Independent numerical y-far operator, exactly the same
        # y volume quadrature as the original split implementation.
        far_pot=np.einsum(
            "ij,jmc->imc",far_k*wy[None,:],vy,optimize=True,
        )
        far_l+=np.einsum(
            "i,imc,inc->mn",
            wx[start:stop],vx[start:stop],far_pot,optimize=True,
        )
    scale=MU0/(4.0*math.pi)
    far_l*=scale
    # Call the reusable physical original-Coulomb operator, not an
    # independently typed copy of the constant-density subtraction.
    sub_l=assemble_coulomb_subtracted(
        x,wx,y,wy,vortex_modes,
        lambda p:2*math.pi*(
            radius**2-np.einsum("nc,nc->n",p,p)/3.0
        ),
        chunk_size=64,
    )
    near_l=scale*np.einsum(
        "i,imc,inc->mn",wx,vx,near_exact,optimize=True,
    )
    exact_x=scale*np.einsum(
        "i,imc,inc->mn",
        wx,vx,vortex_green_potential(x,radius),optimize=True,
    )
    exact=exact_vortex_green(radius)
    far_plus_near=0.5*(far_l+far_l.T)+near_l
    norm=max(np.linalg.norm(exact),1e-30)
    return dict(
        x_exact_rel=float(np.linalg.norm(exact_x-exact)/norm),
        far_y_rel=float(np.linalg.norm(far_plus_near-exact_x)/norm),
        local_l11_rel=(
            float(abs(original_l11-far_plus_near[0,0])/exact[0,0])
            if original_l11 is not None else float("nan")
        ),
        global_constant_subtraction_rel=float(
            np.linalg.norm(sub_l-exact)/norm
        ),
        global_constant_subtraction_l11=float(sub_l[0,0]),
    )


def trial(radius, x_rule, y_rule, local_order, local_radius):
    x,wx=ball_rule(radius,*x_rule)
    y,wy=ball_rule(radius,*y_rule)
    exact=exact_vortex_green(radius)
    raw,skew=assemble_coulomb_split(
        x,wx,y,wy,vortex_modes,
        lambda p:radius-np.linalg.norm(p,axis=1),
        local_radius=local_radius*radius,
        local_fraction=0.6,
        local_radial_order=local_order[0],
        local_polar_order=local_order[1],
        local_azimuthal_order=local_order[2],
        chunk_size=32,return_diagnostics=True,
    )
    difference=raw-exact
    diag_rel=float(np.max(
        np.abs(np.diag(difference))/np.diag(exact),
    ))
    offdiag_rel=float(np.linalg.norm(
        difference-np.diag(np.diag(difference)),
    )/np.linalg.norm(exact))
    err=float(np.linalg.norm(difference)/np.linalg.norm(exact))
    return dict(
        nx=len(x),ny=len(y),norm_error=err,
        diag_error=diag_rel,offdiag_error=offdiag_rel,
        raw_skew=skew,L11=raw[0,0],reference=exact[0,0],
    )


def main():
    radius=1.0
    reference=exact_vortex_green(radius)[0,0]
    print(
        f"p1_exact: domain=conducting_ball, radius_m={radius:.5g},"
        f"modes=3,divJ=0,wall_normal_J=0,"
        f"original_kernel=1/r,analytical_L_diag_basis_SI={reference:.12g},"
        "analytical_offdiag=0,no_softening=True",
        flush=True,
    )
    x0=(3,5,8)
    y0=(4,6,10)
    x1=(5,8,12)
    y1=(6,9,14)
    local=(6,8,12)
    evidence={}
    decomp_both=None
    for name,x,y in (
        ("base",x0,y0),
        ("x_only",x1,y0),
        ("y_only",x0,y1),
        ("both",x1,y1),
    ):
        z=trial(radius,x,y,local,0.4)
        evidence[name]=z
        if name in ("base","both"):
            d=diagnose_original_split(
                radius,x,y,0.4,original_l11=z["L11"],
            )
            if name=="both":
                decomp_both=d
            print(
                f"p1_decomp: variant={name},"
                f"x_exact_rel={d['x_exact_rel']:.7g},"
                f"far_y_rel={d['far_y_rel']:.7g},"
                f"near_local_l11_rel={d['local_l11_rel']:.7g},"
                f"global_const_sub_rel={d['global_constant_subtraction_rel']:.7g},"
                f"global_const_sub_L11={d['global_constant_subtraction_l11']:.12g}",
                flush=True,
            )
        print(
            f"p1_ball: variant={name},x_nodes={z['nx']},"
            f"y_nodes={z['ny']},rel_L={z['norm_error']:.7g},"
            f"diag_rel={z['diag_error']:.7g},"
            f"offdiag_rel={z['offdiag_error']:.7g},"
            f"raw_skew={z['raw_skew']:.7g},"
            f"L11_basis_SI={z['L11']:.12g},"
            f"true_L11_basis_SI={z['reference']:.12g}",
            flush=True,
        )
    # Hold the physical field and both cubatures fixed to isolate the
    # partition-radius sensitivity of the original 1/r numerical integral.
    partition={}
    for a in (0.25,0.55):
        z=trial(radius,x0,y0,local,a)
        partition[a]=z
        print(
            f"p1_partition: near_radius_over_R={a:.3g},"
            f"rel_L={z['norm_error']:.7g},"
            f"diag_rel={z['diag_error']:.7g}",
            flush=True,
        )
    print(
        f"p1_sub_gate: target_rel=0.001,"
        f"global_constant_subtraction_frob_pass="
        f"{decomp_both['global_constant_subtraction_rel']<0.001},"
        f"global_constant_subtraction_rel="
        f"{decomp_both['global_constant_subtraction_rel']:.7g},"
        "domain_scalar_potential=analytic_ball_only",
        flush=True,
    )
    # One predeclared target for this analytically solvable three-mode
    # ball ONLY. It is not a target for the actual BFZI two-box G1 model.
    ref_norm=abs(reference)
    partition_gap=abs(partition[0.25]["L11"]-partition[0.55]["L11"])/max(ref_norm,1e-30)
    both=evidence["both"]
    print(
        f"p1_gate: target_rel=0.001,"
        f"ball_frob_pass={both['norm_error']<0.001},"
        f"ball_diag_pass={both['diag_error']<0.001},"
        f"ball_offdiag_pass={both['offdiag_error']<0.001},"
        f"partition_L11_gap={partition_gap:.7g},"
        f"partition_L11_pass={partition_gap<0.001},"
        "scope=exact_ball_only",
        flush=True,
    )
    print(
        "p1_note: p1_decomp splits x-integration, exactly integrated "
        "local spherical C2 kernel and y-far cubature; the independent "
        "whole-ball constant subtraction has an EXACT ball scalar kernel, "
        "which cannot be reused for two-box geometry without obtaining "
        "its actual scalar geometry potential. Original 1/r is unchanged. "
        "Exact-vortex BALL benchmark tests the ORIGINAL Coulomb "
        "double integral against an analytic three-mode matrix. It does "
        "NOT certify rotated two-box wet-domain L, its near-wall integrals, "
        "or the 229-mode and infinite-ocean physical reference.",
        flush=True,
    )


if __name__=="__main__":
    main()
