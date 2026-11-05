"""Exact spatial Jacobian of non-source-guided BFZI seawater curl modes.

For F_m(x)=g_m(x) * a_m and W_m=curl(F_m)=grad(g_m) cross a_m,
the current Jacobian is DW_m[:,k]=Hess(g_m)[:,k] cross a_m.

The product g_m includes the actual outer sphere envelope, BOTH
independently oriented insulating-box distance factors and the actual
Gaussian/compact mode window. Hessians are analytic in every smooth
piece of the physical seawater domain; box edge bisectors are only
piecewise C2 and require separate convergence checks.

Source-guided F_m=g_m C_port has derivatives of C_port and curl C_port
which are NOT supplied here. It is rejected instead of making an
uncontrolled finite-difference Taylor correction near ideal port wires.
"""
from __future__ import annotations

import numpy as np

from .sea_basis import SeaCurrentBasis


def _multiply(value, grad, hess, f, df, hf):
    """Value, first and Hessian product rule for a scalar factor."""
    mixed=np.einsum("ni,nj->nij",grad,df)
    mixed+=mixed.transpose(0,2,1)
    return (
        value*f,
        grad*f[:,None]+value[:,None]*df,
        hess*f[:,None,None]+mixed+value[:,None,None]*hf,
    )


def _outer_factor(basis,points):
    d=points-np.asarray(basis.scene.outer_center)
    r2=np.einsum("ni,ni->n",d,d)
    r=np.sqrt(r2)
    radius=basis.scene.outer_radius
    eye=np.eye(3)
    if basis.outer_envelope=="quadratic":
        return (
            1-r2/radius**2,
            -2*d/radius**2,
            np.broadcast_to(-2*eye/radius**2,(len(points),3,3)),
        )
    width=basis.outer_taper_width
    t=np.clip((radius-r)/width,0.0,1.0)
    f=t**3*(10+t*(-15+6*t))
    p=30*t*t*(1-t)**2
    pp=60*t*(1-t)*(1-2*t)
    dr=-p/width
    drr=pp/(width*width)
    r_safe=np.maximum(r,1e-30)
    u=d/r_safe[:,None]
    hess=(
        (drr-dr/r_safe)[:,None,None]*
        np.einsum("ni,nj->nij",u,u)
        +(dr/r_safe)[:,None,None]*eye[None,:,:]
    )
    return f,dr[:,None]*u,hess


def _box_factor(assembly,points,scale):
    local=(points-assembly.pose.center)@assembly.pose.rotation
    half=np.asarray(assembly.enclosure.half_extents)
    excess=np.maximum(np.abs(local)-half,0.0)*np.sign(local)
    active=(np.abs(local)>half).astype(float)
    d=np.linalg.norm(excess,axis=1)
    if np.any(d<=0):
        raise ValueError("curl Hessian requires wet points exterior to boxes")
    unit=excess/d[:,None]
    eye=np.eye(3)
    hdist=(
        active[:,:,None]*eye[None,:,:]/d[:,None,None]
        -np.einsum("ni,nj->nij",unit,unit)/d[:,None,None]
    )
    rot=assembly.pose.rotation
    gd=unit@rot.T
    hd=np.einsum("ai,nij,bj->nab",rot,hdist,rot,optimize=True)
    f=d/(d+scale)
    fp=scale/(d+scale)**2
    fpp=-2*scale/(d+scale)**3
    return (
        f,
        fp[:,None]*gd,
        fpp[:,None,None]*np.einsum("ni,nj->nij",gd,gd)
        +fp[:,None,None]*hd,
    )


def _mode_factor(points,mode):
    offset=points-mode.center
    metric=(
        np.eye(3)/mode.width**2
        if mode.gaussian_metric is None
        else np.asarray(mode.gaussian_metric,float)
    )
    q=offset@metric
    r2=np.einsum("ni,ni->n",offset,q)
    dyadic=np.einsum("ni,nj->nij",q,q)
    if mode.profile=="gaussian":
        v=np.exp(-r2)
        return (
            v,
            -2*v[:,None]*q,
            v[:,None,None]*(4*dyadic-2*metric[None,:,:]),
        )
    if mode.profile=="compact":
        r=np.sqrt(np.maximum(r2,0.0))
        w=np.maximum(1-r,0.0)
        value=w**4*(4*r+1)
        return (
            value,
            -20*w[:,None]**3*q,
            -20*w[:,None,None]**3*metric[None,:,:]
            +(60*w**2/np.maximum(r,1e-30))[:,None,None]*dyadic,
        )
    raise ValueError("unknown current mode profile")


def evaluate_nonguided_curl_jacobian(basis:SeaCurrentBasis,positions):
    """Return physical DW[n,mode,component,coordinate] (exact a.e.)."""
    x=np.asarray(positions,float)
    if (x.ndim!=2 or x.shape[1]!=3
            or not np.isfinite(x).all()
            or not np.all(basis.scene.in_seawater(x))):
        raise ValueError("curl Jacobian positions must be wet, finite (n,3)")
    if any(mode.source_port>=0 for mode in basis.modes):
        raise NotImplementedError(
            "Taylor gradient for source-guided C_port curls "
            "requires analytic spatial derivatives of C_port and curl C_port"
        )
    n=len(x)
    factor=np.ones(n)
    grad=np.zeros((n,3))
    hess=np.zeros((n,3,3))
    factor,grad,hess=_multiply(
        factor,grad,hess,*_outer_factor(basis,x),
    )
    for assembly in basis.scene.assemblies:
        factor,grad,hess=_multiply(
            factor,grad,hess,
            *_box_factor(assembly,x,basis.wall_scale),
        )
    result=np.empty((n,basis.size,3,3))
    for i,mode in enumerate(basis.modes):
        _,_,hm=_multiply(
            factor,grad,hess,*_mode_factor(x,mode),
        )
        # hm[n,gradient_component,derivative_coordinate].
        # In each derivative coordinate: DW=cross(Hessian column,a).
        result[:,i]=np.cross(
            hm.transpose(0,2,1),mode.direction,
        ).transpose(0,2,1)
    return result
