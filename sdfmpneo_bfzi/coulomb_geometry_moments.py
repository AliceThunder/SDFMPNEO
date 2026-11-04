"""Exact original-Coulomb scalar and first moments for BFZI finite seawater.

Physical domain: an outer BALL minus two disjoint oriented rectangular
insulating solids. There is NO approximate sphere-cloud current and no
regularization of 1/r.

S(x) = integral_Omega 1/|x-y| dy
M(x) = integral_Omega (y-x)/|x-y| dy

By Gauss' theorem these are respectively one half the outward flux of
(y-x)/|y-x|, and the outward integral of |y-x|. For an oriented box,
the required rectangle face integrals have elementary corner primitives.
For the outer BALL, the moments are analytic for all wet-domain x.

An exact domain moment makes the constant/affine subtraction algebraically
correct, but does NOT certify its *residual* Coulomb integral or the
finite exterior-sphere approximation to an infinite conducting ocean.
"""
from __future__ import annotations

import math
import numpy as np


def _face_primitives(u, v, w):
    """Corner primitives for rectangle integrals of 1/r and r.

    r=sqrt(u²+v²+w²) and w=constant signed face separation.
    The four-corner alternating sum of each primitive gives the 2D
    integral. The absolute value of w gives the correct continuous
    primitive on either side of the face.
    """
    u, v, w = np.broadcast_arrays(
        np.asarray(u,float),np.asarray(v,float),np.asarray(w,float),
    )
    z=np.abs(w)
    r=np.sqrt(u*u+v*v+w*w)
    with np.errstate(divide="ignore",invalid="ignore"):
        as_v=np.arcsinh(v/np.hypot(u,w))
        as_u=np.arcsinh(u/np.hypot(v,w))
        # At u=w=0, u*asinh(v/hypot(u,w)) -> 0 (similarly v).
        u_as_v=np.where(u==0,0.0,u*as_v)
        v_as_u=np.where(v==0,0.0,v*as_u)
        angle=np.arctan2(u*v,z*r)
    inverse=u_as_v+v_as_u-z*angle
    distance=(
        (u*v*r)/3.0
        +(u*u+3*w*w)*u_as_v/6.0
        +(v*v+3*w*w)*v_as_u/6.0
        -(z*z*z)*angle/3.0
    )
    return inverse,distance


def _rectangle_face_integrals(u0, u1, v0, v1, w):
    """Exactly integrate 1/r and r on a face in local (u,v) coords."""
    p11,q11=_face_primitives(u1,v1,w)
    p01,q01=_face_primitives(u0,v1,w)
    p10,q10=_face_primitives(u1,v0,w)
    p00,q00=_face_primitives(u0,v0,w)
    return (p11-p01-p10+p00, q11-q01-q10+q00)


def box_coulomb_moments(points, center, rotation, half_extents):
    """Original 1/r scalar S and vector M of an oriented solid BOX.

    A box is subtracted from the outer BALL when forming the sea moments.
    The box itself can be insulating: these are *geometry* integrals,
    independent of its actual electrical conductivity.
    """
    x=np.asarray(points,float)
    original=x.shape[:-1]
    if x.shape[-1:]!=(3,) or not np.isfinite(x).all():
        raise ValueError("expected finite points with trailing dimension 3")
    center=np.asarray(center,float)
    rotation=np.asarray(rotation,float)
    half=np.asarray(half_extents,float)
    if (center.shape!=(3,) or rotation.shape!=(3,3)
            or half.shape!=(3,) or np.any(half<=0)
            or not np.isfinite(center).all() or
            not np.isfinite(rotation).all() or not np.isfinite(half).all()
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-9)
            or not math.isclose(float(np.linalg.det(rotation)),1.0,abs_tol=1e-9)):
        raise ValueError("oriented box requires a valid proper rotation")
    p=(x.reshape(-1,3)-center)@rotation
    S=np.zeros(len(p))
    M_local=np.zeros((len(p),3))
    for axis in range(3):
        j,k=[d for d in range(3) if d!=axis]
        u0,u1=-half[j]-p[:,j],half[j]-p[:,j]
        v0,v1=-half[k]-p[:,k],half[k]-p[:,k]
        for sign in (-1.0,1.0):
            w=sign*half[axis]-p[:,axis]
            i0,i1=_rectangle_face_integrals(u0,u1,v0,v1,w)
            # ∇_y·((y-x)/r)=2/r, n=sign*e_axis
            S+=0.5*sign*w*i0
            # ∇_y r=(y-x)/r, hence M=∮ r*n dS
            M_local[:,axis]+=sign*i1
    return S.reshape(original), (M_local@rotation.T).reshape(original+(3,))


def ball_coulomb_moments(points,center,radius):
    """Original scalar and first geometric Coulomb moments inside BALL."""
    x=np.asarray(points,float)
    center=np.asarray(center,float)
    if (x.shape[-1:]!=(3,) or center.shape!=(3,)
            or not np.isfinite(x).all() or
            not np.isfinite(center).all() or
            not np.isfinite(radius) or radius<=0):
        raise ValueError("invalid ball geometry or field points")
    d=x-center
    r2=np.sum(d*d,axis=-1)
    if np.any(r2>=radius**2):
        raise ValueError("ball interior formula requires points inside ball")
    return (
        2*math.pi*(radius**2-r2/3.0),
        (4*math.pi/15.0)*(r2-5*radius**2)[...,None]*d,
    )


def sea_coulomb_moments(scene,points):
    """Exact Coulomb S,M of B_outer minus BOTH rotated insulating boxes.

    All three boundary bodies must be disjoint as guaranteed by BfziScene.
    Physical query locations must belong to seawater. Does not involve
    artificial near-surface Gauss orders or induced-current modes.
    """
    x=np.asarray(points,float)
    if x.shape[-1:]!=(3,) or not np.isfinite(x).all():
        raise ValueError("finite (n,3) physical query points required")
    if not np.all(scene.in_seawater(x)):
        raise ValueError("Coulomb geometry moments require wet points")
    S,M=ball_coulomb_moments(x,scene.outer_center,scene.outer_radius)
    for assembly in scene.assemblies:
        s,m=box_coulomb_moments(
            x,assembly.pose.center,assembly.pose.rotation,
            assembly.enclosure.half_extents,
        )
        S=S-s
        M=M-m
    return S,M
