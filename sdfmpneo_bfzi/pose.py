"""Coil-attached package frames and explicit BFZI relative-pose features.

The Euler composition matches the original BFZI geometry._rotation_matrix_3d.
Package axes are expressed relative to the nominal world-aligned boxes.
"""
from __future__ import annotations

import numpy as np


def _rot(axis, angle):
    a=np.deg2rad(float(angle))
    c,s=np.cos(a),np.sin(a)
    if axis=='x':
        return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    if axis=='y':
        return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def axes(coil):
    """World axes of a native BFZI package, identity for nominal coils."""
    R=(_rot('z',getattr(coil,'yaw_deg',0.))@
       _rot('y',getattr(coil,'pitch_deg',0.))@
       _rot('x',getattr(coil,'roll_deg',90.))@
       _rot('z',getattr(coil,'plane_angle_deg',90.)))
    base=_rot('x',90.)@_rot('z',90.)
    return R@base.T


def desc(coils,boxes):
    """3 relative translation, 9 rotation, 6 independent package half-sizes."""
    if len(coils)!=2 or len(boxes)!=2:
        raise ValueError('two BFZI coils and package boxes required')
    A,B=(axes(c) for c in coils)
    p,q=(np.asarray(c.center,float) for c in coils)
    sizes=np.r_[np.asarray(boxes[0][1],float),
                np.asarray(boxes[1][1],float)]
    out=np.r_[A.T@(q-p),(A.T@B).ravel(),sizes]
    if out.shape!=(18,) or not np.isfinite(out).all() or np.any(sizes<=0):
        raise ValueError('BFZI pose/package dimensions invalid')
    return out


def default_desc(geom):
    """Legacy axis-aligned BFZI M1–M4 datasets without explicit pose metadata."""
    g=np.asarray(geom,float)
    if g.shape!=(16,):
        raise ValueError('16 original BFZI geometry values required')
    h=[.122 if g[i]>.11 else .102 for i in range(2)]
    return np.r_[0.,g[10],0.,np.eye(3).ravel(),
                 h[0],.0055,h[0],h[1],.0055,h[1]]


def isinside(pos,center,half,rot):
    """Membership in a rotated package box, using physical local coordinates."""
    p=np.asarray(pos,float).reshape(-1,3)
    return np.all(np.abs((p-center)@rot)<=half+1e-10,axis=1)
