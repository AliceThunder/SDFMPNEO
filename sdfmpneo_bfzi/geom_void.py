"""BFZI native AABB seawater void versus true oriented package geometry.

The original BFZI V150 rectangular seawater volume builder excludes
world-axis-aligned bounding boxes; thermal packages are oriented OBBs.
"""
from __future__ import annotations

import numpy as np


def overlap(mid,half,rot):
    """Separating-axis intersection test for two physical oriented boxes."""
    mid=np.asarray(mid,float)
    half=np.asarray(half,float)
    rot=np.asarray(rot,float)
    if mid.shape!=(2,3) or half.shape!=(2,3) or rot.shape!=(2,3,3):
        raise ValueError("two oriented BFZI packages required")
    d=mid[1]-mid[0]
    a,b=rot
    axes=[*a.T,*b.T]+[
        np.cross(a[:,i],b[:,j]) for i in range(3) for j in range(3)]
    for axis in axes:
        s=float(np.linalg.norm(axis))
        if s<1e-12:
            continue
        n=axis/s
        ra=float(np.sum(half[0]*abs(a.T@n)))
        rb=float(np.sum(half[1]*abs(b.T@n)))
        if abs(d@n)>ra+rb+1e-12:
            return False
    return True


def report(head, mesh):
    """Quantify removed original BFZI sea volume and physical box mismatch."""
    mid=np.asarray(head._pkg_mid,float)
    half=np.asarray(head._pkg_half,float)
    rot=np.asarray(head._pkg_axes,float)
    boxes=mesh["inner_boxes"]
    if len(boxes)!=2:
        raise ValueError("expected original BFZI dual-package void boxes")
    aabb=[]
    for box in boxes:
        lo=np.asarray(box["min_m"],float)
        hi=np.asarray(box["max_m"],float)
        aabb.append((lo,hi))
    volumes=np.array([float(np.prod(hi-lo)) for lo,hi in aabb])
    inter=float(np.prod(np.maximum(
        np.minimum(aabb[0][1],aabb[1][1])-
        np.maximum(aabb[0][0],aabb[1][0]),0.)))
    void=float(volumes.sum()-inter)
    physical=8*np.prod(half,axis=1)
    actual_aabb=np.array([8*np.prod(np.abs(rot[i])@half[i])
                          for i in range(2)])
    return dict(
        original_BFZI_AABB_void_m3=void,
        original_BFZI_AABB_individual_m3=volumes.tolist(),
        original_BFZI_AABB_intersection_m3=inter,
        physical_oriented_individual_m3=physical.tolist(),
        physical_box_sum_m3=float(physical.sum()),
        AABB_void_excess_lower_bound_m3=float(void-physical.sum()),
        BFZI_AABB_over_physical_sum=float(void/physical.sum()),
        physical_packages_overlap=overlap(mid,half,rot),
        physical_box_AABB_m3=actual_aabb.tolist(),
        BFZI_AABB_to_physical_box_ratio=(volumes/physical).tolist())
