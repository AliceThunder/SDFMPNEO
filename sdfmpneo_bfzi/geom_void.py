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



def disjoint_y_shift(mid,half,rot,margin=.005):
    """Additional +world-y displacement for receiver to avoid OBB collision."""
    mid=np.asarray(mid,float)
    if not overlap(mid,half,rot):
        return 0.
    lo,hi=0.,.01
    def collides(dy):
        shifted=mid.copy()
        shifted[1,1]+=dy
        return overlap(shifted,half,rot)
    while collides(hi):
        hi*=2
        if hi>10.:
            raise ValueError("BFZI package geometry cannot be separated")
    for _ in range(45):
        m=.5*(lo+hi)
        if collides(m):
            lo=m
        else:
            hi=m
    return float(hi+margin)


def report(head, mesh):
    """Quantify removed original BFZI sea volume and physical box mismatch."""
    mid=np.asarray(head._pkg_mid,float)
    half=np.asarray(head._pkg_half,float)
    rot=np.asarray(head._pkg_axes,float)
    boxes=mesh["inner_boxes"]
    if len(boxes) not in (2,4):
        raise ValueError("expected two BFZI packages with optional dry risers")
    aabb=[]
    for box in boxes:
        lo=np.asarray(box["min_m"],float)
        hi=np.asarray(box["max_m"],float)
        aabb.append((lo,hi))
    base=[aabb[i] for i in ((0,1) if len(aabb)==2 else (0,2))]
    def union(items):
        total=0.
        for mask in range(1,1<<len(items)):
            idx=[i for i in range(len(items)) if mask & (1<<i)]
            lo=np.max([items[i][0] for i in idx],axis=0)
            hi=np.min([items[i][1] for i in idx],axis=0)
            vol=float(np.prod(np.maximum(hi-lo,0.)))
            total+=vol if len(idx)%2 else -vol
        return total
    void=union(aabb)
    package_void=union(base)
    volumes=np.array([float(np.prod(hi-lo)) for lo,hi in base])
    physical=8*np.prod(half,axis=1)
    actual_aabb=np.array([8*np.prod(np.abs(rot[i])@half[i])
                          for i in range(2)])
    return dict(
        original_BFZI_AABB_void_m3=float(void),
        original_BFZI_package_AABB_union_m3=float(package_void),
        original_BFZI_AABB_individual_m3=volumes.tolist(),
        additional_dry_riser_void_m3=float(void-package_void),
        physical_oriented_individual_m3=physical.tolist(),
        physical_box_sum_m3=float(physical.sum()),
        AABB_void_excess_lower_bound_m3=float(package_void-physical.sum()),
        BFZI_AABB_over_physical_sum=float(package_void/physical.sum()),
        physical_packages_overlap=overlap(mid,half,rot),
        additional_receiver_y_shift_mm=float(1000*disjoint_y_shift(
            mid,half,rot)),
        physical_box_AABB_m3=actual_aabb.tolist(),
        BFZI_AABB_to_physical_box_ratio=(volumes/physical).tolist())
