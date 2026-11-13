"""Clip original BFZI seawater FV cells by physical oriented package solids.

Keeps the original BFZI rectangular FV grid, Green operator, source
projection and two-port solver. Replaces axis-aligned bounding-box water
exclusion with exact convex-polyhedron volume clipping of two rotated
sealed, nonoverlapping packages.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from .geom_void import overlap
from .pose import axes

_EPS=1e-12


def _clip(v,n,b):
    if not len(v):
        return v
    d=v@n-b
    if np.max(d)<=_EPS:
        return v
    if np.min(d)>=-_EPS:
        return np.empty((0,3))
    hull=ConvexHull(v)
    edges=set()
    for tri in hull.simplices:
        for i,j in ((0,1),(1,2),(2,0)):
            edges.add(tuple(sorted((int(tri[i]),int(tri[j])))))
    out=list(v[d<=_EPS])
    for i,j in edges:
        a,b0=d[i],d[j]
        if ((a < -_EPS and b0 > _EPS)
                or (b0 < -_EPS and a > _EPS)):
            out.append(v[i]+(v[j]-v[i])*(a/(a-b0)))
    if len(out)<4:
        return np.empty((0,3))
    return np.unique(np.round(out,14),axis=0)


def _tetra(v):
    if len(v)<4:
        return np.empty((0,3)),np.empty(0)
    tris=v[ConvexHull(v).simplices]
    cen=v.mean(axis=0)
    vol=np.abs(np.einsum(
        "ij,ij->i",tris[:,0]-cen,
        np.cross(tris[:,1]-cen,tris[:,2]-cen)))/6.
    keep=vol>1e-16
    return ((cen+tris.sum(axis=1))*.25)[keep],vol[keep]


def _wet(mn,mx,boxes):
    cube=np.array(list(product(*zip(mn,mx))),float)
    water=[cube]
    for mid,half,rot in boxes:
        planes=[]
        for i in range(3):
            n=rot[:,i]
            planes.extend([(n,float(n@mid+half[i])),
                           (-n,float(-n@mid+half[i]))])
        pieces=[]
        for poly in water:
            core=poly
            for n,b in planes:
                outside=_clip(core,-n,-b)
                if len(outside):
                    pieces.append(outside)
                core=_clip(core,n,b)
                if not len(core):
                    break
        water=pieces
        if not water:
            break
    q=[_tetra(poly) for poly in water]
    valid=[(p,w) for p,w in q if len(w)]
    if not valid:
        return np.empty((0,3)),np.empty(0)
    return (np.concatenate([p for p,w in valid]),
            np.concatenate([w for p,w in valid]))



def _full(mn,mx):
    """Original second-order tensor Gauss points for an uncut FV cell."""
    u=1/np.sqrt(3.)
    q=np.array(list(product((-u,u),repeat=3)),float)
    q=.5*(mn+mx)+.5*(mx-mn)*q
    w=np.full(8,float(np.prod(mx-mn))/8.)
    return q,w


def build(fs,records,meta,coils,sample,cfg,max_nodes=1400):
    """Produce real BFZI FV records with the true rotated sealed-box void.

    The original FV grid lines are reused, including cells previously
    removed by AABB exclusion. Cut cells use positive tetrahedral
    quadrature. Full seawater cells retain original BFZI Gauss quadrature.
    """
    if (len(coils)!=2 or cfg.get("BSR_PACKAGE_TOP_RISER_ENABLE",False)
            or cfg.get("BSR_PACKAGE_WALL_X_EXTRA_M",0.)
            or cfg.get("BSR_PACKAGE_WALL_NEG_Z_EXTRA_M",0.)):
        raise ValueError("oriented sea void needs two sealed BFZI packages")
    mid=np.array([c.center for c in coils],float)
    rot=np.array([axes(c) for c in coils])
    normal=float(cfg.get("BSR_PACKAGE_NORMAL_HALF_M",.0055))
    half=[]
    for p in (1,2):
        pre=f"coil{p}_"
        x=float(sample.get(pre+"package_x_half_mm",102.))*.001
        z=max(float(sample.get(pre+"package_z_pos_half_mm",102.)),
              float(sample.get(pre+"package_z_neg_half_mm",102.)))*.001
        half.append([x,normal,z])
    half=np.asarray(half)
    if overlap(mid,half,rot):
        raise ValueError("physical BFZI packages overlap: adjust receiver pose")
    boxes=list(zip(mid,half,rot))
    lo=np.asarray(meta["box_min_m"],float)
    hi=np.asarray(meta["box_max_m"],float)
    for center,h,R in boxes:
        span=np.abs(R)@h
        if np.any(center-span<lo-1e-12) or np.any(center+span>hi+1e-12):
            raise ValueError("oriented package extends outside BFZI seawater tank")
    # The native FV records include every axis-grid edge; reconstruct
    # the cells that its earlier axis-aligned cutout had removed.
    edges=[]
    for axis in range(3):
        edges.append(np.unique([
            float(x) for r in records
            for x in (r["cell_min"][axis],r["cell_max"][axis])]))
    aabb=[(center-np.abs(R)@h,center+np.abs(R)@h)
          for center,h,R in boxes]
    output=[];pts=[];vol=[];partials=0
    for idx in product(*(range(len(e)-1) for e in edges)):
        mn=np.array([edges[i][idx[i]] for i in range(3)])
        mx=np.array([edges[i][idx[i]+1] for i in range(3)])
        full=float(np.prod(mx-mn))
        subset=[box for box,(amin,amax) in zip(boxes,aabb)
                if np.all(mx>amin) and np.all(mn<amax)]
        if subset:
            q,w=_wet(mn,mx,subset)
            if not len(w):
                continue
            size=float(np.sum(w))
            if abs(size-full)<max(full*1e-10,1e-15):
                q,w=_full(mn,mx)
                size=full
            else:
                partials+=1
        else:
            q,w=_full(mn,mx)
            size=full
        if size<=0:
            continue
        center=np.sum(q*w[:,None],axis=0)/size
        output.append(dict(
            idx=idx,cell_min=mn,cell_max=mx,volume=size,
            full_volume=full,volume_fraction=float(size/full),
            quadrature_points=q,quadrature_weights=w,
            subboxes=[]))
        pts.append(center)
        vol.append(size)
    if len(pts)>max_nodes:
        raise ValueError(
            f"oriented BFZI FV grid {len(pts)} exceeds budget {max_nodes}")
    pts=np.asarray(pts,float).reshape(-1,3)
    vol=np.asarray(vol,float)
    expected=float(np.prod(hi-lo)-np.sum(8*np.prod(half,axis=1)))
    if abs(vol.sum()-expected)>max(1e-7,5e-6*expected):
        raise ValueError(f"oriented BFZI sea volume failed conservation: {vol.sum():.9g} vs {expected:.9g}, grid {[len(e)-1 for e in edges]} expected grid {[meta.get(k) for k in ('grid_nx','grid_ny','grid_nz')]}")
    out=dict(
        meta,geometry="BFZI_native_exact_oriented_package_clip",
        node_count=len(pts),original_node_count=len(pts),
        clipped_cell_count=partials,downsampled=False,
        sea_volume_m3_quadrature=float(vol.sum()),
        sea_volume_m3_expected=expected,
        relative_volume_error=float((vol.sum()-expected)/expected),
        physical_package_half_m=half.tolist(),
        physical_package_center_m=mid.tolist(),
        physical_package_rot=rot.tolist(),
        inner_box_count=2,void_type="oriented")
    return pts,vol,out,output
