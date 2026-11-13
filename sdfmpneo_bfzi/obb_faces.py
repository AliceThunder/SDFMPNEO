"""True wet FV face areas for physical oriented BFZI package voids.

The inherited FV projector uses sqrt(volume_fraction_i*volume_fraction_j)
to approximate an internal shared face. On a rotated cut cell, true wet
face area depends on the section of the package at that face, not on cell
volume. Compute the polygonal section and reuse it in both original BFZI
electric-potential projection and sea thermal diffusion.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as spla


def _clip(poly,n,b):
    """Clip a convex face polygon to one oriented package half-space."""
    if not len(poly):
        return poly
    dist=poly@n-b
    out=[]
    for i in range(len(poly)):
        v=poly[i]
        w=poly[(i+1)%len(poly)]
        a=float(dist[i])
        d=float(dist[(i+1)%len(poly)])
        inside=a<=1e-12
        next_inside=d<=1e-12
        if inside:
            out.append(v)
        if inside!=next_inside:
            out.append(v+(w-v)*(a/(a-d)))
    return np.asarray(out,float).reshape(-1,3)


def _area(poly,axis):
    if len(poly)<3:
        return 0.
    ax=[k for k in range(3) if k!=axis]
    u=poly[:,ax[0]]
    v=poly[:,ax[1]]
    return float(.5*abs(u@np.roll(v,-1)-v@np.roll(u,-1)))


def wet_area(rec1,rec2,axis,boxes):
    """Exact wet area of a common axis-normal native FV face."""
    mn1=np.asarray(rec1["cell_min"],float)
    mx1=np.asarray(rec1["cell_max"],float)
    mn2=np.asarray(rec2["cell_min"],float)
    mx2=np.asarray(rec2["cell_max"],float)
    cross=[i for i in range(3) if i!=axis]
    lo=np.maximum(mn1,mn2)
    hi=np.minimum(mx1,mx2)
    if np.any(hi[cross]<=lo[cross]):
        return 0.
    coord=.5*(mx1[axis]+mn2[axis])
    poly=[]
    for u,v in ((0,0),(1,0),(1,1),(0,1)):
        x=np.empty(3)
        x[axis]=coord
        x[cross[0]]=hi[cross[0]] if u else lo[cross[0]]
        x[cross[1]]=hi[cross[1]] if v else lo[cross[1]]
        poly.append(x)
    poly=np.array(poly)
    full=float(np.prod(hi[cross]-lo[cross]))
    removed=0.
    for mid,half,rot in boxes:
        p=poly
        for k in range(3):
            n=rot[:,k]
            p=_clip(p,n,float(n@mid+half[k]))
            if not len(p):
                break
            p=_clip(p,-n,float(-n@mid+half[k]))
            if not len(p):
                break
        removed+=_area(p,axis)
    a=full-removed
    if a < -max(full*1e-9,1e-12):
        raise ValueError("oriented BFZI face exclusion exceeds full wet face")
    return float(np.clip(a,0.,full))


def apply(proj,records,meta):
    """Update inherited BFZI FV edge areas using actual OBB wet face geometry."""
    if meta.get("void_type")!="oriented":
        return proj
    mid=np.asarray(meta["physical_package_center_m"],float)
    half=np.asarray(meta["physical_package_half_m"],float)
    rot=np.asarray(meta["physical_package_rot"],float)
    boxes=list(zip(mid,half,rot))
    edges=[]
    altered=0
    removed=0
    for i,j,axis,d,area,direction in proj.edges:
        true=wet_area(records[i],records[j],int(axis),boxes)
        if true<=1e-15:
            removed+=1
            continue
        altered+=int(abs(true-area)>max(1e-12,1e-7*area))
        edges.append((int(i),int(j),int(axis),float(d),
                      float(true),float(direction)))
    if not edges:
        raise ValueError("original BFZI true wet faces produce no FV connections")
    count=len(edges)
    rows=np.repeat(np.arange(count),2)
    cols=np.array([(e[0],e[1]) for e in edges],int).ravel()
    vals=np.array([(-1/e[3],1/e[3]) for e in edges],float).ravel()
    B=sp.csr_matrix((vals,(rows,cols)),shape=(count,proj.n))
    weight=np.array([e[3]*e[4] for e in edges])
    L=(B.T@sp.diags(weight)@B).tolil()
    L[0,:]=0
    L[:,0]=0
    L[0,0]=1
    L=L.tocsc()
    proj.edges=edges
    proj.weights=weight
    proj.B=B
    proj.L=L
    proj.solve=spla.factorized(L)
    proj.meta.update(true_oriented_wet_faces=True,
                     face_area_corrected_edges=altered,
                     face_area_removed_edges=removed)
    return proj
