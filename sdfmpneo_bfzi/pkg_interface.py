"""Conservative oriented package-face to seawater FV thermal coupling.

Every physical package face is quadrature-sampled. Conductance follows
package half-cell conduction, surface transfer, and seawater center distance
in series; multiple wet FV neighbors share the same physical face flux.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def face_links(wet, cells, half, rot, dims, start, pkg_k, sea_k, h,
               quad=2, near=4):
    wet=np.asarray(wet,float)
    cells=np.asarray(cells,float)
    half=np.asarray(half,float)
    rot=np.asarray(rot,float)
    dims=tuple(map(int,dims))
    if (wet.ndim!=2 or wet.shape[1]!=3 or
            cells.shape!=(int(np.prod(dims)),3) or
            half.shape!=(3,) or rot.shape!=(3,3) or
            min(dims)<1 or np.min(half)<=0 or
            min(pkg_k,sea_k,h)<=0 or quad<1 or near<1 or
            not np.all(np.isfinite(wet))):
        raise ValueError("positive oriented BFZI interface geometry required")
    size=2*half/np.asarray(dims)
    tree=cKDTree(wet)
    k=min(len(wet),max(16,near*4))
    links=[]
    for t in np.ndindex(*dims):
        a=np.ravel_multi_index(t,dims)
        for ax in range(3):
            cross=[z for z in range(3) if z!=ax]
            for end,sgn in ((0,-1),(dims[ax]-1,1)):
                if t[ax]!=end:
                    continue
                n=sgn*rot[:,ax]
                face=cells[a]+.5*sgn*size[ax]*rot[:,ax]
                area=np.prod(size[cross])/quad**2
                for i in range(quad):
                    for j in range(quad):
                        loc=face+(
                            ((i+.5)/quad-.5)*size[cross[0]]*rot[:,cross[0]]
                            +((j+.5)/quad-.5)*size[cross[1]]*rot[:,cross[1]])
                        d,ids=tree.query(loc,k=k)
                        d=np.atleast_1d(d)
                        ids=np.atleast_1d(ids)
                        valid=((wet[ids]-loc)@n)>1e-12
                        if not np.any(valid):
                            d,ids=tree.query(loc,k=len(wet))
                            d=np.atleast_1d(d)
                            ids=np.atleast_1d(ids)
                            valid=((wet[ids]-loc)@n)>1e-12
                        if not np.any(valid):
                            raise ValueError("no wet BFZI seawater cells outside package face")
                        d=d[valid][:near]
                        ids=ids[valid][:near]
                        wt=1/(d*d+(.1*min(size))**2)
                        wt=wt/wt.sum()
                        dist=float(wt@d)
                        resistance=1/h+.5*size[ax]/pkg_k+dist/sea_k
                        g=area/resistance
                        links.extend((int(b),int(start+a),float(g*w))
                                     for b,w in zip(ids,wt))
    return links



def adjacent_links(wet,records,cells,half,rot,dims,start,pkg_k,
                   sea_k,h,quad=4):
    """Conservative package-face transfer to the actual adjacent wet cut FV.

    Each physical oriented face quadrature point is displaced infinitesimally
    into seawater. Its containing original BFZI wet FV cell receives that
    surface flux; unlike nearest-center transfer, the map follows the wet
    polyhedral cell support. No new volume grid or Green operator is built.
    """
    wet=np.asarray(wet,float)
    cells=np.asarray(cells,float)
    half=np.asarray(half,float)
    rot=np.asarray(rot,float)
    dims=tuple(map(int,dims))
    if (len(records)!=len(wet) or wet.ndim!=2 or wet.shape[1]!=3
            or cells.shape!=(int(np.prod(dims)),3)
            or half.shape!=(3,) or rot.shape!=(3,3)
            or min(dims)<1 or np.any(half<=0)
            or min(pkg_k,sea_k,h)<=0 or quad<1):
        raise ValueError("original BFZI wet FV and package surface mismatch")
    lo=np.array([r["cell_min"] for r in records],float)
    hi=np.array([r["cell_max"] for r in records],float)
    size=2*half/np.asarray(dims)
    span=np.max(hi-lo)
    eps=1e-7*max(span,1.)
    links=[]
    for t in np.ndindex(*dims):
        a=np.ravel_multi_index(t,dims)
        for ax in range(3):
            cross=[j for j in range(3) if j!=ax]
            for end,sgn in ((0,-1),(dims[ax]-1,1)):
                if t[ax]!=end:
                    continue
                n=sgn*rot[:,ax]
                face=cells[a]+.5*sgn*size[ax]*rot[:,ax]
                area=np.prod(size[cross])/(quad*quad)
                for i in range(quad):
                    for j in range(quad):
                        loc=(face+
                             ((i+.5)/quad-.5)*size[cross[0]]*rot[:,cross[0]]+
                             ((j+.5)/quad-.5)*size[cross[1]]*rot[:,cross[1]])
                        probe=loc+eps*n
                        ok=np.all((lo<=probe)&(probe<hi),axis=1)
                        ids=np.flatnonzero(ok)
                        if not len(ids):
                            raise ValueError(
                                "no adjacent wet BFZI cut FV outside package face")
                        if len(ids)>1:
                            dst=np.linalg.norm(wet[ids]-loc,axis=1)
                            idx=int(ids[np.argmin(dst)])
                        else:
                            idx=int(ids[0])
                        dist=float(np.linalg.norm(wet[idx]-loc))
                        res=1/h+.5*size[ax]/pkg_k+dist/sea_k
                        links.append((idx,int(start+a),float(area/res)))
    return links
