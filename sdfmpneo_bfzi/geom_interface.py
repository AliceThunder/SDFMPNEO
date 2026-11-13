"""Physical 3-D package/seawater conductance and oriented-face diagnostics."""
from __future__ import annotations

import numpy as np


def report(head, temp=None):
    """Read original BFZI FV cross-material H without new EM solves."""
    n, ns, m = len(head.cap), head.ns, head.npkg
    if n != ns+2*m+2:
        raise ValueError("BFZI five-material FV dimensions differ")
    ans=[]
    if temp is not None:
        temp=np.asarray(temp,float)
        if temp.shape != (n,):
            raise ValueError("BFZI 3-D temperature state dimensions differ")
    for p in range(2):
        st=ns+p*m
        # Cross-material H_sp contains negative conductances.
        Hsp=head.H[:ns,st:st+m].tocoo()
        row,col=Hsp.row,Hsp.col
        g=-Hsp.data
        if not len(g) or np.any(g<=0):
            raise ValueError("BFZI package-sea FV conductances invalid")
        mirror=(head.H[st:st+m,:ns].T-head.H[:ns,st:st+m]).tocoo()
        reciprocal=float(np.max(abs(mirror.data))) if mirror.nnz else 0.
        half=np.asarray(head._pkg_half[p],float)
        mid=np.asarray(head._pkg_mid[p],float)
        rot=np.asarray(head._pkg_axes[p],float)
        xyz=np.asarray(head.xyz[row],float)
        local=(xyz-mid)@rot
        outside=np.maximum(abs(local)-half,0.)
        dist=np.linalg.norm(outside,axis=1)
        inside=np.all(abs(local)<half-1e-12,axis=1)
        area=8*(half[0]*half[1]+half[1]*half[2]+half[0]*half[2])
        gsum=float(np.sum(g))
        v=dict(package=p+1,links=int(len(g)),wet_cells=int(len(set(row))),
               conductance_W_K=gsum,area_m2=float(area),
               effective_h_W_m2K=gsum/area,
               inside_box_links=int(np.count_nonzero(inside)),
               inside_box_conductance_W_K=float(g[inside].sum()),
               sea_center_surface_distance_median_m=float(np.median(dist)),
               sea_center_surface_distance_max_m=float(np.max(dist)),
               reciprocity_gap_W_K=reciprocal)
        if temp is not None:
            flux=g*(temp[st+col]-temp[row])
            v["package_to_sea_W"]=float(np.sum(flux))
            v["abs_interface_flux_W"]=float(np.sum(abs(flux)))
        ans.append(v)
    return ans
