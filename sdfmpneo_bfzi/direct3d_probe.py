"""3-D A-phi finite-element field sampling for independent D2 benchmarks.

Uses cellwise B_secondary (exact for first-order Nedelec) and barycenter J
(quadrature-consistent cell mean) for mesh-refinement diagnostics.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .direct3d_aphi import MU0, _BARY4


def closed_polyline_magnetic(vertices):
    """Biot-Savart B of a closed oriented polygonal unit-current loop."""
    v=np.asarray(vertices,float)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.array_equal(v[0],v[-1]):
        raise ValueError('Expected exactly closed source path')
    def field(x):
        pts=np.asarray(x,float)
        if pts.ndim!=2 or pts.shape[1]!=3:
            raise ValueError('Positions must be [n,3]')
        bfield=np.zeros_like(pts)
        for a,b in zip(v[:-1],v[1:]):
            first=pts-a; last=pts-b
            r1=np.linalg.norm(first,axis=1);r2=np.linalg.norm(last,axis=1)
            dot=np.sum(first*last,axis=1)
            den=r1*r2*(r1*r2+dot)
            if np.any(den<=0):
                raise ValueError('Biot-Savart probe on ideal wire')
            bfield += MU0/(4*np.pi)*np.cross(first,last)*((r1+r2)/den)[:,None]
        return bfield
    return field


def sample_cell_fields(result, points, *, incident_magnetic=None,
                       prefer_conductor=False, nearest=96):
    """Sample J and B on known physical tetrahedra.

    incident_magnetic: list[callable([n,3]) -> [n,3]] for *unit* ports;
      without it only B_secondary is returned (cannot claim total B).
    Prefer conductor on a face only for deliberately one-sided TEAM-7 J
    probes; do not conflate a sea/air surface with metal interior.
    """
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError('Expected finite [n,3] probe locations')
    mesh=result.mesh
    corners=mesh.nodes[mesh.tetrahedra]
    centres=corners.mean(axis=1)
    k=min(max(1,int(nearest)),len(corners))
    _, candidates=cKDTree(centres).query(p,k=k)
    candidates=np.asarray(candidates).reshape(len(p),-1)
    labels=[]
    probe_bary=[]
    for point,inds in zip(p,candidates):
        hits=[]
        for idx in inds:
            idx=int(idx)
            tet=corners[idx]
            local=np.linalg.solve((tet[1:]-tet[0]).T,point-tet[0])
            bary=np.r_[1.-local.sum(),local]
            if np.min(bary)>=-1e-9 and np.max(bary)<=1.+1e-9:
                hits.append((idx,bary))
        if not hits:
            raise ValueError('Probe outside mesh or requires a larger nearest window')
        if prefer_conductor:
            hits.sort(key=lambda j:mesh.sigma[j[0]],reverse=True)
        labels.append(hits[0][0])
        probe_bary.append(hits[0][1])
    idx=np.asarray(labels,int)
    Bsc=result.magnetic_secondary[:,idx]
    # The solved Nedelec E field is affine per tetrahedron; reconstitute
    # it EXACTLY at each physical probe from the four second-order
    # quadrature field values, rather than using cell-mean J as pointwise J.
    lam=np.asarray(probe_bary)
    quad_weights=lam @ np.linalg.inv(_BARY4)
    E=np.einsum('nq,pnqc->pnc',quad_weights,
                result.electric_at_quadrature[:,idx])
    J=mesh.sigma[idx][None,:,None]*E
    q=.5*mesh.sigma[idx][None,:]*np.sum(abs(E)**2,axis=-1)
    out={'cell_index':idx,'sigma_S_m':mesh.sigma[idx],
         'E_total_V_m':E,'J_A_m2':J,'B_secondary_T':Bsc,
         'q_W_m3':q,'J_cell_mean_A_m2':result.current_at_cells[:,idx]}
    if incident_magnetic is not None:
        if len(incident_magnetic)!=len(Bsc):
            raise ValueError('Expected one incident B provider per port')
        inc=np.array([np.asarray(f(p),dtype=complex) for f in incident_magnetic])
        if inc.shape!=Bsc.shape:
            raise ValueError('Incident B providers must return [n,3]')
        out['B_total_T']=Bsc+inc
    return out
