"""Closed physical port voltages from the 3D Nedelec reaction field.

For peak exp(+iwt), Delta Z[p,q] = +i*w*int_Gamma_p A_react,q.dl.
Closed-path integral is gauge-independent. The commonly used overlap
i*w*int A_source,p.J_sea,q is a *different finite-domain observable*
when the boundary reaction field is nonzero, so neither may be silently
substituted for the other without a boundary/convergence argument.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .direct3d_aphi import _topology, _element_geometry, _EDGES


def reaction_potential_at_points(result, points, *, nearest=32):
    """Interpolate oriented first-order Nedelec A_react at interior points.

    Returns complex array (n_ports, n_points, 3). Rejects points outside
    the reference mesh; never performs nearest-cell extrapolation.
    """
    x=np.asarray(points,float)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
        raise ValueError('Expected finite query positions [n,3]')
    if result.edge_coefficients is None:
        raise ValueError('The solver must retain oriented edge coefficients')
    m=result.mesh
    _,local_edge,signs,*_=_topology(m)
    centres=m.nodes[m.tetrahedra].mean(axis=1)
    tree=cKDTree(centres)
    out=np.zeros((len(result.edge_coefficients),len(x),3),complex)
    for point_id,point in enumerate(x):
        count=min(max(1,int(nearest)),len(centres))
        choice=None
        while choice is None:
            _,ids=tree.query(point,k=count)
            for idx in np.atleast_1d(ids):
                idx=int(idx)
                tet=m.tetrahedra[idx]
                xyz=m.nodes[tet]
                rel=np.linalg.solve((xyz[1:]-xyz[0]).T,point-xyz[0])
                lam=np.r_[1.-rel.sum(),rel]
                if np.min(lam)>=-2e-9 and np.max(lam)<=1.+2e-9:
                    choice=(idx,lam)
                    break
            if choice is not None:break
            if count==len(centres):
                raise ValueError(f'Port sample {point_id} lies outside the 3D mesh')
            count=min(len(centres),count*2)
        cell,lam=choice
        _,grad=_element_geometry(m.nodes[m.tetrahedra[cell]])
        w=np.array([(lam[a]*grad[b]-lam[b]*grad[a])*signs[cell,e]
                    for e,(a,b) in enumerate(_EDGES)])
        out[:,point_id]=result.edge_coefficients[:,local_edge[cell]]@w
    return out


def closed_source_port_impedance(result, frequency_hz, source_strands,
                                 *, quad_order=4, subsegments=1):
    """Return genuine 2-port (or n-port) reaction Z from closed source paths.

    Each source_strands[p] contains (real turn/current weight, vertices)
    pairs, where vertices[n,3] is an exactly closed oriented polygon.
    All paths must lie inside the 3D reference domain. Copper self Z is
    excluded. Refine source chords, 3D mesh, and subsegments separately.
    """
    if (not np.isfinite(frequency_hz) or frequency_hz<=0 or
            not isinstance(quad_order,int) or not 1<=quad_order<=16 or
            not isinstance(subsegments,int) or subsegments<1):
        raise ValueError('Invalid frequency, quadrature or subdivision')
    if result.edge_coefficients is None or len(source_strands)!=len(result.edge_coefficients):
        raise ValueError('Source port paths must match solved unit excitations')
    t,w=np.polynomial.legendre.leggauss(quad_order)
    t=(t+1)/2
    w=w/2
    Z=np.zeros((len(source_strands),len(source_strands)),complex)
    for p,port in enumerate(source_strands):
        if not port:
            raise ValueError('Each port needs a physical closed strand')
        for weight,vertices in port:
            v=np.asarray(vertices,float)
            if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or
                    not np.isfinite(v).all() or not np.array_equal(v[0],v[-1]) or
                    not np.isfinite(weight) or not np.isreal(weight) or
                    np.any(np.linalg.norm(np.diff(v,axis=0),axis=1)<=0)):
                raise ValueError('A source strand must have real weight and exact closure')
            dl=np.diff(v,axis=0)
            sub=(np.arange(subsegments)[:,None]+t[None,:])/subsegments
            pts=v[:-1,None,None,:]+dl[:,None,None,:]*sub[None,:,:,None]
            a=reaction_potential_at_points(result,pts.reshape(-1,3))
            a=a.reshape(len(source_strands),len(dl),subsegments,quad_order,3)
            integral=np.einsum('psrqj,sj,q->p',a,dl,w)/subsegments
            Z[p] += 1j*2*np.pi*frequency_hz*weight*integral
    return Z
