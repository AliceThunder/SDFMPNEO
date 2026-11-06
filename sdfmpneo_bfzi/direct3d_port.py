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


def closed_source_port_impedance_exact(result, frequency_hz, source_strands):
    """Exact source-chord circulation of first-order Nedelec A_react.

    Partition each straight source chord at all intersected tetrahedral
    faces, integrate each cellwise *tangential* Whitney edge field exactly.
    Handles paths on shared faces without double-counting. This eliminates
    Gauss sampling error, NOT mesh error or finite-domain boundary flux.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError('Frequency must be positive')
    if (result.edge_coefficients is None or
            len(source_strands) != len(result.edge_coefficients)):
        raise ValueError('Port paths must match solved incident excitations')
    mesh = result.mesh
    vtx = mesh.nodes[mesh.tetrahedra]
    centers = vtx.mean(axis=1)
    max_radius = float(np.linalg.norm(
        vtx-centers[:, None, :], axis=2).max())
    tree = cKDTree(centers)
    _, edge_ids, signs, *_ = _topology(mesh)
    grads = np.zeros((len(vtx), 4, 3))
    transforms = (vtx[:, 1:] - vtx[:, 0, None, :]).transpose(0, 2, 1)
    grads[:, 1:, :] = np.linalg.inv(transforms)
    grads[:, 0, :] = -grads[:, 1:, :].sum(axis=1)
    out = np.zeros((len(source_strands), len(source_strands)), complex)
    for p, strands in enumerate(source_strands):
        if not strands:
            raise ValueError('Every physical port requires a closed source')
        for weight, path in strands:
            v = np.asarray(path, float)
            if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 4
                    or not np.isfinite(v).all()
                    or not np.array_equal(v[0], v[-1])
                    or not np.isfinite(weight) or not np.isreal(weight)
                    or np.any(np.linalg.norm(np.diff(v, axis=0), axis=1) <= 0)):
                raise ValueError('Source strand must be real-weight and exactly closed')
            for start, stop in zip(v[:-1], v[1:]):
                dx = stop-start
                # Any point on a segment in a tet lies within its vertex
                # radius of the cell centroid; the global bound is safe,
                # even for a graded nonuniform mesh.
                candidates = tree.query_ball_point(
                    (start+stop)/2,
                    float(np.linalg.norm(dx))/2 + 1.00001*max_radius + 1e-12)
                if not candidates:
                    raise ValueError('Source chord outside tetrahedral reference mesh')
                candidates = np.asarray(sorted(candidates), dtype=int)
                g = grads[candidates]
                bary123 = np.einsum('tij,tj->ti', g[:, 1:, :],
                                    start-vtx[candidates, 0])
                lam = np.column_stack((1-bary123.sum(axis=1), bary123))
                slopes = np.einsum('tij,j->ti', g, dx)
                low = np.zeros(len(candidates))
                high = np.ones(len(candidates))
                valid = np.ones(len(candidates), dtype=bool)
                for a in range(4):
                    d = slopes[:, a]
                    flat = abs(d) <= 1e-13
                    valid &= ~(flat & (lam[:, a] < -1e-10))
                    leaving = d < -1e-13
                    entering = d > 1e-13
                    high[leaving] = np.minimum(
                        high[leaving], -lam[leaving, a]/d[leaving])
                    low[entering] = np.maximum(
                        low[entering], -lam[entering, a]/d[entering])
                intersect = valid & (high-low > 1e-11)
                if not np.any(intersect):
                    raise ValueError('Source chord outside tetrahedral reference mesh')
                cells = candidates[intersect]
                lo = low[intersect]
                hi = high[intersect]
                cuts = np.sort(np.r_[0., 1., lo, hi])
                for left, right in zip(cuts[:-1], cuts[1:]):
                    if right-left <= 1e-11:
                        continue
                    mid = .5*(left+right)
                    cover = np.flatnonzero(
                        (lo <= mid+1e-11) & (hi >= mid-1e-11))
                    if not len(cover):
                        raise ValueError('Source chord escapes tetrahedral reference mesh')
                    cell = int(cells[int(cover[0])])
                    gc = grads[cell]
                    dlam = gc@dx
                    b123 = gc[1:]@(start+mid*dx-vtx[cell, 0])
                    b4 = np.r_[1-b123.sum(), b123]
                    tangential = np.array([
                        (b4[i]*dlam[j]-b4[j]*dlam[i])*signs[cell, e]
                        for e, (i, j) in enumerate(_EDGES)])
                    out[p] += weight*(right-left)*(
                        result.edge_coefficients[:, edge_ids[cell]]@tangential)
    return 1j*(2*np.pi*frequency_hz)*out


def medium_vacuum_port_increment(medium_result, frequency_hz,
                                 incident_potentials, source_strands,
                                 *, outer_bc='secondary_zero'):
    """Return (medium reaction Z, vacuum image Z, material-induced Delta Z).

    For total_zero, A_react has a nonzero *vacuum* boundary image even
    when sigma=0. Subtract a vacuum solve using the same mesh, source
    and outer boundary BEFORE calling the remainder seawater/metal Z.
    """
    from .direct3d_aphi import TetraMesh, solve_aphi
    medium_z = closed_source_port_impedance_exact(
        medium_result, frequency_hz, source_strands)
    if outer_bc == 'secondary_zero':
        vacuum_z = np.zeros_like(medium_z)
    elif outer_bc == 'total_zero':
        mesh = medium_result.mesh
        empty = TetraMesh(mesh.nodes, mesh.tetrahedra,
                          np.zeros_like(mesh.sigma))
        empty_solution = solve_aphi(empty, frequency_hz,
                                    incident_potentials,
                                    outer_bc='total_zero')
        vacuum_z = closed_source_port_impedance_exact(
            empty_solution, frequency_hz, source_strands)
    else:
        raise ValueError('outer_bc must be secondary_zero or total_zero')
    return medium_z, vacuum_z, medium_z-vacuum_z
