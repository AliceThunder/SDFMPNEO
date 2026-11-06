"""Independent 3-D Maxwell-MQS A-phi finite-element reference (initial D2 core).

Lowest-order oriented tetrahedral Nedelec edge elements for secondary A,
continuous P1 phi in conductive cells, continuous P1 gauge multiplier over Q.
Incident A comes from externally specified closed source(s); mu=mu0.
The outer boundary uses homogeneous tangential A_secondary=0: finite box only,
NOT an infinite-seawater reference without outer-domain convergence.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Callable
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4e-7 * np.pi
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_A = 0.5854101966249685
_B = (1 - _A) / 3
_BARY4 = np.eye(4)*(_A-_B) + _B


@dataclass(frozen=True)
class TetraMesh:
    nodes: np.ndarray
    tetrahedra: np.ndarray
    sigma: np.ndarray  # real conductivity per tetrahedron, S/m

    def __post_init__(self):
        x = np.asarray(self.nodes, dtype=float)
        t = np.asarray(self.tetrahedra, dtype=int)
        s = np.asarray(self.sigma, dtype=float)
        if (x.ndim != 2 or x.shape[1] != 3 or t.ndim != 2
                or t.shape[1] != 4 or len(t) != len(s) or len(t) < 1):
            raise ValueError('Expected nodes[n,3], tetrahedra[m,4], sigma[m]')
        if (not np.isfinite(x).all() or not np.isfinite(s).all()
                or np.any(s < 0) or np.any(t < 0) or np.any(t >= len(x))
                or any(len(set(row)) != 4 for row in t)):
            raise ValueError('Invalid vertices, tets or conductivity')
        object.__setattr__(self, 'nodes', x)
        object.__setattr__(self, 'tetrahedra', t)
        object.__setattr__(self, 'sigma', s)


def box_tetra_mesh(bounds, divisions, conductivity) -> TetraMesh:
    """Conforming box tet mesh; conductivity(x,y,z) is cellwise.

    Used only for coarse manufactured tests and exploratory voxel geometries.
    A verified TEAM-7 benchmark needs geometry-conforming mesh and real data.
    """
    b = np.asarray(bounds, float)
    n = np.asarray(divisions, int)
    if b.shape != (3,2) or n.shape != (3,) or np.any(n < 1) or np.any(b[:,1] <= b[:,0]):
        raise ValueError('bounds is (3,2); positive integer 3-vector divisions')
    coords = [np.linspace(b[k,0], b[k,1], n[k]+1) for k in range(3)]
    x = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1).reshape(-1,3)
    def index(i,j,k):
        return (i*(n[1]+1)+j)*(n[2]+1)+k
    tetra = []
    sigmas = []
    six = ((0,1,3,7),(0,3,2,7),(0,2,6,7),
           (0,6,4,7),(0,4,5,7),(0,5,1,7))
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                v = (index(i,j,k), index(i+1,j,k), index(i,j+1,k),
                     index(i+1,j+1,k), index(i,j,k+1), index(i+1,j,k+1),
                     index(i,j+1,k+1), index(i+1,j+1,k+1))
                centre = (x[v[0]] + x[v[7]])/2
                sigma = float(conductivity(centre))
                if not np.isfinite(sigma) or sigma < 0:
                    raise ValueError('Conductivity must be nonnegative and finite')
                for t in six:
                    tetra.append([v[a] for a in t])
                    sigmas.append(sigma)
    return TetraMesh(x, np.asarray(tetra), np.asarray(sigmas))


def closed_polyline_potential(vertices):
    """Exact finite straight-segment potential for a CLOSED unit-current loop.

    Polygonal BFZI corners need panel refinement. No point-line Gauss cloud
    or artificial Coulomb kernel softening is used.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError('Expected finite closed polyline [n,3]')
    if not np.array_equal(v[0], v[-1]) or np.any(np.linalg.norm(np.diff(v, axis=0), axis=1) <= 0):
        raise ValueError('Source polyline must be exactly closed and nondegenerate')
    def potential(x):
        p = np.asarray(x, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('Expected evaluation points [n,3]')
        out = np.zeros_like(p)
        for a,b in zip(v[:-1],v[1:]):
            length = np.linalg.norm(b-a)
            d = np.linalg.norm(p-a,axis=1)+np.linalg.norm(p-b,axis=1)-length
            if np.any(d <= 1e-12*length):
                raise ValueError('Sampling on/too close to an ideal source wire')
            out += MU0/(4*np.pi)*(b-a)/length * np.log1p(2*length/d)[:,None]
        return out
    return potential


@dataclass
class AphiResult:
    delta_z: np.ndarray
    power_matrix: np.ndarray   # P(I)=.5 real(I^H matrix I)
    power_from_fields: np.ndarray
    current_at_cells: np.ndarray   # (port,tet,3)
    magnetic_secondary: np.ndarray  # (port,tet,3)
    electric_at_quadrature: np.ndarray  # (port,tet,4,3)
    mesh: TetraMesh
    n_edge_dofs: int
    n_potential_dofs: int
    n_gauge_dofs: int
    edge_coefficients: np.ndarray | None = None  # [port, global edge], includes imposed boundary traces

    def joule_density_at_quadrature(self, currents):
        """Joule heat source [W/m3] for arbitrary complex peak port currents.

        Return (n_tetra, 4) at the field quadrature points; insulation = 0.
        """
        i=np.asarray(currents,dtype=complex)
        if i.shape!=(len(self.electric_at_quadrature),) or not np.isfinite(i).all():
            raise ValueError('Expected one finite complex peak current per port')
        e=np.einsum('p,pnqc->nqc',i,self.electric_at_quadrature)
        return .5*self.mesh.sigma[:,None]*np.sum(abs(e)**2,axis=-1)

    def port_power(self, currents):
        i=np.asarray(currents,dtype=complex)
        if i.shape!=(len(self.delta_z),) or not np.isfinite(i).all():
            raise ValueError('Expected one finite complex current per port')
        return float((.5*np.vdot(i,self.delta_z@i)).real)

    def volume_power(self, currents):
        i=np.asarray(currents,dtype=complex)
        if i.shape!=(len(self.power_from_fields),) or not np.isfinite(i).all():
            raise ValueError('Expected one finite complex current per port')
        return float((.5*np.vdot(i,self.power_from_fields@i)).real)


def _element_geometry(coords):
    transform = (coords[1:] - coords[0]).T
    det = np.linalg.det(transform)
    if abs(det) < 1e-16:
        raise ValueError('Degenerate tetrahedron')
    volume = abs(det)/6
    grads = np.zeros((4,3))
    grads[1:] = np.linalg.inv(transform)
    grads[0] = -np.sum(grads[1:],axis=0)
    return volume,grads


def _topology(mesh):
    tets = mesh.tetrahedra
    edges, faces = {}, {}
    local_ids = []
    local_signs = []
    for tet in tets:
        idx, signs = [], []
        for i,j in _EDGES:
            a,b=int(tet[i]),int(tet[j])
            key=(min(a,b),max(a,b))
            idx.append(edges.setdefault(key,len(edges)))
            signs.append(1 if a < b else -1)
        local_ids.append(idx)
        local_signs.append(signs)
        for ids in _FACES:
            f=tuple(sorted(int(tet[k]) for k in ids))
            faces[f] = faces.get(f,0)+1
    boundary_faces=[f for f,n in faces.items() if n==1]
    if any(n>2 for n in faces.values()):
        raise ValueError('Non-manifold face in tetra mesh')
    boundary_nodes=set(k for f in boundary_faces for k in f)
    boundary_edges=set()
    for f in boundary_faces:
        for e in combinations(f,2):
            boundary_edges.add(edges[tuple(sorted(e))])
    return edges,np.asarray(local_ids),np.asarray(local_signs),boundary_nodes,boundary_edges


def solve_aphi(mesh: TetraMesh, frequency_hz:float,
               incident_potentials: list[Callable], *, mu:float=MU0,
               outer_bc: str = 'secondary_zero') -> AphiResult:
    """Gauge-fixed 3D secondary A-phi, sources=unit-current incident A fields.

    Assumes mu=mu0 everywhere, prescribed incident potentials come from CLOSED
    sources outside conducting cells; copper self impedance is excluded.
    outer_bc='secondary_zero': n x A_react=0 (old finite-domain model).
    outer_bc='total_zero': n x (A_src+A_react)=0; impose projected
    edge traces -int_edge A_src·dl. This matches zero tangential TOTAL
    vector potential on the finite boundary, not an infinite-ocean condition.
    Only a conforming material/geometry mesh is appropriate.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0 or mu != MU0:
        raise ValueError('Positive frequency and currently uniform mu0 required')
    if not incident_potentials:
        raise ValueError('At least one incident port is required')
    if outer_bc not in ('secondary_zero', 'total_zero'):
        raise ValueError('outer_bc must be secondary_zero or total_zero')
    omega=2*np.pi*frequency_hz
    edges,le,sg,bnodes,bedges=_topology(mesh)
    ne=len(edges); nn=len(mesh.nodes)
    interior_nodes=[i for i in range(nn) if i not in bnodes]
    is_cond=mesh.sigma>0
    conductor_nodes=set(int(k) for t in mesh.tetrahedra[is_cond] for k in t)
    parent={k:k for k in conductor_nodes}
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    for tet in mesh.tetrahedra[is_cond]:
        root=int(tet[0])
        for v in tet[1:]:
            a,b=find(root),find(int(v))
            if a != b: parent[b]=a
    fixed={}
    for v in sorted(conductor_nodes):
        root=find(v)
        fixed.setdefault(root,v)
    phi_nodes=sorted(conductor_nodes-set(fixed.values()))
    pe=len(phi_nodes)
    gauge_nodes=interior_nodes
    pg=len(gauge_nodes)
    phimap={n:ne+i for i,n in enumerate(phi_nodes)}
    pmap={n:ne+pe+i for i,n in enumerate(gauge_nodes)}
    size=ne+pe+pg
    rows=[];cols=[];data=[]
    rhs=np.zeros((size,len(incident_potentials)),complex)
    local_vals=[]; source_at_quad=[]
    for ti,tet in enumerate(mesh.tetrahedra):
        coords=mesh.nodes[tet]
        V,g=_element_geometry(coords)
        orient=sg[ti]
        w=np.array([(_BARY4[:,i,None]*g[j]-_BARY4[:,j,None]*g[i])*orient[e]
                    for e,(i,j) in enumerate(_EDGES)]).transpose(1,0,2)
        curl=np.array([2*np.cross(g[i],g[j])*orient[e]
                       for e,(i,j) in enumerate(_EDGES)])
        mass=(V/4)*np.einsum('qic,qjc->ij',w,w)
        curl_stiff=V/mu*curl@curl.T
        wp=(V/4)*np.einsum('qic,jc->ij',w,g)
        ee=le[ti]
        sigma=mesh.sigma[ti]
        local=curl_stiff+1j*omega*sigma*mass
        for i in range(6):
            for j in range(6):
                rows.append(int(ee[i]));cols.append(int(ee[j]));data.append(local[i,j])
        for i in range(6):
            for a,node in enumerate(tet):
                if node in pmap:
                    k=pmap[int(node)]
                    rows.extend((int(ee[i]), k));cols.extend((k,int(ee[i])))
                    data.extend((wp[i,a],wp[i,a]))
                if sigma and node in phimap:
                    k=phimap[int(node)]
                    rows.extend((int(ee[i]),k));cols.extend((k,int(ee[i])))
                    data.extend((sigma*wp[i,a],sigma*wp[i,a]))
        if sigma:
            local_phi=(sigma/(1j*omega))*V*(g@g.T)
            for a,node_a in enumerate(tet):
                if node_a not in phimap:continue
                for b,node_b in enumerate(tet):
                    if node_b in phimap:
                        rows.append(phimap[int(node_a)]);cols.append(phimap[int(node_b)])
                        data.append(local_phi[a,b])
        pts=_BARY4@coords
        a_fields=[]
        for port,field in enumerate(incident_potentials):
            a=np.asarray(field(pts),dtype=complex)
            if a.shape != (4,3) or not np.isfinite(a).all():
                raise ValueError('Incident field must return finite [n,3] vector potentials')
            a_fields.append(a)
            if sigma:
                rhs[ee,port] += -1j*omega*sigma*V/4*np.einsum('qic,qc->i',w,a)
                for v in range(4):
                    if int(tet[v]) in phimap:
                        rhs[phimap[int(tet[v])],port] += -sigma*V/4*np.einsum('qc,c->',a,g[v])
        local_vals.append((V,g,w,curl,ee,tet))
        source_at_quad.append(a_fields)
    K=coo_matrix((np.asarray(data,dtype=complex),(rows,cols)),shape=(size,size)).tocsc()
    free_edges=[i for i in range(ne) if i not in bedges]
    keep=np.array(free_edges+list(range(ne,size)),dtype=int)
    if not len(keep):
        raise ValueError('No interior DOFs in the given mesh')
    Kr=K[keep,:][:,keep]
    bc_values=np.zeros((size,len(incident_potentials)),complex)
    if outer_bc=='total_zero':
        # Nedelec boundary edge DOFs are oriented circulation integrals,
        # not point values. Project each source's tangential A onto the
        # same globally oriented boundary edges with 3-point Gauss.
        t,wg=np.polynomial.legendre.leggauss(3)
        t=(t+1)/2
        wg=wg/2
        by_idx={v:k for k,v in edges.items()}
        indices=np.array(sorted(bedges),dtype=int)
        for start in range(0,len(indices),512):
            ids=indices[start:start+512]
            ends=np.array([by_idx[int(ii)] for ii in ids])
            start_x=mesh.nodes[ends[:,0]]
            tangent=mesh.nodes[ends[:,1]]-start_x
            eval_x=(start_x[:,None,:] + t[None,:,None]*tangent[:,None,:])
            for port,field in enumerate(incident_potentials):
                values=np.asarray(field(eval_x.reshape(-1,3)),dtype=complex)
                if values.shape != (eval_x.shape[0]*3,3) or not np.isfinite(values).all():
                    raise ValueError('Invalid vector potential on outer boundary')
                values=values.reshape(len(ids),3,3)
                bc_values[ids,port]=-np.einsum('eqc,ec,q->e',values,tangent,wg)
    rhsr=rhs[keep]-K[keep,:]@bc_values
    maxrow=np.asarray(abs(Kr).max(axis=1).toarray()).ravel()
    if np.any(maxrow==0):
        raise ValueError('Null row: disconnected conductive or gauge basis')
    scale=1/np.sqrt(maxrow)
    Ks=Kr.multiply(scale[:,None]).multiply(scale[None,:]).tocsc()
    result=spsolve(Ks,scale[:,None]*rhsr)
    if result.ndim==1: result=result[:,None]
    if not np.isfinite(result).all():
        raise ValueError('A-phi solve failed (singular or non-finite)')
    full=bc_values
    full[keep,:]=scale[:,None]*result
    nports=len(incident_potentials)
    jbar=np.zeros((nports,len(mesh.tetrahedra),3),complex)
    bsec=np.zeros_like(jbar)
    e_quad=np.zeros((nports,len(mesh.tetrahedra),4,3),complex)
    zp=np.zeros((nports,nports),complex)
    p_from_fields=np.zeros((nports,nports),complex)
    for ti,(vol,g,w,curl,edgeids,tet) in enumerate(local_vals):
        sig=mesh.sigma[ti]
        As=np.einsum('qic,ip->pqc',w,full[edgeids])
        grad_phi=np.zeros((nports,3),complex)
        for a,node in enumerate(tet):
            if int(node) in phimap:
                grad_phi += full[phimap[int(node)],:,None]*g[a][None,:]
        src=np.asarray(source_at_quad[ti])
        E=-1j*omega*(As+src)-grad_phi[:,None,:]
        e_quad[:,ti]=E
        jbar[:,ti]=sig*E.mean(axis=1)
        bsec[:,ti]=full[edgeids].T@curl
        if sig:
            zp += 1j*omega*vol/4*np.einsum('pqc,rqc->pr',src,sig*E)
            p_from_fields += sig*vol/4*np.einsum('pqc,rqc->pr',E.conj(),E)
    return AphiResult(zp,(zp+zp.conj().T)/2,p_from_fields,
                      jbar,bsec,e_quad,mesh,len(free_edges),pe,pg,full[:ne,:].T.copy())
