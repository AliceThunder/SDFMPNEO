"""Temperature-dependent wrapper around the original BFZI v148 solver.

The independent AliceThunder/BFZI package is imported by the caller and
passed as the 'calc' module, so the original Green/resolvent, source and
Neumann/GMD implementations are reused, not copied into SDFMPNEO.

Original v148 assumes one sigma. This wrapper updates sigma per original
seawater FV cell, reweights the FV scalar-potential projection, and reweights
its original near-source local-mass integral cell by cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from ast import literal_eval
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from .pose import axes

from sdfmpneo_vnext.scene import ConductorMaterial
from .source import BFZISources


class WeightedFV:
    """The original BFZI scalar FV projector with sigma-weighted flux."""

    def __init__(self, base, sigma):
        self.base = base
        self.n = base.n
        self.edges = base.edges
        self.meta = dict(base.meta)
        sig = np.asarray(sigma, float)
        if sig.shape != (self.n,) or not np.all(np.isfinite(sig)) or np.min(sig) <= 0:
            raise ValueError("sigma must be positive per seawater cell")
        cond = np.asarray([2*sig[i]*sig[j]/(sig[i]+sig[j])
                           for i,j,*_ in self.edges])
        self.weights = base.weights*cond
        mat = (base.B.T @ sp.diags(self.weights) @ base.B).tolil()
        mat[0,:] = 0
        mat[:,0] = 0
        mat[0,0] = 1
        self.factor = spla.factorized(mat.tocsc())

    def project_one(self, field):
        base = self.base
        F = np.asarray(field)
        if F.shape != (self.n,3):
            raise ValueError("FV field shape must be (n,3)")
        edges = self.edges
        c = np.asarray([
            .5*(F[i,axis]+F[j,axis])*direction
            for i,j,axis,d,area,direction in edges])
        rhs = np.asarray(base.B.T @ (self.weights*c))
        rhs -= rhs.sum()/self.n
        rhs[0] = 0
        phi = (self.factor(rhs.real)+1j*self.factor(rhs.imag)
               if np.iscomplexobj(F) else self.factor(rhs))
        grad = np.zeros_like(F, dtype=np.result_type(F,float))
        denom = np.zeros((self.n,3),float)
        for i,j,axis,d,area,direction in edges:
            edge_g = (phi[j]-phi[i])*direction/d
            grad[i,axis] += area*edge_g
            grad[j,axis] += area*edge_g
            denom[i,axis] += area
            denom[j,axis] += area
        sel = denom > 0
        grad[sel] /= denom[sel]
        return F-grad

    def project(self, fields):
        return [self.project_one(x) for x in fields]


@dataclass
class V148State:
    sea_z: np.ndarray
    total_z: np.ndarray
    copper_z: np.ndarray
    unit_j: np.ndarray
    sigma: np.ndarray
    residual: object
    weak_q: np.ndarray | None = None  # port-compatible W/m3 per unit-current pair


def _weak_q(C, X, G, GX, sigma, mass, vol, omega, mu):
    """Local BFZI port weak-form real dissipation, W/m3/(A^2).

    Uses exactly the inherited Z_rect(C,X)-Z_rect(G,GX)+Delta Z
    bilinear expression, retaining its local source-mass correction.
    """
    C=np.stack(C,axis=1)
    X=np.stack(X,axis=1)
    G=np.stack(G,axis=1)
    GX=np.stack(GX,axis=1)
    mat=(np.einsum("pic,pjc->pij",C,X,optimize=True)
         -np.einsum("pic,pjc->pij",G,GX,optimize=True))
    mat=.5*(mat+np.swapaxes(mat,1,2))
    w=(omega*mu)**2
    return w*sigma[:,None,None]*(mat.real+mass/vol[:,None,None])


def _oriented_mask(pts,mn,mx,coil,hx,hy,zp,zn,line,rad):
    """True coil-local support of original BFZI wet-cell source moments."""
    center=np.asarray(coil.center,float)
    rot=axes(coil)
    local=(pts-center)@rot
    ext=.5*(mx-mn)@np.abs(rot)
    face=(np.abs(np.abs(local[:,1])-hy)<=ext[:,1]+1e-12)
    footprint=(np.abs(local[:,0])<=hx+ext[:,0]) & (
        local[:,2]<=zp+ext[:,2]) & (local[:,2]>=-zn-ext[:,2])
    dist=cKDTree(np.asarray(line,float)).query(pts)[0]
    return footprint & face & (dist<=rad)


def _mass(data, calc, order=3,mask="legacy",source_radius=.06):
    """Original v148 local mass correction as fixed per-cell source moments."""
    rec = data["records"] or []
    pts = np.asarray(data["points"],float)
    vols = np.asarray(data["volumes"],float)
    n = len(rec)
    fams = data["families"]
    cfg = data["cfg"]
    sample = data["sample"]
    mn = np.asarray([r["cell_min"] for r in rec])
    mx = np.asarray([r["cell_max"] for r in rec])
    sizes = np.median(mx-mn,axis=0)
    rad = .5*np.hypot(sizes[0],sizes[2])
    y_extra = .5*sizes[1]
    if mask not in ("legacy","oriented","physical","all"):
        raise ValueError("BFZI weak-source mask invalid")
    masks = []
    for i,fam in enumerate(fams):
        lp,ldl = calc.base._source_line_for_family(fam,cfg)
        yc = float(np.asarray(fam.coil.center)[1])
        hy = float(cfg.get("BSR_PACKAGE_NORMAL_HALF_M",.0055))
        if len(fams) == 1:
            hx = float(sample.get("package_x_half_mm",102.))*1e-3
            zp = float(sample.get("package_z_pos_half_mm",102.))*1e-3
            zn = float(sample.get("package_z_neg_half_mm",102.))*1e-3
        else:
            prefix = f"coil{i+1}_"
            hx = float(sample.get(prefix+"package_x_half_mm",102.))*1e-3
            zp = float(sample.get(prefix+"package_z_pos_half_mm",102.))*1e-3
            zn = float(sample.get(prefix+"package_z_neg_half_mm",102.))*1e-3
        if mask in ("oriented","physical"):
            local_rad=rad if mask=="oriented" else source_radius
            masks.append(_oriented_mask(
                pts,mn,mx,fam.coil,hx,hy,zp,zn,lp,local_rad))
        else:
            face = ((mn[:,1]<=yc-hy+y_extra)&(mx[:,1]>=yc-hy-y_extra)
                    | (mn[:,1]<=yc+hy+y_extra)&(mx[:,1]>=yc+hy-y_extra))
            dist = calc.base._nearest_dist_xz(pts,lp,chunk=4096)
            masks.append((abs(pts[:,0])<=hx)&(pts[:,2]<=zp)
                         &(pts[:,2]>=-zn)&face&(dist<=rad))
    # Exact original BFZI local source mass correction over every wet
    # FV, without abrupt whole-cell support selection. Same formula,
    # just complete spatial integration rather than heuristic masking.
    if mask=="all":
        masks=[np.ones(n,dtype=bool) for _ in fams]
    out = np.zeros((n,len(fams),len(fams)),float)
    union = np.any(masks,axis=0)
    lead = []
    for fam in fams:
        geom,_ = calc.zlead_line_geom_for_coil(fam.coil,cfg)
        lead.append(geom)
    for p in np.flatnonzero(union):
        q,w = calc.base._record_quadrature(rec[p],order)
        if len(q) == 0:
            continue
        src = []
        for fam,geom in zip(fams,lead):
            Cq = calc.fs.air_projection_to_points_centerline_family_quadratic(
                q,fam,point_chunk=512)
            if len(geom.pos):
                Cq += calc.base._air_projection_to_points(
                    q,np.asarray(geom.pos,float),np.asarray(geom.dl,float),
                    chunk_points=512,chunk_line=1024)
            src.append(Cq)
        for i in range(len(fams)):
            for j in range(i,len(fams)):
                if masks[i][p] and masks[j][p]:
                    exact = np.sum(w*np.einsum("pk,pk->p",src[i],src[j]))
                    lump = vols[p]*np.dot(data["C"][i][p],data["C"][j][p])
                    out[p,i,j] = out[p,j,i] = exact-lump
    return out


class BFZIV148Adapter:
    """Original BFZI v148 two-port EM with locally temperature-dependent sea."""

    def __init__(self, data, calc, *, ambient=293.15, sea_alpha=.02,
                 copper_sigma=5.8e7, copper_alpha=.00393, sea_only=False,
                 heat_mode="j2",fv_faces="approx",source_mask="legacy",
                 source_radius=.06):
        if len(data["families"]) != 2:
            raise ValueError("v148 adapter requires two physical BFZI ports")
        if heat_mode not in ("j2","port-weak"):
            raise ValueError("sea heat mode must be 'j2' or 'port-weak'")
        if fv_faces not in ("approx","exact"):
            raise ValueError("BFZI FV face area mode must be approx or exact")
        if source_mask not in ("legacy","oriented","physical","all"):
            raise ValueError("BFZI weak-source mask must be legacy or oriented")
        if source_radius<=0 or not np.isfinite(source_radius):
            raise ValueError("BFZI physical source support radius must be positive")
        self.heat_mode=heat_mode
        self.source_mask=source_mask
        self.fv_faces=fv_faces
        self.calc = calc
        self.base = dict(data)
        self.ambient = float(ambient)
        self.alpha = float(sea_alpha)
        self.omega = float(data["omega"])
        self.mu = float(data["mu"])
        self.vol = np.asarray(data["volumes"],float)
        self.points = np.asarray(data["points"],float)
        self.sigma0 = float(data["sample"]["sigma_sw_s_per_m"])
        self.proj = calc.core.StrictFVScalarPotentialProjector(
            data["records"] or [],self.points,variant="fv_internal_only")
        if fv_faces=="exact":
            from .obb_faces import apply
            self.proj=apply(self.proj,data["records"],data["vol_meta"])
        self.mass = _mass(data,calc,mask=source_mask,
                          source_radius=source_radius)
        if sea_only:
            # M4 isothermal gate tests only the sea solver. Its expensive
            # Neumann/GMD air calculation is needed for the time history.
            self.air = np.zeros((2,2),float)
            self.copper = None
            return
        self.air_uH,_ = calc.compute_air_matrix_direct_zlead(data)
        self.air = np.asarray(self.air_uH,float)*1e-6
        lines = []
        assemblies = []
        for fam in data["families"]:
            geom,_ = calc._line_for_family_with_zlead(fam,data["cfg"])
            lines.append(SimpleNamespace(line_elements=np.asarray(geom.dl,float)))
            assemblies.append(SimpleNamespace(
                wire_radius=float(fam.coil.wire_radius_m),
                material=ConductorMaterial(
                    copper_sigma,
                    resistance_temperature_coefficient=copper_alpha)))
        self.copper = BFZISources(
            SimpleNamespace(assemblies=assemblies),
            self.omega/(2*np.pi),
            tuple(lines),
            self.air,
        )

    def solve(self, sea_t, cu_t):
        temp = np.asarray(sea_t,float)
        tc = np.asarray(cu_t,float)
        if temp.shape != self.vol.shape or tc.shape != (2,):
            raise ValueError("sea and copper temperature dimensions differ")
        sig = self.sigma0*(1+self.alpha*(temp-self.ambient))
        if np.min(sig)<=0 or not np.all(np.isfinite(sig)):
            raise ValueError("seawater sigma(T) must be finite and positive")
        gamma = 1j*self.omega*self.mu*sig
        vw = gamma*self.vol
        data = dict(self.base,vw=vw)
        proj = WeightedFV(self.proj,sig)
        C = list(data["C"])
        G = [a-b for a,b in zip(C,proj.project(C))]
        X,res = self.calc.core.solve_operator(
            data,proj,C,op_mode="KP",tol=2e-7,maxiter=180)
        infos = res["gmres_infos"]
        if any(int(v)!=0 for v in literal_eval(infos)):
            raise RuntimeError(f"BFZI sea GMRES did not converge: {infos}")
        P = proj.project(X)
        GX = [x-p for x,p in zip(X,P)]
        core = self.calc.core
        z = core.Z_rect(C,X,vw,self.mu,self.omega)
        z -= core.Z_rect(G,GX,vw,self.mu,self.omega)
        dz = -1j*self.omega*self.mu*np.einsum(
            "p,pij->ij",gamma,self.mass,optimize=True)
        z = core.sym_complex(z+dz)
        zcu = (np.zeros(2,complex) if self.copper is None
               else self.copper.conductor_internal_impedance(tc))
        total = z+1j*self.omega*self.air+np.diag(zcu)
        unit_j = np.stack([-gamma[:,None]*px for px in P],axis=1)
        weak=(_weak_q(C,X,G,GX,sig,self.mass,self.vol,self.omega,self.mu)
              if self.heat_mode=="port-weak" else None)
        return V148State(z,total,zcu,unit_j,sig,res,weak)

    def heat(self, state, currents):
        I = np.asarray(currents,complex)
        if I.shape != (2,):
            raise ValueError("currents must have two complex ports")
        if getattr(self,"heat_mode","j2")=="port-weak":
            if state.weak_q is None:
                raise ValueError("weak-form BFZI dissipation matrix missing")
            q=.5*np.real(np.einsum(
                "i,pij,j->p",I.conj(),state.weak_q,I,optimize=True))
            if np.min(q)<0:
                # The original BFZI weak local partition may have a small
                # negative cell even when its total port loss is passive.
                # Keep local Joule heat nonnegative and preserve the exact
                # sea port loss in the original finite-volume mesh.
                power=max(float(.5*np.vdot(I,state.sea_z@I).real),0.)
                q=np.maximum(q,0.)
                total=float(q@self.vol)
                if total>0:
                    q*=power/total
                elif power>1e-12:
                    raise ValueError("BFZI sea loss lacks a spatial Joule source")
        else:
            j=np.einsum("npc,p->nc",state.unit_j,I,optimize=True)
            q=.5*np.sum(abs(j)**2,axis=1)/state.sigma
            # Local source-mass correction also contributes real Joule heat.
            variance=np.real(np.einsum(
                "i,pij,j->p",I.conj(),self.mass,I,optimize=True))
            q+=.5*(self.omega*self.mu)**2*state.sigma*variance/self.vol
        cu=.5*state.copper_z.real*abs(I)**2
        return q,cu
