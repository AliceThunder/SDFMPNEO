"""Transient thermo-electromagnetic loop on the *original* BFZI FV mesh.

Each time step calls BFZIV148Adapter.solve(Tsea,Tcu), then re-solves
the two-port circuit, heats original seawater FV cells, and advances
sea/package/copper temperatures. Sea heat conduction uses the original
BFZI internal FV face geometry, without an unrelated mesh or FEM model.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

from .dynamic_et import ETStep


def circuit_power(voltage, currents, source_r, load_r):
    """Peak-phasor source input, receiver load output and WPT efficiency."""
    v = np.asarray(voltage, complex)
    i = np.asarray(currents, complex)
    if v.shape != (2,) or i.shape != (2,):
        raise ValueError("two complex ports required")
    if source_r <= 0 or load_r <= 0:
        raise ValueError("source and receiver load resistances must be positive")
    p_in = float(.5 * np.real(np.vdot(v, i)))
    p_load = float(.5 * load_r * abs(i[1])**2)
    p_src = float(.5 * source_r * abs(i[0])**2)
    eta = p_load/p_in if p_in > 0 else 0.
    return p_in, p_load, p_src, eta


class BFZIV148ET:
    def __init__(self, model, *, ambient=293.15, external=None,
                 sea_rhoc=4.0e6, sea_k=.6,
                 pkg_rhoc=1.5e6, cu_rhoc=3.45e6,
                 pkg_sea_g=.6, cu_pkg_g=.8, outer_h=10.):
        self.model = model
        self.amb = float(ambient)
        self.count = len(model.vol)
        self.n = self.count + 4
        self.pkg = slice(self.count,self.count+2)
        self.cu = slice(self.count+2,self.count+4)
        self.tree = cKDTree(model.points)
        self.external = (np.zeros((2,2),complex)
                         if external is None else np.asarray(external,complex))
        if self.external.shape != (2,2) or not np.isfinite(self.external).all():
            raise ValueError("external impedance must be a finite 2x2 matrix")
        if np.linalg.eigvalsh((self.external+self.external.conj().T)/2).min() < -1e-12:
            raise ValueError("external circuit must be passive")
        self.cap = np.zeros(self.n)
        self.cap[:self.count] = sea_rhoc*model.vol
        self.boxes = []
        self.cu_pkg_g = float(cu_pkg_g)
        self.pkg_links = []
        sample = model.base["sample"]
        cfg = model.base["cfg"]
        for p,fam in enumerate(model.base["families"]):
            prefix = f"coil{p+1}_" if len(model.base["families"])>1 else ""
            half = np.array([
                float(sample.get(prefix+"package_x_half_mm",102.))*1e-3,
                float(cfg.get("BSR_PACKAGE_NORMAL_HALF_M",.0055)),
                max(float(sample.get(prefix+"package_z_pos_half_mm",102.)),
                    float(sample.get(prefix+"package_z_neg_half_mm",102.)))*1e-3
            ])
            center = np.asarray(fam.coil.center,float)
            self.boxes.append((center,half))
            self.cap[self.count+p] = pkg_rhoc*8*np.prod(half)
            line = model.copper.lines[p].line_elements
            radius = model.copper.scene.assemblies[p].wire_radius
            self.cap[self.count+2+p] = (
                cu_rhoc*np.pi*radius**2*np.linalg.norm(line,axis=1).sum())
        if np.any(self.cap <= 0):
            raise ValueError("invalid heat capacities")
        rows,cols,vals = [],[],[]
        def link(i,j,cond):
            if cond <= 0:
                return
            rows.extend((i,j,i,j))
            cols.extend((i,j,j,i))
            vals.extend((cond,cond,-cond,-cond))
        for i,j,axis,dist,area,direction in model.proj.edges:
            link(i,j,sea_k*area/dist)
        for p,(center,half) in enumerate(self.boxes):
            dist=np.linalg.norm(
                np.maximum(np.abs(model.points-center)-half,0),axis=1)
            take=np.argsort(dist)[:min(12,self.count)]
            g = 1/np.maximum(dist[take],.005)
            g /= g.sum()
            links = [(int(cell),float(pkg_sea_g*weight))
                     for cell,weight in zip(take,g)]
            self.pkg_links.append(links)
            for cell,cond in links:
                link(cell,self.count+p,cond)
            link(self.count+p,self.count+2+p,cu_pkg_g)
        amb_g = np.zeros(self.n)
        for i,axis,side,area,is_outer in model.proj.boundary:
            if is_outer:
                amb_g[i] += outer_h*area
        self.amb_g = amb_g
        self.H = coo_matrix((vals,(rows,cols)),shape=(self.n,self.n)).tocsr()
        self.H += diags(amb_g)
        self._factor = {}

    def zeros(self):
        return np.zeros(self.n)

    def heat_flux(self,state):
        """Current heat flows, W: copper -> package -> seawater -> ambient.

        These exchanges use the same conductances as the FV heat operator.
        Positive values indicate outward flow in the named direction.
        """
        theta=np.asarray(state,float)
        if theta.shape!=(self.n,):
            raise ValueError("invalid thermal state")
        cp=self.cu_pkg_g*(theta[self.cu]-theta[self.pkg])
        ps=np.array([
            sum(cond*(theta[self.count+p]-theta[cell])
                for cell,cond in links)
            for p,links in enumerate(self.pkg_links)])
        amb=float(self.amb_g@theta)
        return cp,ps,amb

    def temperature_at(self,points,state):
        pts = np.asarray(points,float).reshape(-1,3)
        theta=np.asarray(state,float)
        if theta.shape != (self.n,):
            raise ValueError("invalid thermal state")
        out=self.amb+theta[self.tree.query(pts)[1]]
        for p,(center,half) in enumerate(self.boxes):
            inside=np.all(np.abs(pts-center)<=half,axis=1)
            out[inside]=self.amb+theta[self.count+p]
        return out

    def em(self,state,drive):
        theta=np.asarray(state,float)
        v=np.asarray(drive,complex)
        if theta.shape != (self.n,) or v.shape != (2,):
            raise ValueError("invalid thermal state or two-port voltage")
        m=self.model.solve(self.amb+theta[:self.count],
                           self.amb+theta[self.cu])
        I=np.linalg.solve(m.total_z+self.external,v)
        q,cu=self.model.heat(m,I)
        power=np.zeros(self.n)
        power[:self.count]=q*self.model.vol
        power[self.cu]=cu
        return m,I,power,q,cu

    def advance(self,state,drive,dt,*,iterations=16,tol=1e-5,
                budget_tol=1e-4,method="be"):
        if not np.isfinite(dt) or dt<=0:
            raise ValueError("positive thermal time step required")
        if not np.isfinite(budget_tol) or budget_tol<=0:
            raise ValueError("positive heat-balance tolerance required")
        if method not in ("be","cn"):
            raise ValueError("heat integrator must be be or cn")
        old=np.asarray(state,float)
        if old.shape!=(self.n,):
            raise ValueError("invalid thermal state")
        weight=1. if method=="be" else .5
        key=(float(dt),method)
        if key not in self._factor:
            self._factor[key]=factorized(
                (diags(self.cap)+dt*weight*self.H).tocsc())
        solve=self._factor[key]
        if method=="cn":
            _,_,old_heat,_,_=self.em(old,drive)
            rhs=self.cap*old+.5*dt*(old_heat-self.H@old)
        else:
            old_heat=None
            rhs=self.cap*old
        theta=old.copy()
        budget=np.inf
        for _ in range(iterations):
            _,_,heat,_,_=self.em(theta,drive)
            nxt=solve(rhs+dt*weight*heat)
            err=float(np.max(abs(nxt-theta)))
            if not np.isfinite(err):
                raise RuntimeError("BFZI thermal fixed point became nonfinite")
            if err<tol:
                _,_,heat_new,_,_=self.em(nxt,drive)
                if method=="cn":
                    bal=(self.cap*(nxt-old)/dt+
                         .5*(self.H@old+self.H@nxt)-
                         .5*(old_heat+heat_new))
                else:
                    bal=self.cap*(nxt-old)/dt+self.H@nxt-heat_new
                budget=max(float(np.max(abs(bal))),
                           float(abs(np.sum(bal))))
                if budget<=budget_tol:
                    return nxt
            theta=nxt
        raise RuntimeError(
            f"BFZI thermal fixed point did not converge: "
            f"delta={err:.4g} K, heat residual={budget:.4g} W "
            f"after {iterations} iterations")

    def trajectory(self,times,voltage_at,*,state=None):
        times=np.asarray(times,float)
        if (times.ndim!=1 or len(times)<2 or times[0]!=0
                or np.any(np.diff(times)<=0)):
            raise ValueError("times must start at zero and strictly increase")
        theta=self.zeros() if state is None else np.asarray(state,float).copy()
        history=[]
        for k,now in enumerate(times):
            v=np.asarray(voltage_at(float(now)),complex)
            if k:
                # Piecewise-constant drive on [t[k-1],t[k]).
                # The value exactly at the right boundary is used only for
                # the instantaneous EM output, never for the previous step.
                prev_v=np.asarray(voltage_at(float(times[k-1])),complex)
                theta=self.advance(theta,prev_v,float(now-times[k-1]))
            m,I,heat,q,cu=self.em(theta,v)
            j=np.einsum("npc,p->nc",m.unit_j,I,optimize=True)
            history.append(ETStep(
                float(now),theta.copy(),m.total_z,m.sea_z,I,
                float(np.dot(q,self.model.vol)),cu,float(heat.sum()),
                float(self.cap@theta),q.copy(),j.copy(),m.unit_j.copy()))
        return history
