"""Conservative BFZI 3-D thermal field driven by fractional neural EM.

Per geometry, the original BFZI cold unit-current/weak-Joule field and
finite-volume heat operator are exported once. Subsequent 3-D transient
T/J/q/Z/I rollout uses only a frozen fractional neural model, two-port
circuit equations and a sparse thermal solve; no BFZI Green solver is called.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import factorized

from .geom_memory import geom_feat,geom_port,heat_drive
from .frac_memory import zmat


def pack(z):
    a=np.asarray(z,complex)
    return np.array([a[0,0].real,a[0,0].imag,
                     a[0,1].real,a[0,1].imag,
                     a[1,1].real,a[1,1].imag])


class GeomHeat:
    """Spatially conservative sea/dual-3D-package/copper temperature evolution."""

    def __init__(self,geometry,z0,external,vol,sea_xyz,pkg_xyz,cap,H,
                 unit_j,weak_q,*,ambient=293.15,alpha=.02):
        self.geom=np.asarray(geometry,float)
        self.z0=np.asarray(z0,float)
        self.ext=np.asarray(external,complex)
        self.vol=np.asarray(vol,float)
        self.xyz=np.asarray(sea_xyz,float)
        self.pkg_xyz=np.asarray(pkg_xyz,float)
        self.cap=np.asarray(cap,float)
        self.H=csr_matrix(H)
        self.unit=np.asarray(unit_j,complex)
        self.weak=np.asarray(weak_q,float)
        self.amb=float(ambient)
        self.alpha=float(alpha)
        self.ns=len(self.vol)
        self.npkg=self.pkg_xyz.shape[1]
        self.cu=slice(self.ns+2*self.npkg,self.ns+2*self.npkg+2)
        self.port=geom_port(self.geom)
        self.sea0=zmat(self.z0)-1j*self.port.omega*self.port.air-np.diag(
            self.port.copper(np.full(2,self.amb)))
        if self.cap.shape!=(self.cu.stop,) or self.H.shape!=(len(self.cap),)*2:
            raise ValueError("original BFZI thermal FV dimensions mismatch")
        if self.unit.shape!=(self.ns,2,3) or self.weak.shape!=(self.ns,2,2):
            raise ValueError("original BFZI unit currents and weak q dimensions mismatch")
        if self.xyz.shape!=(self.ns,3) or self.pkg_xyz.shape!=(2,self.npkg,3):
            raise ValueError("BFZI world-space material coordinates mismatch")
        if np.any(self.cap<=0) or np.any(self.vol<=0):
            raise ValueError("positive BFZI material thermal capacities required")
        self._lu={}

    @classmethod
    def from_original(cls,em,et,external):
        """Original BFZI calls occur only once at cold geometry preparation."""
        ambient=et.amb
        cold=em.solve(np.full(len(em.vol),ambient),
                      np.full(2,ambient))
        return cls(
            np.asarray([
                *[float(f.coil.outer_radius_m) for f in em.base["families"]],
                *[float(f.coil.rounded_square_core_half_m)
                  for f in em.base["families"]],
                *[float(f.coil.rounded_square_alpha) for f in em.base["families"]],
                *[float(f.coil.turns) for f in em.base["families"]],
                *[float(f.coil.wire_radius_m) for f in em.base["families"]],
                float(np.linalg.norm(np.asarray(em.base["families"][0].coil.center)
                                     -np.asarray(em.base["families"][1].coil.center))),
                em.air[0,0],em.air[0,1],em.air[1,1],
                *np.asarray(et.cap[et.cu],float)
            ],float),
            pack(cold.total_z),external,em.vol,em.points,
            np.asarray(et.pkg_xyz),et.cap,et.H,
            cold.unit_j,cold.weak_q,ambient=ambient,alpha=em.alpha)

    def save(self,path):
        m=self.H.tocsr()
        np.savez_compressed(Path(path),geometry=self.geom,z0=self.z0,
            external=self.ext,vol=self.vol,sea_xyz=self.xyz,
            pkg_xyz=self.pkg_xyz,cap=self.cap,
            h_data=m.data,h_ind=m.indices,h_ptr=m.indptr,h_shape=m.shape,
            unit_j=self.unit,weak_q=self.weak,
            ambient=self.amb,alpha=self.alpha)

    @classmethod
    def load(cls,path):
        with np.load(path) as d:
            mat=csr_matrix((d["h_data"],d["h_ind"],d["h_ptr"]),
                           shape=tuple(d["h_shape"]))
            return cls(d["geometry"],d["z0"],d["external"],
                       d["vol"],d["sea_xyz"],d["pkg_xyz"],
                       d["cap"],mat,d["unit_j"],d["weak_q"],
                       ambient=float(d["ambient"]),alpha=float(d["alpha"]))

    def source(self,theta,I,sea):
        """Original BFZI cold weak source shape at current local sigma(T).

        Positive q is normalized to the *same* complex sea-port loss used by
        the two-port circuit, removing any volume/port energy discrepancy.
        """
        ratio=1+self.alpha*np.asarray(theta[:self.ns])
        if np.any(ratio<=0):
            raise ValueError("nonphysical seawater conductivity")
        raw=.5*np.real(np.einsum(
            "i,pij,j->p",np.conj(I),self.weak,I,optimize=True))
        raw=np.maximum(raw*ratio,0.)
        psea=max(float(.5*np.vdot(I,sea@I).real),0.)
        norm=float(raw@self.vol)
        q=raw*(psea/norm) if norm>1e-20 else np.zeros(self.ns)
        cu=self.port.copper(self.amb+theta[self.cu])
        pc=.5*cu.real*abs(I)**2
        heat=np.zeros(len(self.cap))
        heat[:self.ns]=q*self.vol
        heat[self.cu]=pc
        J=np.einsum("npc,p->nc",self.unit,I,optimize=True)*ratio[:,None]
        return heat,q,J,psea,pc

    def circuit(self,theta,dsea,voltage):
        sea=self.sea0+dsea
        cu=self.port.copper(self.amb+theta[self.cu])
        z=sea+1j*self.port.omega*self.port.air+np.diag(cu)
        I=np.linalg.solve(z+self.ext,np.array([voltage,0],complex))
        return sea,z,I

    def advance(self,theta,dt,voltage,dsea,*,steps=10,tol=2e-9):
        if dt<=0:
            raise ValueError("positive time step required")
        if dt not in self._lu:
            A=(diags(self.cap)+dt*self.H).tocsc()
            self._lu[dt]=factorized(A)
        solve=self._lu[dt]
        guess=np.asarray(theta,float).copy()
        for _ in range(steps):
            sea,z,I=self.circuit(guess,dsea,voltage)
            heat,_,_,_,_=self.source(guess,I,sea)
            nxt=solve(self.cap*theta+dt*heat)
            change=float(np.max(abs(nxt-guess)))
            guess=nxt
            if change<tol:
                break
        sea,z,I=self.circuit(guess,dsea,voltage)
        heat,q,J,ps,pc=self.source(guess,I,sea)
        bal=self.cap*(guess-theta)/dt+self.H@guess-heat
        return guess,z,I,q,J,ps,pc,bal

    @torch.no_grad()
    def rollout(self,net,ck,voltage,times):
        """Causal neural impedance + true 3-D FV heat on any BFZI sample."""
        t=np.asarray(times,float)
        v=np.asarray(voltage,float)
        if t[0]!=0 or v.shape!=t.shape or np.any(np.diff(t)<=0):
            raise ValueError("voltage/time histories must align from zero")
        if not ck.get("port_sea",False):
            raise ValueError("BFZI sea-delta fractional checkpoint required")
        g=torch.tensor(geom_feat(self.geom,self.z0)[None],
                       dtype=torch.float32)
        h=net.init_state(1)
        sc=np.asarray(ck["src_scale"],float)
        ys=np.asarray(ck["y_scale"],float)
        theta=np.zeros(len(self.cap))
        out=[];zout=[];iout=[];qout=[];jout=[];psea=[];pcu=[];gaps=[]
        zprev=zmat(self.z0)
        dsea=np.zeros((2,2),complex)
        for k,tm in enumerate(t):
            if k:
                dt=float(t[k]-t[k-1])
                u=heat_drive(float(v[k]),zprev,self.ext,self.port,
                             self.amb+theta[self.cu],self.cap[self.cu])
                h,y=net.step(
                    h,torch.tensor((u/sc)[None],dtype=torch.float32),
                    dt,g)
                dy=y[0].cpu().numpy()*ys
                dsea=zmat(np.asarray(dy[:6],float))
                theta,z,I,q,J,ps,pc,bal=self.advance(
                    theta,dt,float(v[k]),dsea)
                gaps.append(float(np.max(abs(bal))))
            else:
                sea,z,I=self.circuit(theta,dsea,float(v[k]))
                _,q,J,ps,pc=self.source(theta,I,sea)
                gaps.append(0.)
            zprev=z
            out.append(self.amb+theta.copy())
            zout.append(pack(z))
            iout.append(I)
            qout.append(q)
            jout.append(J)
            psea.append(ps)
            pcu.append(pc)
        return dict(temp_K=np.asarray(out),z6=np.asarray(zout),
                    current_A=np.asarray(iout),qsea_W_m3=np.asarray(qout),
                    Jsea_A_m2=np.asarray(jout),Psea_W=np.asarray(psea),
                    Pcu_W=np.asarray(pcu),fv_node_residual_W=np.asarray(gaps))
