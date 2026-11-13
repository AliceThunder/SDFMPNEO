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
from scipy.spatial import cKDTree

from .geom_memory import geom_feat,geom_port,heat_drive,G_SCALE
from .pose import desc,isinside,default_desc
from .frac_memory import zmat


def pack(z):
    a=np.asarray(z,complex)
    return np.array([a[0,0].real,a[0,0].imag,
                     a[0,1].real,a[0,1].imag,
                     a[1,1].real,a[1,1].imag])


class GeomHeat:
    """Spatially conservative sea/dual-3D-package/copper temperature evolution."""

    def __init__(self,geometry,z0,external,vol,sea_xyz,pkg_xyz,cap,H,
                 unit_j,weak_q,*,ambient=293.15,alpha=.02,
                 pose=None,pkg_axes=None,pkg_mid=None,pkg_half=None):
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
        self._sea_tree=cKDTree(self.xyz)
        self._pkg_tree=[cKDTree(p) for p in self.pkg_xyz]
        mid=self.pkg_xyz.mean(axis=1)
        self._pkg_axes=(np.tile(np.eye(3),(2,1,1)) if pkg_axes is None
                        else np.asarray(pkg_axes,float))
        self._pkg_mid=(mid if pkg_mid is None else np.asarray(pkg_mid,float))
        rel=np.stack([(self.pkg_xyz[j]-self._pkg_mid[j])@
                      self._pkg_axes[j] for j in range(2)])
        span=np.max(abs(rel),axis=1)
        self._pkg_half=(np.maximum(span*np.array([4./3,2.,4./3]),1e-6)
                        if pkg_half is None else np.asarray(pkg_half,float))
        self.pose=(None if pose is None else np.asarray(pose,float))
        if (self._pkg_axes.shape!=(2,3,3) or self._pkg_mid.shape!=(2,3)
                or self._pkg_half.shape!=(2,3)
                or np.any(self._pkg_half<=0)):
            raise ValueError("BFZI oriented dual-package geometry invalid")

    @staticmethod
    def _interp(tree,values,points):
        """Continuous inverse-distance reconstruction on original FV centers."""
        a=np.asarray(values)
        p=np.asarray(points,float).reshape(-1,3)
        if len(a)!=tree.n:
            raise ValueError("field values must match the original BFZI FV cells")
        d,idx=tree.query(p,k=min(8,tree.n))
        d=np.asarray(d).reshape(len(p),-1)
        idx=np.asarray(idx).reshape(len(p),-1)
        w=1/np.maximum(d,1e-9)**2
        w/=w.sum(axis=1,keepdims=True)
        shape=(len(p),len(w[0]))+(1,)*(a.ndim-1)
        y=(a[idx]*w.reshape(shape)).sum(axis=1)
        zero=(d[:,0]<1e-10)
        if np.any(zero):
            y[zero]=a[idx[zero,0]]
        return y

    def query_temp(self,xyz,state):
        """Absolute temperature at arbitrary 3-D sea or package coordinates."""
        p=np.asarray(xyz,float).reshape(-1,3)
        temp=np.asarray(state,float)
        if temp.shape!=(len(self.cap),):
            raise ValueError("full three-material BFZI temperature field required")
        ans=self._interp(self._sea_tree,temp[:self.ns],p)
        for j in range(2):
            inside=isinside(p,self._pkg_mid[j],self._pkg_half[j],
                            self._pkg_axes[j])
            if np.any(inside):
                vals=temp[self.ns+j*self.npkg:self.ns+(j+1)*self.npkg]
                ans[inside]=self._interp(self._pkg_tree[j],vals,p[inside])
        return ans

    def query_sea(self,xyz,values):
        """Interpolate complex J vectors or weak q scalars at arbitrary seawater points."""
        return self._interp(self._sea_tree,values,xyz)

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
            cold.unit_j,cold.weak_q,ambient=ambient,alpha=em.alpha,
            pose=desc([f.coil for f in em.base["families"]],et.boxes),
            pkg_axes=np.asarray(et.pkg_axes),
            pkg_mid=np.asarray([b[0] for b in et.boxes]),
            pkg_half=np.asarray([b[1] for b in et.boxes]))

    def save(self,path):
        m=self.H.tocsr()
        np.savez_compressed(Path(path),geometry=self.geom,z0=self.z0,
            external=self.ext,vol=self.vol,sea_xyz=self.xyz,
            pkg_xyz=self.pkg_xyz,cap=self.cap,
            h_data=m.data,h_ind=m.indices,h_ptr=m.indptr,h_shape=m.shape,
            unit_j=self.unit,weak_q=self.weak,
            pose=self.pose if self.pose is not None else np.empty(0),
            pkg_axes=self._pkg_axes,pkg_mid=self._pkg_mid,
            pkg_half=self._pkg_half,
            ambient=self.amb,alpha=self.alpha)

    @classmethod
    def load(cls,path):
        with np.load(path) as d:
            mat=csr_matrix((d["h_data"],d["h_ind"],d["h_ptr"]),
                           shape=tuple(d["h_shape"]))
            pose=d["pose"] if "pose" in d and d["pose"].size else None
            return cls(d["geometry"],d["z0"],d["external"],
                       d["vol"],d["sea_xyz"],d["pkg_xyz"],
                       d["cap"],mat,d["unit_j"],d["weak_q"],
                       ambient=float(d["ambient"]),alpha=float(d["alpha"]),
                       pose=pose,
                       pkg_axes=d["pkg_axes"] if "pkg_axes" in d else None,
                       pkg_mid=d["pkg_mid"] if "pkg_mid" in d else None,
                       pkg_half=d["pkg_half"] if "pkg_half" in d else None)

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
        if net.ng>len(G_SCALE):
            pose=self.pose if self.pose is not None else default_desc(self.geom)
        else:
            pose=None
        g=torch.tensor(geom_feat(self.geom,self.z0,pose)[None],
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
