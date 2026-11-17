"""Geometry-aware mutual cold port using a dipole-trend inductive descriptor."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .geom_init import _rbf
from .pose import _rot

W=2*np.pi*200000


def _area(a,c):
    # Spiral enclosed-area trend, using its 10 affine rounded-square turns.
    r0=.08885-.001*c
    k0=.001*c
    s=0.
    for j in range(10):
        r=r0-a*.005*(j+.5)
        k=k0-(1-a)*.005*(j+.5)
        s+=np.pi*r*r+8*k*r+4*k*k
    return s


def feat(case):
    g=case['geometry']
    vec=.001*np.asarray(g['shift_mm'],float)+[0.,.04,0.]
    d=np.linalg.norm(vec)
    rot=_rot('z',g['yaw_deg'])@_rot('y',g['pitch_deg'])@_rot('x',g['tilt_deg'])
    n=rot@np.array([0.,1.,0.])
    m=(3*vec[1]*np.dot(n,vec)/d**2-n[1])/d**3
    p=m*_area(g['alpha1'],g['core1_mm'])*_area(g['alpha2'],g['core2_mm'])
    return np.r_[p,g['alpha1'],g['alpha2'],g['core1_mm'],
                  g['core2_mm'],100*vec,n]


class MutFix:
    def __init__(self,data):
        self.idx=np.asarray(data['idx'],int)
        self.x=np.asarray(data['x'],float)
        self.mu=np.asarray(data['mu'],float)
        self.std=np.asarray(data['std'],float)
        self.ym=np.asarray(data['ym'],float)
        self.beta=np.asarray(data['beta'],float)
        self.ell=float(data['ell'])
        self.lam=float(data['lam'])
        self.p0=float(data['p0'])
        self.phys=bool(data['phys'])

    @classmethod
    def fit(cls,rows,idx=None,ell=12.,lam=.001,phys=True):
        if idx is None:
            idx=np.arange(11)
        idx=np.asarray(idx,int)
        feat0=np.stack([feat(r['case']) for r in rows])
        y=np.stack([r['y'][[2,3,7]] for r in rows])
        p0=float(np.mean(feat0[:,0]))
        q=feat0[:,0:1]/p0 if phys else np.ones((len(rows),1))
        a=feat0[:,idx]
        mu=a.mean(0)
        std=np.maximum(a.std(0),1e-7)
        x=(a-mu)/std
        y0=y/q
        ym=y0.mean(0)
        beta=np.linalg.solve(_rbf(x,x,ell)+lam*np.eye(len(x)),y0-ym)
        return cls(dict(idx=idx,x=x,mu=mu,std=std,
                        ym=ym,beta=beta,ell=ell,lam=lam,
                        p0=p0,phys=phys))

    def predict(self,case):
        a=feat(case)
        z=(a[self.idx]-self.mu)/self.std
        pred=self.ym+(_rbf(z[None],self.x,self.ell)@self.beta)[0]
        return pred*(a[0]/self.p0 if self.phys else 1.)

    def apply(self,case,init):
        out=dict(init)
        z=np.asarray(init['z0'],float).copy()
        air=np.asarray(init['air'],float).copy()
        m=self.predict(case)
        z[[2,3]]=m[:2]
        air[1]=m[2]
        out['z0']=z
        out['air']=air
        return out

    def save(self,path):
        np.savez_compressed(path,idx=self.idx,x=self.x,mu=self.mu,
            std=self.std,ym=self.ym,beta=self.beta,ell=self.ell,
            lam=self.lam,p0=self.p0,phys=int(self.phys))

    @classmethod
    def load(cls,path):
        with np.load(Path(path)) as f:
            return cls({k:f[k] for k in f.files})
