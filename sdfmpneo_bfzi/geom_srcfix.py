"""Geometry-only reduced seawater Joule-source field from frozen cold anchors."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .geom_init import _rbf,encode


class SrcFix:
    def __init__(self,raw):
        self.idx=np.asarray(raw['idx'],int)
        self.x=np.asarray(raw['x'],float)
        self.mu=np.asarray(raw['mu'],float)
        self.std=np.asarray(raw['std'],float)
        self.ym=np.asarray(raw['ym'],float)
        self.beta=np.asarray(raw['beta'],float)
        self.ell=float(raw['ell'])
        self.lam=float(raw['lam'])

    @classmethod
    def fit(cls,rows,idx=None,ell=10.,lam=.001):
        if idx is None:
            # Only physical geometry, relative pose, and package size.
            idx=np.arange(12)
        idx=np.asarray(idx,int)
        x=np.stack([encode(r['case'])[idx] for r in rows])
        src=np.stack([r['src'].ravel() for r in rows])
        mu=x.mean(0);std=np.maximum(x.std(0),.01)
        x=(x-mu)/std
        ym=src.mean(0)
        beta=np.linalg.solve(_rbf(x,x,ell)+lam*np.eye(len(x)),src-ym)
        return cls(dict(idx=idx,x=x,mu=mu,std=std,ym=ym,
                        beta=beta,ell=ell,lam=lam))

    def predict(self,case):
        x=(encode(case)[self.idx]-self.mu)/self.std
        z=self.ym+(_rbf(x[None],self.x,self.ell)@self.beta)[0]
        return z.reshape(-1,3)

    def apply(self,case,init):
        out=dict(init)
        z=self.predict(case)
        if z.shape!=out['src'].shape:
            raise ValueError('source ranks must agree')
        out['src']=z
        return out

    def save(self,path):
        np.savez_compressed(path,idx=self.idx,x=self.x,mu=self.mu,
            std=self.std,ym=self.ym,beta=self.beta,
            ell=self.ell,lam=self.lam)

    @classmethod
    def load(cls,path):
        with np.load(Path(path)) as f:
            return cls({k:f[k] for k in f.files})
