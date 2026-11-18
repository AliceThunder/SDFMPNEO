"""Geometry-generalized Cu/package contact-mode coefficients."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .geom_init import _rbf


def xport(case):
    g=case["geometry"]
    return np.array([[g[f"alpha{j}"],g[f"core{j}_mm"],g[f"box{j}_mm"]]
                     for j in (1,2)],dtype=float)


class CuLink:
    """Normalized package contact functionals, one kernel per coil."""

    def __init__(self,raw):
        for key in ("x","mu","std","mean","beta","ell"):
            setattr(self,key,np.asarray(raw[key],dtype=float))
        if (self.x.ndim!=3 or self.x.shape[0]!=2 or
                self.x.shape[2]!=3 or self.beta.shape!=(2,len(self.x[0]),7)):
            raise ValueError("expected two 3D inputs and seven contact coefficients")

    @classmethod
    def fit(cls,rows,ell=(8.,8.),lam=(1e-4,1e-5)):
        x=np.stack([xport(case) for case,y in rows],axis=1)
        target=np.stack([y for case,y in rows],axis=1)
        mu=x.mean(1)
        std=np.maximum(x.std(1),np.array([.01,1.,2.]))
        x=(x-mu[:,None,:])/std[:,None,:]
        ym=target.mean(1)
        beta=[]
        for j in range(2):
            kernel=_rbf(x[j],x[j],ell[j])
            beta.append(np.linalg.solve(
                kernel+lam[j]*np.eye(len(rows)),target[j]-ym[j]))
        return cls(dict(x=x,mu=mu,std=std,mean=ym,beta=np.stack(beta),ell=ell))

    def predict(self,case):
        x=(xport(case)-self.mu)/self.std
        return np.stack([self.mean[j]+
              (_rbf(x[j:j+1],self.x[j],self.ell[j])@self.beta[j])[0]
              for j in (0,1)])

    def save(self,path):
        np.savez_compressed(path,x=self.x,mu=self.mu,std=self.std,
                            mean=self.mean,beta=self.beta,ell=self.ell)

    @classmethod
    def load(cls,path):
        with np.load(Path(path)) as f:
            return cls({k:f[k] for k in f.files})
