"""Port-specific cold two-port self-impedance emulator with air/sea separation.

A learned model for each native terminal preserves BFZI lead differences.
No electromagnetic solve is used to predict a new geometry.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .geom_init import _rbf


class PortSelf:
    def __init__(self, raw):
        self.x=np.asarray(raw['x'],float)
        self.mu=np.asarray(raw['mu'],float)
        self.std=np.asarray(raw['std'],float)
        self.mean=np.asarray(raw['mean'],float)
        self.beta=np.asarray(raw['beta'],float)
        self.ell=float(raw['ell'])
        self.lam=float(raw['lam'])
        self.sea=np.asarray(raw['sea'],float)
        if (self.x.ndim!=3 or self.x.shape[0]!=2 or
            self.beta.shape!=(2,self.x.shape[1],2) or
            self.sea.shape!=(2,)):
            raise ValueError("invalid terminal-specific cold port model")

    @staticmethod
    def input(case, dims):
        g=case['geometry']
        return np.array([[g[f'alpha{j}'],g[f'core{j}_mm'],g[f'box{j}_mm']]
                         for j in (1,2)],float)[:,:dims]

    @classmethod
    def fit(cls, rows, dims=2, ell=5., lam=.01):
        if dims not in (2,3):
            raise ValueError("terminal model supports two or three geometric features")
        x=np.stack([cls.input(r['case'],dims) for r in rows],axis=1)
        y=np.stack([[r['y'][[0,1]],r['y'][[4,5]]] for r in rows],axis=1)
        mu=x.mean(1)
        std=np.maximum(x.std(1),np.array([.01,1.,2.])[:dims])
        norm=(x-mu[:,None,:])/std[:,None,:]
        mean=y.mean(1)
        beta=np.stack([np.linalg.solve(
            _rbf(norm[j],norm[j],ell)+lam*np.eye(len(rows)),
            y[j]-mean[j]) for j in range(2)])
        # Original cold port: Z0_self = Zair_self + Zsea_self + Zcopper.
        # Reconstruct L_air from learned total reactance using the
        # calibrated cold sea+copper imaginary part in each terminal.
        omega=2*np.pi*200000
        sea=np.array([np.median([r['y'][j]-omega*r['y'][k]
                       for r in rows]) for j,k in ((1,6),(5,8))])
        return cls(dict(x=norm,mu=mu,std=std,mean=mean,
                        beta=beta,ell=ell,lam=lam,sea=sea))

    def predict(self,case):
        x=(self.input(case,self.x.shape[2])-self.mu)/self.std
        y=np.stack([self.mean[j]+(_rbf(x[j:j+1],self.x[j],self.ell)
                      @self.beta[j])[0] for j in range(2)])
        return y

    def apply(self,case,init):
        x=dict(init)
        z=np.asarray(x['z0'],float).copy()
        air=np.asarray(x['air'],float).copy()
        val=self.predict(case)
        z[[0,1]]=val[0]
        z[[4,5]]=val[1]
        air[[0,2]]=np.maximum(
            (val[:,1]-self.sea)/(2*np.pi*200000),1.e-9)
        x['z0']=z
        x['air']=air
        return x

    def save(self,path):
        np.savez_compressed(path,x=self.x,mu=self.mu,std=self.std,
            mean=self.mean,beta=self.beta,ell=self.ell,lam=self.lam,
            sea=self.sea)

    @classmethod
    def load(cls,path):
        with np.load(Path(path)) as f:
            return cls({k:f[k] for k in f.files})
