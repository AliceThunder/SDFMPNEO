"""Geometry-only cold port and shared 32-mode thermal operator emulator.

No BFZI solve, geometry-specific FV grid or POD basis is needed at query time.
The frozen training atlas contains geometry-disjoint cold anchors.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def encode(case):
    """The user-controlled M4 geometry, pose, package and thermal parameters."""
    g=case['geometry']
    t=case['thermal']
    vals=[g['alpha1'],g['alpha2'],
          g['core1_mm'],g['core2_mm'],*g['shift_mm'],
          g['tilt_deg'],g['yaw_deg'],g['pitch_deg'],
          g['box1_mm'],g['box2_mm'],
          case['alpha_per_K'],t['ins_k'],t.get('ins_t',.0004),
          t['pkg_k'],t['pkg_sea_h'],
          1. if case['donor']=='g2' else 0.]
    out=np.asarray(vals,float)
    if out.shape!=(18,) or not np.isfinite(out).all():
        raise ValueError('expected 18 finite geometry/material input scalars')
    return out


def _rbf(a,b,ell):
    delta=a[:,None,:]-b[None,:,:]
    d=np.sqrt(np.mean(delta*delta,axis=-1)+1e-20)
    r=np.sqrt(3.)*d/ell
    return (1+r)*np.exp(-r)


def _w(a,b,ell):
    s=_rbf(a,b,ell)
    return s/np.maximum(s.sum(1,keepdims=True),1e-20)


class GeomInit:
    """Cold-state kernel interpolant with conservative SPD thermal atlas."""

    def __init__(self,raw):
        self.x=np.asarray(raw['x'],float)
        self.mu=np.asarray(raw['mu'],float)
        self.std=np.asarray(raw['std'],float)
        self.y=np.asarray(raw['y'],float)
        self.ym=np.asarray(raw['ym'],float)
        self.ys=np.asarray(raw['ys'],float)
        self.beta=np.asarray(raw['beta'],float)
        self.k=np.asarray(raw['k'],float)
        self.src=np.asarray(raw['src'],float)
        self.ext=np.asarray(raw['ext'],complex)
        self.ell=float(raw['ell'])
        self.lam=float(raw['lam'])
        self.sx=np.asarray(raw.get('sx',[]),float)
        self.smu=np.asarray(raw.get('smu',np.zeros(3)),float)
        self.sstd=np.asarray(raw.get('sstd',np.ones(3)),float)
        self.sy=np.asarray(raw.get('sy',np.zeros(2)),float)
        self.sbeta=np.asarray(raw.get('sbeta',np.zeros((0,2))),float)
        self.sell=float(raw.get('sell',1.))
        self.slam=float(raw.get('slam',0.))
        self.rank=int(self.k.shape[1])
        if (self.x.shape[1]!=18 or self.rank<5 or
                self.k.shape[1:]!=(self.rank,self.rank) or
                self.src.shape[1:]!=(self.rank,3)):
            raise ValueError('invalid cold/thermal atlas dimensions')

    @staticmethod
    def fit(x,y,k,src,ext,ell=1.5,lam=.03):
        x=np.asarray(x,float)
        y=np.asarray(y,float)
        mu=x.mean(0)
        std=np.maximum(x.std(0),np.array([.02]*2+[1.]*2+[1.]*3+
                              [1.]*3+[2.]*2+[.001,.01,.00001,.01,1.,.15]))
        x=(x-mu)/std
        ym=y.mean(0)
        ys=np.maximum(y.std(0),1e-10)
        z=(y-ym)/ys
        h=_rbf(x,x,ell)
        beta=np.linalg.solve(h+lam*np.eye(len(x)),z)
        return GeomInit(dict(x=x,mu=mu,std=std,y=y,ym=ym,ys=ys,
                             beta=beta,k=k,src=src,ext=ext,
                             ell=ell,lam=lam))

    def fit_self(self,rows,ell,lam):
        x=[];y=[]
        for row in rows:
            g=row['case']['geometry']
            for j in (1,2):
                x.append([g[f'alpha{j}'],g[f'core{j}_mm'],g[f'box{j}_mm']])
                idx=0 if j==1 else 4
                y.append(row['y'][idx:idx+2])
        x=np.asarray(x,float)
        y=np.asarray(y,float)
        self.smu=x.mean(0)
        self.sstd=np.maximum(x.std(0),[.01,1.,2.])
        self.sx=(x-self.smu)/self.sstd
        self.sy=y.mean(0)
        scale=np.maximum(y.std(0),1e-8)
        self.sbeta=np.linalg.solve(_rbf(self.sx,self.sx,ell)+
                   lam*np.eye(len(x)),(y-self.sy)/scale)*scale
        self.sell=float(ell)
        self.slam=float(lam)
        return self

    def predict(self,case,external=None):
        a=(encode(case)-self.mu)/self.std
        a=a[None]
        h=_rbf(a,self.x,self.ell)
        y=self.ym+(h@self.beta)[0]*self.ys
        w=_w(a,self.x,self.ell)[0]
        # Geometric convex interpolation retains symmetry and nonnegative
        # thermal diffusion eigenvalues without a new FV heat construction.
        k=np.einsum('n,nij->ij',w,self.k,optimize=True)
        k=.5*(k+k.T)
        src=np.einsum('n,nij->ij',w,self.src,optimize=True)
        # The external matching network belongs to a known circuit family.
        # Do not interpolate the calibrated external reactance across families.
        donor=(self.x[:,-1]*self.std[-1]+self.mu[-1]>.5)
        family=case['donor']=='g2'
        inds=np.flatnonzero(donor==family)
        ext=self.ext[inds[0]] if len(inds) else self.ext[np.argmax(w)]
        z=y[:6].copy()
        if self.sx.size:
            g=case['geometry']
            q=np.array([[g[f'alpha{i}'],g[f'core{i}_mm'],g[f'box{i}_mm']]
                        for i in (1,2)])
            t=_rbf((q-self.smu)/self.sstd,self.sx,self.sell)
            values=self.sy+t@self.sbeta
            z[[0,1]]=values[0]
            z[[4,5]]=values[1]
        air=y[6:9]
        cap=np.exp(np.clip(y[9:14],-30,30))
        if external is not None:
            ext=np.asarray(external,complex)
            if ext.shape!=(2,2):
                raise ValueError("external circuit must be 2x2 complex matrix")
        return dict(z0=z,air=air,cap=cap,k=k,src=src,ext=ext,
                    norm_weights=w)

    def save(self,path):
        np.savez_compressed(path,x=self.x,mu=self.mu,std=self.std,
              y=self.y,ym=self.ym,ys=self.ys,beta=self.beta,
              k=self.k,src=self.src,ext=self.ext,
              ell=self.ell,lam=self.lam,sx=self.sx,smu=self.smu,
              sstd=self.sstd,sy=self.sy,sbeta=self.sbeta,
              sell=self.sell,slam=self.slam)

    @classmethod
    def load(cls,path):
        with np.load(Path(path)) as f:
            raw={k:f[k] for k in f.files}
        return cls(raw)
