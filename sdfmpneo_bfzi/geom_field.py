"""Continuous BFZI seawater/package temperature decoder for variable FV grids.

Original BFZI geometry and finite-volume coordinates define local inputs.
A shared fractional-memory model supplies the eight transient material
observables. Training learns dimensionless temperature *shapes* only.
The reference-grid reconstruction preserves predicted material means and
hotspots at every time, including powered-off thermal memory.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .geom_memory import geom_feat

SCALE=np.array([.2,5.,20.,20.,30.,30.,20.,100.])
N_FEAT=53


def space_feat(x,geom,z0,pkg_xyz,material):
    """Original global BFZI xyz, coil distances and 3-D package coordinates."""
    p=np.asarray(x,float).reshape(-1,3)
    g=np.asarray(geom,float)
    cells=np.asarray(pkg_xyz,float)
    if g.shape!=(16,) or cells.ndim!=3 or cells.shape[0]!=2 or cells.shape[2]!=3:
        raise ValueError("BFZI geometric descriptors/package cells required")
    mid=cells.mean(axis=1)
    half=np.maximum(np.max(abs(cells-mid[:,None,:]),axis=1),.001)
    v=[p/np.array([.3,.2,.3])]
    for k in range(2):
        dx=p-mid[k]
        rel=dx/half[k]
        core=g[2+k]
        r=np.sqrt(np.maximum(abs(dx[:,0])-core,0.)**2+
                  np.maximum(abs(dx[:,2])-core,0.)**2)
        outer=max(g[k]-core,1e-6)
        v.append(np.stack((
            (r-outer)/.1, dx[:,1]/.2,
            r/.15, (np.linalg.norm(
                np.maximum(abs(rel)-1.,0.),axis=1))),axis=-1))
    for k in range(2):
        v.append((p-mid[k])/half[k])
        v.append(np.linalg.norm(
            (p-mid[k])/half[k],axis=1,keepdims=True))
    gg=np.broadcast_to(geom_feat(g,z0), (len(p),22))
    v.append(gg)
    mat=np.zeros((len(p),3))
    mat[:,int(material)]=1.
    v.append(mat)
    arr=np.concatenate(v,axis=1)
    if arr.shape[1]!=44:
        raise RuntimeError("original BFZI coordinate dimension mismatch")
    return arr.astype(np.float32)


def time_feat(summary,t):
    """Condition spatial shape on the current causal eight material states."""
    a=np.asarray(summary,float)
    if a.shape!=(8,):
        raise ValueError("eight shared BFZI neural material states required")
    return np.r_[np.log1p(np.maximum(a,0))/np.log1p(SCALE),
                 float(t)/600.].astype(np.float32)


class SpaceTemp(nn.Module):
    """Query original BFZI sea/package FV points or arbitrary 3-D coordinates."""

    def __init__(self,hidden=96):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(N_FEAT,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),
            nn.Linear(hidden,1))

    def forward(self,x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def shape(net,x,batch=32768):
    """Predict a positive bounded dimensionless spatial temperature field."""
    f=np.asarray(x,np.float32)
    with torch.no_grad():
        return np.concatenate([
            net(torch.from_numpy(f[j:j+batch])).numpy()
            for j in range(0,len(f),batch)],axis=0)


def calibrate(raw,mean,hot,weight=None):
    """Nonnegative field with exactly the material mean and max on the FV grid."""
    r=np.asarray(raw,float)
    if r.ndim!=1 or not len(r):
        raise ValueError("nonempty physical FV shape required")
    hot=max(float(hot),0.)
    if hot<1e-10:
        return np.zeros_like(r)
    target=np.clip(float(mean)/hot,1/len(r),1.)
    w=(np.ones(len(r)) if weight is None else np.asarray(weight,float))
    w=w/w.sum()
    # Physical peak/minimum support: monotone power map, renormalized to 1.
    r=(r-r.min())/max(float(r.max()-r.min()),1e-10)
    if np.max(r)<1e-10:
        return np.full(len(r),min(float(mean),hot))
    lo,hi=.005,256.
    for _ in range(28):
        mid=.5*(lo+hi)
        current=float(w@(r**mid))
        if current>target:
            lo=mid
        else:
            hi=mid
    return hot*r**(.5*(lo+hi))


def infer_field(net,static,summary,times,sea_vol=None):
    """FV grid Tsea/Tpkg and continuous query values at supplied xyz positions.

    static: 3 material feature tensors in order [sea,pkg1,pkg2], already
    evaluated on desired coordinates. Full-grid calibration uses those points.
    """
    seas=[];boxes=[]
    for a,t in zip(np.asarray(summary,float),np.asarray(times,float)):
        dyn=time_feat(a,t)
        raw=[shape(net,np.column_stack(
                 (x,np.broadcast_to(dyn,(len(x),len(dyn))))))
             for x in static]
        sea=calibrate(raw[0],a[0],a[1],sea_vol)
        pk=[calibrate(raw[j+1],a[j+2],a[j+4])
            for j in range(2)]
        seas.append(sea)
        boxes.append(pk)
    return np.asarray(seas),np.asarray(boxes)
