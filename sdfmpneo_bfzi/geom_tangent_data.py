"""Derive BFZI 8-mode seawater-Z tangents from saved original direct histories.

This creates physical-trajectory supervision with *zero new perturbed-EM
solves*.  The identifiable heat-mode subspace is reported explicitly; this
is an inexpensive alternative teacher for cross-geometry tangent transfer.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .geom_tangent import SeaTangent,basis


def data(head,modes,path):
    with np.load(Path(path)) as d:
        t=np.asarray(d["temp_K"],float)
        z=np.asarray(d["z6"],float)
        if "geometry" in d and not np.allclose(
                d["geometry"],head.geom,rtol=1e-9,atol=1e-10):
            raise ValueError("BFZI trajectory and cold geometry differ")
    if (t.ndim!=3 or t.shape[2]!=len(head.cap) or
            z.shape!=t.shape[:2]+(6,) or t.shape[1]<2):
        raise ValueError("original BFZI time/3D temperature/Z6 shape mismatch")
    x=np.einsum("ctn,nj->ctj",t[:,:,:head.ns]-head.amb,
                head.vol[:,None]*modes,optimize=True)
    y=z-head.z0[None,None,:]
    cu0=head.port.copper(np.full(2,head.amb))
    cu=np.stack([head.port.copper(v[head.cu])
                 for v in t.reshape(-1,t.shape[-1])])
    dc=(cu-cu0).reshape(*t.shape[:2],2)
    y=y.copy()
    y[:,:,0]-=dc[:,:,0].real
    y[:,:,1]-=dc[:,:,0].imag
    y[:,:,4]-=dc[:,:,1].real
    y[:,:,5]-=dc[:,:,1].imag
    return x[:,1:],y[:,1:]


def solve(x,y,ratio):
    """Dimension-balanced truncated/Tikhonov multi-output heat sensitivity."""
    scale=np.maximum(np.linalg.norm(x,axis=0),1e-14)
    a=x/scale
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    if s[0]<=0:
        raise ValueError("original BFZI teacher has no sea-temperature variation")
    floor=max(float(s[0])*1e-8,1e-14)
    gain=np.where(s>floor,s/(s*s+(ratio*s[0])**2),0.)
    j=(vt.T*gain)@(u.T@y)
    rank=int(np.count_nonzero(s>s[0]*1e-3))
    return j/scale[:,None],rank


def fit(head,phi,teacher,rank=8):
    """Fit a 2x2 reciprocal Zsea Jacobian with geometry-contained validation."""
    modes=basis(head.vol,np.asarray(phi)[:head.ns],rank)
    x,y=data(head,modes,teacher)
    if len(x)<2:
        raise ValueError("at least two original BFZI histories needed")
    choices=(1e-5,1e-3,1e-2,.1)
    scores=[]
    for r in choices:
        j,_=solve(x[:-1].reshape(-1,len(modes.T)),
                  y[:-1].reshape(-1,6),r)
        delta=x[-1]@j-y[-1]
        scores.append(float(np.linalg.norm(delta)/
                            max(np.linalg.norm(y[-1]),1e-15)))
    k=int(np.argmin(scores))
    coef,effective=solve(x.reshape(-1,len(modes.T)),
                         y.reshape(-1,6),choices[k])
    jac=np.zeros((len(coef),2,2),complex)
    jac[:,0,0]=coef[:,0]+1j*coef[:,1]
    jac[:,0,1]=jac[:,1,0]=coef[:,2]+1j*coef[:,3]
    jac[:,1,1]=coef[:,4]+1j*coef[:,5]
    tan=SeaTangent(modes,jac,head.vol,head.sea0,head.geom)
    tan.method="trajectory-ls"
    tan.amplitude=0.
    tan.nsolve=0
    err=float(np.linalg.norm(x.reshape(-1,len(coef))@coef-
                             y.reshape(-1,6))/
              max(np.linalg.norm(y),1e-15))
    stats=dict(teacher=str(teacher),original_BFZI_trajectories=len(x),
               original_BFZI_steps=int(np.prod(x.shape[:2])),
               thermal_modes=len(coef),identified_modes=effective,
               directional_EM_solves=0,
               ridge=float(choices[k]),train_sea_delta_relative_L2=err,
               heldout_history_sea_delta_relative_L2=scores[k])
    return tan,stats
