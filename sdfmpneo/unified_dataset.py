"""Compact solution-label-free training data for the neural Maxwell accelerator."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def operator_encoding(A,B,V):
    """Encode one physical operator and its full-space residual quadratic form."""
    AV=A@V; Ar=V.conj().T@AV; br=V.conj().T@B; r=V.shape[1]; tiny=np.finfo(float).tiny
    ascale=max(float(np.linalg.norm(Ar))/np.sqrt(max(r,1)),tiny); bscale=np.maximum(np.linalg.norm(br,axis=0),tiny)
    Ah=Ar/ascale; Bh=br/bscale[None,:]
    features=np.concatenate([Ah.real.ravel(),Ah.imag.ravel(),Bh.real.ravel(),Bh.imag.ravel(),[np.log(ascale)],np.log(bscale)])
    d=np.diag(Ar); ds=max(float(np.max(np.abs(d))),tiny); d=np.where(np.abs(d)>1e-12*ds,d,ds+0j); base=br/d[:,None]; cscale=bscale/ascale
    baseline=np.concatenate([base.real.T,base.imag.T],axis=1)
    Q=AV.conj().T@AV; S=AV.conj().T@B; Qr,Qim=Q.real,Q.imag; qblock=np.block([[Qr,-Qim],[Qim,Qr]])
    sreal=np.concatenate([S.real.T,S.imag.T],axis=1); norm2=np.sum(np.abs(B)**2,axis=0).real
    return tuple(np.asarray(x,np.float64) for x in (features,baseline,cscale,qblock,sreal,norm2))


@dataclass
class MaxwellOperatorDataset:
    features: np.ndarray
    baseline: np.ndarray
    coefficient_scale: np.ndarray
    residual_gram: np.ndarray
    residual_linear: np.ndarray
    rhs_norm2: np.ndarray
    split: np.ndarray
    reduced_rank: int
    n_rhs: int
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("wb") as f:
            np.savez_compressed(f,features=self.features,baseline=self.baseline,coefficient_scale=self.coefficient_scale,residual_gram=self.residual_gram,residual_linear=self.residual_linear,rhs_norm2=self.rhs_norm2,split=self.split,reduced_rank=np.array(self.reduced_rank),n_rhs=np.array(self.n_rhs))
        return path
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            return cls(d["features"],d["baseline"],d["coefficient_scale"],d["residual_gram"],d["residual_linear"],d["rhs_norm2"],d["split"],int(d["reduced_rank"]),int(d["n_rhs"]))
    def indices(self,name):
        return np.flatnonzero(self.split=={"train":0,"validation":1,"test":2}[name])


def generate_operator_dataset(background,V,geometry_samples,state_samples,*,seed=0,monitor=None):
    pairs=list(zip(geometry_samples,state_samples)); n=len(pairs)
    if n<10: raise ValueError("at least ten operator samples are required")
    rows=[]
    for i,(g,a) in enumerate(pairs):
        if monitor is not None: monitor.checkpoint()
        ctx=background.geometry_context(g); A=background.em_operator(ctx,a); B=background.rhs_matrix(ctx); rows.append(operator_encoding(A,B,V))
        if monitor is not None:
            with monitor._lock: monitor.data.update(phase="maxwell_operator_samples",training_points=i+1)
        if i==0 or (i+1)%max(1,n//20)==0 or i+1==n:
            print(f"生成 Maxwell 残差训练算子……{100*(i+1)/n:5.1f}% ({i+1}/{n})",flush=True)
    features,base,scale,Q,S,N=map(np.asarray,zip(*rows)); rng=np.random.default_rng(seed); order=rng.permutation(n); split=np.zeros(n,np.int8); nv=max(1,int(round(.15*n))); nt=max(1,int(round(.15*n))); split[order[:nv]]=1; split[order[nv:nv+nt]]=2
    return MaxwellOperatorDataset(features,base,scale,Q,S,N,split,V.shape[1],base.shape[1])

__all__=["MaxwellOperatorDataset","generate_operator_dataset","operator_encoding"]
