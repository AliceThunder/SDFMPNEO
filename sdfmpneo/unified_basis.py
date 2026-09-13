"""Residual-driven common Maxwell space for the unified background."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.linalg


def safe_diag(A):
    d=np.asarray(A.diagonal(),complex); scale=max(float(np.max(np.abs(d))),np.finfo(float).tiny)
    return np.where(np.abs(d)>1e-12*scale,d,scale+0j)


def orthonormal_append(V,v):
    q=np.asarray(v,complex).copy()
    if V.size:
        for _ in range(2): q-=V@(V.conj().T@q)
    n=float(np.linalg.norm(q))
    if n<=1e-12*max(float(np.linalg.norm(v)),1.0): return V,False
    q/=n
    return (q[:,None] if V.size==0 else np.column_stack([V,q])),True


def reduced_solution(A,B,V):
    if V.shape[1]==0: return np.zeros((A.shape[0],B.shape[1]),complex)
    Ar=V.conj().T@(A@V); br=V.conj().T@B
    try: c=scipy.linalg.solve(Ar,br,assume_a="gen")
    except np.linalg.LinAlgError: c=np.linalg.lstsq(Ar,br,rcond=None)[0]
    return V@c


@dataclass(frozen=True)
class BasisReport:
    basis_dimension: int
    maximum_anchor_relative_residual: float
    target_relative_residual: float
    sample_count: int


def build_residual_basis(background,geometry_samples,state_samples,*,max_rank=96,target_relative_residual=1e-2,monitor=None):
    """Build V using only physical operators, RHS vectors, and residual lifts."""
    max_rank=int(max_rank); target=float(target_relative_residual)
    if max_rank<1 or not 0<target<1: raise ValueError("invalid basis settings")
    pairs=list(zip(geometry_samples,state_samples))
    if not pairs: raise ValueError("basis samples cannot be empty")
    V=np.empty((background.n_edges,0),complex)
    for si,(g,a) in enumerate(pairs):
        if monitor is not None: monitor.checkpoint()
        ctx=background.geometry_context(g); A=background.em_operator(ctx,a); B=background.rhs_matrix(ctx); diag=safe_diag(A)
        for p in range(B.shape[1]):
            while V.shape[1]<max_rank:
                x=reduced_solution(A,B[:,[p]],V)[:,0]; r=B[:,p]-A@x; denom=max(float(np.linalg.norm(B[:,p])),np.finfo(float).tiny); rel=float(np.linalg.norm(r)/denom)
                if rel<=target: break
                V,added=orthonormal_append(V,r/diag)
                if not added: break
        if monitor is not None:
            with monitor._lock: monitor.data.update(phase="maxwell_basis",training_points=si+1)
        print(f"构建统一 Maxwell 空间……{100*(si+1)/len(pairs):5.1f}%  rank={V.shape[1]}",flush=True)
        if V.shape[1]>=max_rank: break
    worst=0.0
    for g,a in pairs:
        ctx=background.geometry_context(g); A=background.em_operator(ctx,a); B=background.rhs_matrix(ctx); X=reduced_solution(A,B,V); den=np.maximum(np.linalg.norm(B,axis=0),np.finfo(float).tiny); worst=max(worst,float(np.max(np.linalg.norm(B-A@X,axis=0)/den)))
    return V,BasisReport(V.shape[1],worst,target,len(pairs))

__all__=["BasisReport","build_residual_basis","reduced_solution","safe_diag"]
