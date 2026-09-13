"""Neural initial guess plus true sparse Maxwell residual correction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse.linalg as spla
from .unified_basis import safe_diag
from .unified_dataset import operator_encoding

@dataclass(frozen=True)
class MaxwellSolveReport:
    initial_relative_residual: tuple
    final_relative_residual: tuple
    correction_iterations: tuple
    direct_corrections: int

class NeuralMaxwellAccelerator:
    def __init__(self,network,basis,*,residual_tolerance=1e-7,max_iterations=200):
        self.network=network; self.basis=np.asarray(basis,complex); self.residual_tolerance=float(residual_tolerance); self.max_iterations=int(max_iterations)
        if self.basis.ndim!=2 or self.basis.shape[1]<1: raise ValueError("basis must be a nonempty matrix")
        if self.residual_tolerance<=0 or self.max_iterations<1: raise ValueError("invalid Maxwell correction settings")
    def guess(self,A,B):
        import torch
        features,base,scale,_,_,_=operator_encoding(A,B,self.basis); parameter=next(self.network.parameters()); x=torch.as_tensor(features,dtype=parameter.dtype,device=parameter.device).unsqueeze(0)
        with torch.no_grad(): y=self.network(x).detach().cpu().double().numpy().reshape((B.shape[1],2*self.basis.shape[1]))
        cvec=base+scale[:,None]*y; r=self.basis.shape[1]; coeff=(cvec[:,:r]+1j*cvec[:,r:]).T
        return self.basis@coeff
    def solve(self,A,B):
        B=np.asarray(B,complex); X=self.guess(A,B); denom=np.maximum(np.linalg.norm(B,axis=0),np.finfo(float).tiny); R=B-A@X; initial=np.linalg.norm(R,axis=0)/denom; iterations=[]; failed=[]; diag=safe_diag(A); M=spla.LinearOperator(A.shape,matvec=lambda x:x/diag,dtype=complex)
        for j in range(B.shape[1]):
            if initial[j]<=self.residual_tolerance: iterations.append(0); continue
            counter=[0]
            def callback(_): counter[0]+=1
            try: dx,info=spla.gmres(A,R[:,j],M=M,rtol=self.residual_tolerance*.25,atol=0.0,restart=min(60,A.shape[0]),maxiter=self.max_iterations,callback=callback,callback_type="pr_norm")
            except TypeError: dx,info=spla.gmres(A,R[:,j],M=M,tol=self.residual_tolerance*.25,restart=min(60,A.shape[0]),maxiter=self.max_iterations,callback=callback)
            X[:,j]+=dx; iterations.append(counter[0]); rel=float(np.linalg.norm(B[:,j]-A@X[:,j])/denom[j])
            if info!=0 or rel>self.residual_tolerance: failed.append(j)
        direct=0
        if failed:
            lu=spla.splu(A.tocsc())
            for j in failed: X[:,j]+=lu.solve(B[:,j]-A@X[:,j]); direct+=1
        final=np.linalg.norm(B-A@X,axis=0)/denom
        if np.any(final>max(5*self.residual_tolerance,1e-12)): raise RuntimeError(f"Maxwell residual correction failed: max relative residual={float(np.max(final)):.3e}")
        return X,MaxwellSolveReport(tuple(map(float,initial)),tuple(map(float,final)),tuple(map(int,iterations)),direct)

__all__=["MaxwellSolveReport","NeuralMaxwellAccelerator"]
