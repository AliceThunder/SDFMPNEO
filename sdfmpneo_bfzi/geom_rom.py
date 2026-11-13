"""Energy-conserving thermal ROM for original-BFZI geometry-anchored surrogates.

Build one basis from cold weak Joule shapes and copper heat responses using
only the saved FV H/C operators. No dynamic reference trajectories are used.
Five material indicator modes enforce individual sea/package/copper budgets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

from .geom_heat import GeomHeat


def groups(head):
    ns,npkg=head.ns,head.npkg
    return (slice(0,ns),slice(ns,ns+npkg),
            slice(ns+npkg,ns+2*npkg),
            slice(ns+2*npkg,ns+2*npkg+1),
            slice(ns+2*npkg+1,ns+2*npkg+2))


def make_basis(head,rank=32):
    """POD of physically forced FV impulse responses, plus five budget modes."""
    n=len(head.cap)
    if rank<5 or rank>n:
        raise ValueError("BFZI thermal ROM needs 5 <= rank <= FV states")
    w=np.sqrt(head.cap)
    mark=np.zeros((n,5))
    for j,sl in enumerate(groups(head)):
        mark[sl,j]=w[sl]
    mark/=np.linalg.norm(mark,axis=0)
    forcing=[]
    for a,b in ((0,0),(0,1),(1,1)):
        f=np.zeros(n)
        f[:head.ns]=head.weak[:,a,b]*head.vol
        if np.linalg.norm(f)>0:
            forcing.append(f)
    for j in range(2):
        f=np.zeros(n)
        f[head.cu.start+j]=1.
        forcing.append(f)
    if not forcing:
        raise ValueError("no original BFZI Joule/copper source modes")
    times=np.array([1.,4.,12.,30.,75.,180.,420.,900.,1800.,3600.])
    snapshots=[]
    factors={}
    def factor(dt):
        if dt not in factors:
            factors[dt]=factorized((diags(head.cap)+dt*head.H).tocsc())
        return factors[dt]
    for f in forcing:
        temp=np.zeros(n)
        prev=0.
        for tm in times:
            dt=float(tm-prev)
            temp=factor(dt)(head.cap*temp+dt*f)
            z=w*temp
            size=float(np.linalg.norm(z))
            if size>1e-20:
                snapshots.append(z/size)
            prev=tm
        for dt in (30.,150.,600.):
            temp=factor(dt)(head.cap*temp)
            z=w*temp
            size=float(np.linalg.norm(z))
            if size>1e-20:
                snapshots.append(z/size)
    A=np.stack(snapshots,axis=1)
    A-=mark@(mark.T@A)
    U,s,_=np.linalg.svd(A,full_matrices=False)
    n_extra=int(np.count_nonzero(s>max(float(s[0])*1e-11,1e-12)))
    basis=np.column_stack((mark,U[:,:min(rank-5,n_extra)]))/w[:,None]
    return basis


class GeomHeatROM(GeomHeat):
    """Full-field output with reduced implicit thermal solve and material budgets."""

    def __init__(self,head,phi):
        self.__dict__.update(head.__dict__.copy())
        self.phi=np.asarray(phi,float)
        n=len(self.cap)
        if self.phi.ndim!=2 or self.phi.shape[0]!=n or self.phi.shape[1]<5:
            raise ValueError("invalid BFZI spatial heat ROM basis")
        gram=self.phi.T@(self.cap[:,None]*self.phi)
        if np.max(abs(gram-np.eye(len(gram))))>1e-7:
            raise ValueError("BFZI thermal ROM capacity orthogonality failed")
        self.rank=self.phi.shape[1]
        self.K=self.phi.T@(self.H@self.phi)
        self._rsolves={}
        self.budgets=[]

    @classmethod
    def build(cls,head,rank=32):
        return cls(head,make_basis(head,rank))

    def save(self,path):
        np.savez_compressed(path,phi=self.phi,cap=self.cap,
                            geometry=self.geom,
                            pose=self.pose if self.pose is not None else np.empty(0),
                            pkg_axes=self._pkg_axes,pkg_half=self._pkg_half)

    @classmethod
    def load(cls,head,path):
        with np.load(Path(path)) as d:
            if (not np.array_equal(d["geometry"],head.geom) or
                not np.allclose(d["cap"],head.cap,rtol=1e-12,atol=1e-12)):
                raise ValueError("BFZI ROM basis geometry/capacity mismatch")
            if "pose" in d:
                p=head.pose if head.pose is not None else np.empty(0)
                if (not np.allclose(d["pose"],p,atol=1e-12,rtol=0)
                        or not np.allclose(d["pkg_axes"],head._pkg_axes,
                                           atol=1e-12,rtol=0)
                        or not np.allclose(d["pkg_half"],head._pkg_half,
                                           atol=1e-12,rtol=0)):
                    raise ValueError("BFZI ROM pose and package sizes differ")
            return cls(head,d["phi"])

    def advance(self,theta,dt,voltage,dsea,*,steps=10,tol=2e-9):
        if dt<=0 or not np.isfinite(dt):
            raise ValueError("positive thermal step required")
        if dt not in self._rsolves:
            self._rsolves[dt]=cho_factor(
                np.eye(self.rank)+dt*self.K,check_finite=False)
        fac=self._rsolves[dt]
        old=self.phi.T@(self.cap*np.asarray(theta,float))
        a=old.copy()
        for _ in range(steps):
            guess=self.phi@a
            sea,z,I=self.circuit(guess,dsea,voltage)
            heat,_,_,_,_=self.source(guess,I,sea)
            nxt=cho_solve(fac,old+dt*(self.phi.T@heat),
                          check_finite=False)
            change=float(np.max(abs(self.phi@(nxt-a))))
            a=nxt
            if change<tol:
                break
        new=self.phi@a
        sea,z,I=self.circuit(new,dsea,voltage)
        heat,q,J,ps,pc=self.source(new,I,sea)
        bal=self.cap*(new-theta)/dt+self.H@new-heat
        group=np.array([np.sum(bal[sl]) for sl in groups(self)])
        self.budgets.append(group)
        return new,z,I,q,J,ps,pc,bal

    def rollout(self,net,ck,voltage,times):
        self.budgets=[]
        r=super().rollout(net,ck,voltage,times)
        r["material_budget_W"]=np.asarray(self.budgets)
        r["rom_rank"]=self.rank
        return r
