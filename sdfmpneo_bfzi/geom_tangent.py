"""Original-BFZI low-rank seawater impedance Jacobian from cold thermal modes.

For a new geometry, derivatives are sampled once from the ORIGINAL BFZI
electromagnetic direct solver. Arbitrary thermal histories then require only
small tensor contractions, not time-stepped Green/GMRES solves.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def basis(vol,phi,rank):
    """Sea part of physical thermal ROM, orthonormal in FV volume."""
    w=np.sqrt(np.asarray(vol,float))
    A=w[:,None]*np.asarray(phi,float)
    U,s,_=np.linalg.svd(A,full_matrices=False)
    if not len(s) or s[0]<=0:
        raise ValueError("BFZI physical heat ROM has no sea modes")
    keep=min(int(rank),int(np.count_nonzero(s>s[0]*1e-8)))
    if keep<1:
        raise ValueError("at least one physical sea thermal mode required")
    return U[:,:keep]/w[:,None]


class SeaTangent:
    def __init__(self,modes,jac,vol,cold,geom):
        self.modes=np.asarray(modes,float)
        self.jac=np.asarray(jac,complex)
        self.vol=np.asarray(vol,float)
        self.cold=np.asarray(cold,complex)
        self.geom=np.asarray(geom,float)
        self.method="central"
        self.amplitude=1.
        self.nsolve=2*len(self.jac)
        if (self.modes.shape[0]!=len(self.vol)
                or self.modes.shape[1]!=len(self.jac)
                or self.jac.shape[1:]!=(2,2)
                or self.cold.shape!=(2,2)):
            raise ValueError("original BFZI sea impedance tangent shape mismatch")

    @classmethod
    def build(cls,em,head,phi,rank=8,amplitude=1.,method="central"):
        """Build BFZI sea thermal Jacobian with 1 or 2 EM solves per mode.

        forward: use already available cold Z from head, then one positive
        perturbation per mode. central: original two-sided baseline.
        The underlying original BFZI solver, thermal basis and output
        coordinate convention are unchanged.
        """
        if amplitude<=0 or not np.isfinite(amplitude):
            raise ValueError("positive finite BFZI thermal perturbation required")
        if method not in ("central","forward"):
            raise ValueError("BFZI finite-difference method must be central/forward")
        modes=basis(head.vol,np.asarray(phi)[:head.ns],rank)
        jac=[]
        cu=np.full(2,head.amb)
        cold=np.asarray(head.sea0,complex)
        nsolve=0
        for j in range(modes.shape[1]):
            span=float(amplitude/max(np.max(np.abs(modes[:,j])),1e-12))
            dt=span*modes[:,j]
            plus=em.solve(np.full(head.ns,head.amb)+dt,cu).sea_z
            nsolve+=1
            if method=="central":
                minus=em.solve(np.full(head.ns,head.amb)-dt,cu).sea_z
                nsolve+=1
                deriv=(plus-minus)/(2*span)
            else:
                deriv=(plus-cold)/span
            jac.append(.5*(deriv+deriv.T))
        tan=cls(modes,jac,head.vol,cold,head.geom)
        tan.method=method
        tan.amplitude=float(amplitude)
        tan.nsolve=nsolve
        return tan

    def delta(self,theta):
        """Reciprocal 2-port sea delta for an actual 3-D seawater FV state."""
        t=np.asarray(theta,float)
        if t.shape!=self.vol.shape:
            raise ValueError("original BFZI FV seawater temperature required")
        a=self.modes.T@(self.vol*t)
        dz=np.einsum("j,jab->ab",a,self.jac,optimize=True)
        return .5*(dz+dz.T)

    def capture(self,theta):
        """FV-volume-weighted projection error of a sea temperature field."""
        t=np.asarray(theta,float)
        a=self.modes.T@(self.vol*t)
        recon=self.modes@a
        norm=np.linalg.norm(np.sqrt(self.vol)*t)
        return float(np.linalg.norm(np.sqrt(self.vol)*(t-recon))/
                     max(norm,1e-12))

    def save(self,path):
        np.savez_compressed(Path(path),modes=self.modes,jac=self.jac,
                            vol=self.vol,cold=self.cold,geometry=self.geom,
                            fd_method=self.method,fd_amplitude_K=self.amplitude,
                            em_directional_solves=self.nsolve)

    @classmethod
    def load(cls,path,head=None):
        with np.load(path) as d:
            obj=cls(d["modes"],d["jac"],d["vol"],d["cold"],d["geometry"])
            if "fd_method" in d:
                obj.method=str(d["fd_method"])
                obj.amplitude=float(d["fd_amplitude_K"])
                obj.nsolve=int(d["em_directional_solves"])
        if head is not None:
            if (not np.allclose(obj.geom,head.geom,rtol=0,atol=1e-12)
                    or not np.allclose(obj.cold,head.sea0,atol=1e-10,rtol=1e-8)
                    or not np.allclose(obj.vol,head.vol,atol=1e-15,rtol=1e-10)):
                raise ValueError("original BFZI tangent and 3-D FV geometry differ")
        return obj
