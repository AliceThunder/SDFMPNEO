"""Portable exact BFZI copper-skin and air-inductance port assembly.

Extract constants once from the independent original BFZI checkout.
Subsequent neural inference needs only this small NPZ and SciPy special.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.special import ive


class PortModel:
    def __init__(self,omega,air,length,radius,sigma,alpha,mu,ref):
        self.omega=float(omega)
        self.air=np.asarray(air,float)
        self.length=np.asarray(length,float)
        self.radius=np.asarray(radius,float)
        self.sigma=np.asarray(sigma,float)
        self.alpha=np.asarray(alpha,float)
        self.mu=np.asarray(mu,float)
        self.ref=np.asarray(ref,float)
        if (self.air.shape!=(2,2) or
            any(a.shape!=(2,) for a in
                (self.length,self.radius,self.sigma,self.alpha,self.mu,self.ref))):
            raise ValueError("two-port BFZI physical model dimensions required")
        if (self.omega<=0 or
            np.min(self.length)<=0 or np.min(self.radius)<=0 or
            np.min(self.sigma)<=0 or np.min(self.mu)<=0):
            raise ValueError("invalid copper/air material parameters")

    @classmethod
    def from_adapter(cls,em):
        src=em.copper
        if src is None:
            raise ValueError("original BFZI copper model must be enabled")
        wire=src.scene.assemblies
        lines=src.lines
        return cls(
            em.omega,em.air,
            [np.linalg.norm(line.line_elements,axis=1).sum() for line in lines],
            [a.wire_radius for a in wire],
            [a.material.conductivity for a in wire],
            [a.material.resistance_temperature_coefficient for a in wire],
            [4e-7*np.pi*a.material.relative_permeability for a in wire],
            [a.material.reference_temperature for a in wire])

    def copper(self,temp):
        t=np.asarray(temp,float)
        if t.shape!=(2,):
            raise ValueError("BFZI copper temperatures must be two values")
        fac=1+self.alpha*(t-self.ref)
        if np.min(fac)<=0:
            raise ValueError("negative copper conductivity")
        sig=self.sigma/fac
        k=np.sqrt(1j*self.omega*self.mu*sig)
        a=k*self.radius
        unit=np.where(abs(a)<1e-3,
            1/(np.pi*self.radius**2*sig)+1j*self.omega*self.mu/(8*np.pi),
            k*ive(0,a)/(2*np.pi*self.radius*sig*ive(1,a)))
        return unit*self.length

    def total(self,sea,temp):
        z=np.asarray(sea,complex)
        if z.shape!=(2,2):
            raise ValueError("seawater impedance must be full 2x2")
        return z+1j*self.omega*self.air+np.diag(self.copper(temp))

    def save(self,path):
        np.savez_compressed(
            path,omega=self.omega,air=self.air,
            length=self.length,radius=self.radius,
            sigma=self.sigma,alpha=self.alpha,mu=self.mu,ref=self.ref)

    @classmethod
    def load(cls,path):
        with np.load(Path(path)) as d:
            return cls(*(d[k] for k in
                ("omega","air","length","radius","sigma","alpha","mu","ref")))
