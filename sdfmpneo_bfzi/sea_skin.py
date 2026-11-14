"""Passive half-order seawater thermal memory for an unresolved surface layer.

For a semi-infinite water halfspace, the flux-to-surface-temperature
impedance is 1/sqrt(k*rho*c*s). A positive exponential quadrature provides
causal fractional memory, plus fast and slow endpoint corrections.

Fluxes are per unit area (W/m2), temperatures K, time s. This describes
only the local halfspace; coupling to finite FV water requires separate
bulk exchange and energy accounting.
"""
from __future__ import annotations

import numpy as np


class SeaSkin:
    def __init__(self, *, k=.6, rhoc=4e6, dt_min=.5,
                 t_max=3600., modes=32, span=500., shape=()):
        if (min(k,rhoc,dt_min,t_max,span)<=0 or
                modes<4 or t_max<=dt_min):
            raise ValueError("positive seawater skin physical/time domain required")
        lo=1/(t_max*span)
        hi=span/dt_min
        dx=np.log(hi/lo)/modes
        self.lam=np.exp(np.log(lo)+(np.arange(modes)+.5)*dx)
        eff=np.sqrt(k*rhoc)
        self.w=dx*np.sqrt(self.lam)/(np.pi*eff)
        self.slow_w=2*np.sqrt(lo)/(np.pi*eff)
        self.fast_r=2/(np.pi*np.sqrt(hi)*eff)
        self.k=float(k)
        self.rhoc=float(rhoc)
        self.shape=tuple(shape)
        self.x=np.zeros((modes,)+self.shape)
        self.slow=np.zeros(self.shape)
        self.temp=np.zeros(self.shape)
        self.energy=np.zeros(self.shape)

    def _coeff(self, dt):
        if dt<=0 or not np.isfinite(dt):
            raise ValueError("positive time step required")
        e=np.exp(-self.lam*dt)
        g=-np.expm1(-self.lam*dt)/self.lam
        return (e.reshape((-1,)+(1,)*len(self.shape)),
                g.reshape((-1,)+(1,)*len(self.shape)))

    def step(self, flux, dt):
        """Piecewise-constant W/m2 flux; return skin-side temperature rise."""
        q=np.asarray(flux,float)
        if q.shape!=self.shape or not np.isfinite(q).all():
            raise ValueError("seawater surface flux dimensions differ")
        e,g=self._coeff(dt)
        self.x=e*self.x+g*q
        self.slow+=dt*q
        self.energy+=dt*q
        self.temp=(self.fast_r*q+self.slow_w*self.slow+
                   np.tensordot(self.w,self.x,axes=(0,0)))
        return self.temp.copy()

    def contact(self, delta, dt, resistance):
        """Implicit halfspace surface flux for a package-water temperature gap.

        Uses T_package - T_far = resistance*q + T_skin with all memory.
        """
        d=np.asarray(delta,float)
        r=np.asarray(resistance,float)
        if (d.shape!=self.shape or r.shape not in ((),self.shape)
                or np.any(r<0)):
            raise ValueError("invalid seawater contact thermal resistance")
        e,g=self._coeff(dt)
        old=self.slow_w*self.slow+np.tensordot(
            self.w,e*self.x,axes=(0,0))
        gain=(self.fast_r+self.slow_w*dt+
              np.sum(self.w*g.reshape(-1),axis=0))
        q=(d-old)/(r+gain)
        self.step(q,dt)
        return q

    def impedance(self, s):
        """Positive-real passive area-specific thermal impedance."""
        z=complex(s)
        if z==0:
            raise ValueError("positive Laplace frequency required")
        return (self.fast_r+self.slow_w/z+
                np.sum(self.w/(z+self.lam)))


def exact_step(flux, time, *, k=.6, rhoc=4e6):
    """Surface rise from constant W/m2 flux into infinite halfspace."""
    return 2*np.asarray(flux)*np.sqrt(time/(np.pi*k*rhoc))
