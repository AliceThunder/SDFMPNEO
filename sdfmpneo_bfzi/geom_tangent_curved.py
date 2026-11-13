"""Quadratic correction to original BFZI cold-response sea impedance tangent.

Uses a second, larger temperature amplitude only for selected directions.
No further BFZI Green/EM solves are needed for transient prediction.
"""
from __future__ import annotations

import time
import numpy as np


class CurvedTangent:
    def __init__(self,base,ids,curv,h1):
        self.base=base
        self.ids=np.asarray(ids,int)
        self.curv=np.asarray(curv,complex)
        self.h1=np.asarray(h1,float)
        self.nsolve=base.nsolve+len(self.ids)
        if (self.curv.shape!=(len(self.ids),2,2)
                or self.h1.shape!=(len(self.ids),)
                or len(set(self.ids.tolist()))!=len(self.ids)
                or np.any(self.ids<0)
                or np.any(self.ids>=base.modes.shape[1])):
            raise ValueError("BFZI second-order tangent directions mismatch")

    def delta(self,theta):
        t=np.asarray(theta,float)
        if t.shape!=self.base.vol.shape:
            raise ValueError("original BFZI thermal modes require full sea temperature")
        a=self.base.modes.T@(self.base.vol*t)
        z=self.base.delta(t)
        coeff=a[self.ids]
        z+=np.einsum("k,kab->ab",
                      coeff*(coeff-self.h1),self.curv,optimize=True)
        return .5*(z+z.T)


def build(em,head,tan,ids=(0,1),amplitude=5.):
    """Add one original-BFZI positive-temperature solve per selected mode."""
    if not np.isfinite(amplitude) or amplitude<=tan.amplitude:
        raise ValueError("large BFZI curvature probe must exceed tangent probe")
    ids=tuple(int(x) for x in ids)
    cu=np.full(2,head.amb)
    curv=[];small=[]
    start=time.perf_counter()
    for i in ids:
        mode=tan.modes[:,i]
        peak=float(np.max(abs(mode)))
        h1=tan.amplitude/peak
        h2=amplitude/peak
        state=em.solve(head.amb+h2*mode,cu)
        d2=(state.sea_z-head.sea0)/h2
        b=(d2-tan.jac[i])/(h2-h1)
        curv.append(.5*(b+b.T))
        small.append(h1)
    obj=CurvedTangent(tan,ids,np.asarray(curv),np.asarray(small))
    return obj,dict(extra_directional_solves=len(ids),
                    extra_directional_sec=time.perf_counter()-start,
                    large_amplitude_K=amplitude,corrected_modes=list(ids))
