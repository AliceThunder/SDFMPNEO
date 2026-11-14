"""Physical near-source spatial current and conservative weak-Joule profile."""
from __future__ import annotations

import numpy as np


def local_j(jbar, sigma, current, src, avg, cell, omega_mu):
    """Restore unresolved source variation while preserving FV mean J."""
    jbar=np.asarray(jbar,complex)
    sig=np.asarray(sigma,float)
    cur=np.asarray(current,complex)
    src=np.asarray(src,float)
    avg=np.asarray(avg,float)
    cell=np.asarray(cell,int)
    if (jbar.shape!=(len(sig),3) or avg.shape!=(len(sig),2,3) or
            src.shape!=(len(cell),2,3) or cur.shape!=(2,) or
            np.any(cell<0) or np.any(cell>=len(sig)) or np.any(sig<=0)):
        raise ValueError("original wet FV near-source field dimensions invalid")
    delta=np.einsum('npc,p->nc',src-avg[cell],cur,optimize=True)
    return jbar[cell]-1j*omega_mu*sig[cell,None]*delta


def local_q(j, sigma, qcell, vol, cell, weights=None, *,scale=None):
    """Nonnegative source-shaped heat with original per-FV weak power."""
    j=np.asarray(j,complex)
    sig=np.asarray(sigma,float)
    qcell=np.asarray(qcell,float)
    vol=np.asarray(vol,float)
    cell=np.asarray(cell,int)
    if (j.shape!=(len(cell),3) or sig.shape!=qcell.shape or
            sig.shape!=vol.shape or np.any(sig<=0) or np.any(qcell<0)):
        raise ValueError("invalid spatial weak-Joule inputs")
    raw=.5*np.sum(abs(j)**2,axis=1)/sig[cell]
    if scale is None:
        weights=np.asarray(weights,float)
        if weights.shape!=(len(cell),) or np.any(weights<=0):
            raise ValueError("native wet FV quadrature weights required")
        tot=np.bincount(cell,weights=weights*raw,minlength=len(sig))
        energy=qcell*vol
        if np.any((tot<=0)&(energy>1e-14)):
            raise ValueError("nonzero BFZI weak Joule requires spatial current")
        scale=np.divide(energy,tot,out=np.zeros_like(energy),where=tot>0)
    else:
        scale=np.asarray(scale,float)
        if scale.shape!=qcell.shape:
            raise ValueError("weak-Joule scale must be per native wet FV")
    return raw*scale[cell],scale
