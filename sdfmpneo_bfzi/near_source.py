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

def local_j_var(jbar, sigq, current, src, ref, cell, omega_mu):
    """Temperature-resolved near-source J with original wet FV mean fixed."""
    jbar=np.asarray(jbar,complex)
    sigq=np.asarray(sigq,float)
    cur=np.asarray(current,complex)
    src=np.asarray(src,float)
    ref=np.asarray(ref,float)
    cell=np.asarray(cell,int)
    if (jbar.shape!=(len(ref),3) or ref.shape!=(len(ref),2,3) or
            sigq.shape!=(len(cell),) or src.shape!=(len(cell),2,3) or
            cur.shape!=(2,) or np.any(cell<0) or np.any(cell>=len(ref)) or
            not np.all(np.isfinite(sigq)) or np.any(sigq<=0)):
        raise ValueError("invalid temperature-resolved BFZI source field")
    delta=sigq[:,None,None]*src-ref[cell]
    return jbar[cell]-1j*omega_mu*np.einsum(
        "qpc,p->qc",delta,cur,optimize=True)


def split_heat(qcell,vol,bulk_vol,raw_bulk,cell,node,weights,
               raw_skin,n_skin):
    """Positive skin/bulk power preserving every original BFZI wet FV."""
    qcell=np.asarray(qcell,float)
    vol=np.asarray(vol,float)
    bulk_vol=np.asarray(bulk_vol,float)
    raw_bulk=np.asarray(raw_bulk,float)
    cell=np.asarray(cell,int)
    node=np.asarray(node,int)
    weights=np.asarray(weights,float)
    raw_skin=np.asarray(raw_skin,float)
    n=len(vol)
    if (qcell.shape!=(n,) or bulk_vol.shape!=(n,) or raw_bulk.shape!=(n,) or
            any(x.shape!=(len(cell),) for x in (node,weights,raw_skin)) or
            np.any(cell<0) or np.any(cell>=n) or
            np.any(node<0) or np.any(node>=n_skin) or
            np.any(vol<=0) or np.any(bulk_vol<=0) or
            np.any(qcell<0) or np.any(raw_bulk<0) or
            np.any(weights<=0) or np.any(raw_skin<0)):
        raise ValueError("invalid skin weak Joule partition")
    s=weights*raw_skin
    skin_by_cell=np.bincount(cell,weights=s,minlength=n)
    b=bulk_vol*raw_bulk
    den=b+skin_by_cell
    absent=(den==0)&(qcell>0)
    if np.any(absent):
        b[absent]=bulk_vol[absent]
        s=np.where(absent[cell],weights,s)
        den=b+np.bincount(cell,weights=s,minlength=n)
    ratio=np.divide(qcell*vol,den,out=np.zeros(n),where=den>0)
    bulk=b*ratio
    skin=np.bincount(node,weights=s*ratio[cell],minlength=n_skin)
    return bulk,skin
