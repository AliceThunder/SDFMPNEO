"""Analytic BFZI M4 copper length, capacity and insulated heat conductance."""
from __future__ import annotations
import numpy as np
from .geom_canon import labels

N=10.
OUTER=.08885
PITCH=.005
WIRE=.000790569415
CV=3.45e6


def properties(case):
    """Continuous affine rounded-square winding and radial insulation."""
    g=case["geometry"]
    length=[]
    for j in (1,2):
        alpha=float(g[f"alpha{j}"])
        core=float(g[f"core{j}_mm"])*.001
        r0=OUTER-core
        step=core/N+alpha*PITCH
        per0=8*core+2*np.pi*r0
        slope=8*(1-alpha)+2*np.pi*alpha
        length.append(N*per0-.5*N*N*step*slope)
    length=np.asarray(length)
    if np.min(length)<=0:
        raise ValueError("invalid continuous winding length")
    t=case["thermal"]
    ins=float(t["ins_k"])
    thick=float(t.get("ins_t",.0004))
    if ins<=0 or thick<=0:
        raise ValueError("positive insulation conductivity and thickness required")
    cap=CV*np.pi*WIRE**2*length
    cond=2*np.pi*ins*length/np.log1p(thick/WIRE)
    return length,cap,cond


def apply(case,init,exchange=False):
    """Use analytic Cu heat capacities; optionally enforce insulated exchange.

    Interpolate other heat coordinates unchanged in physical conductance
    coordinates H=M K M and preserve the non-Cu Schur complement if the
    copper/package coupling is modified.
    """
    _,cap_cu,cond=properties(case)
    out=dict(init)
    cold=np.asarray(init["cap"],float)
    cap=cold.copy()
    cap[3:5]=cap_cu
    lab=labels(len(init["k"]))
    k=np.asarray(init["k"],float)
    old=np.sqrt(cold[lab])
    h=k*old[:,None]*old[None,:]
    if exchange:
        cu=np.array([3,4])
        rest=np.array([i for i in range(len(k)) if i not in cu])
        d=h[np.ix_(cu,cu)]
        b=h[np.ix_(rest,cu)]
        bnew=b*(cond/np.diag(d))[None,:]
        for col,p in enumerate((1,2)):
            row=int(np.flatnonzero(rest==p)[0])
            bnew[row,col]=-cond[col]
        dnew=np.diag(cond)
        h[np.ix_(rest,rest)]+=(
            bnew@np.linalg.solve(dnew,bnew.T)
            -b@np.linalg.solve(d,b.T))
        h[np.ix_(rest,cu)]=bnew
        h[np.ix_(cu,rest)]=bnew.T
        h[np.ix_(cu,cu)]=dnew
    mass=np.sqrt(cap[lab])
    out["cap"]=cap
    out["k"]=.5*(h/(mass[:,None]*mass[None,:])+
                  h.T/(mass[:,None]*mass[None,:]))
    return out
