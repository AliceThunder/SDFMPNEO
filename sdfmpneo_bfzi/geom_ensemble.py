"""BFZI geometry-extrapolation ensemble sharing one conservative thermal solve.

Average only learned sea-Z and temperature observations. The original BFZI
complex port/copper and 3-D heat propagation remain physically assembled once.
"""
from __future__ import annotations
import numpy as np
import torch


class Multi:
    def __init__(self,nets,coeff):
        if not nets:
            raise ValueError("at least one BFZI memory network is required")
        w=np.asarray(coeff,float)
        if w.shape!=(len(nets),) or np.any(w<0) or w.sum()<=0:
            raise ValueError("BFZI ensemble weights must be nonnegative")
        self.nets=nets
        self.ng=max(net.ng for net in nets)
        self.w=torch.tensor(w/w.sum(),dtype=torch.float32)

    def init_state(self,n):
        return [net.init_state(n) for net in self.nets]

    def step(self,h,u,dt,geom):
        states=[]
        values=[]
        for net,state in zip(self.nets,h):
            nxt,y=net.step(state,u,dt,geom[:,:net.ng])
            states.append(nxt)
            values.append(y)
        return states,torch.sum(
            torch.stack(values)*self.w[:,None,None],dim=0)


def load(paths,coeff=None):
    """Combine 22D/40D BFZI checkpoints with common source/target scales."""
    from .geom_memory import GeomFrac
    paths=list(paths)
    if not paths:
        raise ValueError("specify at least one neural BFZI checkpoint")
    nets=[];cks=[]
    for path in paths:
        ck=torch.load(path,map_location="cpu",weights_only=True)
        if not ck.get("port_sea"):
            raise ValueError("BFZI ensemble needs sea-delta neural heads")
        net=GeomFrac(ng=int(ck["ng"]))
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append(net)
        cks.append(ck)
    base=cks[0]
    for ck in cks[1:]:
        for name in ("src_scale","y_scale"):
            if not np.allclose(np.asarray(base[name]),np.asarray(ck[name]),
                               rtol=1e-6,atol=1e-9):
                raise ValueError("BFZI ensemble physical scale mismatch")
    return Multi(nets,coeff if coeff is not None else np.ones(len(nets))),base
