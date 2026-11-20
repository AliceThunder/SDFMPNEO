"""Residual hot-state impedance map trained from fixed-temperature BFZI labels."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn


class HotResidual(nn.Module):
    def __init__(self,gm,gs,ass,ys,hidden=96):
        super().__init__()
        self.register_buffer("gm",torch.as_tensor(gm,dtype=torch.float32))
        self.register_buffer("gs",torch.as_tensor(gs,dtype=torch.float32))
        self.register_buffer("ass",torch.as_tensor(ass,dtype=torch.float32))
        self.register_buffer("ys",torch.as_tensor(ys,dtype=torch.float32))
        self.net=nn.Sequential(
            nn.Linear(45+48,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),
            nn.Linear(hidden,6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,g,a):
        if g.shape!=(len(a),45) or a.shape!=(len(g),48):
            raise ValueError("hot residual expects geometry [B,45] and state [B,48]")
        xg=(g-self.gm)/self.gs
        xa=a/self.ass
        x=torch.cat((xg,xa),dim=-1)
        x0=torch.cat((xg,torch.zeros_like(xa)),dim=-1)
        return (self.net(x)-self.net(x0))*self.ys


class HotZ(nn.Module):
    shared_geom=True

    def __init__(self,base,residual):
        super().__init__()
        self.base=base
        self.residual=residual

    def forward(self,g,a):
        return self.base(g,a)+self.residual(g,a)


def restore(path):
    ck=torch.load(Path(path),map_location="cpu",weights_only=True)
    net=HotResidual(ck["gm"],ck["gs"],ck["ass"],ck["ys"],
                    hidden=int(ck.get("hidden",96)))
    net.load_state_dict(ck["state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    from .geom_skintan import restore_rank48
    base=restore_rank48(ck["skin_tangent"],ck["skin_base"])
    return HotZ(base,net)
