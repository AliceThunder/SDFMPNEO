"""Geometry-conditioned correction of 14 package temperature shape modes."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


PKG = slice(18,32)


class PkgFix(nn.Module):
    def __init__(self,ng=45,scale=None,hidden=64):
        super().__init__()
        a=np.ones(32) if scale is None else np.asarray(scale,float)
        if a.shape!=(32,) or np.min(a)<=0:
            raise ValueError("expected 32 positive heat scales")
        self.register_buffer("scale",torch.as_tensor(a,dtype=torch.float32))
        self.ng=int(ng)
        self.net=nn.Sequential(nn.Linear(ng+32,hidden),nn.SiLU(),
                                nn.Linear(hidden,14))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,g,a):
        if g.shape!=(len(a),self.ng) or a.shape!=(len(a),32):
            raise ValueError("package correction expects geometry and 32 heat modes")
        x=torch.cat((g,a/self.scale),dim=-1)
        cold=torch.cat((g,torch.zeros_like(a)),dim=-1)
        return self.net(x)-self.net(cold)


def restore(path):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    model=PkgFix(ng=int(ck['ng']),scale=np.asarray(ck['scale']))
    model.load_state_dict(ck['state'])
    model.eval()
    return model
