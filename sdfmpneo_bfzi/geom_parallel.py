"""Compact parallel impedance head sharing the 45D geometry/condition encoder.

The head is linear in five material-mean temperature rises, so its output
vanishes at ambient. It corrects TX/RX self impedance and preserves mutual Z.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ParZ(nn.Module):
    shared_geom = True

    def __init__(self, gm, gs, a_scale, y_scale, hidden=32, strength=1.):
        super().__init__()
        self.register_buffer("gm", torch.as_tensor(gm,dtype=torch.float32))
        self.register_buffer("gs", torch.as_tensor(gs,dtype=torch.float32))
        self.register_buffer("a_scale", torch.as_tensor(a_scale,dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_scale,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(45,hidden),nn.SiLU(),
                               nn.Linear(hidden,4*5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.strength=float(strength)

    def forward(self,g,a,t=None,v=None):
        if g.ndim!=2 or g.shape[1]!=45 or a.ndim!=2 or a.shape[1]!=48 or len(g)!=len(a):
            raise ValueError("ParZ expects shared geometry [B,45] and thermal state [B,48]")
        slope=self.net((g-self.gm)/self.gs).reshape(-1,4,5)
        delta=(slope*(a[:,:5]/self.a_scale)[:,None,:]).sum(-1)*self.y_scale
        z=torch.zeros((len(a),6),dtype=delta.dtype,device=delta.device)
        z[:,0:2]=delta[:,0:2]*self.strength
        z[:,4:6]=delta[:,2:4]*self.strength
        return z

    def save(self,path):
        torch.save(dict(state=self.state_dict(),
                        hidden=self.net[0].out_features,
                        strength=self.strength),path)

    @classmethod
    def load(cls,path):
        ck=torch.load(path,map_location="cpu",weights_only=True)
        sd=ck["state"]
        model=cls(sd["gm"],sd["gs"],sd["a_scale"],sd["y_scale"],
                  hidden=ck["hidden"],strength=ck["strength"])
        model.load_state_dict(sd)
        model.eval()
        return model
