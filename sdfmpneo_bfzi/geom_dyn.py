"""Causal geometry-only residual dynamic-impedance head."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class DynFix(nn.Module):
    def __init__(self,mu,std,scale,ys,hidden=80,geom_strength=1.):
        super().__init__()
        self.register_buffer('mu',torch.as_tensor(mu,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.register_buffer('ys',torch.as_tensor(ys,dtype=torch.float32))
        self.register_buffer('gw',torch.tensor(float(geom_strength)))
        self.net=nn.Sequential(nn.Linear(18+32+2,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,g,a,t,v):
        if (g.shape!=(len(a),18) or a.shape!=(len(a),32) or
            t.shape!=(len(a),1) or v.shape!=(len(a),1)):
            raise ValueError("dynamic Z requires 18 geometry, 32 heat, time, voltage")
        x=((g-self.mu)/self.std)*self.gw
        aa=a/self.scale
        tm=torch.log1p(t.clamp_min(0.))/np.log1p(600.)
        vv=v/100.
        a1=torch.cat((x,aa,tm,vv),dim=-1)
        a0=torch.cat((x,torch.zeros_like(aa),tm,vv),dim=-1)
        return (self.net(a1)-self.net(a0))*self.ys


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    net=DynFix(ck['mu'],ck['std'],ck['scale'],ck['ys'],
               geom_strength=float(ck.get('geom_strength',1.)))
    state=dict(ck['state'])
    if 'gw' not in state:
        state['gw']=net.gw
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net
