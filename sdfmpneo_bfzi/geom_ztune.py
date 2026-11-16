"""Nonlinear residual thermal-to-seawater-impedance head for GeomFracET."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

IDX=[0,*range(5,18)]


class ZFix(nn.Module):
    def __init__(self,ng=45,sa=None,hidden=64):
        super().__init__()
        self.ng=int(ng)
        sc=np.ones(len(IDX)) if sa is None else np.asarray(sa,float)[IDX]
        self.register_buffer('scale',torch.as_tensor(sc,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(self.ng+2*len(IDX),hidden),
           nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,g,a):
        if g.shape!=(len(a),self.ng) or a.ndim!=2 or a.shape[1]!=32:
            raise ValueError("ZFix requires geometry and 32 neural heat modes")
        h=a[:,IDX]/self.scale
        x=torch.cat((g,h,.15*h*h),dim=1)
        z=torch.zeros_like(h)
        cold=torch.cat((g,z,z),dim=1)
        return self.net(x)-self.net(cold)


def restore(path):
    from examples.bfzi_z_tangent import restore as load_tan
    ck=torch.load(path,map_location='cpu',weights_only=True)
    net=ZFix(ng=int(ck['ng']),sa=np.asarray(ck['sa']))
    net.load_state_dict(ck['state'])
    net.eval()
    for p in net.parameters():p.requires_grad_(False)
    base=load_tan(ck['base'])
    sy=torch.as_tensor(np.asarray(ck['sy'])[:6],dtype=torch.float32)
    def predict(g,a):
        return base(g,a)+net(g,a)*sy
    return predict
