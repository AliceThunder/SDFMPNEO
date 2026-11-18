"""Residual TX/RX cold self-port head on top of frozen PortSelf V5."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .geom_coldnet import OMEGA, feat
from .geom_portself import PortSelf


def _input(g):
    a=feat(g)
    return np.stack((a[[0,2,4,12,13]],
                     a[[1,3,5,14,15]]))


class ColdRes(nn.Module):
    """Train per-terminal residuals, preserve existing mutual port physics."""

    def __init__(self,base_path,xm,xs,ys,hidden=64):
        super().__init__()
        self.base_path=str(base_path)
        self.base=PortSelf.load(self.base_path)
        self.register_buffer('xm',torch.as_tensor(xm,dtype=torch.float32))
        self.register_buffer('xs',torch.as_tensor(xs,dtype=torch.float32))
        self.register_buffer('ys',torch.as_tensor(ys,dtype=torch.float32))
        self.head=nn.ModuleList([
            nn.Sequential(nn.Linear(5,hidden),nn.SiLU(),
                nn.Linear(hidden,hidden),nn.SiLU(),
                nn.Linear(hidden,2))
            for _ in range(2)])
        for net in self.head:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self,x):
        out=[]
        for j in range(2):
            v=(x[:,j]-self.xm[j])/self.xs[j]
            out.append(self.head[j](v)*self.ys[j])
        return torch.stack(out,dim=1)

    def predict(self,case):
        g=case['geometry'] if 'geometry' in case else case
        with torch.no_grad():
            delta=self(torch.as_tensor(_input(g)[None]))[0].numpy()
        return self.base.predict({'geometry':g})+delta

    def apply(self,case,init):
        out=self.base.apply(case,init)
        old=self.base.predict(case)
        new=self.predict(case)
        air=np.asarray(out['air'],float).copy()
        z=np.asarray(out['z0'],float).copy()
        for j,idx in enumerate(([0,1],[4,5])):
            air[[0,2][j]]+=float(new[j,1]-old[j,1])/OMEGA
            z[idx]=new[j]
        out['z0']=z
        out['air']=air
        return out

    def save(self,path):
        torch.save(dict(state=self.state_dict(),
                        base_path=self.base_path,
                        hidden=self.head[0][0].out_features),path)

    @classmethod
    def load(cls,path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        st=ck['state']
        obj=cls(ck['base_path'],st['xm'],st['xs'],st['ys'],ck['hidden'])
        obj.load_state_dict(st)
        obj.eval()
        return obj
