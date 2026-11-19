"""Pose-conditioned cold self-port residuals on top of frozen PortSelf V5."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .geom_coldnet import OMEGA, feat
from .geom_coldres import ColdRes
from .geom_portself import PortSelf


def _input_pose(g):
    a=feat(g)
    local=([0,2,4,12,13],[1,3,5,14,15])
    return np.stack([
        np.r_[a[idx],a[[k for k in range(len(a)) if k not in idx]]]
        for idx in local
    ]).astype(np.float32)


class ColdResPose(nn.Module):
    """Per-terminal residual heads conditioned on both coils and relative pose."""

    def __init__(self,base_path,xm,xs,ys,hidden=64):
        super().__init__()
        self.base_path=str(base_path)
        self.base=PortSelf.load(self.base_path)
        self.register_buffer('xm',torch.as_tensor(xm,dtype=torch.float32))
        self.register_buffer('xs',torch.as_tensor(xs,dtype=torch.float32))
        self.register_buffer('ys',torch.as_tensor(ys,dtype=torch.float32))
        self.head=nn.ModuleList([
            nn.Sequential(nn.Linear(16,hidden),nn.SiLU(),
                nn.Linear(hidden,hidden),nn.SiLU(),
                nn.Linear(hidden,2))
            for _ in range(2)])
        for net in self.head:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    @classmethod
    def from_old(cls,old,xm,xs):
        hidden=old.head[0][0].out_features
        obj=cls(old.base_path,xm,xs,old.ys,hidden)
        with torch.no_grad():
            for j in range(2):
                obj.head[j][0].weight[:,5:].zero_()
                obj.head[j][0].weight[:,:5].copy_(old.head[j][0].weight)
                obj.head[j][0].bias.copy_(old.head[j][0].bias)
                obj.head[j][2].load_state_dict(old.head[j][2].state_dict())
                obj.head[j][4].load_state_dict(old.head[j][4].state_dict())
        return obj

    def forward(self,x):
        out=[]
        for j in range(2):
            v=(x[:,j]-self.xm[j])/self.xs[j]
            out.append(self.head[j](v)*self.ys[j])
        return torch.stack(out,dim=1)

    def predict(self,case):
        g=case['geometry'] if 'geometry' in case else case
        with torch.no_grad():
            delta=self(torch.as_tensor(_input_pose(g)[None]))[0].numpy()
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
        torch.save(dict(state=self.state_dict(),base_path=self.base_path,
                        hidden=self.head[0][0].out_features,
                        pose_context=True),path)

    @classmethod
    def load(cls,path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        st=ck['state']
        obj=cls(ck['base_path'],st['xm'],st['xs'],st['ys'],ck['hidden'])
        obj.load_state_dict(st)
        obj.eval()
        return obj


def load_cold(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('pose_context',False):
        return ColdResPose.load(path)
    return ColdRes.load(path)
