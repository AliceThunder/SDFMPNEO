"""Terminal-wise blending of frozen cold self-port emulators."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geom_portself import PortSelf


class PortBlend:
    def __init__(self,base,new,tx=0.,rx=0.):
        self.base=base
        self.new=new
        self.w=np.asarray([tx,rx],float)
        if self.w.shape!=(2,) or np.any(self.w<0) or np.any(self.w>1):
            raise ValueError('TX/RX blend weights must lie in [0,1]')

    def apply(self,case,init):
        a=self.base.apply(case,init)
        b=self.new.apply(case,init)
        out=dict(a)
        z=np.asarray(a['z0'],float).copy()
        air=np.asarray(a['air'],float).copy()
        for j,ids in enumerate(([0,1],[4,5])):
            w=self.w[j]
            z[ids]=(1-w)*a['z0'][ids]+w*b['z0'][ids]
            air[[0,2][j]]=(1-w)*a['air'][[0,2][j]]+w*b['air'][[0,2][j]]
        out['z0']=z
        out['air']=air
        return out

    def save(self,path,old,new):
        Path(path).write_text(json.dumps(dict(
            base=str(old),new=str(new),tx=float(self.w[0]),rx=float(self.w[1])),
            indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        cfg=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return cls(PortSelf.load(cfg['base']),PortSelf.load(cfg['new']),
                   cfg['tx'],cfg['rx'])
