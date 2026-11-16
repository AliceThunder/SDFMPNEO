"""Geometry/time-conditioned 16-mode seawater-skin thermal state."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SkinFix(nn.Module):
    def __init__(self,ng=45,scale=None,hidden=96):
        super().__init__()
        sc=np.ones(32) if scale is None else np.asarray(scale,float)
        if sc.shape!=(32,) or np.min(sc)<=0:
            raise ValueError("32 positive neural thermal scales required")
        self.ng=int(ng)
        self.register_buffer('scale',torch.as_tensor(sc,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(ng+34,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,16))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,g,a,time,voltage):
        if (a.ndim!=2 or a.shape[1]!=32 or
                g.shape!=(len(a),self.ng) or
                time.shape!=(len(a),1) or voltage.shape!=(len(a),1)):
            raise ValueError("skin predictor requires geometry, 32 states, time and voltage")
        t=torch.log1p(time.clamp_min(0.))/np.log1p(600.)
        v=voltage/100.
        f=torch.cat((g,a/self.scale,t,v),dim=1)
        cold=torch.cat((g,torch.zeros_like(a),t,v),dim=1)
        return self.net(f)-self.net(cold)


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=SkinFix(ng=int(ck['ng']),scale=np.asarray(ck['scale']))
    model.load_state_dict(ck['state'])
    model.eval()
    return model

@torch.no_grad()
def reconstruct(rom,temp,state,time,voltage,model):
    """Improve only the spatial sea-skin temperature output of a V6 trajectory."""
    from .geom_canon import basis,mode_scale
    t=np.asarray(temp,float)
    a=np.asarray(state,float)
    times=np.asarray(time,float)
    vin=np.asarray(voltage,float)
    if t.shape!=(len(times),len(rom.cap)) or a.shape!=(len(times),32):
        raise ValueError("inconsistent thermal output and 32 heat states")
    v=np.r_[0.,vin[:-1]]
    from .s5_bridge import feature
    geom=np.broadcast_to(feature(rom,model.ng),(len(times),model.ng)).copy()
    dtype=model.scale.dtype
    corr=model(torch.as_tensor(geom,dtype=dtype),
               torch.as_tensor(a,dtype=dtype),
               torch.as_tensor(times[:,None],dtype=dtype),
               torch.as_tensor(v[:,None],dtype=dtype)).numpy()
    phi=basis(rom,48)[:,32:]*mode_scale(rom,48)[32:]
    return t+corr@phi.T
