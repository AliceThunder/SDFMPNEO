"""Port-factorized seawater impedance correction using local skin modes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

IDX=[0,*range(5,18)]


def _head(n,h):
    net=nn.Sequential(nn.Linear(n,h),nn.SiLU(),
        nn.Linear(h,h),nn.SiLU(),nn.Linear(h,2))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class SkinTan(nn.Module):
    """Separate TX, mutual and RX thermal corrections."""
    def __init__(self,ng=45,sa=None,ss=None,hidden=64):
        super().__init__()
        aa=np.ones(32) if sa is None else np.asarray(sa,float)
        sk=np.ones(16) if ss is None else np.asarray(ss,float)
        if aa.shape!=(32,) or sk.shape!=(16,) or np.min(aa)<=0 or np.min(sk)<=0:
            raise ValueError("invalid thermal feature scales")
        self.register_buffer("sa",torch.as_tensor(aa[IDX],dtype=torch.float32))
        self.register_buffer("ss",torch.as_tensor(sk,dtype=torch.float32))
        self.tx=_head(ng+14+8,hidden)
        self.mx=_head(ng+14+16,hidden)
        self.rx=_head(ng+14+8,hidden)

    @staticmethod
    def _run(net,g,a,s):
        x=torch.cat((g,a,s),dim=-1)
        z=torch.cat((g,torch.zeros_like(a),torch.zeros_like(s)),dim=-1)
        return net(x)-net(z)

    def forward(self,g,a,skin):
        if (g.ndim!=2 or a.shape!=(len(g),32) or
                skin.shape!=(len(g),16)):
            raise ValueError("skin tangent expects [B,ng], [B,32], [B,16]")
        aa=a[:,IDX]/self.sa
        ss=skin/self.ss
        tx=self._run(self.tx,g,aa,ss[:,:8])
        mx=self._run(self.mx,g,aa,ss)
        rx=self._run(self.rx,g,aa,ss[:,8:])
        return torch.cat((tx,mx,rx),dim=-1)


def restore(path,base_path,skin_path):
    from .geom_ztune import restore as load_base
    from .geom_skin import restore as load_skin
    ck=torch.load(Path(path),map_location="cpu",weights_only=True)
    net=SkinTan(ng=int(ck["ng"]),sa=np.asarray(ck["sa"]),
                ss=np.asarray(ck["ss"]))
    net.load_state_dict(ck["state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    base=load_base(base_path)
    skin=load_skin(skin_path)
    sy=torch.as_tensor(np.asarray(ck["sy"])[:6],dtype=torch.float32)

    def predict(g,a,tm,voltage):
        t=torch.full((len(a),1),float(tm),dtype=a.dtype,device=a.device)
        v=torch.full((len(a),1),float(voltage),dtype=a.dtype,device=a.device)
        s=skin(g,a,t,v)
        return base(g,a)+net(g,a,s)*sy

    predict.time_conditioned=True
    return predict
