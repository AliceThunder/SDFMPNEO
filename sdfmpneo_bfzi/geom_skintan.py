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


def encode_geom_v3(g):
    """Keep physical coil, relative translation, package and thermal descriptors.

    Cold-Z and explicit orientation are omitted: the base tangent already
    carries cold geometry, while the local skin modes carry pose-dependent
    thermal-field information.
    """
    if g.ndim!=2 or g.shape[1]!=45:
        raise ValueError("geometry encoder expects [B,45]")
    return torch.cat((g[:,:16],g[:,22:25],g[:,34:45]),dim=-1)


def encode_geom(g):
    """Encode the relative coil rotation by periodic Euler-angle features.

    Input layout is the existing 45-D geometry descriptor:
    16 coil descriptors, 6 cold-Z values, 3 translations, 9 rotation
    matrix entries, 6 package half-sizes and 5 thermal descriptors.
    Output replaces the 9 matrix entries with sin/cos of yaw, pitch, tilt.
    """
    if g.ndim!=2 or g.shape[1]!=45:
        raise ValueError("geometry encoder expects [B,45]")
    r=g[:,25:34].reshape(-1,3,3)
    pitch=torch.asin(torch.clamp(-r[:,2,0],-1.,1.))
    yaw=torch.atan2(r[:,1,0],r[:,0,0])
    tilt=torch.atan2(r[:,2,1],r[:,2,2])
    ang=torch.stack((yaw,pitch,tilt),dim=-1)
    rot=torch.stack((torch.sin(ang),torch.cos(ang)),dim=-1).flatten(1)
    return torch.cat((g[:,:25],rot,g[:,34:]),dim=-1)


class SkinTan(nn.Module):
    """Separate TX, mutual and RX thermal corrections (V1)."""
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
        if (g.ndim!=2 or g.shape[1]!=45 or a.shape!=(len(g),32) or
                skin.shape!=(len(g),16)):
            raise ValueError("skin tangent expects [B,45], [B,32], [B,16]")
        aa=a[:,IDX]/self.sa
        ss=skin/self.ss
        tx=self._run(self.tx,g,aa,ss[:,:8])
        mx=self._run(self.mx,g,aa,ss)
        rx=self._run(self.rx,g,aa,ss[:,8:])
        return torch.cat((tx,mx,rx),dim=-1)


class SkinTanV2(SkinTan):
    """V2 uses periodic SO(3) angle features instead of raw matrix entries."""
    def __init__(self,ng=45,sa=None,ss=None,hidden=64):
        if ng!=45:
            raise ValueError("V2 expects the original 45-D geometry descriptor")
        nn.Module.__init__(self)
        aa=np.ones(32) if sa is None else np.asarray(sa,float)
        sk=np.ones(16) if ss is None else np.asarray(ss,float)
        if aa.shape!=(32,) or sk.shape!=(16,) or np.min(aa)<=0 or np.min(sk)<=0:
            raise ValueError("invalid thermal feature scales")
        self.ng=45
        self.register_buffer("sa",torch.as_tensor(aa[IDX],dtype=torch.float32))
        self.register_buffer("ss",torch.as_tensor(sk,dtype=torch.float32))
        self.tx=_head(42+14+8,hidden)
        self.mx=_head(42+14+16,hidden)
        self.rx=_head(42+14+8,hidden)

    def forward(self,g,a,skin):
        if (g.ndim!=2 or g.shape[1]!=45 or a.shape!=(len(g),32) or
                skin.shape!=(len(g),16)):
            raise ValueError("skin tangent V2 expects [B,45], [B,32], [B,16]")
        xg=encode_geom(g)
        aa=a[:,IDX]/self.sa
        ss=skin/self.ss
        tx=self._run(self.tx,xg,aa,ss[:,:8])
        mx=self._run(self.mx,xg,aa,ss)
        rx=self._run(self.rx,xg,aa,ss[:,8:])
        return torch.cat((tx,mx,rx),dim=-1)


class SkinTanV3(SkinTan):
    """V3 conditions local thermal residuals on compact physical descriptors."""
    def __init__(self,ng=45,sa=None,ss=None,hidden=64):
        if ng!=45:
            raise ValueError("V3 expects the original 45-D geometry descriptor")
        nn.Module.__init__(self)
        aa=np.ones(32) if sa is None else np.asarray(sa,float)
        sk=np.ones(16) if ss is None else np.asarray(ss,float)
        if aa.shape!=(32,) or sk.shape!=(16,) or np.min(aa)<=0 or np.min(sk)<=0:
            raise ValueError("invalid thermal feature scales")
        self.ng=45
        self.register_buffer("sa",torch.as_tensor(aa[IDX],dtype=torch.float32))
        self.register_buffer("ss",torch.as_tensor(sk,dtype=torch.float32))
        self.tx=_head(30+14+8,hidden)
        self.mx=_head(30+14+16,hidden)
        self.rx=_head(30+14+8,hidden)

    def forward(self,g,a,skin):
        if (g.ndim!=2 or g.shape[1]!=45 or a.shape!=(len(g),32) or
                skin.shape!=(len(g),16)):
            raise ValueError("skin tangent V3 expects [B,45], [B,32], [B,16]")
        xg=encode_geom_v3(g)
        aa=a[:,IDX]/self.sa
        ss=skin/self.ss
        tx=self._run(self.tx,xg,aa,ss[:,:8])
        mx=self._run(self.mx,xg,aa,ss)
        rx=self._run(self.rx,xg,aa,ss[:,8:])
        return torch.cat((tx,mx,rx),dim=-1)


class SkinTanV4(SkinTan):
    """V4 geometry-free local thermal correction; base tangent carries geometry."""
    def __init__(self,ng=45,sa=None,ss=None,hidden=32):
        if ng!=45:
            raise ValueError("V4 expects the original 45-D geometry descriptor")
        nn.Module.__init__(self)
        aa=np.ones(32) if sa is None else np.asarray(sa,float)
        sk=np.ones(16) if ss is None else np.asarray(ss,float)
        if aa.shape!=(32,) or sk.shape!=(16,) or np.min(aa)<=0 or np.min(sk)<=0:
            raise ValueError("invalid thermal feature scales")
        self.ng=45
        self.register_buffer("sa",torch.as_tensor(aa[IDX],dtype=torch.float32))
        self.register_buffer("ss",torch.as_tensor(sk,dtype=torch.float32))
        self.tx=_head(14+8,hidden)
        self.mx=_head(14+16,hidden)
        self.rx=_head(14+8,hidden)

    def forward(self,g,a,skin):
        if (g.ndim!=2 or g.shape[1]!=45 or a.shape!=(len(g),32) or
                skin.shape!=(len(g),16)):
            raise ValueError("skin tangent V4 expects [B,45], [B,32], [B,16]")
        xg=g.new_zeros((len(g),0))
        aa=a[:,IDX]/self.sa
        ss=skin/self.ss
        tx=self._run(self.tx,xg,aa,ss[:,:8])
        mx=self._run(self.mx,xg,aa,ss)
        rx=self._run(self.rx,xg,aa,ss[:,8:])
        return torch.cat((tx,mx,rx),dim=-1)


def restore(path,base_path,skin_path):
    from .geom_ztune import restore as load_base
    from .geom_skin import restore as load_skin
    ck=torch.load(Path(path),map_location="cpu",weights_only=True)
    cls={"v2":SkinTanV2,"v3":SkinTanV3,"v4":SkinTanV4}.get(ck.get("model_version"),SkinTan)
    net=cls(ng=int(ck["ng"]),sa=np.asarray(ck["sa"]),
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


def restore_rank48(path,base_path):
    """Use the 16 directly predicted canonical skin states in a rank48 rollout."""
    ck=torch.load(Path(path),map_location="cpu",weights_only=True)
    cls={"v2":SkinTanV2,"v3":SkinTanV3,"v4":SkinTanV4}.get(
        ck.get("model_version"),SkinTan)
    net=cls(ng=int(ck["ng"]),sa=np.asarray(ck["sa"]),
            ss=np.asarray(ck["ss"]))
    net.load_state_dict(ck["state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    from .geom_ztune import restore as load_base
    base=load_base(base_path)
    sy=torch.as_tensor(np.asarray(ck["sy"])[:6],dtype=torch.float32)

    def predict(g,a,tm,voltage):
        if a.ndim!=2 or a.shape[1]!=48:
            raise ValueError("rank48 skin tangent requires 48 canonical heat states")
        global_state=a[:,:32]
        skin_state=a[:,32:48]
        return base(g,global_state)+net(g,global_state,skin_state)*sy

    predict.time_conditioned=True
    return predict
