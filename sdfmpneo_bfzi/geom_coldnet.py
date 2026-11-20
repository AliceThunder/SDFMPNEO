"""Geometry-only cold two-port neural surrogate with physical air/sea split."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

OMEGA=2*np.pi*200000


def feat(g):
    keys=('alpha1','alpha2','core1_mm','core2_mm','box1_mm','box2_mm',
          'tilt_deg','yaw_deg','pitch_deg')
    a=[float(g[k]) for k in keys]
    a.extend([float(v) for v in g['shift_mm']])
    for j in (1,2):
        alpha=float(g[f'alpha{j}'])
        core=float(g[f'core{j}_mm'])*.001
        lens=[]
        areas=[]
        for k in range(10):
            r=.08885-core-alpha*.005*(k+.5)
            c=core-(1-alpha)*.005*(k+.5)
            areas.append(np.pi*r*r+8*c*r+4*c*c)
            lens.append(2*np.pi*r+8*c)
        a.extend((sum(areas),sum(lens)))
    return np.asarray(a,dtype=np.float32)


def target(row):
    z=np.asarray(row['z0'],float)
    air=np.asarray(row['air'],float)
    if air.shape==(3,):
        l=air
    elif air.shape==(2,2):
        l=np.array([air[0,0],air[0,1],air[1,1]])
    else:
        raise ValueError('invalid cold inductance components')
    return np.array([z[0],z[1]-OMEGA*l[0],
                     z[2],z[3]-OMEGA*l[1],
                     z[4],z[5]-OMEGA*l[2],
                     OMEGA*l[0],OMEGA*l[1],OMEGA*l[2]],dtype=np.float32)


def copper_target(row):
    """BFZI cold terminal copper: [TX/RX] x [real, imaginary]."""
    if "copper" not in row:
        return None
    cu=np.asarray(row["copper"],dtype=np.float32)
    if cu.shape!=(2,2) or not np.isfinite(cu).all():
        raise ValueError("invalid BFZI cold copper label")
    return cu.ravel()


def unpack(y):
    y=np.asarray(y,float)
    z=np.stack((y[...,0],y[...,1]+y[...,6],
                y[...,2],y[...,3]+y[...,7],
                y[...,4],y[...,5]+y[...,8]),axis=-1)
    air=np.stack((y[...,6],y[...,7],y[...,8]),axis=-1)/OMEGA
    return z,air


class ColdNet(nn.Module):
    def __init__(self,xm,xs,ym,ys,hidden=96):
        super().__init__()
        if len(ym) not in (9,13):
            raise ValueError('cold port model requires 9 or 13 outputs')
        self.register_buffer('xm',torch.as_tensor(xm,dtype=torch.float32))
        self.register_buffer('xs',torch.as_tensor(xs,dtype=torch.float32))
        self.register_buffer('ym',torch.as_tensor(ym,dtype=torch.float32))
        self.register_buffer('ys',torch.as_tensor(ys,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(len(xm),hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden),
            nn.SiLU(),nn.Linear(hidden,len(ym)))

    def forward(self,x):
        return self.net((x-self.xm)/self.xs)*self.ys+self.ym

    def predict_parts(self,case):
        g=case['geometry'] if 'geometry' in case else case
        with torch.no_grad():
            y=self(torch.as_tensor(feat(g)[None])).numpy()[0]
        z,air=unpack(y)
        cu=None
        if len(y)==13:
            a=y[9:13].reshape(2,2)
            cu=a[:,0]+1j*a[:,1]
        return z,air,cu

    def predict(self,case):
        return self.predict_parts(case)[:2]

    def apply(self,case,init):
        z,air,cu=self.predict_parts(case)
        out=dict(init)
        out['z0']=z
        out['air']=air
        if cu is not None:
            out['copper0']=cu
        return out

    def save(self,path):
        torch.save(dict(state=self.state_dict(),hidden=self.net[0].out_features),path)

    @classmethod
    def load(cls,path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        sd=ck['state']
        net=cls(sd['xm'],sd['xs'],sd['ym'],sd['ys'],ck['hidden'])
        net.load_state_dict(sd)
        net.eval()
        return net
