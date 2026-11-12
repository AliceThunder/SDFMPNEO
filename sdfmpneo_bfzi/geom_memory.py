"""Geometry-conditioned fractional thermal history for variable BFZI FV grids.

The original BFZI circular/rounded-rectangle geometry determines the cold
complex port and 16 physical descriptors. A shared fractional memory predicts
geometry-independent observables, with no assumption of a fixed sea mesh.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .frac_memory import zmat
from .frac_port import PortModel

G_SCALE=np.array([
    .1,.1,.05,.05,1.,1.,10.,10.,.001,.001,.05,
    20e-6,10e-6,20e-6,40.,40.,
    1.,30.,1.,15.,1.,30.
],float)
N_OUT=14


def geom_feat(g,z0):
    """16 original BFZI coil/air/thermal descriptors + cold complex port."""
    a=np.r_[np.asarray(g,float),np.asarray(z0,float)]
    if a.shape!=(len(G_SCALE),):
        raise ValueError("expected BFZI physical geometry and original cold Z")
    return a/G_SCALE


def geom_port(g):
    """Exact BFZI copper/air model from physical geometry and copper capacity."""
    a=np.asarray(g,float)
    if a.shape!=(16,):
        raise ValueError("original BFZI requires 16 two-coil parameters")
    r=a[8:10]
    length=a[14:16]/(3.45e6*np.pi*r*r)
    air=np.array([[a[11],a[12]],[a[12],a[13]]])
    return PortModel(
        2*np.pi*200000,air,length,r,
        np.full(2,5.8e7),np.full(2,.00393),
        np.full(2,4e-7*np.pi),np.full(2,293.15))


def heat_drive(voltage,z,ext,port,tc,cap):
    """Original BFZI energy sources: sea W, copper heating K/s for each port."""
    if voltage==0:
        return np.zeros(3)
    v=np.array([voltage,0.],complex)
    I=np.linalg.solve(z+ext,v)
    cu=port.copper(tc)
    sea=z-1j*port.omega*port.air-np.diag(cu)
    qsea=max(.5*float(np.vdot(I,sea@I).real),0.)
    qcu=.5*cu.real*abs(I)**2
    return np.r_[qsea,qcu/np.asarray(cap,float)]


def thermal_summary(t,nsea,npkg,vol,amb):
    """Eight comparable temperatures despite different sea mesh sizes."""
    a=np.asarray(t,float)
    if a.ndim!=2 or a.shape[1]!=nsea+2*npkg+2:
        raise ValueError("BFZI time-history thermal state dimension mismatch")
    sea=a[:,:nsea]
    pkg=a[:,nsea:nsea+2*npkg].reshape(-1,2,npkg)
    copper=a[:,nsea+2*npkg:]
    w=np.asarray(vol,float)
    return np.column_stack((
        (sea@w)/w.sum()-amb,
        np.max(sea,axis=1)-amb,
        pkg.mean(axis=2)[:,0]-amb,
        pkg.mean(axis=2)[:,1]-amb,
        pkg.max(axis=2)[:,0]-amb,
        pkg.max(axis=2)[:,1]-amb,
        copper[:,0]-amb,copper[:,1]-amb))


class GeomFrac(nn.Module):
    """Shared SOE fractional/exponential memory plus geometry conditioning."""

    def __init__(self,ng=len(G_SCALE),hidden=128,nrate=14):
        super().__init__()
        self.ng=ng
        self.register_buffer("rates",torch.tensor(
            np.geomspace(1/2400.,1/12.,nrate),dtype=torch.float32))
        self.raw=nn.Parameter(torch.tensor([-.9,0.,.9]))
        self.raw_tau=nn.Parameter(torch.tensor([-.9,0.,.9]))
        self.net=nn.Sequential(
            nn.Linear(9+ng,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),
            nn.Linear(hidden,N_OUT))

    def init_state(self,batch):
        return self.rates.new_zeros((batch,3,len(self.rates)+1))

    def orders(self):
        return .12+.83*torch.sigmoid(self.raw)

    def step(self,h,u,dt,geom):
        if geom.shape!=(len(u),self.ng):
            raise ValueError("one BFZI geometry feature vector per history")
        hf,he=h[...,:-1],h[...,-1:]
        df=torch.exp(-self.rates*dt)
        hf=hf*df+(1-df)*u[...,None]
        w=torch.softmax(
            -self.orders()[:,None]*torch.log(self.rates),dim=-1)
        mf=torch.sum(hf*w[None,:,:],dim=-1)
        lam=torch.exp(np.log(1/2400.)+
              torch.sigmoid(self.raw_tau)*np.log(2400/12.))
        de=torch.exp(-lam*dt)
        he=he*de[None,:,None]+(1-de)[None,:,None]*u[...,None]
        me=he[...,0]
        amp=.5*(mf.mean(-1,keepdim=True)+me.mean(-1,keepdim=True))
        y=self.net(torch.cat((u,mf,me,geom),dim=-1))
        # Physical Joule history starts from ambient and adds nonnegative
        # material temperature rises. Dynamic Z may change either sign.
        z=y[:,:6]*amp
        temp=F.softplus(y[:,6:])*amp
        return torch.cat((hf,he),dim=-1),torch.cat((z,temp),dim=-1)

    def forward(self,src,times,geom):
        h=self.init_state(len(src))
        pred=[src.new_zeros((len(src),N_OUT))]
        for k in range(1,len(times)):
            h,y=self.step(h,src[:,k,:],
                          float(times[k]-times[k-1]),geom)
            pred.append(y)
        return torch.stack(pred,dim=1)


@torch.no_grad()
def rollout(net,ck,voltage,times,geometry,z0,external):
    """Neural port/circuit rollout for an unseen BFZI grid/geometry."""
    t=np.asarray(times,float)
    v=np.asarray(voltage,float)
    z0=np.asarray(z0,float)
    ex=np.asarray(external,complex)
    if (t[0]!=0 or v.shape!=t.shape or np.any(np.diff(t)<=0)
            or ex.shape!=(2,2)):
        raise ValueError("invalid BFZI two-port time history")
    geom=geom_feat(geometry,z0)
    port=geom_port(geometry)
    g=torch.tensor(geom[None,:],dtype=torch.float32)
    h=net.init_state(1)
    out=np.zeros((len(t),N_OUT))
    src=np.zeros((len(t),3))
    z=zmat(z0)
    zhist=[z0.copy()]
    ys=np.asarray(ck["y_scale"],float)
    us=np.asarray(ck["src_scale"],float)
    sea_mode=bool(ck.get("port_sea",False))
    cu0=port.copper(np.full(2,293.15))
    def pack(mat):
        return np.array([mat[0,0].real,mat[0,0].imag,
                         mat[0,1].real,mat[0,1].imag,
                         mat[1,1].real,mat[1,1].imag])
    for k in range(1,len(t)):
        tc=np.array([293.15,293.15])+out[k-1,-2:]
        u=heat_drive(float(v[k]),z,ex,port,tc,
                     np.asarray(geometry)[14:16])
        src[k]=u
        h,y=net.step(h,torch.tensor((u/us)[None,:],
                                   dtype=torch.float32),
                     float(t[k]-t[k-1]),g)
        out[k]=y[0].cpu().numpy()*ys
        z=zmat(z0+out[k,:6])
        if sea_mode:
            # Output 6 encodes sea delta; exact original BFZI skin
            # impedance supplies the varying copper part of the full Z.
            hot=port.copper(293.15+out[k,-2:])
            z+=np.diag(hot-cu0)
        zhist.append(pack(z))
    out[:,:6]=np.asarray(zhist)
    curr=np.asarray([
        np.linalg.solve(zmat(zi)+ex,np.array([a,0.],complex))
        for zi,a in zip(out[:,:6],v)])
    return dict(z6=out[:,:6],temp_rise_K=out[:,6:],
                current_A=curr,source=src)
