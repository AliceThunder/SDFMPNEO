"""Causal fractional diffusive-memory neural surrogate for original BFZI M4.

The state is a logarithmic sum-of-exponentials realization of a fractional
Volterra heat-memory kernel. Port currents and electromagnetic losses are
recomputed at every rollout step from the network's predicted complex Z.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def zmat(v):
    """Unique reciprocal 2-port complex entries -> full matrix."""
    a=np.asarray(v,float)
    z11=a[...,0]+1j*a[...,1]
    z12=a[...,2]+1j*a[...,3]
    z22=a[...,4]+1j*a[...,5]
    return np.stack([np.stack([z11,z12],axis=-1),
                     np.stack([z12,z22],axis=-1)],axis=-2)


def sources(volt,z,ext):
    """Causal two-port Joule/coil-current drive from the previous Z."""
    if not volt:
        return np.zeros(3)
    cur=np.linalg.solve(z+ext,np.array([volt,0.],complex))
    p=.5*float(np.vdot(cur,z@cur).real)
    return np.array([max(p,0),.5*abs(cur[0])**2,
                     .5*abs(cur[1])**2],float)


class FracNet(nn.Module):
    """Three thermal source channels with learnable fractional memory orders."""

    def __init__(self, nout=20, mode="frac", hidden=96, nrate=14):
        super().__init__()
        if mode not in ("frac","exp","hybrid"):
            raise ValueError("mode must be frac, exp or hybrid")
        self.mode=mode
        self.nout=int(nout)
        self.register_buffer("rates",torch.tensor(
            np.geomspace(1/2400.,1/12.,nrate),dtype=torch.float32))
        self.raw=nn.Parameter(torch.tensor([-0.9,0.,0.9]))
        if mode=="hybrid":
            self.raw_tau=nn.Parameter(torch.tensor([-0.9,0.,0.9]))
        self.net=nn.Sequential(
            nn.Linear(9 if mode=="hybrid" else 6,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),
            nn.Linear(hidden,self.nout))

    def init_state(self,batch=1):
        n=(len(self.rates)+1 if self.mode=="hybrid" else
           len(self.rates) if self.mode=="frac" else 1)
        return self.rates.new_zeros((batch,3,n))

    def orders(self):
        if self.mode in ("frac","hybrid"):
            return .12+.83*torch.sigmoid(self.raw)
        # Conventional single exponential with learned time constants.
        return torch.exp(np.log(1/2400.)+
                         torch.sigmoid(self.raw)*np.log(2400/12.))

    def step(self,h,u,dt):
        """Advance only the newly elapsed interval; no full-history storage."""
        if self.mode in ("frac","hybrid"):
            if self.mode=="hybrid":
                hf,he=h[...,:-1],h[...,-1:]
            else:
                hf=h
            dec=torch.exp(-self.rates*dt)
            hf=hf*dec+(1-dec)*u[...,None]
            alpha=self.orders()
            # Log-lambda quadrature of t^(alpha-1) / Gamma(alpha).
            w=torch.softmax(
                -alpha[:,None]*torch.log(self.rates[None,:]),dim=-1)
            mem=torch.sum(hf*w[None,:,:],dim=-1)
            if self.mode=="hybrid":
                lam=torch.exp(
                    np.log(1/2400.)+torch.sigmoid(self.raw_tau)
                    *np.log(2400/12.))
                ae=torch.exp(-lam*dt)
                he=he*ae[None,:,None]+(1-ae)[None,:,None]*u[...,None]
                mexp=he[...,0]
                h=torch.cat([hf,he],dim=-1)
                inp=torch.cat([u,mem,mexp],dim=-1)
                amp=.5*(mem.mean(dim=-1,keepdim=True)
                         +mexp.mean(dim=-1,keepdim=True))
            else:
                h=hf
                inp=torch.cat([u,mem],dim=-1)
                amp=mem.mean(dim=-1,keepdim=True)
        else:
            lam=self.orders()
            dec=torch.exp(-lam*dt)
            h=h*dec[None,:,None]+(1-dec)[None,:,None]*u[...,None]
            mem=h[...,0]
            inp=torch.cat([u,mem],dim=-1)
            amp=mem.mean(dim=-1,keepdim=True)
        y=self.net(inp)*amp
        return h,y

    def forward(self,src,times):
        batch,nt,_=src.shape
        h=self.init_state(batch)
        out=[src.new_zeros((batch,self.nout))]
        for k in range(1,nt):
            h,y=self.step(h,src[:,k,:],float(times[k]-times[k-1]))
            out.append(y)
        return torch.stack(out,dim=1)


def build_sources(volt,z6,ext):
    """Teacher inputs use the previous true Z and the current voltage."""
    v=np.asarray(volt,float)
    z=np.asarray(z6,float)
    out=np.zeros(v.shape+(3,))
    for i in range(len(v)):
        for k in range(1,len(v[i])):
            out[i,k]=sources(float(v[i,k]),zmat(z[i,k-1]),ext)
    return out


@torch.no_grad()
def predict(model, ckpt, volts, times, *, field=None, port=None):
    """Causal surrogate; optional physical BFZI sea+air+copper assembly."""
    if port is not None and field is None:
        raise ValueError("physical port assembly needs the seawater field head")
    v=np.asarray(volts,float)
    t=np.asarray(times,float)
    if v.ndim!=1 or t.shape!=v.shape or t[0]!=0 or np.any(np.diff(t)<=0):
        raise ValueError("voltage and time must be aligned from zero")
    ex=ckpt["external_z"].cpu().numpy()
    z0=ckpt["z0"].cpu().numpy()
    ts=ckpt["t_scale"].cpu().numpy()
    basis=ckpt["basis"].cpu().numpy()
    scale=ckpt["y_scale"].cpu().numpy()
    src_scale=ckpt["src_scale"].cpu().numpy()
    state=model.init_state(1)
    temp=[]; z=[]; zraw=[]; curr=[]; eta=[]; src=[]; zsea=[]; zcu=[]
    amb=float(ckpt["ambient"])
    last=(port.total(field.sea_z(np.full(len(field.vol),amb)),
                     np.full(2,amb))
          if port is not None else zmat(z0))
    def pack(x):
        return np.array([x[0,0].real,x[0,0].imag,
                         x[0,1].real,x[0,1].imag,
                         x[1,1].real,x[1,1].imag])
    for k,tm in enumerate(t):
        if k:
            inp=sources(float(v[k]),last,ex)
            state,y=model.step(
                state,torch.tensor((inp/src_scale)[None,:],
                                   dtype=torch.float32),
                float(t[k]-t[k-1]))
            yy=y[0].cpu().numpy()*scale
            th=(yy[:-6]@basis.T)*ts+amb
            zn=z0+yy[-6:]
            if port is not None:
                zs=field.sea_z(th[:len(field.vol)])
                zc=port.copper(th[-2:])
                last=zs+1j*port.omega*port.air+np.diag(zc)
                zsea.append(zs)
                zcu.append(zc)
                zk=pack(last)
            else:
                zk=zn
                last=zmat(zk)
        else:
            inp=np.zeros(3)
            th=np.full(len(ts),amb)
            zn=z0.copy()
            if port is not None:
                zs=field.sea_z(th[:len(field.vol)])
                zc=port.copper(th[-2:])
                zsea.append(zs)
                zcu.append(zc)
            zk=pack(last) if port is not None else zn
        ii=np.linalg.solve(last+ex,np.array([v[k],0],complex))
        pin=.5*np.real(np.vdot([v[k],0],ii))
        pl=.5*float(ex[1,1].real)*abs(ii[1])**2
        temp.append(th)
        z.append(zk)
        zraw.append(zn)
        curr.append(ii)
        eta.append(float(pl/pin) if pin>0 else 0.)
        src.append(inp)
    result=dict(temp_K=np.asarray(temp),z6=np.asarray(z),
                current_A=np.asarray(curr),eta=np.asarray(eta),
                source=np.asarray(src))
    if port is not None:
        result.update(z6_network=np.asarray(zraw),
                      Zsea_ohm=np.asarray(zsea),
                      Zcu_ohm=np.asarray(zcu))
    return result
