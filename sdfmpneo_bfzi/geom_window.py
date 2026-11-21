"""Differentiable short-window counterpart of the geometry-only BFZI loop.

The SOE history is warmed with complete physical history. Only a small rollout
window builds a gradient graph. Cold circuitry and the EM temperature map stay
frozen when training the thermal head.
"""
from __future__ import annotations

import numpy as np
import torch

from .geom_canon import labels
from .geom_free import geometry
from .geom_memory import geom_port


def zmat(y):
    a=torch.complex(y[...,0],y[...,1])
    b=torch.complex(y[...,2],y[...,3])
    c=torch.complex(y[...,4],y[...,5])
    return torch.stack((torch.stack((a,b),-1),
                        torch.stack((b,c),-1)),-2)


def pack(z):
    return torch.stack((z[0,0].real,z[0,0].imag,
                        z[0,1].real,z[0,1].imag,
                        z[1,1].real,z[1,1].imag))


class HeatWindow:
    """Torch-only physical circuit, 43D shape and five material budgets."""

    def __init__(self,case,init,net,ck,tan,par,shape_mix=.5,ambient=293.15,
                 phys=None,grad_mix=0.):
        if net.rank!=48:
            raise ValueError("shared rank48 heat model required")
        self.net=net
        self.tan=tan
        self.par=par
        self.mix=float(shape_mix)
        if not 0<=self.mix<=1:
            raise ValueError("invalid thermal shape mix")
        self.ambient=ambient
        self.grad_mix=float(grad_mix)
        if not 0<=self.grad_mix<=1:
            raise ValueError("physical Jacobian blend must be in [0,1]")
        if self.grad_mix and phys is None:
            raise ValueError("physical Jacobian requires BFZI training gains")
        self.alpha=float(case["alpha_per_K"])
        self.rgain=(torch.as_tensor(phys["gain"],dtype=torch.float32)
                    if phys is not None else None)
        self.xgain=(torch.as_tensor(phys["react_gain"],dtype=torch.float32)
                    if phys is not None else None)
        geom,feat=geometry(case,init)
        self.port=geom_port(geom)
        self.g=torch.as_tensor(feat[None],dtype=torch.float32)
        self.cap=torch.as_tensor(init["cap"],dtype=torch.float32)
        self.scale=torch.sqrt(self.cap[labels(48)])
        self.k=torch.as_tensor(init["k"],dtype=torch.float32)
        self.src=torch.as_tensor(init["src"],dtype=torch.float32)
        self.ext=torch.as_tensor(init["ext"],dtype=torch.complex64)
        self.sx=torch.as_tensor(ck["src_scale"],dtype=torch.float32)
        self.sa=torch.as_tensor(ck["a_scale"],dtype=torch.float32)
        self.z0=torch.as_tensor(init["z0"],dtype=torch.float32)
        self.air=torch.as_tensor(1j*self.port.omega*
                                 np.array([[init["air"][0],init["air"][1]],
                                           [init["air"][1],init["air"][2]]]),
                                 dtype=torch.complex64)
        self.cold=torch.as_tensor(self.port.copper(np.full(2,ambient)),
                                  dtype=torch.complex64)
        self.lead=torch.as_tensor(np.asarray(
            init.get("copper0",self.port.copper(np.full(2,ambient))),
            complex)-self.port.copper(np.full(2,ambient)),
            dtype=torch.complex64)
        self.sea0=zmat(self.z0)-self.air-torch.diag(self.cold+self.lead)
        self.jac=torch.eye(5)/1.875+self.k[:5,:5]
        self.guard=(torch.as_tensor(init["guard"],dtype=torch.float32)
                    if init.get("guard") is not None else None)
        self.gmat=(torch.as_tensor(init["gmat"],dtype=torch.long)
                   if self.guard is not None else None)
        self.ref_temp=None

    def anchor(self,a):
        """Local Taylor model of exact BFZI copper skin impedance."""
        temp=(self.ambient+a[3:5]/self.scale[3:5]).detach().numpy()
        h=.1
        cu=self.port.copper(temp)
        slope=(self.port.copper(temp+h)-self.port.copper(temp-h))/(2*h)
        self.ref_temp=torch.as_tensor(temp,dtype=torch.float32)
        self.ref_cu=torch.as_tensor(cu,dtype=torch.complex64)
        self.cu_slope=torch.as_tensor(slope,dtype=torch.complex64)

    def phys_z(self,a):
        """BFZI weak-source regional complex susceptibility at this state."""
        pieces=torch.stack((a[:1]@self.src[:1],
                            a[1:32]@self.src[1:32],
                            a[32:48]@self.src[32:48]))
        r=(pieces*self.rgain).sum(dim=0)
        x=(pieces*self.xgain).sum(dim=0)
        return self.alpha*torch.stack((r[0],x[0],r[1],x[1],r[2],x[2]))

    def circuit(self,a,voltage,time):
        if self.ref_temp is None:
            raise ValueError("initialize copper tangent using anchor()")
        state=(a/self.scale)[None]
        dz=self.tan(self.g,state)[0]
        if self.par is not None:
            t=torch.as_tensor([[time]],dtype=torch.float32)
            v=torch.as_tensor([[voltage]],dtype=torch.float32)
            dz=dz+self.par(self.g,state,t,v)[0]
        if self.grad_mix:
            # Preserve the exact frozen V8 forward sea Z. Use the physical
            # regional susceptibility, not learned compensation, for dZ/da.
            phy=self.phys_z(a)
            dz=((1-self.grad_mix)*dz+self.grad_mix*
                (dz.detach()+phy-phy.detach()))
        temp=self.ambient+a[3:5]/self.scale[3:5]
        cu=self.ref_cu+self.cu_slope*(temp-self.ref_temp)
        sea=self.sea0+zmat(dz)
        z=sea+self.air+torch.diag(cu+self.lead)
        drive=torch.tensor([voltage,0.],dtype=torch.float32).to(torch.complex64)
        current=torch.linalg.solve(z+self.ext,drive)
        psea=(.5*(current.conj() @ (sea.real.to(torch.complex64) @ current)).real).clamp_min(0)
        pcu=.5*cu.real*current.abs().square()
        return z,current,psea,pcu

    def source(self,a,voltage,time):
        z,I,psea,pcu=self.circuit(a,voltage,time)
        amp=torch.stack((.5*I[0].abs().square(),
                         (I[0].conj()*I[1]).real,
                         .5*I[1].abs().square()))
        q=self.src@amp
        raw=q[0]*self.scale[0]
        fac=torch.where(raw>1e-20,psea/raw.clamp_min(1e-20),
                        torch.zeros_like(raw))
        q=q*fac
        q=q+torch.cat((torch.zeros(3),pcu/self.scale[3:5],
                        torch.zeros(43)))
        u=torch.stack((psea,pcu[0]/self.cap[3],pcu[1]/self.cap[4]))
        return q,u

    def balance(self,old,guess,dt,voltage,time):
        mat=torch.eye(5)/dt+self.k[:5,:5]
        a=guess
        for _ in range(12):
            q,_=self.source(a,voltage,time)
            rhs=old[:5]/dt+q[:5]-self.k[:5,5:]@a[5:]
            mean=torch.linalg.solve(mat,rhs)
            a=torch.cat((mean,guess[5:]))
        return a

    def positivity(self,old,a,dt,voltage,time):
        if self.guard is None:
            return a
        for _ in range(5):
            fluct=self.guard@(a[5:]/self.scale[5:])
            bad=fluct<0
            if not bool(bad.any()):
                break
            mean=a[:5]/self.scale[:5]
            ratio=torch.where(bad,mean[self.gmat].clamp_min(0)/
                              (-fluct).clamp_min(1e-12),
                              torch.ones_like(fluct))
            beta=ratio.min().clamp(max=1.)
            if float(beta.detach())>=1-1e-9:
                break
            clipped=torch.cat((a[:5],a[5:]*(beta*.999)))
            a=self.balance(old,clipped,dt,voltage,time)
        return a

    def step(self,h,old,dt,voltage,time):
        _,u=self.source(old,voltage,time-dt)
        h,_,next_a=self.net.step_et(h,(u/self.sx)[None],
                                   dt,self.g,(old/self.scale)[None],
                                   self.sa)
        guess=next_a[0]*self.scale
        if self.mix:
            q,_=self.source(guess,voltage,time)
            mat=torch.eye(43)/dt+self.k[5:,5:]
            rhs=old[5:]/dt+q[5:]-self.k[5:,:5]@guess[:5]
            shape=torch.linalg.solve(mat,rhs)
            guess=torch.cat((guess[:5],
                             (1-self.mix)*guess[5:]+self.mix*shape))
        a=self.balance(old,guess,dt,voltage,time)
        a=self.positivity(old,a,dt,voltage,time)
        return h,a

    def warm(self,row,start,pred_src=None,pred_state=None,mix=0.):
        """Warm full SOE memory using consistent teacher/autonomous exposure."""
        if not 0<=mix<=1:
            raise ValueError("SOE exposure must be in [0,1]")
        times=np.asarray(row["time"],float)
        source=torch.as_tensor(row["src"],dtype=torch.float32)
        states=torch.as_tensor(row["a"],dtype=torch.float32)
        if pred_src is None:
            pred_src=source
        else:
            pred_src=torch.as_tensor(pred_src,dtype=torch.float32)
        if pred_state is None:
            pred_state=states
        else:
            pred_state=torch.as_tensor(pred_state,dtype=torch.float32)
        if pred_src.shape!=source.shape or pred_state.shape!=states.shape:
            raise ValueError("autonomous SOE prefix shape mismatch")
        h=self.net.init_state(1)
        with torch.no_grad():
            for i in range(1,start+1):
                dt=float(times[i]-times[i-1])
                u=(1-mix)*source[i]+mix*pred_src[i]
                old=(1-mix)*states[i-1:i]+mix*pred_state[i-1:i]
                h,_,_=self.net.step_et(h,(u/self.sx)[None],
                                        dt,self.g,old,self.sa)
        return h.detach()
