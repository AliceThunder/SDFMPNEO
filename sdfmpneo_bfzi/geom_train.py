"""Differentiable BFZI closed-loop teacher/rollout training.

Use the original BFZI two-port cold Z, air L and copper skin-impedance
slope. The copper tangent is only used during backpropagation; dynamic
inference keeps the exact original Bessel copper impedance.
"""
from __future__ import annotations

import numpy as np
import torch


def zmat(x):
    """Six real reciprocal-Z coordinates to a complex symmetric matrix."""
    z0=torch.complex(x[...,0],x[...,1])
    z1=torch.complex(x[...,2],x[...,3])
    z2=torch.complex(x[...,4],x[...,5])
    return torch.stack((z0,z1,z1,z2),dim=-1).reshape(x.shape[:-1]+(2,2))


def port_batch(rows,port_fn):
    """Geometry-constant original BFZI port data for each history."""
    cold,ext,air,cu,slope,cap,volts=([] for _ in range(7))
    for d,i in rows:
        port=port_fn(d["geom"])
        tc=np.full(2,293.15)
        cold.append(d["z0"])
        ext.append(d["ext"])
        air.append(port.omega*port.air)
        cu.append(port.copper(tc))
        slope.append((port.copper(tc+.1)-port.copper(tc-.1))/.2)
        cap.append(np.asarray(d["geom"])[14:16])
        volts.append(d["v"][i])
    return dict(
        cold=torch.tensor(np.stack(cold),dtype=torch.float32),
        ext=torch.tensor(np.stack(ext),dtype=torch.complex64),
        air=torch.tensor(np.stack(air),dtype=torch.complex64),
        cu=torch.tensor(np.stack(cu),dtype=torch.complex64),
        slope=torch.tensor(np.stack(slope),dtype=torch.complex64),
        cap=torch.tensor(np.stack(cap),dtype=torch.float32),
        volts=torch.tensor(np.stack(volts),dtype=torch.float32))


def source(z,tc,v,p):
    """Causal BFZI heating sources from predicted complex port and copper."""
    cu=p["cu"]+p["slope"]*tc
    sea=z-1j*p["air"]-torch.diag_embed(cu)
    b=torch.stack((v,torch.zeros_like(v)),dim=-1).to(torch.complex64)
    I=torch.linalg.solve(z+p["ext"],b)
    ps=.5*torch.einsum("bi,bij,bj->b",I.conj(),sea,I).real
    pc=.5*cu.real*I.abs().square()
    return torch.cat((ps.clamp_min(0)[:,None],pc/p["cap"]),dim=-1)


def closed(net,geom,t,teacher,src_scale,y_scale,p,mix=1.,detach=False):
    """Causal rollout, 0=teacher forcing, 1=fully autonomous heating input."""
    if not 0<=mix<=1:
        raise ValueError("BFZI causal source mixture must be within [0,1]")
    if geom.shape!=(len(p["volts"]),net.ng):
        raise ValueError("one geometry vector per BFZI history")
    h=net.init_state(len(geom))
    sc=torch.as_tensor(src_scale,dtype=torch.float32,device=geom.device)
    ys=torch.as_tensor(y_scale,dtype=torch.float32,device=geom.device)
    z0=zmat(p["cold"])
    z=z0
    tc=torch.zeros((len(geom),2),dtype=torch.float32,device=geom.device)
    out=[torch.zeros((len(geom),14),device=geom.device)]
    for k in range(1,len(t)):
        physical=source(z,tc,p["volts"][:,k],p)
        u=(1-mix)*(teacher[:,k]*sc)+mix*physical
        if detach:
            u=u.detach()
        h,y=net.step(h,u/sc,float(t[k]-t[k-1]),geom)
        y=y*ys
        tc=y[:,12:14]
        z=z0+zmat(y[:,:6])+torch.diag_embed(p["slope"]*tc)
        out.append(y/ys)
    return torch.stack(out,dim=1)


def weights(rows):
    """Assign the same total supervised loss weight to each BFZI geometry."""
    count={}
    for d,_ in rows:
        name=d["sample"]
        count[name]=count.get(name,0)+1
    w=np.array([1/count[d["sample"]] for d,_ in rows],float)
    return torch.tensor(w/w.mean(),dtype=torch.float32)


def loss(pred,target,deriv,y_scale,w,phys=.65,pkg=.9,sea=.15):
    """Geometry-balanced complex-port Jacobian and material-hotspot loss."""
    err=pred-target
    rows=err.square().mean((1,2))+.45*err[:,:,:6].square().mean((1,2))
    if pkg:
        rows=rows+pkg*err[:,:,8:12].square().mean((1,2))
    if sea:
        rows=rows+sea*err[:,:,6:8].square().mean((1,2))
    if phys:
        scale=torch.as_tensor(y_scale,dtype=pred.dtype,device=pred.device)
        absolute=err*scale
        ez=absolute[:,:,:6].clone()
        dt=absolute[:,:,12:14]
        ez[:,:,0]+=deriv[:,:,0]*dt[:,:,0]
        ez[:,:,1]+=deriv[:,:,1]*dt[:,:,0]
        ez[:,:,4]+=deriv[:,:,2]*dt[:,:,1]
        ez[:,:,5]+=deriv[:,:,3]*dt[:,:,1]
        rows=rows+phys*(ez/scale[:6]).square().mean((1,2))
    return (rows*w).mean()
