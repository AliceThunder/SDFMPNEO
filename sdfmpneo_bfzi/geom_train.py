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
        physical=source(z,tc,p["volts"][:,k-1],p)
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


def shared_step(prev,guess,dt,k,q,shape_mix=.5):
    """Differentiable shared-atlas Kss/Ksm shape and Kmm/Kms mean update.

    State and source vectors use the 48-mode energy coordinates. q is held
    fixed over this step and includes sea and copper heating in its first
    five material coordinates.
    """
    if dt<=0:
        raise ValueError("thermal step must be positive")
    if not 0.<=shape_mix<=1.:
        raise ValueError("shape_mix must be within [0,1]")
    if prev.shape!=guess.shape or prev.shape!=q.shape:
        raise ValueError("thermal state and source shapes must match")
    rank=prev.shape[-1]
    if k.shape!=(rank,rank) or rank<=5:
        raise ValueError("shared thermal operator shape mismatch")
    if shape_mix:
        mat=torch.eye(rank-5,dtype=k.dtype,device=k.device)/dt+k[5:,5:]
        rhs=(prev[...,5:]/dt+q[...,5:]
             -guess[...,:5]@k[:5,5:])
        phys=torch.linalg.solve(mat,rhs.unsqueeze(-1)).squeeze(-1)
        shape=(1.-shape_mix)*guess[...,5:]+shape_mix*phys
    else:
        shape=guess[...,5:]
    mat=torch.eye(5,dtype=k.dtype,device=k.device)/dt+k[:5,:5]
    rhs=prev[...,:5]/dt+q[...,:5]-shape@k[:5,5:].T
    mean=torch.linalg.solve(mat,rhs.unsqueeze(-1)).squeeze(-1)
    return torch.cat((mean,shape),dim=-1)


def make_ctx(init,feat,port,par_geom=None,rank=48,ambient=293.15,dtype=torch.float32):
    """Pack one frozen deployment atlas prediction for differentiable training."""
    from .geom_canon import labels
    cap=np.asarray(init["cap"],float)
    if cap.shape!=(5,) or np.min(cap)<=0 or rank not in (32,48,64):
        raise ValueError("shared training needs five positive material capacities")
    scale=np.sqrt(cap[labels(rank)])
    cold=np.asarray(port.copper(np.full(2,ambient)),complex)
    lead=np.asarray(init.get("copper0",cold),complex)-cold
    air=np.asarray(init["air"],float)
    airmat=np.array([[air[0],air[1]],[air[1],air[2]]],float)
    dev=torch.device("cpu")
    cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
    z0=zmat(torch.as_tensor(init["z0"],dtype=dtype,device=dev))
    air_t=torch.as_tensor(airmat,dtype=dtype,device=dev)
    cold_t=torch.as_tensor(cold,dtype=cdtype,device=dev)
    lead_t=torch.as_tensor(lead,dtype=cdtype,device=dev)
    sea0=z0-1j*float(port.omega)*air_t-torch.diag(cold_t+lead_t)
    guard=np.asarray(init.get("guard",[]),float)
    gmat=np.asarray(init.get("gmat",[]),int)
    if guard.size and (guard.ndim!=2 or guard.shape[1]!=rank-5
                       or gmat.shape!=(len(guard),)):
        raise ValueError("shared atlas positivity guard dimensions mismatch")
    return dict(
        g=torch.as_tensor(np.asarray(feat,float)[None],dtype=dtype,device=dev),
        gp=torch.as_tensor(np.asarray(feat if par_geom is None else par_geom,float)[None],
                           dtype=dtype,device=dev),
        k=torch.as_tensor(np.asarray(init["k"],float),dtype=dtype,device=dev),
        src=torch.as_tensor(np.asarray(init["src"],float),dtype=dtype,device=dev),
        cap=torch.as_tensor(cap,dtype=dtype,device=dev),
        scale=torch.as_tensor(scale,dtype=dtype,device=dev),
        z0=z0,ext=torch.as_tensor(init["ext"],dtype=cdtype,device=dev),
        air=air_t,sea0=sea0,cold=cold_t,lead=lead_t,
        omega=float(port.omega),port=port,ambient=float(ambient),
        guard=(torch.as_tensor(guard,dtype=dtype,device=dev) if guard.size else None),
        gmat=(torch.as_tensor(gmat,dtype=torch.long,device=dev) if gmat.size else None))


def copper_t(temp,port):
    """Exact differentiable modified-Bessel copper impedance."""
    if temp.shape[-1]!=2:
        raise ValueError("two copper temperatures are required")
    dtype=temp.dtype
    cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
    dev=temp.device
    def par(name):
        return torch.as_tensor(getattr(port,name),dtype=dtype,device=dev)
    fac=1.+par("alpha")*(temp-par("ref"))
    sig=par("sigma")/fac
    mu=par("mu")
    radius=par("radius")
    wave=torch.sqrt((1j*port.omega)*mu*sig)
    arg=wave*radius
    x=arg.square()*.25
    t0=torch.ones_like(arg)
    t1=torch.ones_like(arg)
    i0=t0
    i1s=t1
    for n in range(1,49):
        t0=t0*x/(n*n)
        t1=t1*x/(n*(n+1))
        i0=i0+t0
        i1s=i1s+t1
    i1=arg*.5*i1s
    exact=wave*i0/(2.*np.pi*radius*sig*i1)
    low=1./(np.pi*radius.square()*sig)+1j*port.omega*mu/(8.*np.pi)
    unit=torch.where(torch.abs(arg)<1e-3,low,exact)
    return unit*par("length")


def _circuit_t(state,ctx,tan,par,tm,voltage):
    """Differentiable counterpart of geom_free.circuit."""
    scale=ctx["scale"]
    cap=ctx["cap"]
    mean=state[:,:5]/torch.sqrt(cap)[None]
    cu=copper_t(ctx["ambient"]+mean[:,3:5],ctx["port"])
    st=state/scale[None]
    dz=torch.zeros((len(state),6),dtype=state.dtype,device=state.device)
    if tan is not None:
        if getattr(tan,"time_conditioned",False):
            dz=tan(ctx["g"],st,float(tm),float(voltage))
        else:
            dz=tan(ctx["g"],st)
    if par is not None:
        tt=state.new_full((len(state),1),float(tm))
        vv=state.new_full((len(state),1),float(voltage))
        dz=dz+par(ctx["gp"],st,tt,vv)
    sea=ctx["sea0"][None]+zmat(dz)
    z=sea+1j*ctx["omega"]*ctx["air"]+torch.diag_embed(cu+ctx["lead"][None])
    drive=torch.zeros((len(state),2),dtype=z.dtype,device=z.device)
    drive[:,0]=float(voltage)
    cur=torch.linalg.solve(z+ctx["ext"][None],drive.unsqueeze(-1)).squeeze(-1)
    ps=.5*torch.einsum("bi,bij,bj->b",cur.conj(),
                       sea.real.to(cur.dtype),cur).real.clamp_min(0.)
    pc=.5*cu.real*cur.abs().square()
    return z,cur,ps,pc,sea


def _source_t(state,ctx,tan,par,tm,voltage,copper=True):
    """Atlas weak-source projection scaled to the live BFZI sea power."""
    z,cur,ps,pc,sea=_circuit_t(state,ctx,tan,par,tm,voltage)
    amp=torch.stack((.5*cur[:,0].abs().square(),
                     (cur[:,0].conj()*cur[:,1]).real,
                     .5*cur[:,1].abs().square()),dim=-1)
    q=amp@ctx["src"].T
    sea_in=q[:,0]*torch.sqrt(ctx["cap"][0])
    factor=torch.where(sea_in>1e-20,
        ps/sea_in.clamp_min(1e-20),torch.zeros_like(ps))
    q=q*factor[:,None]
    if copper:
        extra=torch.zeros_like(q)
        extra[:,3:5]=pc/torch.sqrt(ctx["cap"][3:5])[None]
        q=q+extra
    return q,z,cur,ps,pc,sea


def _drive_t(zprev,state,ctx,voltage):
    """Exact deployment heat_drive from the previous port and copper state."""
    scale=ctx["scale"]
    tc=ctx["ambient"]+state[:,3:5]/scale[None,3:5]
    cu=copper_t(tc,ctx["port"])
    sea=zprev-1j*ctx["omega"]*ctx["air"]-torch.diag_embed(
        cu+ctx["lead"][None])
    drive=torch.zeros((len(state),2),dtype=zprev.dtype,device=zprev.device)
    drive[:,0]=float(voltage)
    cur=torch.linalg.solve(zprev+ctx["ext"][None],drive.unsqueeze(-1)).squeeze(-1)
    ps=.5*torch.einsum("bi,bij,bj->b",cur.conj(),
                       sea.real.to(cur.dtype),cur).real.clamp_min(0.)
    pc=.5*cu.real*cur.abs().square()
    return torch.cat((ps[:,None],pc/ctx["cap"][3:5][None]),dim=-1)


def closure_t(prev,guess,dt,ctx,tan,par,tm,voltage,shape_mix=.5):
    """Differentiable deployment-equivalent shape, mean and positivity update."""
    if not 0.<=shape_mix<=1.:
        raise ValueError("shape_mix must be in [0,1]")
    k=ctx["k"]
    rank=prev.shape[-1]
    q,*_=_source_t(guess,ctx,tan,par,tm,voltage,copper=False)
    mat_s=torch.eye(rank-5,dtype=k.dtype,device=k.device)/dt+k[5:,5:]
    rhs_s=prev[:,5:]/dt+q[:,5:]-guess[:,:5]@k[:5,5:]
    phys=torch.linalg.solve(mat_s,rhs_s.unsqueeze(-1)).squeeze(-1)
    shape=(1.-shape_mix)*guess[:,5:]+shape_mix*phys
    mat_m=torch.eye(5,dtype=k.dtype,device=k.device)/dt+k[:5,:5]

    def solve_mean(target,shape_now):
        for _ in range(12):
            q,*_=_source_t(target,ctx,tan,par,tm,voltage,copper=True)
            rhs=prev[:,:5]/dt+q[:,:5]-shape_now@k[:5,5:].T
            mean=torch.linalg.solve(mat_m,rhs.unsqueeze(-1)).squeeze(-1)
            target=torch.cat((mean,shape_now),dim=-1)
        return target

    target=solve_mean(torch.cat((guess[:,:5],shape),dim=-1),shape)
    if ctx["guard"] is not None:
        for _ in range(5):
            fluct=(target[:,5:]/ctx["scale"][None,5:])@ctx["guard"].T
            bad=fluct<0.
            if not bool(torch.any(bad)):
                break
            mean=target[:,:5]/torch.sqrt(ctx["cap"])[None]
            ratio=(mean[:,ctx["gmat"]].clamp_min(0.)/
                   (-fluct).clamp_min(1e-12))
            beta=torch.where(bad,ratio,torch.full_like(ratio,float("inf"))).min()
            beta=beta.clamp(0.,1.)
            if float(beta.detach())>=1.-1e-9:
                break
            shape=target[:,5:]*torch.clamp(beta*.999,min=0.)
            target=solve_mean(torch.cat((target[:,:5],shape),dim=-1),shape)
    return target


def hist_loss(net,row,ctx,sx,sa,tan=None,par=None,shape_mix=.5,
              src_mix=1.,state_weight=1.,z_weight=1.,return_metrics=False):
    """One fully autonomous shared-atlas history with deployment feedback."""
    if net.rank!=ctx["k"].shape[0] or net.rank!=48:
        raise ValueError("shared closure training currently requires rank 48")
    if not 0.<=src_mix<=1.:
        raise ValueError("source curriculum mix must be in [0,1]")
    dtype=ctx["g"].dtype
    t=np.asarray(row["time"],float)
    volts=np.asarray(row["voltage"],float)
    teacher=torch.as_tensor(np.asarray(row["src"],float),dtype=dtype)
    target_a=torch.as_tensor(np.asarray(row["a"],float),dtype=dtype)
    target_z=torch.as_tensor(np.asarray(row["z6"],float),dtype=dtype)
    sx_t=torch.as_tensor(np.asarray(sx,float),dtype=dtype)
    sa_t=torch.as_tensor(np.asarray(sa,float),dtype=dtype)
    h=net.init_state(1)
    prev=torch.zeros((1,net.rank),dtype=dtype)
    zprev,*_=_circuit_t(prev,ctx,tan,par,0.,float(volts[0]))
    states=[prev[0]/ctx["scale"]]
    zs=[torch.stack((zprev[0,0,0].real,zprev[0,0,0].imag,
                     zprev[0,0,1].real,zprev[0,0,1].imag,
                     zprev[0,1,1].real,zprev[0,1,1].imag))]
    for i in range(1,len(t)):
        dt=float(t[i]-t[i-1])
        up=_drive_t(zprev,prev,ctx,float(volts[i-1]))
        u=(1.-src_mix)*teacher[i:i+1]+src_mix*up
        h,_,guess=net.step_et(h,u/sx_t[None],dt,ctx["g"],
                              prev/ctx["scale"][None],sa_t)
        guess=guess*ctx["scale"][None]
        prev=closure_t(prev,guess,dt,ctx,tan,par,float(t[i]),
                       float(volts[i-1]),shape_mix)
        zprev,*_=_circuit_t(prev,ctx,tan,par,float(t[i]),float(volts[i]))
        states.append(prev[0]/ctx["scale"])
        zs.append(torch.stack((zprev[0,0,0].real,zprev[0,0,0].imag,
                               zprev[0,0,1].real,zprev[0,0,1].imag,
                               zprev[0,1,1].real,zprev[0,1,1].imag)))
    pred_a=torch.stack(states)
    pred_z=torch.stack(zs)
    err=(pred_a-target_a)/sa_t
    blocks=(slice(0,5),slice(5,18),slice(18,25),
            slice(25,32),slice(32,40),slice(40,48))
    block_loss=torch.stack([err[:,sl].square().mean() for sl in blocks])
    state_loss=block_loss.mean()
    true_d=target_z-target_z[:1]
    pred_d=pred_z-pred_z[:1]
    z_loss=(pred_d-true_d).square().sum()/true_d.square().sum().clamp_min(1e-12)
    loss=state_weight*state_loss+z_weight*z_loss
    if not return_metrics:
        return loss
    raw=pred_a-target_a
    metrics=dict(loss=float(loss.detach()),state_loss=float(state_loss.detach()),
        z_rel=float(torch.sqrt(z_loss.detach())),
        state_rmse_K=float(torch.sqrt(raw.square().mean()).detach()),
        blocks_K=[float(torch.sqrt(raw[:,sl].square().mean()).detach())
                  for sl in blocks])
    return loss,metrics


def shared_score(net,rows,contexts,sx,sa,tan=None,par=None,shape_mix=.5,
                 src_mix=1.,state_weight=1.,z_weight=1.,
                 hist_weight="step",return_metrics=False):
    """Geometry-balanced collection of deployment-equivalent histories."""
    vals=[]
    spans=[]
    metrics=[]
    for row in rows:
        loss,info=hist_loss(net,row,contexts[row["file"]],sx,sa,tan,par,
            shape_mix,src_mix,state_weight,z_weight,True)
        vals.append(loss)
        spans.append(len(row["time"])-1)
        metrics.append(dict(file=row["file"],**info))
    v=torch.stack(vals)
    if hist_weight=="equal":
        total=v.mean()
    elif hist_weight=="step":
        w=v.new_tensor(spans)
        total=(v*w).sum()/w.sum()
    else:
        raise ValueError("history weighting must be equal or step")
    return (total,metrics) if return_metrics else total
