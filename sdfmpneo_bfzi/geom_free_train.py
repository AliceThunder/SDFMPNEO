"""Differentiable BFZI-free electothermal rollout matching geom_free.rollout.

Atlas, port models and impedance tangent are frozen. Gradients reach the
GeomFracET thermal head through the circuit and implicit shared-rank heat update.
"""
from __future__ import annotations

import numpy as np
import torch

from .frac_memory import zmat
from .geom_canon import labels
from .geom_free import geometry
from .geom_memory import geom_port


def _v6(z):
    a,b,c,d,e,f = z.unbind(-1)
    return torch.stack((torch.stack((torch.complex(a,b),
                                    torch.complex(c,d))),
                        torch.stack((torch.complex(c,d),
                                     torch.complex(e,f)))))


def _z6(z):
    return torch.stack((z[0,0].real,z[0,0].imag,z[0,1].real,
                        z[0,1].imag,z[1,1].real,z[1,1].imag))


def _cu(port, temp):
    """Exact forward copper skin impedance, local temperature derivative."""
    t=temp.detach().cpu().numpy()
    z=port.copper(t)
    eps=.01
    d=(port.copper(t+eps)-port.copper(t-eps))/(2*eps)
    return (torch.as_tensor(z,dtype=torch.complex128)
            +(temp-temp.detach()).to(torch.complex128)
              *torch.as_tensor(d,dtype=torch.complex128))


def _circuit(a, d, voltage, tm, heat_voltage=None):
    cap=d['cap']
    port=d['port']
    cu=_cu(port, d['amb']+a[3:5]/torch.sqrt(cap[3:5]))
    g=d['g']
    st=(a/d['scale']).to(torch.float32)[None]
    tan=d['tan']
    if tan is None:
        dz=torch.zeros(6,dtype=torch.float32)
    elif getattr(tan,'time_conditioned',False):
        hv=voltage if heat_voltage is None else heat_voltage
        dz=tan(g,st,float(tm),float(hv))[0]
    else:
        dz=tan(g,st)[0]
    sea=d['sea0']+_v6(dz).to(torch.complex128)
    z=sea+d['airz']+torch.diag(cu)
    v=torch.as_tensor([voltage,0.],dtype=torch.complex128)
    cur=torch.linalg.solve(z+d['ext'],v)
    ps=(.5*torch.vdot(cur,sea.real.to(torch.complex128)@cur).real).clamp_min(0.)
    pc=.5*cu.real*cur.abs().square()
    return z,cur,ps,pc,sea


def _source(a, d, voltage, tm):
    z,cur,ps,pc,_=_circuit(a,d,voltage,tm)
    amps=torch.stack((.5*cur[0].abs().square(),
                      (cur[0].conj()*cur[1]).real,
                      .5*cur[1].abs().square()))
    q=d['src']@amps
    sea=q[0]*torch.sqrt(d['cap'][0])
    q=torch.where(sea>1e-20,q*ps/sea.clamp_min(1e-20),
                  torch.zeros_like(q))
    return q,pc


def _shape(prev,guess,dt,voltage,tm,d):
    q,_=_source(guess,d,voltage,tm)
    k=d['k']
    mat=torch.eye(k.shape[0]-5,dtype=k.dtype,device=k.device)/dt+k[5:,5:]
    rhs=prev[5:]/dt+q[5:]-k[5:,:5]@guess[:5]
    return torch.linalg.solve(mat,rhs)


def _thermal(prev,guess,dt,voltage,tm,d):
    k=d['k']
    s=torch.sqrt(d['cap'])
    jac=torch.eye(5,dtype=torch.float64)/dt+k[:5,:5]
    target=guess
    for _ in range(12):
        q,pc=_source(target,d,voltage,tm)
        q=torch.cat((q[:3],q[3:5]+pc/s[3:5],q[5:]))
        rhs=prev[:5]/dt+q[:5]-k[:5,5:]@target[5:]
        nxt=torch.linalg.solve(jac,rhs)
        done=bool(torch.max(torch.abs(nxt-target[:5])/s).detach()<1e-10)
        target=torch.cat((nxt,target[5:]))
        if done:
            break
    q,pc=_source(target,d,voltage,tm)
    q=torch.cat((q[:3],q[3:5]+pc/s[3:5],q[5:]))
    budget=s*((target[:5]-prev[:5])/dt+k[:5]@target-q[:5])
    return target,budget


def _positive(a,d):
    if d['guard'] is None:
        return a.new_tensor(1.)
    mean=a[:5]/torch.sqrt(d['cap'])
    fluct=d['guard']@(a[5:]/d['scale'][5:])
    bad=fluct<0
    ratio=torch.where(bad,mean[d['gmat']].clamp_min(0.)
                      /(-fluct).clamp_min(1e-20),
                      torch.ones_like(fluct))
    return ratio.min().clamp(max=1.)


def _drive(v,z,old,d):
    if v==0.:
        return old.new_zeros(3)
    port=d['port']
    cu=_cu(port,d['amb']+old[3:5]/d['scale'][3:5])
    cur=torch.linalg.solve(z+d['ext'],
                           torch.as_tensor([v,0.],dtype=torch.complex128))
    sea=z-d['airz']-torch.diag(cu)
    ps=(.5*torch.vdot(cur,sea@cur).real).clamp_min(0.)
    pc=.5*cu.real*cur.abs().square()
    return torch.cat((ps[None],pc/d['cap'][3:5]))


def setup(case,init,tan,ambient=293.15):
    """Precompute frozen geometry, port and atlas tensors for one case."""
    geom,feat=geometry(case,init)
    port=geom_port(geom)
    cap=torch.as_tensor(init['cap'],dtype=torch.float64)
    sc=torch.sqrt(cap[labels(init['k'].shape[0])])
    air=init['air']
    air=np.array([[air[0],air[1]],[air[1],air[2]]],float)
    airz=torch.as_tensor(1j*port.omega*air,dtype=torch.complex128)
    cold=torch.as_tensor(port.copper(np.full(2,ambient)),
                         dtype=torch.complex128)
    sea0=(torch.as_tensor(zmat(init['z0']),dtype=torch.complex128)
          -airz-torch.diag(cold))
    guard=init.get('guard')
    mat=init.get('gmat')
    return dict(g=torch.as_tensor(feat[None],dtype=torch.float32),
                k=torch.as_tensor(init['k'],dtype=torch.float64),
                src=torch.as_tensor(init['src'],dtype=torch.float64),
                cap=cap,scale=sc,port=port,tan=tan,airz=airz,
                sea0=sea0,ext=torch.as_tensor(init['ext'],
                dtype=torch.complex128),amb=float(ambient),
                guard=None if guard is None else torch.as_tensor(guard,dtype=torch.float64),
                gmat=None if mat is None else torch.as_tensor(mat,dtype=torch.long),
                z0=torch.as_tensor(zmat(init['z0']),dtype=torch.complex128))


def rollout(d,net,ck,times,voltage,shape_mix=0.):
    """Autonomous prediction with physical source and implicit heat corrections."""
    t=np.asarray(times,float)
    v=np.asarray(voltage,float)
    if t.ndim!=1 or len(t)==0 or t[0]!=0 or v.shape!=t.shape or np.any(np.diff(t)<=0):
        raise ValueError('invalid voltage/time schedule')
    if not 0<=shape_mix<=1:
        raise ValueError('invalid shape mix')
    hist=net.init_state(1)
    old=torch.zeros(d['k'].shape[0],dtype=torch.float64)
    zprev=d['z0']
    src_scale=torch.as_tensor(ck['src_scale'],dtype=torch.float32)
    a_scale=torch.as_tensor(ck['a_scale'],dtype=torch.float32)
    zs=[];temps=[];states=[];budgets=[]
    for i,tm in enumerate(t):
        if i:
            dt=float(tm-t[i-1])
            q=_drive(float(v[i-1]),zprev,old,d)
            hist,_,pred=net.step_et(hist,(q.to(torch.float32)/src_scale)[None],
                                   dt,d['g'],(old/d['scale']).to(torch.float32)[None],
                                   a_scale)
            guess=pred[0].to(torch.float64)*d['scale']
            if shape_mix:
                phys=_shape(old,guess,dt,float(v[i-1]),float(tm),d)
                guess=torch.cat((guess[:5],(1-shape_mix)*guess[5:]
                                 +shape_mix*phys))
            full,budget=_thermal(old,guess,dt,float(v[i-1]),float(tm),d)
            for _ in range(5):
                beta=_positive(full,d)
                if bool(beta.detach()>=1-1e-9):
                    break
                fac=(beta*.999).clamp_min(0.)
                damp=torch.cat((full[:5],full[5:]*fac))
                full,budget=_thermal(old,damp,dt,float(v[i-1]),float(tm),d)
            old=full
        else:
            budget=old.new_zeros(5)
        z,_,_,_,_=_circuit(old,d,float(v[i]),float(tm),
                          float(v[i-1]) if i else 0.)
        zprev=z
        zs.append(_z6(z))
        temps.append(d['amb']+old[:5]/torch.sqrt(d['cap']))
        states.append(old/d['scale'])
        budgets.append(budget)
    return dict(z6=torch.stack(zs),temp_group_K=torch.stack(temps),
                heat_state=torch.stack(states),material_budget_W=torch.stack(budgets))


def loss(rows,states,net,ck,shape_mix=.5,temp_weight=0.01):
    """Mean per-history squared relative dynamic Z, plus group-temperature loss."""
    errs=[]
    for r in rows:
        name=r['file'].split('#')[0].replace('\\','/').split('/')[-1].split('_')[2]
        out=rollout(states[name],net,ck,r['time'],r['voltage'],shape_mix)
        pz=out['z6']-out['z6'][0]
        z=torch.as_tensor(r['z6']-r['z6'][0],dtype=torch.float64)
        ze=(pz-z).square().sum()/z.square().sum().clamp_min(1e-20)
        if temp_weight:
            true=torch.as_tensor(r['y'][:,[6,8,9,12,13]]+293.15,
                                 dtype=torch.float64)
            te=((out['temp_group_K']-true)/10.).square().mean()
            ze=ze+temp_weight*te
        errs.append(ze)
    return torch.stack(errs).mean()
