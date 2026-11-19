"""BFZI-free, shared-coordinate autonomous electothermal rollout.

The geometry-only atlas predicts the cold port and thermal operators.
No BFZI cold solve, FV grid, geometry POD or anchored GeomHeatROM is loaded.
"""
from __future__ import annotations

import numpy as np
import torch

from .frac_memory import zmat
from .geom_canon import labels
from .geom_heat import pack
from .geom_init import GeomInit,encode
from .geom_memory import geom_feat, geom_port, heat_drive
from .geom_neural import GeomFracET
from .pose import _rot
from .s5_bridge import THERM_SCALE


def geometry(case,init):
    """Provide V9's 45 features using only the predicted cold state."""
    c=case['geometry']
    a=np.r_[.08885,.08885,
            .001*c['core1_mm'],.001*c['core2_mm'],
            c['alpha1'],c['alpha2'],
            10.,10.,.000790569415,.000790569415,
            np.linalg.norm(np.array(c['shift_mm'])*.001+[0,.04,0]),
            init['air'],init['cap'][3:]]
    pos=np.array(c['shift_mm'],float)*.001+[0.,.04,0.]
    rot=(_rot('z',c['yaw_deg']) @ _rot('y',c['pitch_deg']) @
         _rot('x',c['tilt_deg']))
    boxes=np.r_[c['box1_mm']*.0005,.0055,c['box1_mm']*.0005,
                c['box2_mm']*.0005,.0055,c['box2_mm']*.0005]
    pose=np.r_[pos,rot.ravel(),boxes]
    th=case['thermal']
    thermal=np.array([case['alpha_per_K'],th['ins_k'],
                      th.get('ins_t',.0004),th['pkg_k'],th['pkg_sea_h']])
    feat=np.r_[geom_feat(a,init['z0'],pose),thermal/THERM_SCALE]
    if a.shape!=(16,) or feat.shape!=(45,):
        raise ValueError("invalid native BFZI geometry-only feature layout")
    return a,feat


def circuit(a,init,port,state,tan,g,scale,voltage,ambient,extra=None,
            tm=0.,heat_voltage=None):
    """Nonlinear two-port circuit with temperature-sensitive sea and copper."""
    mean=state[:5]/np.sqrt(init['cap'])
    cu=port.copper(ambient+mean[3:5])
    cold=port.copper(np.full(2,ambient))
    air=np.array([[init['air'][0],init['air'][1]],
                  [init['air'][1],init['air'][2]]])
    sea0=zmat(init['z0'])-1j*port.omega*air-np.diag(cold)
    if tan is None:
        dsea=np.zeros((2,2),complex)
    else:
        st=torch.as_tensor((state/scale)[None],dtype=g.dtype)
        if getattr(tan,"time_conditioned",False):
            hv=voltage if heat_voltage is None else heat_voltage
            dz=tan(g,st,float(tm),float(hv))
        else:
            dz=tan(g,st)
        dsea=zmat(dz[0].detach().cpu().numpy())
    if extra is not None:
        corr=extra(torch.as_tensor((state/scale)[None],dtype=g.dtype),
                   tm,voltage)
        dsea+=zmat(np.asarray(corr,float))
    sea=sea0+dsea
    z=sea+1j*port.omega*air+np.diag(cu)
    cur=np.linalg.solve(z+init['ext'],np.array([voltage,0.],complex))
    psea=max(0.,float(.5*np.vdot(cur,sea.real@cur).real))
    pcu=.5*cu.real*abs(cur)**2
    return z,cur,psea,pcu,sea


def positivity(a,init,scale):
    """Basis-shape positivity from the nearest three frozen geometry anchors."""
    mat=init.get('gmat')
    shape=init.get('guard')
    if shape is None or mat is None:
        return 1.
    mean=a[:5]/np.sqrt(init['cap'])
    fluctu=shape@(a[5:]/scale[5:])
    bad=fluctu<0
    if not np.any(bad):
        return 1.
    return min(1.,float(np.min(
        np.maximum(mean[mat[bad]],0.)/(-fluctu[bad]))))


def shape_step(prev,guess,dt,voltage,init,port,tan,g,scale,amb,extra=None,tm=0.):
    """Shared physically forced thermal shape: (rank-5)D implicit dissipative step."""
    z,cur,ps,pc,sea=circuit(guess,init,port,guess,tan,g,scale,
                            voltage,amb,extra,tm)
    amp=np.array([.5*abs(cur[0])**2,
                  (cur[0].conjugate()*cur[1]).real,
                  .5*abs(cur[1])**2])
    q=init['src']@amp
    oldsea=q[0]*np.sqrt(init['cap'][0])
    if oldsea<=1e-20:
        q[:]=0.
    else:
        q*=ps/oldsea
    k=init['k']
    mat=np.eye(len(guess)-5)/dt+k[5:,5:]
    rhs=prev[5:]/dt+q[5:]-k[5:,:5]@guess[:5]
    return np.linalg.solve(mat,rhs)


def thermal_step(prev,guess,dt,voltage,init,port,tan,g,scale,amb,extra=None,tm=0.):
    """Implicit correction of five budgets, explicit neural shape coordinates."""
    cap=init['cap']
    k=init['k']
    s=np.sqrt(cap)
    z=cur=pc=None
    target=np.asarray(guess,float).copy()
    jac=np.eye(5)/dt+k[:5,:5]
    for _ in range(12):
        z,cur,ps,pc,sea=circuit(target,init,port,target,tan,g,scale,
                                voltage,amb,extra,tm)
        # Cold weak-source projection with positive total sea loss.
        q=init['src']@np.array([.5*abs(cur[0])**2,
                   (cur[0].conjugate()*cur[1]).real,
                   .5*abs(cur[1])**2])
        volpower=q[0]*s[0]
        q=q*ps/volpower if volpower>1e-20 else np.zeros_like(q)
        q[3]+=pc[0]/s[3]
        q[4]+=pc[1]/s[4]
        rhs=prev[:5]/dt +q[:5]-k[:5,5:]@target[5:]
        nxt=np.linalg.solve(jac,rhs)
        if np.max(abs(nxt-target[:5])/s)<1e-10:
            target[:5]=nxt
            break
        target[:5]=nxt
    z,cur,ps,pc,sea=circuit(target,init,port,target,tan,g,scale,
                             voltage,amb,extra,tm)
    q=init['src']@np.array([.5*abs(cur[0])**2,
                           (cur[0].conjugate()*cur[1]).real,
                           .5*abs(cur[1])**2])
    p0=q[0]*s[0]
    q=q*ps/p0 if p0>1e-20 else np.zeros_like(q)
    q[3]+=pc[0]/s[3];q[4]+=pc[1]/s[4]
    residual=s*((target[:5]-prev[:5])/dt+
                k[:5,:]@target-q[:5])
    return target,residual


def rom_step(prev,dt,voltage,init,port,tan,g,scale,amb,extra=None,tm=0.):
    """Fully implicit, nonlinear 32D shared heat-ROM step with port feedback."""
    k=init['k']
    cap=init['cap']
    sc=np.sqrt(cap)
    mat=np.eye(len(prev))/dt+k
    target=prev.copy()
    for _ in range(12):
        _,cur,ps,pc,_=circuit(target,init,port,target,tan,g,scale,
                              voltage,amb,extra,tm)
        amp=np.array([.5*abs(cur[0])**2,
                      (cur[0].conjugate()*cur[1]).real,
                      .5*abs(cur[1])**2])
        q=init['src']@amp
        sea_in=q[0]*sc[0]
        q=q*ps/sea_in if sea_in>1e-20 else np.zeros_like(q)
        q[3]+=pc[0]/sc[3]
        q[4]+=pc[1]/sc[4]
        nxt=np.linalg.solve(mat,prev/dt+q)
        if np.max(abs(nxt-target)/scale)<1e-9:
            target=nxt
            break
        target=nxt
    _,cur,ps,pc,_=circuit(target,init,port,target,tan,g,scale,
                          voltage,amb,extra,tm)
    amp=np.array([.5*abs(cur[0])**2,
                  (cur[0].conjugate()*cur[1]).real,
                  .5*abs(cur[1])**2])
    q=init['src']@amp
    sea_in=q[0]*sc[0]
    q=q*ps/sea_in if sea_in>1e-20 else np.zeros_like(q)
    q[3]+=pc[0]/sc[3];q[4]+=pc[1]/sc[4]
    budget=sc*((target[:5]-prev[:5])/dt+k[:5]@target-q[:5])
    return target,budget


@torch.no_grad()
def rollout(case,atlas,net,ck,times,voltage,tan=None,ambient=293.15,
            init_override=None,extra=None,shape_mix=0.,heat_mode="neural",rom_mix=0.):
    """Predict port Z and material temperatures with no BFZI anchor or mesh."""
    if not isinstance(atlas,GeomInit) or not isinstance(net,GeomFracET):
        raise TypeError("geometry-only atlas and GeomFracET required")
    if (atlas.rank not in (32,48,64) or net.rank!=atlas.rank
            or not ck.get('canonical',False)):
        raise ValueError("atlas and neural head must share a supported canonical rank")
    if not 0.<=shape_mix<=1.:
        raise ValueError("thermal shape mixing coefficient must be in [0,1]")
    if heat_mode not in ("neural","rom","hybrid"):
        raise ValueError("heat mode must be neural, rom or hybrid")
    if not 0.<=rom_mix<=1.:
        raise ValueError("physical heat mix must be in [0,1]")
    t=np.asarray(times,float);v=np.asarray(voltage,float)
    if t.ndim!=1 or not len(t) or t[0]!=0 or v.shape!=t.shape or np.any(np.diff(t)<=0):
        raise ValueError("invalid voltage/time schedule")
    init=atlas.predict(case) if init_override is None else init_override
    geom,feat=geometry(case,init)
    port=geom_port(geom)
    g=torch.as_tensor(feat[None],dtype=torch.float32)
    if extra is not None:
        eg=(g if getattr(extra,'shared_geom',False) else
            torch.as_tensor(encode(case)[None],dtype=g.dtype))
        def ext_fn(state,tm,vin):
            tim=torch.as_tensor([[tm]],dtype=g.dtype)
            vol=torch.as_tensor([[vin]],dtype=g.dtype)
            return extra(eg,state,tim,vol)[0].detach().cpu().numpy()
    else:
        ext_fn=None
    cap=init['cap']
    rank=atlas.rank
    scale=np.sqrt(cap[labels(rank)])
    cs=np.asarray(ck['src_scale'],float)
    ys=np.asarray(ck['y_scale'],float)
    ac=np.asarray(ck['a_scale'],float)
    hist=net.init_state(1)
    old=np.zeros(rank)
    full=np.zeros(rank)
    zprev=zmat(init['z0'])
    out=dict(z6=[],current_A=[],temp_group_K=[],heat_state=[],
             material_budget_W=[],Psea_W=[],Pcu_W=[])
    for i,tm in enumerate(t):
        if i:
            dt=float(tm-t[i-1])
            if heat_mode=="rom":
                full,budget=rom_step(old,dt,float(v[i-1]),init,port,
                                     tan,g,scale,ambient,ext_fn,tm)
            else:
                u=heat_drive(float(v[i-1]),zprev,init['ext'],port,
                    ambient+full[3:5]/scale[3:5],cap[3:5])
                hist,_,pred=net.step_et(hist,
                    torch.as_tensor((u/cs)[None],dtype=g.dtype),
                    dt,g,torch.as_tensor((old/scale)[None],dtype=g.dtype),ac)
                guess=pred[0].cpu().numpy()*scale
                if shape_mix:
                    phys=shape_step(old,guess,dt,float(v[i-1]),
                        init,port,tan,g,scale,ambient,ext_fn,tm)
                    guess[5:]=(1.-shape_mix)*guess[5:]+shape_mix*phys
                if heat_mode=="hybrid" and rom_mix:
                    implicit,_=rom_step(old,dt,float(v[i-1]),init,port,
                                        tan,g,scale,ambient,ext_fn,tm)
                    guess[5:]=(1.-rom_mix)*guess[5:]+rom_mix*implicit[5:]
                full,budget=thermal_step(old,guess,dt,float(v[i-1]),
                                        init,port,tan,g,scale,ambient,ext_fn,tm)
            for _ in range(5):
                beta=positivity(full,init,scale)
                if beta>=1.-1e-9:
                    break
                damped=full.copy()
                damped[5:]*=max(beta*.999,0.)
                full,budget=thermal_step(old,damped,dt,float(v[i-1]),
                                          init,port,tan,g,scale,ambient,ext_fn,tm)
            old=full.copy()
        else:
            budget=np.zeros(5)
        hv=float(v[i-1]) if i else 0.
        z,I,ps,pc,_=circuit(full,init,port,full,tan,g,scale,
                            float(v[i]),ambient,ext_fn,tm,hv)
        zprev=z
        for key,val in (
            ('z6',pack(z)),('current_A',I),
            ('temp_group_K',ambient+full[:5]/np.sqrt(cap)),
            ('heat_state',full/scale),
            ('material_budget_W',budget),('Psea_W',ps),('Pcu_W',pc)):
            out[key].append(np.asarray(val).copy())
    res={name:np.asarray(val) for name,val in out.items()}
    res.update(heat_rank=rank,geometry_only=True,heat_mode=heat_mode,rom_mix=rom_mix,full_FV_states=0,
               cold_BFZI_solves=0,heat_basis_builds=0,
               heat_FV_solves=0,ambient=ambient)
    return res
