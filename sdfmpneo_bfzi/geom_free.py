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
from .geom_init import GeomInit
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


def circuit(a,init,port,state,tan,g,scale,voltage,ambient):
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
        dsea=zmat(tan(g,st)[0].detach().cpu().numpy())
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


def thermal_step(prev,guess,dt,voltage,init,port,tan,g,scale,amb):
    """Implicit correction of five budgets, explicit neural shape coordinates."""
    cap=init['cap']
    k=init['k']
    s=np.sqrt(cap)
    z=cur=pc=None
    target=np.asarray(guess,float).copy()
    jac=np.eye(5)/dt+k[:5,:5]
    for _ in range(12):
        z,cur,ps,pc,sea=circuit(target,init,port,target,tan,g,scale,
                                voltage,amb)
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
    z,cur,ps,pc,sea=circuit(target,init,port,target,tan,g,scale,voltage,amb)
    q=init['src']@np.array([.5*abs(cur[0])**2,
                           (cur[0].conjugate()*cur[1]).real,
                           .5*abs(cur[1])**2])
    p0=q[0]*s[0]
    q=q*ps/p0 if p0>1e-20 else np.zeros_like(q)
    q[3]+=pc[0]/s[3];q[4]+=pc[1]/s[4]
    residual=s*((target[:5]-prev[:5])/dt+
                k[:5,:]@target-q[:5])
    return target,residual


@torch.no_grad()
def rollout(case,atlas,net,ck,times,voltage,tan=None,ambient=293.15,
            init_override=None):
    """Predict port Z and material temperatures with no BFZI anchor or mesh."""
    if not isinstance(atlas,GeomInit) or not isinstance(net,GeomFracET):
        raise TypeError("geometry-only atlas and GeomFracET required")
    if atlas.rank!=32 or net.rank!=32 or not ck.get('canonical',False):
        raise ValueError("current autonomous shared-coordinate head uses 32 modes")
    t=np.asarray(times,float);v=np.asarray(voltage,float)
    if t.ndim!=1 or not len(t) or t[0]!=0 or v.shape!=t.shape or np.any(np.diff(t)<=0):
        raise ValueError("invalid voltage/time schedule")
    init=atlas.predict(case) if init_override is None else init_override
    geom,feat=geometry(case,init)
    port=geom_port(geom)
    g=torch.as_tensor(feat[None],dtype=torch.float32)
    cap=init['cap']
    scale=np.sqrt(cap[labels(32)])
    cs=np.asarray(ck['src_scale'],float)
    ys=np.asarray(ck['y_scale'],float)
    ac=np.asarray(ck['a_scale'],float)
    hist=net.init_state(1)
    old=np.zeros(32)
    full=np.zeros(32)
    zprev=zmat(init['z0'])
    out=dict(z6=[],current_A=[],temp_group_K=[],heat_state=[],
             material_budget_W=[],Psea_W=[],Pcu_W=[])
    for i,tm in enumerate(t):
        if i:
            dt=float(tm-t[i-1])
            u=heat_drive(float(v[i-1]),zprev,init['ext'],port,
                ambient+full[3:5]/scale[3:5],cap[3:5])
            hist,_,pred=net.step_et(hist,
                torch.as_tensor((u/cs)[None],dtype=g.dtype),
                dt,g,torch.as_tensor((old/scale)[None],dtype=g.dtype),ac)
            guess=pred[0].cpu().numpy()*scale
            full,budget=thermal_step(old,guess,dt,float(v[i-1]),
                                    init,port,tan,g,scale,ambient)
            for _ in range(5):
                beta=positivity(full,init,scale)
                if beta>=1.-1e-9:
                    break
                damped=full.copy()
                damped[5:]*=max(beta*.999,0.)
                full,budget=thermal_step(old,damped,dt,float(v[i-1]),
                                          init,port,tan,g,scale,ambient)
            old=full.copy()
        else:
            budget=np.zeros(5)
        z,I,ps,pc,_=circuit(full,init,port,full,tan,g,scale,
                            float(v[i]),ambient)
        zprev=z
        for key,val in (
            ('z6',pack(z)),('current_A',I),
            ('temp_group_K',ambient+full[:5]/np.sqrt(cap)),
            ('heat_state',full/scale),
            ('material_budget_W',budget),('Psea_W',ps),('Pcu_W',pc)):
            out[key].append(np.asarray(val).copy())
    res={name:np.asarray(val) for name,val in out.items()}
    res.update(heat_rank=32,geometry_only=True,full_FV_states=0,
               cold_BFZI_solves=0,heat_basis_builds=0,
               heat_FV_solves=0,ambient=ambient)
    return res
