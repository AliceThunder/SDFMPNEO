"""BFZI V150 tangent offline-cost / held-out full-3D precision study.

P01/P02/P04/P05/P06: original BFZI EM tangent training teachers.
P03/P07: complete held-out 8-vs-16 EM directions, true-temperature
and independent fractional-neural + conservative 32-state thermal rollout.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import originals,create
from examples.bfzi_tangent_eval import score
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import SeaTangent,basis
from sdfmpneo_bfzi.geom_tangent_transfer import TangentTransfer

TRAIN=("P01","P02","P04","P05","P06")
HOLD=("P03","P07")


def cold_em(bf,v150,head,meta):
    c=np.asarray(meta["coil_center_m"],float)
    dr=(c[1]-c[0])*1000
    data,_=create(bf,v150,"M4",
        alpha1=meta["alpha"][0],alpha2=meta["alpha"][1],
        core1_mm=meta["core_mm"][0],core2_mm=meta["core_mm"][1],
        shift_mm=dr,tilt_deg=meta["roll_deg"][1]-meta["roll_deg"][0],
        yaw_deg=meta["yaw_deg"][1]-meta["yaw_deg"][0],
        pitch_deg=meta["pitch_deg"][1]-meta["pitch_deg"][0],
        box1_mm=meta["box1_mm"],box2_mm=meta["box2_mm"])
    em=BFZIV148Adapter(data,bf,heat_mode="port-weak")
    if (not np.allclose(em.vol,head.vol,rtol=1e-10,atol=1e-14)
            or not np.allclose(em.points,head.xyz,rtol=0,atol=1e-10)):
        raise ValueError("reconstructed original BFZI geometry differs from anchor")
    cold=em.solve(np.full(head.ns,head.amb),np.full(2,head.amb))
    if not np.allclose(cold.sea_z,head.sea0,rtol=2e-6,atol=1e-9):
        raise ValueError("original BFZI cold complex sea impedance changed")
    return em


def solve_modes(em,head,modes,method,h):
    if method not in ("forward","central") or h<=0:
        raise ValueError("choose finite positive BFZI tangent amplitude")
    jac=[]
    t0=time.perf_counter()
    cu=np.full(2,head.amb)
    for j in range(modes.shape[1]):
        amp=h/max(np.max(abs(modes[:,j])),1e-12)
        dt=amp*modes[:,j]
        zp=em.solve(np.full(head.ns,head.amb)+dt,cu).sea_z
        if method=="central":
            zm=em.solve(np.full(head.ns,head.amb)-dt,cu).sea_z
            dz=(zp-zm)/(2*amp)
        else:
            dz=(zp-head.sea0)/amp
        jac.append(.5*(dz+dz.T))
    sec=time.perf_counter()-t0
    tan=SeaTangent(modes,np.array(jac),head.vol,head.sea0,head.geom)
    tan.method=method
    tan.amplitude=h
    tan.nsolve=len(modes.T)*(2 if method=="central" else 1)
    return tan,sec


def full_score(head,rom,tan,net,ck,teacher):
    with np.load(teacher) as d:
        t=np.asarray(d["time_s"],float)
        v=np.asarray(d["voltage_v"],float)
        tt=np.asarray(d["temp_K"],float)
        zz=np.asarray(d["z6"],float)
        cases=json.loads(str(d["cases_json"]))
    rows=[]
    for i in range(len(v)):
        start=time.perf_counter()
        pred=rom.rollout(net,ck,v[i],t,tangent=tan,blend=1.)
        elapsed=time.perf_counter()-start
        err=pred["temp_K"]-tt[i]
        ns=head.ns;nc=head.cu.start
        rows.append(dict(case=cases[i]["id"],voltage_V=cases[i]["voltage_v"],
            dynamic_Z_relative_L2=float(np.linalg.norm(pred["z6"]-zz[i])/
                max(np.linalg.norm(zz[i]-zz[i,0]),1e-12)),
            sea_max_abs_K=float(np.max(abs(err[:,:ns]))),
            pkg_max_abs_K=float(np.max(abs(err[:,ns:nc]))),
            cu_max_abs_K=float(np.max(abs(err[:,nc:]))),
            q_power_gap_W=float(np.max(abs(pred["qsea_W_m3"]@head.vol-
                                             pred["Psea_W"]))),
            material_budget_gap_W=float(np.max(abs(pred["material_budget_W"]))),
            dynamic_infer_sec=elapsed))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--amplitude",type=float,default=.2)
    p.add_argument("--epochs",type=int,default=350)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_tangent_study.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    train=[];hold={};cost={};timer=time.perf_counter()
    for name in (*TRAIN,*HOLD):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        head=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        start=time.perf_counter()
        em=cold_em(bf,v150,head,meta)
        build_sec=time.perf_counter()-start
        if name in HOLD:
            rom=GeomHeatROM.load(head,pre.with_name(pre.name+"_rom.npz"))
            ref=SeaTangent.load(pre.with_name(
                pre.name+"_tangent_anchor_sea_tangent.npz"),head)
            modes=ref.modes
        else:
            rom=GeomHeatROM.build(head,rank=32)
            modes=basis(head.vol,rom.phi[:head.ns],rank=8)
        central,sc=solve_modes(em,head,modes,"central",a.amplitude)
        cfile=a.runs/f"bfzi_pose_{name.lower()}_fd_central.npz"
        central.save(cfile)
        row=dict(setup_sec=build_sec,central_sec=sc,
                 central_em_solves=central.nsolve)
        if name in HOLD:
            fast,sf=solve_modes(em,head,modes,"forward",a.amplitude)
            ffile=a.runs/f"bfzi_pose_{name.lower()}_fd_forward.npz"
            fast.save(ffile)
            row.update(forward_sec=sf,forward_em_solves=fast.nsolve,
                       direction_speedup=sc/max(sf,1e-12))
            hold[name]=(head,rom,central,fast)
        else:
            train.append((name,head,central))
        cost[name]=row
        print(name,json.dumps(row),flush=True)
    start=time.perf_counter()
    model=TangentTransfer.train(train,epochs=a.epochs)
    train_sec=time.perf_counter()-start
    model_path=a.runs/"bfzi_tangent_transfer.npz"
    model.save(model_path)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",map_location="cpu",
                  weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"])
    net.eval()
    out={"method":"original BFZI thermal modes + aligned 40D neural tangent residual",
         "train_geometries":list(TRAIN),"heldout_geometries":list(HOLD),
         "amplitude_K":a.amplitude,"offline_sec":cost,
         "training_sec":train_sec,"network_file":str(model_path),
         "training_LOO_tangent_errors":model.cv.tolist(),
         "chosen_neural_residual_weight":model.weight,"heldout":{}}
    for name,(head,rom,central,fast) in hold.items():
        predicted=model.predict(head,rom)
        predicted.save(a.runs/f"bfzi_pose_{name.lower()}_neural_tangent.npz")
        files=[a.runs/f"bfzi_pose_{name.lower()}_holdout.npz"]
        if name=="P07":
            files.append(a.runs/"bfzi_pose_p07_15s_holdout.npz")
        rows={}
        for teacher in files:
            rows[teacher.name]={}
            for label,tan in (("central16",central),("forward8",fast),
                              ("neural0",predicted)):
                rows[teacher.name][label]=dict(
                    oracle_true_T=score(head,tan,teacher),
                    autonomous_3D=full_score(head,rom,tan,net,ck,teacher))
        out["heldout"][name]=rows
        print("HELDOUT",name,json.dumps(rows),flush=True)
    out["total_wall_sec"]=time.perf_counter()-timer
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
