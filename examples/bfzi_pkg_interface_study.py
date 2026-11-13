"""Real original BFZI V150 thermal-interface refinement study.

Uses original BFZI warm EM direct solver with both historical and oriented
surface-resistance interface on identical coil/sea FV geometry. Compares the
new physical direct trajectories with a fresh k=6 conservative ROM surrogate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import originals
from examples.bfzi_tangent_study import cold_em
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_rom import GeomHeatROM,groups
from sdfmpneo_bfzi.geom_tangent import basis
from sdfmpneo_bfzi.geom_tangent_subspace import build
from sdfmpneo_bfzi.geom_interface import report
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D


def physical(et, t, v):
    def volt(now):
        k=int(np.argmin(abs(t-now)))
        return np.array([v[k],0.],complex)
    seq=et.trajectory(t,volt)
    T=et.amb+np.stack([s.temp for s in seq])
    z=np.stack([pack(s.impedance) for s in seq])
    flux=np.stack([et.heat_flux(s.temp)[1] for s in seq])
    budget=[]
    for k in range(1,len(t)):
        theta=seq[k].temp
        _,_,heat,_,_=et.em(theta,volt(t[k]))
        bal=et.cap*(theta-seq[k-1].temp)/(t[k]-t[k-1])+et.H@theta-heat
        budget.append([float(sum(bal[g])) for g in groups(
            type("Group",(),dict(ns=et.count,npkg=et.pkg_n,cu=et.cu))())])
    return T,z,flux,float(np.max(np.abs(budget))) if budget else 0.




def warm_field(em,head,T,voltage):
    """Independent warm original BFZI J/q at a powered physical-time sample."""
    state=em.solve(T[:head.ns],T[head.cu])
    I=np.linalg.solve(state.total_z+head.ext,
                      np.array([voltage,0.],complex))
    truth_j=np.einsum("npc,p->nc",state.unit_j,I,optimize=True)
    truth_q,_=em.heat(state,I)
    _,q,j,psea,_=head.source(T-head.amb,I,state.sea_z)
    def rel(a,b):
        w=head.vol.reshape(-1,*([1]*(np.ndim(a)-1)))
        return float(np.sqrt(np.sum(w*abs(a-b)**2)/
                             max(np.sum(w*abs(b)**2),1e-30)))
    return dict(
        J_volume_relative_L2=rel(j,truth_j),
        q_volume_relative_L2=rel(q,truth_q),
        original_q_port_gap_W=float(abs(
            truth_q@head.vol-.5*np.vdot(I,state.sea_z@I).real)),
        proxy_q_port_gap_W=float(abs(q@head.vol-psea)))

def metrics(pred,temp,z,head):
    d=pred["temp_K"]-temp
    return dict(
        dynamic_Z_relative_L2=float(np.linalg.norm(pred["z6"]-z)/
            max(np.linalg.norm(z-z[0]),1e-12)),
        sea_max_abs_K=float(np.max(abs(d[:,:head.ns]))),
        pkg_max_abs_K=float(np.max(abs(d[:,head.ns:head.cu.start]))),
        cu_max_abs_K=float(np.max(abs(d[:,head.cu]))),
        q_port_gap_W=float(np.max(abs(pred["qsea_W_m3"]@head.vol-
                                      pred["Psea_W"]))),
        material_budget_gap_W=float(np.max(abs(pred["material_budget_W"]))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--h",type=float,default=20.,
                   help="Physical package surface transfer W/(m2 K)")
    p.add_argument("--near",type=int,default=4)
    p.add_argument("--quad",type=int,default=2)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_pkg_interface_study.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"])
    net.eval()
    out=dict(original="BFZI V150 full warm electromagnetic direct method",
             interface="conservative oriented surface face quadrature",
             surface_h_W_m2K=a.h,neighbor_count=a.near,quadrature=a.quad,
             geometries={})
    for name in ("P03","P07"):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        old=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        em=cold_em(bf,v150,old,meta)
        cfg=dict(external=old.ext,pkg_shape=(4,2,4),
                 pkg_k=.25,pkg_sea_model="face-resistance",
                 pkg_sea_h=a.h,pkg_sea_near=a.near)
        qstudy={}
        for q in sorted(set([1,2,3,a.quad])):
            obj=BFZIV148Pkg3D(em,**dict(cfg,pkg_sea_quad=q))
            qstudy[str(q)]=[
                float(sum(edge[2] for edge in links))
                for links in obj.pkg_links]
            if q==a.quad:
                et=obj
        head=GeomHeat.from_original(em,et,old.ext)
        if (not np.allclose(old.z0,head.z0,atol=1e-10,rtol=1e-9)
                or not np.allclose(old.xyz,head.xyz,atol=1e-12)):
            raise ValueError("BFZI EM/FV geometry must be unchanged")
        rom=GeomHeatROM.build(head,rank=32)
        modes=basis(head.vol,rom.phi[:head.ns],rank=8)
        tan,diag=build(em,head,rom,modes,6,amplitude=.2)
        tan_path=a.runs/f"bfzi_pose_{name.lower()}_interface_rank6.npz"
        tan.save(tan_path)
        files=[pre.with_name(pre.name+"_holdout.npz")] if name=="P03" else [
            a.runs/"bfzi_pose_p07_15s_holdout.npz"]
        geom=dict(cold_interface=report(old),
                  refined_interface=report(head),
                  face_quadrature_total_G_W_K=qstudy,
                  directional_tangent=diag,
                  new_tangent=str(tan_path),programs={})
        for file in files:
            with np.load(file) as d:
                t=d["time_s"].copy()
                v=d["voltage_v"].copy()
                z0=d["z6"].copy()
                T0=d["temp_K"].copy()
                cases=json.loads(str(d["cases_json"]))
            rows=[]
            for i in range(len(v)):
                ts=time.perf_counter()
                temp,z,flux,budget=physical(et,t,v[i])
                direct_sec=time.perf_counter()-ts
                ts=time.perf_counter()
                pred=rom.rollout(net,ck,v[i],t,tangent=tan,blend=1.)
                pred_sec=time.perf_counter()-ts
                delta=temp-T0[i]
                powered=np.flatnonzero(v[i]>0)
                last=int(powered[-1]) if len(powered) else 0
                spatial=warm_field(em,head,temp[last],v[i,last])
                rows.append(dict(
                    warm_spatial_time_s=float(t[last]),
                    warm_spatial=spatial,
                    case=int(cases[i]["id"]),voltage_V=float(cases[i]["voltage_v"]),
                    time_step_s=float(t[1]-t[0]),
                    original_BFZI_direct_sec=direct_sec,
                    reduced_neural_inference_sec=pred_sec,
                    physical_interface_change_max_K=dict(
                        sea=float(np.max(abs(delta[:,:head.ns]))),
                        package=float(np.max(abs(
                            delta[:,head.ns:head.cu.start]))),
                        copper=float(np.max(abs(delta[:,head.cu])))),
                    physical_interface_change_Z_relative_L2=float(
                        np.linalg.norm(z-z0[i])/
                        max(np.linalg.norm(z0[i]-z0[i,0]),1e-12)),
                    direct_interface_flux_W_max=flux.max(axis=0).tolist(),
                    original_BFZI_material_budget_gap_W=budget,
                    surrogate_vs_refined_physics=metrics(pred,temp,z,head)))
                print("GEOM",name,"CASE",cases[i]["id"],
                      "old_new_pkg_K",rows[-1][
                          "physical_interface_change_max_K"]["package"],
                      "new_surrogate_Z_percent",rows[-1][
                          "surrogate_vs_refined_physics"][
                              "dynamic_Z_relative_L2"]*100,flush=True)
            geom["programs"][file.name]=rows
        out["geometries"][name]=geom
        print("GEOM",name,"G_by_quad",qstudy,flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
