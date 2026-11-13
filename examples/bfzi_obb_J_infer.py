"""End-to-end optional FV charge-conserving J inference on corrected OBBs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.geom_void import disjoint_y_shift
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D


def rel(a,b,vol):
    w=vol[:,None]
    return float(np.sqrt(np.sum(w*abs(a-b)**2)/
                         max(np.sum(w*abs(b)**2),1e-30)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_obb_J_infer.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"]);net.eval()
    out={}
    for name in ("P03","P07"):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        old=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(geo["shift_mm"],float)
        geo["shift_mm"][1]+=1000*disjoint_y_shift(
            old._pkg_mid,old._pkg_half,old._pkg_axes)
        d,_=create(bf,v150,"M4",**geo,physical_box_only=True,
                   sea_void="oriented",grid_xz=.14,grid_y=.12)
        em=BFZIV148Adapter(d,bf,heat_mode="port-weak")
        path=a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz"
        prev=GeomHeat.load(path)
        et=BFZIV148Pkg3D(
            em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
            pkg_sea_model="face-resistance",pkg_sea_h=20.,
            pkg_sea_quad=2,pkg_sea_near=4)
        head=GeomHeat.from_original(em,et,prev.ext)
        if not np.allclose(head.z0,prev.z0,rtol=1e-11,atol=1e-11):
            raise ValueError("original BFZI cold state changed across J upgrade")
        rom=GeomHeatROM.load(head,a.runs/
                  f"bfzi_pose_{name.lower()}_disjoint_rom.npz")
        tan=SeaTangent.load(a.runs/
                  f"bfzi_pose_{name.lower()}_disjoint_k6.npz",head)
        with np.load(a.runs/
             f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz") as file:
            t=file["time_s"].copy()
            v=file["voltage_v"].copy()
            T=file["temp_K"].copy()
        t0=time.perf_counter()
        original=rom.rollout(net,ck,v,t,tangent=tan,blend=1.,correct_J=False)
        old_sec=time.perf_counter()-t0
        t0=time.perf_counter()
        upgraded=rom.rollout(net,ck,v,t,tangent=tan,blend=1.,correct_J=True)
        new_sec=time.perf_counter()-t0
        k=int(np.flatnonzero(v>0)[-1])
        st=em.solve(T[k,:head.ns],T[k,head.cu])
        I=np.linalg.solve(st.total_z+head.ext,np.array([v[k],0.],complex))
        true=np.einsum("npc,p->nc",st.unit_j,I,optimize=True)
        corrected=head.corrected_J(T[k]-head.amb,I)
        baseline=np.einsum("npc,p->nc",head.unit,I,optimize=True)*(
            1+head.alpha*(T[k,:head.ns]-head.amb))[:,None]
        result=dict(
            selected_time_s=float(t[k]),
            spatial_fixed_true_temperature=dict(
                uncorrected_J_relative_L2=rel(baseline,true,head.vol),
                corrected_J_relative_L2=rel(corrected,true,head.vol)),
            autonomous_prediction=dict(
                uncorrected_J_relative_L2=rel(
                    original["Jsea_A_m2"][k],true,head.vol),
                corrected_J_relative_L2=rel(
                    upgraded["Jsea_A_m2"][k],true,head.vol)),
            source_q_difference_W_m3=float(np.max(
                abs(original["qsea_W_m3"]-upgraded["qsea_W_m3"]))),
            dynamic_Z_difference=float(np.max(abs(
                original["z6"]-upgraded["z6"]))),
            old_inference_sec=old_sec,new_inference_sec=new_sec,
            FV_projection_edges=len(head.sea_proj["edges"]))
        out[name]=result
        head.save(path)
        print(name,"Jtrue_err",
              round(result["spatial_fixed_true_temperature"][
                   "uncorrected_J_relative_L2"]*100,5),
              round(result["spatial_fixed_true_temperature"][
                   "corrected_J_relative_L2"]*100,5),
              "Jpred_err",
              round(result["autonomous_prediction"][
                   "uncorrected_J_relative_L2"]*100,5),
              round(result["autonomous_prediction"][
                   "corrected_J_relative_L2"]*100,5),
              "time_sec",round(old_sec,3),round(new_sec,3),flush=True)
    a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
