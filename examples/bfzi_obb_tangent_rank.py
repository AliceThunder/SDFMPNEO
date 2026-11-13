"""Diagnose BFZI exact-oriented FV k6/k8/c16 tangent error on new poses."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_pkg_interface_study import metrics
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import SeaTangent,basis
from sdfmpneo_bfzi.geom_tangent_subspace import build as tangent_build
from sdfmpneo_bfzi.geom_void import disjoint_y_shift
from sdfmpneo_bfzi.geom_memory import GeomFrac


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfzi-root",type=Path,required=True)
    ap.add_argument("--runs",type=Path,default=Path("runs"))
    ap.add_argument("--poses",nargs="+",default=["P03","P07"])
    ap.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_obb_tangent_rank.json"))
    a=ap.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"]);net.eval()
    out={}
    for name in a.poses:
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
        head=GeomHeat.load(a.runs/
             f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        rom=GeomHeatROM.load(head,a.runs/
             f"bfzi_pose_{name.lower()}_disjoint_rom.npz")
        with np.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz") as f:
            t=f["time_s"].copy()
            v=f["voltage_v"].copy()
            temp=f["temp_K"].copy()
            z=f["z6"].copy()
        sea=basis(head.vol,rom.phi[:head.ns],rank=8)
        methods={"cold6":SeaTangent.load(
            a.runs/f"bfzi_pose_{name.lower()}_disjoint_k6.npz",head)}
        build_stats={}
        for rank in (8,):
            tan,stat=tangent_build(em,head,rom,sea,rank,amplitude=.2)
            methods[f"cold{rank}"]=tan
            build_stats[f"cold{rank}"]=stat
        t0=time.perf_counter()
        methods["central16"]=SeaTangent.build(
            em,head,rom.phi,rank=8,amplitude=.2,method="central")
        build_stats["central16"]={"directional_sec":time.perf_counter()-t0}
        result={}
        for method,tan in methods.items():
            t0=time.perf_counter()
            pred=rom.rollout(net,ck,v,t,tangent=tan,blend=1.)
            result[method]=dict(
                metrics=metrics(pred,temp,z,head),
                inference_sec=time.perf_counter()-t0,
                EM_solves=tan.nsolve,
                EM_time_sec=build_stats.get(method,{}).get("directional_sec"))
            print(name,method,"Zerr_percent",
                  round(result[method]["metrics"][
                        "dynamic_Z_relative_L2"]*100,6),
                  "seaK",round(result[method]["metrics"]["sea_max_abs_K"],6),
                  flush=True)
        out[name]=result
    a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
