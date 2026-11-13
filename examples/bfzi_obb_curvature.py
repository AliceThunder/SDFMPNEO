"""BFZI V150 true-OBB heldout nonlinear sea-Z curvature experiment.

Reuses the same original six thermal modes and one additional large-amplitude
BFZI solve per corrected direction. True 3-D trajectories are evaluation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_pkg_interface_study import metrics
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.geom_tangent_curved import CurvedTangent,build
from sdfmpneo_bfzi.geom_void import disjoint_y_shift
from sdfmpneo_bfzi.geom_memory import GeomFrac


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--amplitude",type=float,default=5.)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_obb_curvature.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"]);net.eval()
    out=dict(extra_probe_peak_temperature_K=a.amplitude,
             reference="full original BFZI warm 3-D EM/thermal direct",
             geometries={})
    for name in ("P03","P07"):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        old=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(geo["shift_mm"],float)
        geo["shift_mm"][1]+=1000*disjoint_y_shift(
            old._pkg_mid,old._pkg_half,old._pkg_axes)
        data,_=create(bf,v150,"M4",**geo,sea_void="oriented",
                      physical_box_only=True,grid_xz=.14,grid_y=.12)
        em=BFZIV148Adapter(data,bf,heat_mode="port-weak")
        head=GeomHeat.load(a.runs/
            f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        rom=GeomHeatROM.load(head,a.runs/
            f"bfzi_pose_{name.lower()}_disjoint_rom.npz")
        base=SeaTangent.load(a.runs/
            f"bfzi_pose_{name.lower()}_disjoint_k6.npz",head)
        with np.load(a.runs/
             f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz") as f:
            t=f["time_s"].copy()
            v=f["voltage_v"].copy()
            T=f["temp_K"].copy()
            Z=f["z6"].copy()
        curved,info=build(em,head,base,ids=range(6),
                          amplitude=a.amplitude)
        vals={}
        for rank in (0,1,2,4,6):
            tangent=(base if rank==0 else CurvedTangent(
                base,curved.ids[:rank],curved.curv[:rank],
                curved.h1[:rank]))
            pred=rom.rollout(net,ck,v,t,tangent=tangent,blend=1.)
            vals[str(rank)]=dict(
                originalBFZI_EM_directional_solves=base.nsolve+rank,
                dynamic=metrics(pred,T,Z,head))
            print(name,"extra",rank,"Z_err_percent",
                  round(vals[str(rank)]["dynamic"][
                      "dynamic_Z_relative_L2"]*100,7),flush=True)
        out["geometries"][name]=dict(
            extra_probes=info,candidates=vals)
    a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
