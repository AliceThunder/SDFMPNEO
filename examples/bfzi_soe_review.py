"""Independent BFZI S5 cross-time-grid autonomous inference review.

Reports deviations at matching clock times and physical BFZI teacher error.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_canon import make_rom
from sdfmpneo_bfzi.geom_neural import GeomFracET,rollout


def review(anchor,model,teacher,out):
    ck=torch.load(model,map_location="cpu",weights_only=True)
    rom=make_rom(GeomHeat.load(anchor),int(ck["rank"]))
    net=GeomFracET(ng=int(ck["ng"]),rank=int(ck["rank"]),
                   soe_quad=ck.get("soe_quad","legacy"),
                   heat_dt=float(ck.get("heat_dt",0.)))
    net.load_state_dict(ck["state"])
    net.eval()
    with np.load(teacher) as data:
        ref_z=data["z6"]
        ref_t=data["temp_K"][:,rom.state_order] if hasattr(rom,"state_order") else data["temp_K"]
    outgrid={}
    coarse=None
    for dt in (15.,7.5,3.75,1.875):
        times=np.arange(0.,60.+dt*.1,dt)
        voltage=np.where(times<30.,75.,0.)
        tic=time.perf_counter()
        pred=rollout(rom,net,ck,voltage,times)
        cost=time.perf_counter()-tic
        ratio=int(round(15./dt))
        shared_z=pred["z6"][::ratio]
        shared_t=pred["temp_K"][::ratio]
        if coarse is None:
            coarse=(shared_z,shared_t)
        dynamic=np.linalg.norm(ref_z-ref_z[0])
        outgrid[f"{dt:g}"]=dict(
            online_seconds=round(cost,5),
            temperature_error_vs_truth_K=float(np.max(abs(shared_t-ref_t))),
            dynamic_Z_relative_L2_vs_truth=float(
                np.linalg.norm(shared_z-ref_z)/max(dynamic,1e-12)),
            temperature_diff_vs_15s_K=float(np.max(abs(shared_t-coarse[1]))),
            dynamic_Z_diff_vs_15s=float(
                np.linalg.norm(shared_z-coarse[0])/max(dynamic,1e-12)),
            budget_max_W=float(np.max(abs(pred["material_budget_W"]))))
    result=dict(model=str(model),anchor=str(anchor),soe_quad=ck.get("soe_quad","legacy"),
                heat_dt=ck.get("heat_dt",0.),grid=outgrid)
    Path(out).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--anchor",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    review(a.anchor,a.model,a.teacher,a.out)
