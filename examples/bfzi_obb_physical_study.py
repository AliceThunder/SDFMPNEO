"""Physical BFZI V150 dual-package noncollision and exact-OBB sea study.

Rebuilds genuine original BFZI ports/Green/weak Joule on corrected disjoint
P03/P07 poses; compares original AABB sea void with polyhedral OBB cutout.
Then tests the existing k6 fractional-thermal surrogate against warm direct
BFZI with the new physically coherent geometry.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_native_sea_convergence import direct,compare,pack_save
from examples.bfzi_pkg_interface_study import metrics,warm_field
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import basis
from sdfmpneo_bfzi.geom_tangent_subspace import build as tangent_build
from sdfmpneo_bfzi.geom_void import disjoint_y_shift,overlap,report as void_report
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D


CASES={"P03":("bfzi_pose_p03_holdout.npz",29),
       "P07":("bfzi_pose_p07_15s_holdout.npz",18)}
LEVELS={"aabb":("aabb",None,None),
        "oriented":("oriented",None,None),
        "oriented_medium":("oriented",.14,.12)}


def load_program(path,which):
    with np.load(path) as d:
        cases=json.loads(str(d["cases_json"]))
        i=next(i for i,v in enumerate(cases) if int(v["id"])==which)
        return d["time_s"].copy(),d["voltage_v"][i].copy()


def run(bf,v150,geo,void,xz,y,t,v,ex=None):
    ts=time.perf_counter()
    data,info=create(bf,v150,"M4",**geo,sea_void=void,
                     physical_box_only=True,grid_xz=xz,grid_y=y)
    em=BFZIV148Adapter(data,bf,heat_mode="port-weak")
    if ex is None:
        ex=np.diag([.5-1j*em.omega*em.air[0,0],
                    .5-1j*em.omega*em.air[1,1]])
    et=BFZIV148Pkg3D(
        em,external=ex,pkg_shape=(4,2,4),pkg_k=.25,
        pkg_sea_model="face-resistance",pkg_sea_h=20.,
        pkg_sea_quad=2,pkg_sea_near=4)
    head=GeomHeat.from_original(em,et,ex)
    setup=time.perf_counter()-ts
    phy=direct(et,t,v)
    return data,em,et,head,phy,setup


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfzi-root",type=Path,required=True)
    ap.add_argument("--runs",type=Path,default=Path("runs"))
    ap.add_argument("--poses",nargs="+",choices=tuple(CASES),
                    default=("P03","P07"))
    ap.add_argument("--out",type=Path,
                    default=Path("runs/bfzi_obb_physical_study.json"))
    a=ap.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"]);net.eval()
    out=dict(
        method="original BFZI V150 warm physical EM with native oriented OBB void",
        physical_interface="face-resistance",
        training_geometry_touched=False,
        regimes=["axis-aligned AABB vs exact oriented polyhedron",
                 "new six-direction physical k6 hybrid"],
        geometries={})
    if a.out.exists():
        out=json.loads(a.out.read_text())
    for name in a.poses:
        stem=a.runs/f"bfzi_pose_{name.lower()}"
        old=GeomHeat.load(stem.with_name(stem.name+"_anchor.npz"))
        meta=json.loads(stem.with_name(stem.name+"_anchor.json").read_text())
        dY=disjoint_y_shift(old._pkg_mid,old._pkg_half,old._pkg_axes)
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(geo["shift_mm"],float)
        geo["shift_mm"][1]+=1000*dY
        path,cid=CASES[name]
        t,v=load_program(a.runs/path,cid)
        item=dict(case_id=cid,original_angle_deg=meta["physical_relative_tilt_deg"],
                  additional_y_shift_mm=float(dY*1000),
                  new_shift_mm=geo["shift_mm"].tolist(),
                  program_voltage_V=float(np.max(v)),
                  duration_s=float(t[-1]),step_s=float(t[1]-t[0]),
                  levels={})
        runs={}
        external=None
        for label,(void,xz,y) in LEVELS.items():
            data,em,et,head,phys,setup=run(
                bf,v150,geo,void,xz,y,t,v,ex=external)
            external=head.ext
            if overlap(head._pkg_mid,head._pkg_half,head._pkg_axes):
                raise ValueError("P03/P07 oriented packages still overlap")
            f=a.runs/f"bfzi_pose_{name.lower()}_disjoint_{label}.npz"
            pack_save(f,phys)
            s=dict(sea_cells=head.ns,sea_volume_m3=float(sum(head.vol)),
                   sea_volume_error=data["vol_meta"]["relative_volume_error"],
                   void_method=data["vol_meta"].get("void_type","aabb"),
                   thermal_budget_W=phys["max_budget_W"],
                   direct_sec=phys["elapsed_sec"],setup_sec=setup,
                   cold_z6=head.z0.tolist(),
                   package_to_sea_flux_W=phys["max_pkg_flux_W"],
                   data_file=str(f))
            if label=="aabb":
                s["original_aabb_geometry"]=void_report(head,data["vol_meta"])
            if label=="oriented_medium":
                rom=GeomHeatROM.build(head,rank=32)
                modes=basis(head.vol,rom.phi[:head.ns],rank=8)
                tan,stat=tangent_build(em,head,rom,modes,6,amplitude=.2)
                ts=time.perf_counter()
                pred=rom.rollout(net,ck,v,t,tangent=tan,blend=1.)
                s["neural_k6"]=dict(
                    **metrics(pred,phys["temp_K"],phys["z6"],head),
                    inference_sec=time.perf_counter()-ts,
                    tangent_EM_solves=tan.nsolve,
                    tangent_EM_sec=stat["directional_sec"])
                powered=np.flatnonzero(v>0)
                pick=int(powered[-1]) if len(powered) else 0
                s["warm_field"]=warm_field(em,head,phys["temp_K"][pick],v[pick])
                s["warm_field_time_s"]=float(t[pick])
                tan.save(a.runs/f"bfzi_pose_{name.lower()}_disjoint_k6.npz")
                head.save(a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
                rom.save(a.runs/f"bfzi_pose_{name.lower()}_disjoint_rom.npz")
            item["levels"][label]=s
            runs[label]=phys
            print(name,label,"n",head.ns,"setup",round(setup,3),
                  "direct",round(phys["elapsed_sec"],3),
                  "sea_volume",round(s["sea_volume_m3"],7),
                  flush=True)
        for label,phy in runs.items():
            item["levels"][label]["vs_oriented_medium"]=compare(
                phy,runs["oriented_medium"])
        out["geometries"][name]=item
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
        print("RESULT",name,"baseline_vs_obb_deltaZ_pct",
              round(item["levels"]["aabb"]["vs_oriented_medium"][
                  "thermal_delta_Z_relative_L2"]*100,5),
              "k6_err_pct",round(item["levels"]["oriented_medium"][
                  "neural_k6"]["dynamic_Z_relative_L2"]*100,6),flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
