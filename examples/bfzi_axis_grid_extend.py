"""Extend nonoverlapping original BFZI axis-aligned sea-FV convergence.

Reuses archived 384/272/312/400-cell physical direct trajectories and
adds only selected finer ORIGINAL BFZI volume-integrated FV grids.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import values
from examples.bfzi_native_sea_convergence import direct,compare,pack_save
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_void import report as void_report
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import basis
from sdfmpneo_bfzi.geom_tangent_subspace import build as tangent_build
from sdfmpneo_bfzi.geom_memory import GeomFrac
from examples.bfzi_pkg_interface_study import metrics

GRIDS={"face_finer":(.09,.075),
       "face_extra":(.075,.06),
       "face_ultra":(.065,.055)}


def load_data(path):
    with np.load(path) as d:
        return dict(time_s=d["time_s"].copy(),voltage_v=d["voltage_v"].copy(),
                    temp_K=d["temp_K"].copy(),z6=d["z6"].copy(),
                    sea_xyz=d["sea_xyz"].copy(),nsea=len(d["sea_xyz"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_axis_grid_control.json"))
    p.add_argument("--levels",nargs="+",choices=tuple(GRIDS),
                   default=("face_finer","face_extra"))
    p.add_argument("--hybrid",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    out=json.loads(a.out.read_text())
    ref=load_data(a.runs/"bfzi_axis_grid_legacy.npz")
    t=ref["time_s"]
    volt=ref["voltage_v"]
    for label in a.levels:
        xz,y=GRIDS[label]
        ts=time.perf_counter()
        d,_=create(bf,v150,"M4",physical_box_only=True,
                   grid_xz=xz,grid_y=y,grid_layers=0)
        em=BFZIV148Adapter(d,bf,heat_mode="port-weak")
        ex=np.diag([.5-1j*em.omega*em.air[0,0],
                    .5-1j*em.omega*em.air[1,1]])
        et=BFZIV148Pkg3D(em,external=ex,pkg_shape=(4,2,4),
                         pkg_k=.25,pkg_sea_model="face-resistance",
                         pkg_sea_h=20.,pkg_sea_quad=2,pkg_sea_near=4)
        head=GeomHeat.from_original(em,et,ex)
        void=void_report(head,d["vol_meta"])
        if abs(void["BFZI_AABB_over_physical_sum"]-1.)>1e-6:
            raise ValueError("oriented and native sea void volumes differ")
        cold_sec=time.perf_counter()-ts
        s=direct(et,t,volt)
        pack_save(a.runs/f"bfzi_axis_grid_{label}.npz",s)
        row=dict(grid_xz_m=xz,grid_y_m=y,
                 nsea=len(em.vol),downsampled=d["vol_meta"]["downsampled"],
                 mesh_rel_volume_error=d["vol_meta"]["relative_volume_error"],
                 cold_z6=head.z0.tolist(),
                 setup_sec=cold_sec,direct_sec=s["elapsed_sec"],
                 max_material_budget_W=s["max_budget_W"],
                 package_interface_flux_W=s["max_pkg_flux_W"],
                 interface=values(head,em.points),
                 oriented_void=void,physical_file=str(
                     a.runs/f"bfzi_axis_grid_{label}.npz"))
        if a.hybrid:
            rom=GeomHeatROM.build(head,rank=32)
            sea=basis(head.vol,rom.phi[:head.ns],rank=8)
            tan,stat=tangent_build(em,head,rom,sea,6,amplitude=.2)
            now=time.perf_counter()
            ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                          map_location="cpu",weights_only=True)
            net=GeomFrac(ng=int(ck["ng"]))
            net.load_state_dict(ck["state"])
            net.eval()
            pred=rom.rollout(net,ck,volt,t,tangent=tan,blend=1.)
            row["hybrid_k6"]=dict(
                **metrics(pred,s["temp_K"],s["z6"],head),
                inference_sec=time.perf_counter()-now,
                directional_EM_solves=tan.nsolve,
                directional_EM_sec=stat["directional_sec"])
            tan.save(a.runs/f"bfzi_axis_grid_{label}_k6.npz")
            head.save(a.runs/f"bfzi_axis_grid_{label}_anchor.npz")
            rom.save(a.runs/f"bfzi_axis_grid_{label}_rom.npz")
        out["levels"][label]=row
        print(label,"n",len(em.vol),"cold",round(cold_sec,3),
              "direct",round(s["elapsed_sec"],3),flush=True)
        a.out.write_text(json.dumps(out,indent=2))
    reference=a.levels[-1]
    last=load_data(a.runs/f"bfzi_axis_grid_{reference}.npz")
    for label in out["levels"]:
        file=a.runs/f"bfzi_axis_grid_{label}.npz"
        s=load_data(file)
        out["levels"][label][f"vs_{reference}"]=compare(s,last)
    a.out.write_text(json.dumps(out,indent=2))
    print("COMPARISON",reference,[
        (label,round(r[f"vs_{reference}"][
            "thermal_delta_Z_relative_L2"]*100,5),
         round(r[f"vs_{reference}"]["package_max_abs_K"],5))
        for label,r in out["levels"].items()],flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
