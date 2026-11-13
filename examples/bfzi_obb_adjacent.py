"""Compare conservative true wet cut-FV package interface across OBB meshes.

Uses original BFZI V150 EM and the same previously evaluated voltage programs.
Only package-to-seawater FV trace coupling changes. New physical direct data
is compared to the archived nearest-center interface at the SAME mesh.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_native_sea_convergence import direct,compare,pack_save
from examples.bfzi_obb_refine import load
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D

GRIDS={"medium":(.14,.12),"extra":(.075,.06)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),default=["P03","P07"])
    p.add_argument("--levels",nargs="+",choices=tuple(GRIDS),
                   default=["medium","extra"])
    p.add_argument("--preview",action="store_true")
    p.add_argument("--quad",type=int,default=4)
    p.add_argument("--out",type=Path,default=Path("runs/bfzi_obb_adjacent.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    study=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    out=json.loads(a.out.read_text()) if a.out.exists() else dict(
        method="true cut-cell adjacent face interface",
        source="original BFZI V150 direct EM heat",quad=a.quad,
        geometries={})
    for name in a.poses:
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        prev=GeomHeat.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(
            study["geometries"][name]["new_shift_mm"],float)
        t=load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz")
        g=out["geometries"].setdefault(name,dict(levels={}))
        for key in a.levels:
            xz,y=GRIDS[key]
            ts=time.perf_counter()
            data,_=create(bf,v150,"M4",**geo,physical_box_only=True,
                          sea_void="oriented",grid_xz=xz,grid_y=y)
            em=BFZIV148Adapter(data,bf,heat_mode="port-weak")
            et=BFZIV148Pkg3D(
                em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
                pkg_sea_model="face-adjacent",pkg_sea_h=20.,
                pkg_sea_quad=a.quad,pkg_sea_near=4)
            head=GeomHeat.from_original(em,et,prev.ext)
            row=dict(nsea=head.ns,precompute_sec=time.perf_counter()-ts,
                     total_G_W_K=[float(sum(v for _,_,v in links))
                                  for links in et.pkg_links],
                     link_count=[len(links) for links in et.pkg_links])
            if not a.preview:
                s=direct(et,t["time_s"],t["voltage_v"])
                file=a.runs/f"bfzi_pose_{name.lower()}_adj_{key}.npz"
                pack_save(file,s)
                oldkey="oriented_medium" if key=="medium" else key
                legacy=a.runs/f"bfzi_pose_{name.lower()}_disjoint_{oldkey}.npz"
                row.update(
                    direct_sec=s["elapsed_sec"],
                    max_budget_W=s["max_budget_W"],
                    physical_file=str(file),
                    same_mesh_vs_old=compare(s,load(legacy)))
            g["levels"][key]=row
            print(name,key,"nodes",head.ns,
                  "G",[round(v,5) for v in row["total_G_W_K"]],
                  "links",row["link_count"],
                  "old_delta_pct",(round(
                      row["same_mesh_vs_old"]["thermal_delta_Z_relative_L2"]*100,5)
                       if not a.preview else None),flush=True)
            a.out.parent.mkdir(parents=True,exist_ok=True)
            a.out.write_text(json.dumps(out,indent=2))
        if not a.preview and all(k in g["levels"] and
                "physical_file" in g["levels"][k] for k in ("medium","extra")):
            fine=load(a.runs/f"bfzi_pose_{name.lower()}_adj_extra.npz")
            for k in ("medium","extra"):
                path=a.runs/f"bfzi_pose_{name.lower()}_adj_{k}.npz"
                g["levels"][k]["vs_adj_extra"]=compare(load(path),fine)
            a.out.write_text(json.dumps(out,indent=2))
            print("CONVERGENCE",name,round(
                g["levels"]["medium"]["vs_adj_extra"][
                    "thermal_delta_Z_relative_L2"]*100,5),flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
