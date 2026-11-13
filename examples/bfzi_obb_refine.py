"""Refine true-OBB original BFZI V150 sea-FV mesh on disjoint heldout poses.

Reuses archived physical programs and warm FV reference at 572 wet cells.
Computes only new original BFZI grids, no fitting to heldout references.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args, values
from examples.bfzi_native_sea_convergence import direct, compare, pack_save
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D

GRIDS={"fine":(.11,.09),"finer":(.09,.075),"extra":(.075,.06)}


def load(path):
    with np.load(path) as f:
        return dict(time_s=f["time_s"].copy(),
                    voltage_v=f["voltage_v"].copy(),
                    temp_K=f["temp_K"].copy(),
                    z6=f["z6"].copy(),
                    sea_xyz=f["sea_xyz"].copy(),
                    nsea=len(f["sea_xyz"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),
                   default=["P03","P07"])
    p.add_argument("--levels",nargs="+",choices=tuple(GRIDS),
                   default=["fine","finer","extra"])
    p.add_argument("--mesh-only",action="store_true")
    p.add_argument("--max-nodes",type=int,default=1400)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_obb_refine.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    main_study=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    out=json.loads(a.out.read_text()) if a.out.exists() else dict(
        method="original BFZI V150 true-OBB sea FV grid convergence",
        geometry_training_used=False,geometries={})
    for name in a.poses:
        oldpre=a.runs/f"bfzi_pose_{name.lower()}"
        meta=json.loads(oldpre.with_name(oldpre.name+"_anchor.json").read_text())
        prev=GeomHeat.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        info=main_study["geometries"][name]
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(info["new_shift_mm"],float)
        ref=load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz")
        g=out["geometries"].setdefault(name,dict(
            reference="oriented_medium",reference_nodes=ref["nsea"],
            program_id=info["case_id"],levels={}))
        for label in a.levels:
            xz,y=GRIDS[label]
            t0=time.perf_counter()
            if a.mesh_only:
                d,m=create(bf,v150,"M4",**geo,sea_void="oriented",
                           grid_xz=xz,grid_y=y,grid_max_nodes=a.max_nodes,
                           mesh_only=True)
                record=dict(nodes=len(d["points"]),
                            volume_m3=float(sum(d["volumes"])),
                            volume_error=float(d["vol_meta"]["relative_volume_error"]),
                            mesh_sec=time.perf_counter()-t0)
            else:
                try:
                    d,_=create(bf,v150,"M4",**geo,sea_void="oriented",
                               grid_xz=xz,grid_y=y,grid_max_nodes=a.max_nodes)
                except ValueError as e:
                    if "exceeds budget" not in str(e):
                        raise
                    record=dict(skipped=str(e))
                    g["levels"][label]=record
                    print(name,label,record["skipped"],flush=True)
                    continue
                em=BFZIV148Adapter(d,bf,heat_mode="port-weak")
                et=BFZIV148Pkg3D(
                    em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
                    pkg_sea_model="face-resistance",pkg_sea_h=20.,
                    pkg_sea_quad=2,pkg_sea_near=4)
                head=GeomHeat.from_original(em,et,prev.ext)
                pre_sec=time.perf_counter()-t0
                run=direct(et,ref["time_s"],ref["voltage_v"])
                f=a.runs/f"bfzi_pose_{name.lower()}_disjoint_{label}.npz"
                pack_save(f,run)
                record=dict(
                    nodes=head.ns,
                    volume_m3=float(sum(head.vol)),
                    volume_error=float(d["vol_meta"]["relative_volume_error"]),
                    cold_precompute_sec=pre_sec,
                    direct_sec=float(run["elapsed_sec"]),
                    max_budget_W=float(run["max_budget_W"]),
                    interface=values(head,em.points),
                    cold_z6=head.z0.tolist(),
                    physical_file=str(f),
                    vs_oriented_medium=compare(run,ref))
                print(name,label,"nodes",head.ns,
                      "direct_sec",round(record["direct_sec"],2),
                      "deltaZ_vs_572_pct",round(
                          record["vs_oriented_medium"]["thermal_delta_Z_relative_L2"]*100,5),
                      "pkg_K",round(
                          record["vs_oriented_medium"]["package_max_abs_K"],5),
                      flush=True)
            g["levels"][label]=record
            if a.mesh_only:
                print(name,label,"nodes",record["nodes"],
                      "vol_err",record["volume_error"],flush=True)
            a.out.parent.mkdir(parents=True,exist_ok=True)
            a.out.write_text(json.dumps(out,indent=2))
    if not a.mesh_only:
        for name in a.poses:
            g=out["geometries"][name]
            ok=[label for label in a.levels
                if "physical_file" in g["levels"].get(label,{})]
            if len(ok)<2:
                continue
            hi=ok[-1]
            ref=load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_{hi}.npz")
            for label in ("oriented_medium",)+tuple(ok):
                f=(a.runs/f"bfzi_pose_{name.lower()}_disjoint_{label}.npz")
                g["levels"].setdefault(label,{})[
                    f"vs_{hi}"]=compare(load(f),ref)
            a.out.write_text(json.dumps(out,indent=2))
            print("HI_REF",name,hi,[(label,round(
                g["levels"][label][f"vs_{hi}"][
                    "thermal_delta_Z_relative_L2"]*100,5))
                for label in ("oriented_medium",)+tuple(ok)],flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
