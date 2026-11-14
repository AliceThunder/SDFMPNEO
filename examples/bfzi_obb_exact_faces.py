"""Physical BFZI V150 original-FV face-area correction on true OBB sea void.

Keeps original Green, port, weak-q and thermal solver. Compares the physically
wetted common FV face area against the inherited sqrt(volume fractions).
Uses disjoint tilted heldout P03/P07 572/1312 sea-cell grids.
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
from examples.bfzi_obb_refine import load
from examples.bfzi_pkg_interface_study import metrics
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import basis
from sdfmpneo_bfzi.geom_tangent_subspace import build as build_tan
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D

GRIDS={"medium":(.14,.12),"extra":(.075,.06)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),
                   default=["P03","P07"])
    p.add_argument("--levels",nargs="+",choices=("medium","extra"),
                   default=["medium","extra"])
    p.add_argument("--hybrid",action="store_true")
    p.add_argument("--source-mask",choices=("legacy","oriented"),
                   default="legacy")
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_obb_exact_faces.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    study=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    ck=net=None
    if a.hybrid:
        ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                      map_location="cpu",weights_only=True)
        net=GeomFrac(ng=int(ck["ng"]))
        net.load_state_dict(ck["state"])
        net.eval()
    out=(json.loads(a.out.read_text()) if a.out.exists()
         else dict(method="original BFZI V150 true OBB wet FV face polygon",
                   original_Green_solver=True,geometries={}))
    for name in a.poses:
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        prev=GeomHeat.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(
            study["geometries"][name]["new_shift_mm"],float)
        program=load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz")
        g=out["geometries"].setdefault(name,dict(levels={}))
        prefix=("exact" if a.source_mask=="legacy" else "source")
        for key in a.levels:
            xz,y=GRIDS[key]
            t0=time.perf_counter()
            data,_=create(bf,v150,"M4",**geo,
                          sea_void="oriented",physical_box_only=True,
                          grid_xz=xz,grid_y=y)
            em=BFZIV148Adapter(
                data,bf,heat_mode="port-weak",fv_faces="exact",
                source_mask=a.source_mask)
            et=BFZIV148Pkg3D(
                em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
                pkg_sea_model="face-resistance",pkg_sea_h=20.,
                pkg_sea_quad=2,pkg_sea_near=4)
            head=GeomHeat.from_original(em,et,prev.ext)
            setup=time.perf_counter()-t0
            phy=direct(et,program["time_s"],program["voltage_v"])
            path=a.runs/f"bfzi_pose_{name.lower()}_{prefix}_{key}.npz"
            pack_save(path,phy)
            old="oriented_medium" if key=="medium" else key
            refpath=(a.runs/f"bfzi_pose_{name.lower()}_exact_{key}.npz"
                     if a.source_mask=="oriented" else
                     a.runs/f"bfzi_pose_{name.lower()}_disjoint_{old}.npz")
            ref=load(refpath)
            row=dict(
                nsea=head.ns,setup_sec=setup,direct_sec=phy["elapsed_sec"],
                material_budget_W=phy["max_budget_W"],
                FV_edges=len(em.proj.edges),
                exact_face_corrected_edges=int(
                    em.proj.meta["face_area_corrected_edges"]),
                exact_face_removed_edges=int(
                    em.proj.meta["face_area_removed_edges"]),
                cold_z6=head.z0.tolist(),
                weak_mass_cells=int(np.count_nonzero(
                    np.any(abs(em.mass)>1e-18,axis=(1,2)))),
                weak_mass_l2=float(np.linalg.norm(em.mass)),
                thermal_file=str(path),
                interface_G_W_K=[
                    float(sum(x[2] for x in li)) for li in et.pkg_links],
                same_mesh_vs_volume_fraction=compare(phy,ref))
            if a.hybrid and key=="extra":
                rom=GeomHeatROM.build(head,rank=32)
                sea=basis(head.vol,rom.phi[:head.ns],rank=8)
                tan,stat=build_tan(em,head,rom,sea,6,amplitude=.2)
                t0=time.perf_counter()
                pred=rom.rollout(net,ck,program["voltage_v"],
                                 program["time_s"],tangent=tan,blend=1.)
                row["hybrid_k6"]=dict(
                    **metrics(pred,phy["temp_K"],phy["z6"],head),
                    inference_sec=time.perf_counter()-t0,
                    directional_EM_solves=tan.nsolve,
                    directional_EM_sec=stat["directional_sec"])
                head.save(a.runs/f"bfzi_pose_{name.lower()}_{prefix}_anchor.npz")
                rom.save(a.runs/f"bfzi_pose_{name.lower()}_{prefix}_rom.npz")
                tan.save(a.runs/f"bfzi_pose_{name.lower()}_{prefix}_k6.npz")
            g["levels"][key]=row
            print(name,key,"n",head.ns,"modified_faces",
                  row["exact_face_corrected_edges"],
                  "old_vs_exact_dZ_pct",round(row[
                      "same_mesh_vs_volume_fraction"][
                          "thermal_delta_Z_relative_L2"]*100,5),
                  "direct",round(row["direct_sec"],3),flush=True)
            a.out.parent.mkdir(parents=True,exist_ok=True)
            a.out.write_text(json.dumps(out,indent=2))
        if all(k in g["levels"] for k in ("medium","extra")):
            coarse=load(a.runs/f"bfzi_pose_{name.lower()}_{prefix}_medium.npz")
            fine=load(a.runs/f"bfzi_pose_{name.lower()}_{prefix}_extra.npz")
            g["levels"]["medium"]["vs_exact_extra"]=compare(coarse,fine)
            print("CONVERGENCE",name,"dZ_pct",round(
                g["levels"]["medium"]["vs_exact_extra"][
                    "thermal_delta_Z_relative_L2"]*100,6),
                "pkg_K",round(g["levels"]["medium"]["vs_exact_extra"][
                    "package_max_abs_K"],6),flush=True)
            a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
