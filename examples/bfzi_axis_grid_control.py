"""Grid-convergence control with two nonoverlapping axis-aligned packages.

This control switches off the original BFZI dry riser and extra wall AABBs,
so the original BFZI water void exactly matches both sealed thermal boxes.
All values use original BFZI V150 temperature-dependent EM solves.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import values
from examples.bfzi_native_sea_convergence import GRIDS,direct,compare,pack_save
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_void import report as void_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_axis_grid_control.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    t=np.arange(0.,601.,15.)
    volt=np.full(len(t),150.)
    old=None
    runs={}
    out=dict(source="original BFZI V150 physical 3D direct EM-thermal",
             control="equal 204 mm axis-aligned boxes, 40 mm center distance",
             physical_box_only=True,voltage_V=150.,
             time_step_s=15.,duration_s=600.,levels={})
    for label,xz,y in GRIDS:
        start=time.perf_counter()
        d,info=create(
            bf,v150,"M4",grid_xz=xz,grid_y=y,
            grid_layers=0,physical_box_only=True)
        em=BFZIV148Adapter(d,bf,heat_mode="port-weak")
        ex=np.diag([.5-1j*em.omega*em.air[0,0],
                    .5-1j*em.omega*em.air[1,1]])
        et=BFZIV148Pkg3D(
            em,external=ex,pkg_shape=(4,2,4),pkg_k=.25,
            pkg_sea_model="face-resistance",pkg_sea_h=20.,
            pkg_sea_quad=2,pkg_sea_near=4)
        head=GeomHeat.from_original(em,et,ex)
        if old is None:
            old=head
        v=void_report(head,d["vol_meta"])
        if v["physical_packages_overlap"]:
            raise ValueError("axis-aligned physical control packages overlap")
        if abs(v["BFZI_AABB_over_physical_sum"]-1.)>1e-6:
            raise ValueError("native BFZI void must match axis-aligned physical package")
        setup=time.perf_counter()-start
        s=direct(et,t,volt)
        runs[label]=s
        path=a.runs/f"bfzi_axis_grid_{label}.npz"
        pack_save(path,s)
        out["levels"][label]=dict(
            grid_xz_m=xz,grid_y_m=y,nsea=len(em.vol),
            downsampled=d["vol_meta"]["downsampled"],
            mesh_rel_volume_error=d["vol_meta"]["relative_volume_error"],
            cold_z6=head.z0.tolist(),
            setup_sec=setup,direct_sec=s["elapsed_sec"],
            max_material_budget_W=s["max_budget_W"],
            package_interface_flux_W=s["max_pkg_flux_W"],
            interface=values(head,em.points),
            oriented_void=v,physical_file=str(path))
        print(label,"n",len(em.vol),"cold",round(setup,3),
              "direct",round(s["elapsed_sec"],3),
              "G",[round(q["interface_G_W_K"],5)
                   for q in out["levels"][label]["interface"]],flush=True)
    fine=runs["face_fine"]
    for label,s in runs.items():
        out["levels"][label]["vs_face_fine"]=compare(s,fine)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2))
    print("COMPARE",{label:dict(
        coldZ_pct=round(d["vs_face_fine"]["cold_Z_relative_L2"]*100,5),
        thermalZ_pct=round(d["vs_face_fine"]["thermal_delta_Z_relative_L2"]*100,5),
        pkgK=round(d["vs_face_fine"]["package_max_abs_K"],5))
        for label,d in out["levels"].items()},flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
