"""D: cross-geometry held-out P03/P07/G3 and true 8-probe offline comparison."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_layer_study import GRIDS
from examples.bfzi_obb_subcell_study import measure
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat, pack
from sdfmpneo_bfzi.heat_rom import HeatROM
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D
from sdfmpneo_bfzi.geom_shared_rom import SharedROM

TARGET={"p03":"g1","p07":"g2","g3":"g1"}
def make_geo(runs,target):
    parent="p07" if target=="p07" else "p03"
    meta=json.loads((runs/f"bfzi_pose_{parent}_anchor.json").read_text())
    geo=build_args(meta)
    old=json.loads((runs/"bfzi_obb_physical_study.json").read_text())
    shift=np.array(old["geometries"][parent.upper()]["new_shift_mm"])
    if target=="g3":
        shift[1]+=10.
        geo.update(alpha2=.88,core2_mm=16.,box2_mm=228.,
                   tilt_deg=22.,yaw_deg=6.,pitch_deg=8.)
    geo["shift_mm"]=shift
    return parent,geo

def build_et(data,bf,ext):
    em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                       fv_faces="exact",source_mask="subcell")
    et=BFZIV148Skin3D(em,external=ext,pkg_shape=(4,2,4),
                      pkg_k=.25,skin_depth=.002,skin_layers=8,
                      skin_ratio=1.4,skin_quad=2,skin_near=4,
                      skin_map=2,pkg_sea_h=20.,skin_heat="source")
    return et

def baseline(data,bf,ext,geom_sec):
    """Original BFZI cold geometry + 32-mode H + eight dynamic EM directions."""
    t0=time.perf_counter()
    et=build_et(data,bf,ext)
    heat_sec=time.perf_counter()-t0
    t1=time.perf_counter()
    _,_,power,_,_=et.em(np.zeros(et.n),np.array([1.,0.],complex))
    cold_sec=time.perf_counter()-t1
    # Cold physical power response, never the held-out dynamic truth.
    t_rom=time.perf_counter()
    A=diags(et.cap)+30.*et.H
    snap=np.stack([np.full(et.n,et.amb),
                   et.amb+np.asarray(spsolve(A,30*power*3000.))])
    rom=HeatROM(et.cap,et.H,snap,rank=32,ambient=et.amb)
    rom_sec=time.perf_counter()-t_rom
    t2=time.perf_counter()
    for j in range(8):
        phi=rom.basis[:,j]
        sc=1./max(float(np.max(abs(phi))),1e-30)
        et.em(sc*phi,np.array([1.,0.],complex))
    em_sec=time.perf_counter()-t2
    stats=dict(cold_geometry_sec=float(geom_sec),
               thermal_setup_sec=float(heat_sec),
               rom_build_sec=float(rom_sec),
               cold_solve_sec=float(cold_sec),
               eight_em_directions_sec=float(em_sec),
               train_sec=0.,
               total_sec=float(geom_sec+heat_sec+cold_sec+rom_sec+em_sec))
    return et,stats

def eval_pred(pred,z,temp,ns):
    dz=z-z[0]
    d=pred["z6"]-pred["z6"][0]
    x=pred["temp_K"]-temp
    return dict(delta_Z_rel_L2=float(np.linalg.norm(d-dz) /
                                    max(np.linalg.norm(dz),1e-30)),
                pkg_max_abs_K=float(np.max(abs(x[:,ns:ns+64]))),
                copper_max_abs_K=float(np.max(abs(x[:,ns+64:ns+66]))),
                max_total_budget_W=float(np.max(abs(pred["total_budget_W"]))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--targets",nargs="+",choices=tuple(TARGET),default=tuple(TARGET))
    p.add_argument("--baseline",action="store_true")
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    out=(json.loads((a.runs/"bfzi_d_shared.json").read_text())
         if (a.runs/"bfzi_d_shared.json").exists() else {})
    xz,y=GRIDS["extra"]
    for name in a.targets:
        parent,geo=make_geo(a.runs,name)
        donor=TARGET[name]
        prev=GeomHeat.load(a.runs/f"bfzi_pose_{parent}_disjoint_anchor.npz")
        t0=time.perf_counter()
        data,info=create(bf,v150,"M4",**geo,
                         sea_void="oriented",physical_box_only=True,
                         grid_xz=xz,grid_y=y)
        geom_sec=time.perf_counter()-t0
        t1=time.perf_counter()
        em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                           fv_faces="exact",source_mask="legacy")
        mass_sec=time.perf_counter()-t1
        t2=time.perf_counter()
        cold=em.solve(np.full(len(em.vol),293.15),np.full(2,293.15))
        cold_sec=time.perf_counter()-t2
        pack_z=pack(cold.total_z)
        shared=SharedROM.load(a.runs,donor)
        prep_sec=time.perf_counter()-t0
        if name=="g3":
            t=np.arange(0.,601.,30.)
            v=np.full_like(t,57.5)
            v[(t>=150.)&(t<300.)]=0.
            v[t>=450.]=57.5
        else:
            with np.load(a.runs/f"bfzi_pose_{name}_skin_source_subcell_extra.npz") as x:
                t=x["time_s"].copy()
                v=x["voltage_v"].copy()
                refz=x["z6"].copy()
                reft=x["temp_K"].copy()
        t3=time.perf_counter()
        result=shared.rollout(pack_z,t,v,sea_xyz=em.points)
        infer_sec=time.perf_counter()-t3
        geo_json={k:(v.tolist() if isinstance(v,np.ndarray) else v)
                  for k,v in geo.items()}
        item=dict(donor=donor,geometry=geo_json,
                  offline=dict(cold_geometry_sec=float(geom_sec),
                               cold_mass_sec=float(mass_sec),
                               cold_solve_sec=float(cold_sec),
                               ROM_share_sec=float(prep_sec-geom_sec-mass_sec-cold_sec),
                               new_geometry_train_sec=0.,
                               total_sec=float(prep_sec)),
                  infer_sec=float(infer_sec),
                  target_cells=len(em.vol))
        if a.baseline or name=="g3":
            et,b=baseline(data,bf,prev.ext,geom_sec)
            item["baseline"]=b
            item["offline_reduction"]=float(1.-prep_sec/b["total_sec"])
        if name=="g3":
            direct=measure(et,t,v)
            refz=direct["z6"]
            reft=direct["temp_K"]
            np.savez_compressed(
                a.runs/"bfzi_d_g3_truth.npz",
                time_s=t,voltage_v=v,z6=refz,temp_K=reft,
                Psea_W=direct["Psea_W"],Pcu_W=direct["Pcu_W"],
                sea_xyz=em.points,sea_vol=em.vol,
                nsea=len(em.vol),n_pkg=et.pkg_n)
            item["direct_sec"]=float(direct["direct_sec"])
        item["heldout"]=eval_pred(result,refz,reft,len(em.vol))
        item["heldout"]["cold_delta_Z_abs"]=float(
            np.linalg.norm(shared.target(pack_z).z0-refz[0]))
        out[name]=item
        (a.runs/"bfzi_d_shared.json").write_text(json.dumps(out,indent=2))
        print("D",name,json.dumps({k:v for k,v in item.items()
                                   if k!="geometry"}),flush=True)
    print("SAVED",a.runs/"bfzi_d_shared.json",flush=True)

if __name__=="__main__":
    main()
