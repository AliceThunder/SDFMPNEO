"""Build D geometry-only donors G1/G2 from A2 training samples.

P03/P07 are never consulted as dynamic training targets.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_layer_study import GRIDS
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.frac_rom import FracROM
from sdfmpneo_bfzi.heat_rom import HeatROM
from sdfmpneo_bfzi.geom_heat import GeomHeat, pack
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D

CASES={"g1":("p03",20.),"g2":("p07",25.)}
def get_geo(runs,key):
    ref,dy=CASES[key]
    meta=json.loads((runs/f"bfzi_pose_{ref}_anchor.json").read_text())
    geo=build_args(meta)
    old=json.loads((runs/"bfzi_obb_physical_study.json").read_text())
    shift=np.asarray(old["geometries"][ref.upper()]["new_shift_mm"],float)
    shift[1]+=dy
    geo["shift_mm"]=shift
    return geo,ref

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--targets",nargs="+",choices=tuple(CASES),default=tuple(CASES))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    xz,y=GRIDS["extra"]
    rep={}
    for key in a.targets:
        geo,anchor=get_geo(a.runs,key)
        prev=GeomHeat.load(a.runs/f"bfzi_pose_{anchor}_disjoint_anchor.npz")
        t0=time.perf_counter()
        data,_=create(bf,v150,"M4",**geo,sea_void="oriented",
                      physical_box_only=True,grid_xz=xz,grid_y=y)
        t1=time.perf_counter()
        em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                           fv_faces="exact",source_mask="subcell")
        et=BFZIV148Skin3D(
            em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
            skin_depth=.002,skin_layers=8,skin_ratio=1.4,
            skin_quad=2,skin_near=4,skin_map=2,pkg_sea_h=20.,
            skin_heat="source")
        t2=time.perf_counter()
        with np.load(a.runs/f"bfzi_a2_{key}_extra.npz") as x:
            snaps=x["temp_K"].copy()
        rom=HeatROM(et.cap,et.H,snaps,rank=32,ambient=et.amb)
        rom.save(a.runs/f"bfzi_d_{key}_rom.npz")
        t3=time.perf_counter()
        fr=FracROM.build(et,rom)
        fr.save(a.runs/f"bfzi_d_{key}_tangent.npz")
        t4=time.perf_counter()
        # Original BFZI inexpensive cold source mask for future geometry use.
        leg=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                            fv_faces="exact",source_mask="legacy")
        cold=pack(leg.solve(np.full(len(leg.vol),et.amb),
                            np.full(2,et.amb)).total_z)
        corr=fr.z0-cold
        np.savez_compressed(a.runs/f"bfzi_d_{key}_cold.npz",
                            cold_correction=corr,sea_xyz=em.points,
                            rank=rom.rank,donor=key)
        rep[key]=dict(geometry_sec=t1-t0,thermal_setup_sec=t2-t1,
                      thermal_rom_sec=t3-t2,
                      em_32_probe_sec=t4-t3,
                      train_sec=0.,cold_legacy_offset6=corr.tolist(),
                      donor_z0=fr.z0.tolist(),thermal_nodes=et.n)
        print(key,json.dumps(rep[key]),flush=True)
    (a.runs/"bfzi_d_donor.json").write_text(json.dumps(rep,indent=2))

if __name__=="__main__":
    main()
