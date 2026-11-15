"""Generate non-frozen S5 full-physics teachers and a multi-geometry manifest.

One BFZI V150 model per geometry; separate excitation histories share its
cold anchor. This script never trains on frozen U01-U12 acceptance cases.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create, originals
from examples.bfzi_physics_recheck import GRIDS, build, drive, march
from examples.bfzi_geom_basis_check import metrics
from sdfmpneo_bfzi.geom_heat import GeomHeat, pack
from sdfmpneo_bfzi.s5_bridge import from_skin, teacher_temp


def check_plan(plan):
    cases = plan["cases"]
    tr, val = list(plan["train"]), list(plan["val"])
    if not tr or not val or len(set(tr+val)) != len(tr+val):
        raise ValueError("training and validation must use disjoint geometries")
    frozen = {f"U{i:02d}" for i in range(1,13)}
    if (set(tr+val) & frozen) or (set(tr+val)-set(cases)):
        raise ValueError("training/validation keys must be non-frozen, defined cases")
    seen = {}
    for key in tr+val:
        c = cases[key]
        for field in ("geometry","thermal","donor","alpha_per_K","drives"):
            if field not in c:
                raise ValueError(f"{key}: missing {field}")
        if not c["drives"]:
            raise ValueError(f"{key}: at least one complete excitation required")
        signature=json.dumps(c["geometry"],sort_keys=True)
        if signature in seen:
            raise ValueError(f"duplicate physical geometry: {seen[signature]} and {key}")
        seen[signature]=key
        for drive_spec in c["drives"]:
            if (drive_spec["voltage"]<=0 or
                any(float(t1)<=float(t0) for t0,t1 in drive_spec["off"])):
                raise ValueError(f"{key}: invalid voltage or off interval")
    return tr, val


def generate(plan, *, bfzi_root, runs, grid="ultra", heat_step=1.875,
             limit_skin=None, limit_pkg=None, duration=600., tag=""):
    tr, val = check_plan(plan)
    if (heat_step <= 0 or grid not in GRIDS or duration <= 0
            or abs(duration/15.-round(duration/15.))>1e-9):
        raise ValueError("invalid time step or BFZI grid")
    bf, v150 = originals(bfzi_root)
    runs=Path(runs)
    runs.mkdir(parents=True,exist_ok=True)
    manifest={"train":[],"val":[]}
    report={"cases":{},"teacher":"BFZI V150 five-point-order S5"}
    suffix=f"_{tag}" if tag else ""
    for key in tr+val:
        start=time.perf_counter()
        c=plan["cases"][key]
        parent="p03" if c["donor"]=="g1" else "p07"
        ext=GeomHeat.load(runs/f"bfzi_pose_{parent}_disjoint_anchor.npz").ext
        xz,y=GRIDS[grid]
        data,_=create(bf,v150,"M4",**c["geometry"],
                      sea_void="oriented",physical_box_only=True,
                      grid_xz=xz,grid_y=y,grid_max_nodes=4000)
        mat=c["thermal"]
        opts=dict(alpha=float(c["alpha_per_K"]),source_order=5,
                  skin=mat.get("skin_m",.002),ins_k=mat.get("ins_k",.25),
                  ins_t=mat.get("ins_t"),pkg_k=mat.get("pkg_k",.25),
                  pkg_sea_h=mat.get("pkg_sea_h",20.),
                  sea_max=373.15,cu_max=1357.77,pkg_max=None)
        et=build(data,bf,ext,opts)
        ins_t=float(opts["ins_t"]) if opts["ins_t"] is not None else (
            float(np.mean(et.cu_ins_layers)))
        thermal=np.array([opts["alpha"],opts["ins_k"],ins_t,
                          opts["pkg_k"],opts["pkg_sea_h"]])
        head=from_skin(et.model,et,ext,thermal=thermal)
        # The physical teacher and the surrogate cold anchor must agree.
        warm,*_=et.em(et.zeros(),np.array([1.,0.]))
        if not np.allclose(pack(warm.total_z),head.z0,atol=1e-7,rtol=1e-6):
            raise RuntimeError(f"{key}: cold EM anchor differs from S5 teacher")
        anchor=runs/f"bfzi_s5_{key}_{grid}{suffix}_anchor.npz"
        head.save(anchor)
        traces=[]
        checks=[]
        for j,exc in enumerate(c["drives"]):
            spec=dict(c)
            spec.update(voltage=float(exc["voltage"]),off=exc["off"],
                        duration=float(duration))
            status,raw=march(et,spec,15.,inner_step=heat_step)
            if status["status"]!="candidate" or len(raw["time_s"])!=int(round(duration/15.))+1:
                raise RuntimeError(f"{key} history {j}: physical teacher "
                                   f"did not complete 600s: {status['status']}, "
                                   f"{status.get('failed_at')}")
            metric=metrics(head,teacher_temp(raw["temp_K"],head),rank=32)
            if limit_skin is not None and (
                  metric["regions"]["sea_skin"]["max_abs_K"]>limit_skin):
                raise RuntimeError(f"{key} history {j}: thermal skin basis error "
                                   f"{metric['regions']['sea_skin']['max_abs_K']:.5g}K")
            if limit_pkg is not None and any(
                  metric["regions"][k]["max_abs_K"]>limit_pkg
                  for k in ("package_tx","package_rx")):
                raise RuntimeError(f"{key} history {j}: package basis error "
                                   "exceeds requested limit")
            voltage=np.array([drive(spec,float(t))[0].real
                              for t in raw["time_s"]])
            teacher=runs/f"bfzi_s5_{key}_{grid}{suffix}_drive{j+1}.npz"
            np.savez_compressed(teacher,**raw,voltage_v=voltage)
            traces.append(str(teacher))
            checks.append(dict(teacher=str(teacher),physics=status,
                               basis=metric))
        role="train" if key in tr else "val"
        manifest[role].append(dict(anchor=str(anchor),teacher=traces))
        report["cases"][key]=dict(split=role,anchor=str(anchor),
                                   traces=checks,
                                   seconds=time.perf_counter()-start)
        print("S5DATA",key,len(traces),"histories",flush=True)
    manifest["canonical"]=True
    manifest["rank"]=32
    manifest["duration_s"]=float(duration)
    (runs/f"geomfrac_s5{suffix}_manifest.json").write_text(
        json.dumps(manifest,indent=2),encoding="utf-8")
    (runs/f"geomfrac_s5{suffix}_build_report.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8")
    return manifest,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_s5_plan.json"))
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--grid",default="ultra",choices=("ultra","superfine"))
    p.add_argument("--inner-step",type=float,default=1.875)
    p.add_argument("--max-skin-K",type=float,default=None)
    p.add_argument("--max-pkg-K",type=float,default=None)
    p.add_argument("--duration",type=float,default=600.)
    p.add_argument("--tag",default="")
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding="utf-8"))
    generate(plan,bfzi_root=a.bfzi_root,runs=a.runs,grid=a.grid,
             heat_step=a.inner_step,limit_skin=a.max_skin_K,
             limit_pkg=a.max_pkg_K,duration=a.duration,tag=a.tag)


if __name__=="__main__":
    main()
