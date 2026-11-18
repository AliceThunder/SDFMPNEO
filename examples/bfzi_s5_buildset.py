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
    if plan.get("invalid_for_physics"):
        raise ValueError("invalid physical plan; use "+str(plan.get("superseded_by","corrected plan")))
    cases = plan["cases"]
    tr, val = list(plan["train"]), list(plan["val"])
    test=list(plan.get("test",[]))
    keys=tr+val+test
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("all geometry splits must be disjoint")
    frozen = {f"U{i:02d}" for i in range(1,13)}
    if (set(keys) & frozen) or (set(keys)-set(cases)):
        raise ValueError("training/validation keys must be non-frozen, defined cases")
    seen = {}
    for key in keys:
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


def _save(path, obj):
    """Persist progress after every attempt; a restart can resume finished work."""
    path=Path(path)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    tmp.replace(path)


def generate(plan, *, bfzi_root, runs, grid="ultra", heat_step=1.875,
             limit_skin=None, limit_pkg=None, duration=600., tag=""):
    tr,val=check_plan(plan)
    if (heat_step<=0 or grid not in GRIDS or duration<=0
            or abs(duration/15.-round(duration/15.))>1e-9):
        raise ValueError("invalid time step or BFZI grid")
    bf,v150=originals(bfzi_root)
    runs=Path(runs)
    runs.mkdir(parents=True,exist_ok=True)
    suffix=f"_{tag}" if tag else ""
    mf=runs/f"geomfrac_s5{suffix}_manifest.json"
    rf=runs/f"geomfrac_s5{suffix}_build_report.json"
    manifest=json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {
        "train":[],"val":[],"test":[]}
    report=json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else {
        "cases":{},"teacher":"BFZI V150 five-point-order S5"}
    done=failed=0

    for key in tr+val+plan.get("test",[]):
        start=time.perf_counter()
        c=plan["cases"][key]
        role="train" if key in tr else ("val" if key in val else "test")
        anchor=runs/f"bfzi_s5_{key}_{grid}{suffix}_anchor.npz"
        paths=[runs/f"bfzi_s5_{key}_{grid}{suffix}_drive{j+1}.npz"
               for j in range(len(c["drives"]))]
        prior=report.get("cases",{}).get(key,{})
        old={str(item.get("teacher")):item
             for item in prior.get("traces",[])}
        valid=lambda path: (Path(path).is_file() and
                 old.get(str(path),{}).get("physics",{}).get("status")=="candidate")
        if anchor.is_file() and all(valid(p) for p in paths):
            print("S5DATA",key,"SKIP_COMPLETE",len(paths),flush=True)
            done+=len(paths)
            continue

        checks=[]
        traces=[]
        head=et=None
        try:
            parent="p03" if c["donor"]=="g1" else "p07"
            ext=GeomHeat.load(runs/f"bfzi_pose_{parent}_disjoint_anchor.npz").ext
            xz,y=GRIDS[grid]
            bfdata,_=create(bf,v150,"M4",**c["geometry"],
                            sea_void="oriented",physical_box_only=True,
                            grid_xz=xz,grid_y=y,grid_max_nodes=4000)
            mat=c["thermal"]
            opts=dict(alpha=float(c["alpha_per_K"]),source_order=5,
                      skin=mat.get("skin_m",.002),ins_k=mat.get("ins_k",.25),
                      ins_t=mat.get("ins_t"),pkg_k=mat.get("pkg_k",.25),
                      pkg_sea_h=mat.get("pkg_sea_h",20.),
                      sea_max=373.15,cu_max=1357.77,pkg_max=None)
            et=build(bfdata,bf,ext,opts)
            ins_t=float(opts["ins_t"]) if opts["ins_t"] is not None else (
                float(np.mean(et.cu_ins_layers)))
            thermal=np.array([opts["alpha"],opts["ins_k"],ins_t,
                              opts["pkg_k"],opts["pkg_sea_h"]])
            head=from_skin(et.model,et,ext,thermal=thermal)
            warm,*_=et.em(et.zeros(),np.array([1.,0.]))
            if not np.allclose(pack(warm.total_z),head.z0,atol=1e-7,rtol=1e-6):
                raise RuntimeError(f"{key}: cold EM anchor differs from S5 teacher")
            head.save(anchor)
        except Exception as exc:
            failed+=len(paths)
            msg=f"{type(exc).__name__}: {exc}"
            report["cases"][key]=dict(split=role,anchor=str(anchor),
                traces=[],status="anchor_error",error=msg,
                seconds=time.perf_counter()-start)
            _save(rf,report)
            print("S5DATA",key,"ANCHOR_FAILED",msg,flush=True)
            continue

        for j,exc in enumerate(c["drives"]):
            path=paths[j]
            if valid(path):
                checks.append(old[str(path)])
                traces.append(str(path))
                done+=1
                print("S5DATA",key,j+1,"SKIP_COMPLETE",flush=True)
                continue
            try:
                spec=dict(c)
                spec.update(voltage=float(exc["voltage"]),off=exc["off"],
                            duration=float(exc.get("duration",duration)))
                status,raw=march(et,spec,15.,inner_step=heat_step)
                expected=int(round(spec["duration"]/15.))+1
                if status["status"]!="candidate" or len(raw["time_s"])!=expected:
                    raise RuntimeError(
                        f"physical status={status['status']}, "
                        f"failed_at={status.get('failed_at')}, "
                        f"points={len(raw['time_s'])}/{expected}")
                metric=metrics(head,teacher_temp(raw["temp_K"],head),rank=32)
                if limit_skin is not None and (
                    metric["regions"]["sea_skin"]["max_abs_K"]>limit_skin):
                    raise RuntimeError(f"skin basis error exceeds {limit_skin} K")
                if limit_pkg is not None and any(
                    metric["regions"][k]["max_abs_K"]>limit_pkg
                    for k in ("package_tx","package_rx")):
                    raise RuntimeError(f"package basis error exceeds {limit_pkg} K")
                voltage=np.array([drive(spec,float(t))[0].real
                                  for t in raw["time_s"]])
                np.savez_compressed(path,**raw,voltage_v=voltage)
                checks.append(dict(teacher=str(path),physics=status,basis=metric))
                traces.append(str(path))
                done+=1
                print("S5DATA",key,j+1,"OK",len(raw["time_s"]),flush=True)
            except Exception as err:
                failed+=1
                msg=f"{type(err).__name__}: {err}"
                checks.append(dict(teacher=str(path),physics={
                    "status":"failed","reason":msg},basis=None))
                print("S5DATA",key,j+1,"FAILED",msg,flush=True)
            # Commit each history's outcome before starting the next long solve.
            report["cases"][key]=dict(split=role,anchor=str(anchor),
                traces=checks.copy(),seconds=time.perf_counter()-start,
                status=("complete" if len(traces)==len(paths) else "partial"))
            for section in ("train","val","test"):
                manifest[section]=[entry for entry in manifest.get(section,[])
                                   if entry["anchor"]!=str(anchor)]
            if traces:
                manifest[role].append(dict(anchor=str(anchor),teacher=traces.copy()))
            manifest.update(canonical=True,rank=32,duration_s=float(duration))
            _save(mf,manifest)
            _save(rf,report)
        print("S5DATA",key,len(traces),"/",len(paths),"histories",flush=True)
    print("S5DATA_SUMMARY",f"success={done}",f"failed={failed}",
          f"total={done+failed}",flush=True)
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
    p.add_argument("--cases",nargs="+",default=None,
                   help="build only selected non-frozen geometry IDs")
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding="utf-8"))
    if a.cases:
        sel=set(a.cases)
        if sel-set(plan["train"]+plan["val"]+plan.get("test",[])):
            raise ValueError("unknown geometry in --cases")
        plan={**plan,
              "train":[k for k in plan["train"] if k in sel],
              "val":[k for k in plan["val"] if k in sel],
              "test":[k for k in plan.get("test",[]) if k in sel]}
    generate(plan,bfzi_root=a.bfzi_root,runs=a.runs,grid=a.grid,
             heat_step=a.inner_step,limit_skin=a.max_skin_K,
             limit_pkg=a.max_pkg_K,duration=a.duration,tag=a.tag)


if __name__=="__main__":
    main()
