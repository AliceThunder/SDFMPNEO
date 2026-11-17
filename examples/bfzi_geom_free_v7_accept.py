"""Frozen independent BFZI-free V6 acceptance after true S5 teachers exist.

Build teachers first with examples.bfzi_s5_buildset --tag free7.
This entrypoint never trains the model or changes the frozen geometry split.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight(sel):
    plan=load(sel["plan"])
    keys=sel["test"]
    if (plan["test"]!=keys or plan["train"] or plan["val"] or
            list(plan["cases"])!=keys):
        raise ValueError("independent test geometry split was modified")
    if sel["shape_mix"]!=0.5 or sel["durations_s"]!=[60,300,600]:
        raise ValueError("frozen V6 operating configuration changed")
    if len(set(keys))!=6:
        raise ValueError("expected six distinct new geometries")
    for key in keys:
        got=[int(d["duration"]) for d in plan["cases"][key]["drives"]]
        if got!=sel["durations_s"]:
            raise ValueError(f"{key}: expected 60/300/600 second histories")
    for key in ("atlas","heat_model","z_fix_model","field_atlas",
                "skin_model","port_self_model","mutual_model","source_model"):
        if not Path(sel[key]).is_file():
            raise FileNotFoundError(f"frozen V6 model missing: {sel[key]}")
    mf=load(sel["manifest"])
    report=load(sel["physical_report"])
    specs=mf.get("test",[])
    if mf.get("train") or mf.get("val") or len(specs)!=len(keys):
        raise ValueError("geometry-independent teacher manifest mismatch")
    by={}
    for spec in specs:
        name=Path(spec["anchor"]).stem.split("_")[2]
        if name in by or name not in keys:
            raise ValueError("teacher anchor is repeated or not frozen")
        by[name]=spec
    phys={}
    for name in keys:
        if name not in by or name not in report["cases"]:
            raise ValueError(f"{name}: S5 physical reference not ready")
        entry=report["cases"][name]
        spec=by[name]
        if (entry["split"]!="test" or
                entry["anchor"]!=spec["anchor"] or
                len(spec["teacher"])!=3 or
                len(entry["traces"])!=3):
            raise ValueError(f"{name}: teacher reports disagree")
        if not Path(spec["anchor"]).is_file():
            raise FileNotFoundError(spec["anchor"])
        checks=[]
        for dur,path,trace in zip(sel["durations_s"],
                                  spec["teacher"],entry["traces"]):
            status=trace["physics"]
            if (trace["teacher"]!=path or status["status"]!="candidate" or
                    not np.isclose(status["completed_s"],dur) or
                    not Path(path).is_file()):
                raise ValueError(f"{name}: incomplete {dur}s BFZI S5 teacher")
            checks.append(dict(duration_s=dur,
                               status=status["status"],
                               teacher=path,
                               elapsed_s=status["elapsed_s"],
                               basis=trace["basis"]))
        phys[name]=dict(build_seconds=entry["seconds"],traces=checks)
    return plan,mf,phys


def group(rows,keys):
    out={}
    for key in keys:
        a=[r for r in rows if r["geometry"]==key]
        if len(a)!=3:
            raise ValueError(f"{key}: expected three compared histories")
        out[key]=dict(
            dynamic_z_rel_L2=float(np.mean([r["dynamic_z_rel_L2"] for r in a])),
            cold_z_rel_L2=a[0]["cold_z_rel_L2"],
            current_rel_L2=float(np.mean([r["current_rel_L2"] for r in a])),
            field_rmse_K=float(np.mean([r["temperature_rmse_K"] for r in a])),
            field_max_K=float(max(r["temperature_max_K"] for r in a)),
            first_new_geometry_s=a[0]["first_new_geometry_s"],
            repeat_infer_s=float(np.mean([r["repeat_s"] for r in a])))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",type=Path,
                   default=Path("runs/geomfrac_free_v7_selection.json"))
    p.add_argument("--out",type=Path,default=None)
    a=p.parse_args()
    sel=load(a.selection)
    plan,mf,phys=preflight(sel)
    path=a.out or Path(sel["out"])
    raw=path.with_name(path.stem+"_raw.json")
    cmd=[sys.executable,"-m","examples.bfzi_geom_free_eval",
         "--plan",sel["plan"],"--manifest",sel["manifest"],
         "--split","test","--atlas",sel["atlas"],
         "--model",sel["heat_model"],"--z-fix-model",sel["z_fix_model"],
         "--field-atlas",sel["field_atlas"],
         "--skin-model",sel["skin_model"],
         "--port-self-model",sel["port_self_model"],
         "--mutual-model",sel["mutual_model"],
         "--source-model",sel["source_model"],
         "--shape-mix",str(sel["shape_mix"]),
         "--out",str(raw)]
    start=time.perf_counter()
    subprocess.run(cmd,check=True)
    wall=time.perf_counter()-start
    result=load(raw)
    if (result["geometry_count"]!=6 or
            result["history_count"]!=18):
        raise ValueError("not all frozen geometries and histories were evaluated")
    rows=result["rows"]
    for key in sel["test"]:
        if sorted(int(round(float(r["file"].split("_drive")[1].split(".")[0])))
                  for r in rows if r["geometry"]==key)!=[1,2,3]:
            raise ValueError(f"{key}: incomplete or duplicated history")
    per_geo=group(rows,sel["test"])
    sm=result["summary"]
    source=dict(
        primary="dynamic thermal impedance and circuit coupling",
        cold_z_mean=sm["cold_z_rel_L2_mean"],
        dynamic_z_mean=sm["dynamic_z_rel_L2_mean"],
        current_mean=sm["current_rel_L2_mean"],
        temperature_rmse_K=sm["temperature_rmse_K_mean"],
        sea_skin_rmse_K=sm["sea_skin_rmse_K_mean"],
        package_tx_rmse_K=sm["package_tx_rmse_K_mean"],
        package_rx_rmse_K=sm["package_rx_rmse_K_mean"])
    out=dict(status="frozen_new_geometry_complete",
             geometry_count=6,history_count=18,
             physics_reference=phys,
             frozen_selection=str(a.selection),
             model="BFZI-free V6 with 32D shared heat, 48-mode 3D field",
             shape_mix=sel["shape_mix"],
             cold_BFZI_online=0,geometry_specific_heat_basis_builds=0,
             evaluation_wall_s=wall,metrics=sm,
             source_diagnostics=source,
             per_geometry=per_geo,rows=rows)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("FREE_FROZEN_ACCEPT",path)
    print("DYNAMIC_Z_PCT",round(100*sm["dynamic_z_rel_L2_mean"],4),
          "CURRENT_PCT",round(100*sm["current_rel_L2_mean"],4),
          "FIELD_RMSE_K",round(sm["temperature_rmse_K_mean"],4),
          "BUDGET_W",sm["energy_budget_W_mean"],
          "REPEAT_S",round(sm["repeat_s_mean"],4),flush=True)


if __name__=="__main__":
    main()
