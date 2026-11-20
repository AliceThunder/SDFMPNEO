"""Collect BFZI physically realized hot states for geometry-disjoint EM training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import geometry
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path,rows):
    np.savez_compressed(path,
        name=np.array([r["name"] for r in rows]),
        g=np.array([r["g"] for r in rows],np.float32),
        a=np.array([r["a"] for r in rows],np.float32),
        sea=np.array([r["sea"] for r in rows],np.float32),
        total=np.array([r["total"] for r in rows],np.float32),
        time_s=np.array([r["time_s"] for r in rows],np.float32),
        voltage_v=np.array([r["voltage_v"] for r in rows],np.float32),
        history=np.array([r["history"] for r in rows]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_full_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8_full_manifest.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v9_hotstates_physical.npz"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v9_hotstates_physical.json"))
    a=p.parse_args()

    torch.set_num_threads(2)
    plan=read(a.plan)
    mf=read(a.manifest)
    cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
    src=SrcFix.load(cfg["source_model"])
    train=set(plan["train"])
    specs=mf["train"]
    names={Path(s["anchor"].replace("\\","/")).stem.split("_")[2] for s in specs}
    if names != train or len(specs) != len(train):
        raise ValueError("manifest train geometries and V8 train plan differ")

    rows=[]
    report=dict(status="running",geometries={},failures={})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for spec in specs:
        name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
        try:
            case=plan["cases"][name]
            init=thermal_mix(atlas,src.apply(case,cold.apply(
                case,atlas.predict(case))),.25)
            g=geometry(case,init)[1]
            hist=prepare(spec,48)
            count=0
            for j,row in enumerate(hist):
                z=np.asarray(row["z6"],float)
                a48=np.asarray(row["a"],float)
                sea=np.asarray(row["y"],float)[:,:6]
                time=np.asarray(row["time"],float)
                volts=np.asarray(row["voltage"],float)
                if z.shape!=(len(time),6) or a48.shape!=(len(time),48):
                    raise ValueError("teacher row shape mismatch")
                if len(time)<2 or not np.isfinite(sea).all():
                    raise ValueError("empty/nonfinite physical history")
                for i in range(1,len(time)):
                    rows.append(dict(name=name,g=g,a=a48[i],sea=sea[i],
                        total=z[i]-z[0],time_s=time[i],
                        voltage_v=volts[i],history=j))
                    count+=1
            report["geometries"][name]=dict(histories=len(hist),states=count)
            save(a.out,rows)
            print("V9_PHYS",name,len(hist),count,flush=True)
        except Exception as exc:
            report["failures"][name]=f"{type(exc).__name__}: {exc}"
            print("V9_PHYS_FAILED",name,report["failures"][name],flush=True)
        report["sample_count"]=len(rows)
        a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    report["status"]="complete" if len(report["geometries"])==len(train) and not report["failures"] else "partial"
    report["sample_count"]=len(rows)
    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V9_PHYS_DONE",len(rows),len(report["geometries"]),
          len(report["failures"]),flush=True)


if __name__=="__main__":
    main()
