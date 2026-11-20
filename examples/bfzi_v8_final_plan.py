"""Freeze final V8 dynamic geometries from unseen V7 cold test partition."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from examples.bfzi_v8_dyn_expand import pick
from examples.bfzi_cold_v7_dyn_plan import vec
from sdfmpneo_bfzi.geom_cold_domain import check,load


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v7_plan.json"))
    p.add_argument("--expand",type=Path,default=Path("runs/geomfrac_v8_dyn_expand_plan.json"))
    p.add_argument("--old-core",type=Path,default=Path("runs/geomfrac_cold_v7_dyn_plan.json"))
    p.add_argument("--domain",type=Path,default=Path("runs/geomfrac_cold_v7_domain.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v8_dyn_holdout_plan.json"))
    p.add_argument("--count",type=int,default=4)
    a=p.parse_args()
    cold=read(a.cold)
    exp=read(a.expand)
    old=read(a.old_core)
    dom=load(a.domain)
    forbidden={r["source_test"] for r in old.get("selected_new",[])}
    seeds=[vec(c["geometry"]) for c in
           list(exp["cases"].values())+list(old["cases"].values())]
    pool=[n for n in cold["test"]
          if n not in forbidden and cold["strata"][n]=="core"
          and check(cold["cases"][n],dom)["accepted"]]
    chosen=pick(pool,cold,a.count,seeds)
    cases={}
    names=[]
    for i,src in enumerate(chosen,1):
        name=f"V8E{i:04d}"
        template=copy.deepcopy(old["cases"]["C0416" if i%2 else "C0725"])
        template["geometry"]=copy.deepcopy(cold["cases"][src])
        cases[name]=template
        names.append(name)
    out=dict(description="Sealed new-geometry V8 final BFZI V150 dynamic acceptance",
             version=8,train=[],val=[],test=names,cases=cases,
             source_test=dict(zip(names,chosen)),frozen_from="V7 cold test",
             note="Do not use these geometries for V8 training or hyperparameter selection.")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise FileExistsError("final V8 holdout already frozen")
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("V8_FINAL_HOLDOUT",json.dumps(out["source_test"]),flush=True)


if __name__=="__main__":
    main()
