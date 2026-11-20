"""Freeze a second untouched geometry holdout for the V8 parallel head."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from examples.bfzi_v8_dyn_expand import pick
from examples.bfzi_cold_v7_dyn_plan import vec
from sdfmpneo_bfzi.geom_cold_domain import check, load


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v7_plan.json"))
    p.add_argument("--expand",type=Path,default=Path("runs/geomfrac_v8_dyn_expand_plan.json"))
    p.add_argument("--old-core",type=Path,default=Path("runs/geomfrac_cold_v7_dyn_plan.json"))
    p.add_argument("--domain",type=Path,default=Path("runs/geomfrac_cold_v7_domain.json"))
    p.add_argument("--used",type=Path,default=Path("runs/geomfrac_v8_dyn_holdout_plan.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v8_par48_holdout_plan.json"))
    p.add_argument("--count",type=int,default=4)
    p.add_argument("--used-extra",type=Path,default=None)
    p.add_argument("--prefix",type=str,default="V8P")
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError(f"holdout already frozen: {a.out}")
    cold=read(a.cold)
    exp=read(a.expand)
    old=read(a.old_core)
    used=read(a.used)
    dom=load(a.domain)
    forbidden={x["source_test"] for x in old.get("selected_new",[])}
    forbidden.update(used.get("source_test",{}).values())
    if a.used_extra:
        forbidden.update(read(a.used_extra).get("source_test",{}).values())
    seeds=[vec(c["geometry"]) for c in
           list(exp["cases"].values())+list(old["cases"].values())]
    for src in used.get("source_test",{}).values():
        seeds.append(vec(cold["cases"][src]))
    pool=[n for n in cold["test"] if n not in forbidden and
          cold["strata"][n]=="core" and
          check(cold["cases"][n],dom)["accepted"]]
    if len(pool)<a.count:
        raise RuntimeError(f"only {len(pool)} unused core geometries remain")
    chosen=pick(pool,cold,a.count,seeds)
    cases={}
    names=[]
    for i,src in enumerate(chosen,1):
        name=f"{a.prefix}{i:04d}"
        template=copy.deepcopy(old["cases"]["C0416" if i%2 else "C0725"])
        template["geometry"]=copy.deepcopy(cold["cases"][src])
        cases[name]=template
        names.append(name)
    out=dict(description="Second untouched V8 full-physics holdout for full-state parallel impedance head",
        version=8,train=[],val=[],test=names,cases=cases,
        source_test=dict(zip(names,chosen)),
        frozen_from="unused V7 cold test geometries",
        excludes=list(forbidden),
        note="Do not use these geometries for training, model selection or hyperparameter tuning.")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("V8_PAR48_HOLDOUT",json.dumps(out["source_test"]),flush=True)


if __name__=="__main__":
    main()
