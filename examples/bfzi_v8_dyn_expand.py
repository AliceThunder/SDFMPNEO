"""Freeze new V8 core-domain dynamic geometries from V7 cold train/dev splits.

Select diverse geometry-only samples without touching V7 test geometries or
historical six-case dynamic holdouts. BFZI buildset can resume each sample.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from examples.bfzi_cold_v7_dyn_plan import vec
from sdfmpneo_bfzi.geom_cold_domain import check,load


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pick(names,cold,num,seeds):
    pool=sorted(names)
    chosen=[]
    for _ in range(num):
        name=max(pool,key=lambda n:(min(np.linalg.norm(
            vec(cold["cases"][n])-v) for v in seeds),n))
        seeds.append(vec(cold["cases"][name]))
        chosen.append(name)
        pool.remove(name)
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v7_plan.json"))
    p.add_argument("--old",type=Path,default=Path("runs/geomfrac_pose42_plan.json"))
    p.add_argument("--old-manifest",type=Path,default=Path("runs/geomfrac_s5_v6_36g_hold6_manifest.json"))
    p.add_argument("--core-plan",type=Path,default=Path("runs/geomfrac_cold_v7_dyn_plan.json"))
    p.add_argument("--domain",type=Path,default=Path("runs/geomfrac_cold_v7_domain.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v8_dyn_expand_plan.json"))
    p.add_argument("--train",type=int,default=12)
    p.add_argument("--val",type=int,default=4)
    a=p.parse_args()
    cold=read(a.cold)
    old=read(a.old)
    mf=read(a.old_manifest)
    core=read(a.core_plan)
    dom=load(a.domain)
    train_old={Path(s["anchor"].replace("\\","/")).stem.split("_")[2]
               for s in mf["train"]}
    seeds=[vec(old["cases"][n]["geometry"]) for n in train_old]
    pools={}
    for part in ("train","val"):
        pools[part]=[n for n in cold[part]
            if cold["strata"][n]=="core" and
               check(cold["cases"][n],dom)["accepted"]]
    tr=pick(pools["train"],cold,a.train,seeds)
    va=pick(pools["val"],cold,a.val,seeds)
    cases={}
    names={"train":[],"val":[]}
    for part,source in (("train",tr),("val",va)):
        for i,name in enumerate(source,1):
            key=("V8T" if part=="train" else "V8V")+f"{i:04d}"
            example=copy.deepcopy(core["cases"]["C0416" if i%2 else "C0725"])
            example["geometry"]=copy.deepcopy(cold["cases"][name])
            cases[key]=example
            names[part].append(key)
    out=dict(description="V8 diverse core-domain dynamic BFZI teachers; existing V7 test never used",
             version=8,train=names["train"],val=names["val"],test=[],
             cases=cases,selected={
                "train":dict(zip(names["train"],tr)),
                "val":dict(zip(names["val"],va))},
             base_train_geometry=len(train_old),
             source="frozen V7 cold train and validation geometry splits",
             note="Full 60s and pulsed 300s excitation per geometry. BFZI V150 S5 physics.")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("V8_DYN_PLAN",json.dumps(dict(train=tr,val=va,
        existing_train=len(train_old))),flush=True)


if __name__=="__main__":
    main()
