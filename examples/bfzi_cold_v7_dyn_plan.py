"""Select six core-domain dynamic holdouts for V7, using geometry-only coverage."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np
from examples.bfzi_cold_v7_plan import read,category,BOUNDS
from sdfmpneo_bfzi.geom_cold_domain import load,check

def vec(g):
    x=[g["alpha1"],g["alpha2"],g["core1_mm"],g["core2_mm"],
       *g["shift_mm"],g["tilt_deg"],g["yaw_deg"],g["pitch_deg"],
       g["box1_mm"],g["box2_mm"]]
    return np.asarray([(v-lo)/(hi-lo) for v,(_,lo,hi) in zip(x,BOUNDS)],float)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold-plan",type=Path,default=Path("runs/geomfrac_cold_v7_plan.json"))
    p.add_argument("--old-plan",type=Path,default=Path("runs/geomfrac_cold_dyn6_plan.json"))
    p.add_argument("--domain",type=Path,default=Path("runs/geomfrac_cold_v7_domain.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_cold_v7_dyn_plan.json"))
    p.add_argument("--seed",type=int,default=72028)
    a=p.parse_args()
    cold=read(a.cold_plan);old=read(a.old_plan);dom=load(a.domain)
    fixed=["C0416","C0725"]
    if any(category(old["cases"][k]["geometry"])!="core" for k in fixed):
        raise ValueError("existing core dynamic anchors changed domain class")
    candidates=[]
    for name in cold["test"]:
        if cold["strata"][name]!="core":continue
        g=cold["cases"][name]
        if check(g,dom)["accepted"]:candidates.append((name,g))
    if len(candidates)<4:
        raise RuntimeError(f"only {len(candidates)} accepted core test geometries")
    chosen=[]
    base=[vec(old["cases"][k]["geometry"]) for k in fixed]
    pool=list(candidates)
    while len(chosen)<4:
        best=max(pool,key=lambda item:min(
            np.linalg.norm(vec(item[1])-x) for x in base))
        chosen.append(best);base.append(vec(best[1]));pool.remove(best)
    cases={k:copy.deepcopy(old["cases"][k]) for k in fixed}
    for i,(src,g) in enumerate(chosen,1):
        name=f"V7D{i:04d}"
        template=copy.deepcopy(old["cases"]["C0416" if i%2 else "C0725"])
        template["geometry"]=g
        cases[name]=template
    names=fixed+[f"V7D{i:04d}" for i in range(1,5)]
    out=dict(description="V7 core-only independent dynamic acceptance plan",
        version=7,seed=a.seed,train=[],val=[],test=names,cases=cases,
        selected_new=[dict(name=f"V7D{i:04d}",source_test=chosen[i-1][0])
                      for i in range(1,5)],
        domain_model=str(a.domain),
        note="Two existing core dynamic geometries plus four new core geometries from the frozen V7 cold test split.")
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("V7_DYN_PLAN",json.dumps(out["selected_new"]),flush=True)
if __name__=="__main__":main()
