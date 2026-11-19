"""Build the V7 stratified cold-only data plan from existing BFZI V150 labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from examples.bfzi_cold_plan import sample
from examples.bfzi_param import originals
from examples.bfzi_cold_lead_guard import gap

BOUNDS=[
 ("alpha1",.70,1.),("alpha2",.70,1.),
 ("core1_mm",0.,25.),("core2_mm",0.,25.),
 ("shift_x",-16.,16.),("shift_y",80.,116.),("shift_z",-12.,12.),
 ("tilt_deg",-20.,20.),("yaw_deg",-20.,20.),("pitch_deg",-18.,18.),
 ("box1_mm",210.,242.),("box2_mm",210.,242.)]
TARGET={"train":{"core":1050,"boundary":300,"extreme":150},
        "val":{"core":105,"boundary":30,"extreme":15},
        "test":{"core":105,"boundary":30,"extreme":15}}
SOURCES={
 "base":"runs/geomfrac_cold500/data.json",
 "local1":"runs/geomfrac_c0260_local/data.json",
 "local2":"runs/geomfrac_c0260_local2/data.json"}
OLD="runs/geomfrac_cold_ports_v1.json"

def key(g):
    return json.dumps(g,sort_keys=True,separators=(",",":"))

def category(g):
    x=[g["alpha1"],g["alpha2"],g["core1_mm"],g["core2_mm"],
       *g["shift_mm"],g["tilt_deg"],g["yaw_deg"],g["pitch_deg"],
       g["box1_mm"],g["box2_mm"]]
    edge=sum(min((v-lo)/(hi-lo),(hi-v)/(hi-lo))<.05
             for v,(_,lo,hi) in zip(x,BOUNDS))
    return "extreme" if edge>=3 else ("core" if edge==0 else "boundary")

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def clear(bf,g):
    try:return all(x["clear"] for x in gap(bf,g))
    except Exception:return False

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_cold_v7_plan.json"))
    p.add_argument("--seed",type=int,default=72026)
    p.add_argument("--batch",type=int,default=500)
    p.add_argument("--max-batches",type=int,default=24)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(f"frozen V7 plan already exists: {a.out}")
    bf,_=originals(a.bfzi_root)
    old=read(OLD)
    data={k:read(v) for k,v in SOURCES.items()}
    existing={s:[] for s in ("train","val","test")}
    occupied=[]
    for d in data.values():
        for s in existing:
            existing[s].extend(d.get(s,[]))
            occupied.extend(r["geometry"] for r in d.get(s,[]))
    legacy_train=[r for r in old.get("train",[]) if clear(bf,r["geometry"])]
    legacy_val=[r for r in old.get("dev",[]) if clear(bf,r["geometry"])]
    for s in ("train","dev","test","bb_diagnostic"):
        occupied.extend(r["geometry"] for r in old.get(s,[]) if "geometry" in r)
    existing["train"].extend(legacy_train)
    existing["val"].extend(legacy_val)
    # Existing reference sets must already be geometry-disjoint.
    seen={}
    for split,rows in existing.items():
        for r in rows:
            k=key(r["geometry"])
            if k in seen:raise ValueError(f"existing geometry overlaps {seen[k]} and {split}: {r['name']}")
            seen[k]=split
    counts={s:{c:0 for c in ("core","boundary","extreme")} for s in existing}
    for s,rows in existing.items():
        for r in rows:counts[s][category(r["geometry"])]+=1
    need={}
    for s in ("train","val","test"):
        need[s]={c:max(0,TARGET[s][c]-counts[s][c]) if s=="train" else TARGET[s][c]
                 for c in ("core","boundary","extreme")}
        if s=="train" and any(counts[s][c]>TARGET[s][c] for c in counts[s]):
            raise ValueError(f"existing train strata exceed V7 targets: {counts[s]}")
    totals={c:sum(need[s][c] for s in need) for c in ("core","boundary","extreme")}
    pool={c:[] for c in totals}
    rng=np.random.default_rng(a.seed)
    batches=0;attempts=0
    while any(len(pool[c])<totals[c] for c in pool):
        if batches>=a.max_batches:
            raise RuntimeError(f"sampling incomplete: need={totals}, got={{k:len(v) for k,v in pool.items()}}")
        batch=sample(a.seed+batches,a.batch,existing=occupied,accept=lambda g:clear(bf,g))
        batches+=1;attempts+=int(batch["attempts"])
        for g in batch["cases"].values():
            occupied.append(g)
            pool[category(g)].append(g)
        print("V7_PLAN_BATCH",batches,
              {k:len(v) for k,v in pool.items()},flush=True)
    cases={};strata={};splits={s:[] for s in ("train","val","test")}
    for c in pool:rng.shuffle(pool[c])
    serial={"train":0,"val":0,"test":0}
    for s in ("train","val","test"):
        for c in ("core","boundary","extreme"):
            for g in pool[c][:need[s][c]]:
                serial[s]+=1
                prefix={"train":"V7T","val":"V7V","test":"V7E"}[s]
                name=f"{prefix}{serial[s]:04d}"
                cases[name]=g;strata[name]=c;splits[s].append(name)
            pool[c]=pool[c][need[s][c]:]
    for s in splits:
        got={c:sum(strata[n]==c for n in splits[s]) for c in totals}
        if got!=need[s]:raise RuntimeError(f"bad split {s}: {got} != {need[s]}")
    plan=dict(
        description="V7 stratified BFZI V150 cold-only reference plan",
        version=7,seed=a.seed,candidate_batches=batches,
        candidate_count=batches*a.batch,lead_clearance_checked=batches*a.batch,
        sampling_attempts=attempts,
        existing_sources=SOURCES,legacy_source=OLD,
        legacy_included={"train":[r["name"] for r in legacy_train],
                         "val":[r["name"] for r in legacy_val]},
        existing_counts={s:dict(total=len(existing[s]),strata=counts[s])
                         for s in existing},
        new_counts={s:dict(total=len(splits[s]),
                           strata={c:sum(strata[n]==c for n in splits[s])
                                   for c in totals}) for s in splits},
        target_counts=TARGET,
        domain=dict(
            global_bounds={n:[lo,hi] for n,lo,hi in BOUNDS},
            core="all 12 normalized geometry coordinates are within [0.05, 0.95]",
            boundary="one or two coordinates enter the outer 5% of the global bounds",
            extreme="three or more coordinates enter the outer 5% of the global bounds",
            inference="core only; final nearest-neighbour coverage threshold is calibrated from V7 validation core samples"),
        train=splits["train"],val=splits["val"],test=splits["test"],
        cases=cases,strata=strata,
        solver=dict(source="original BFZI V150",frequency_hz=200000,
                    geometry="M4 two-coil",medium="original M4 seawater",
                    ambient_K=293.15,grid="ultra",source_order=5))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    print("V7_PLAN_DONE",json.dumps({
        "new":{s:len(splits[s]) for s in splits},
        "existing":{s:len(existing[s]) for s in existing},
        "train_final":sum(len(existing["train"]) for _ in [0])+len(splits["train"]),
        "candidate_batches":batches,"seed":a.seed}),flush=True)

if __name__=="__main__":main()
