"""Generate a local cold-only geometry batch around a difficult held-out pose."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from examples.bfzi_cold_lead_guard import gap
from examples.bfzi_param import originals


def key(g):
    return json.dumps(g,sort_keys=True,separators=(",",":"))


def make(center,n=50,seed=62028,bf=None,scale=1.0,prefix='L',existing=()):
    rng=np.random.default_rng(seed)
    seen={key(g) for g in existing}
    cases={}
    tries=0
    for _ in range(12000):
        tries+=1
        if len(cases)>=n:break
        g=dict(
            alpha1=round(float(np.clip(rng.normal(center["alpha1"],.022*scale),.70,.999)),4),
            alpha2=round(float(np.clip(rng.normal(center["alpha2"],.018*scale),.70,.999)),4),
            core1_mm=int(np.clip(round(rng.normal(center["core1_mm"],1.8*scale)),1,25)),
            core2_mm=int(np.clip(round(rng.normal(center["core2_mm"],1.5*scale)),1,25)),
            shift_mm=[int(np.clip(round(rng.normal(v,s)),lo,hi))
                      for v,s,lo,hi in zip(center["shift_mm"],(2.8*scale,3.0*scale,2.4*scale),(-16,80,-12),(16,116,12))],
            tilt_deg=int(np.clip(round(rng.normal(center["tilt_deg"],3.0*scale)),-20,20)),
            yaw_deg=int(np.clip(round(rng.normal(center["yaw_deg"],3.0*scale)),-20,20)),
            pitch_deg=int(np.clip(round(rng.normal(center["pitch_deg"],3.0*scale)),-18,18)),
            box1_mm=int(np.clip(round(rng.normal(center["box1_mm"],3.5*scale)),210,242)),
            box2_mm=int(np.clip(round(rng.normal(center["box2_mm"],3.0*scale)),210,242)))
        if key(g) in seen:continue
        seen.add(key(g))
        try:
            if not (0.08885-g["core1_mm"]*.001-
                    g["alpha1"]*.05>1.25*.000790569415 and
                    g["core1_mm"]*.001-(1-g["alpha1"])*.05>=0 and
                    0.08885-g["core2_mm"]*.001-
                    g["alpha2"]*.05>1.25*.000790569415 and
                    g["core2_mm"]*.001-(1-g["alpha2"])*.05>=0):
                continue
            if not all(x["clear"] for x in gap(bf,g)):continue
        except Exception:
            continue
        cases[f"{prefix}{len(cases)+1:04d}"]=g
    if len(cases)<n:
        raise RuntimeError(f"only {len(cases)} valid local geometries from {tries} attempts")
    names=list(cases)
    rng.shuffle(names)
    nt=int(round(n*.6));nv=int(round(n*.2))
    return dict(description="BFZI V150 local cold-only batch around held-out C0260",
        version=1,seed=seed,center="C0260",attempts=tries,scale=float(scale),
        train=names[:nt],val=names[nt:nt+nv],test=names[nt+nv:],cases=cases,
        solver=dict(frequency_hz=200000,grid="ultra",medium="original M4 seawater",source_order=5))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--center-plan",type=Path,default=Path("runs/geomfrac_cold_dyn6_plan.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_c0260_local_plan.json"))
    p.add_argument("--samples",type=int,default=50)
    p.add_argument("--seed",type=int,default=62028)
    p.add_argument("--scale",type=float,default=1.0)
    p.add_argument("--prefix",default="L")
    p.add_argument("--existing",type=Path,nargs="*",default=[
        Path("runs/geomfrac_cold500_plan.json"),
        Path("runs/geomfrac_c0260_local_plan.json")])
    a=p.parse_args()
    if a.samples<20 or a.scale<=0:p.error("samples must be >=20 and scale positive")
    plan=json.loads(a.center_plan.read_text(encoding="utf-8-sig"))
    center=plan["cases"]["C0260"]["geometry"]
    bf,_=originals(a.bfzi_root)
    existing=[]
    for f in a.existing:
        if not f.exists():continue
        obj=json.loads(f.read_text(encoding="utf-8-sig"))
        existing.extend(obj.get("cases",{}).values())
    data=make(center,a.samples,a.seed,bf,a.scale,a.prefix,existing)
    a.out.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print("COLD_LOCAL_PLAN","train",len(data["train"]),"val",len(data["val"]),
          "test",len(data["test"]),"attempts",data["attempts"],
          "scale",data["scale"],"prefix",a.prefix,flush=True)


if __name__=="__main__":main()
