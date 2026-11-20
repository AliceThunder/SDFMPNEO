"""Extend the shared rank48 geometry-thermal atlas with new V8 cold anchors.

This is an offline atlas update, not geometry-dependent work at inference.
Preserve old learned cold-self head and thermal interpolation rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_geom_init import pack,records
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_init import GeomInit


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def metrics(atlas,rows):
    out=[]
    for r in rows:
        init=atlas.predict(r["case"])
        out.append(dict(
            k=float(np.linalg.norm(init["k"]-r["k"])/
                    max(np.linalg.norm(r["k"]),1e-12)),
            src=float(np.linalg.norm(init["src"]-r["src"])/
                    max(np.linalg.norm(r["src"]),1e-12)),
            cap=float(np.linalg.norm(init["cap"]-r["cap"])/
                    max(np.linalg.norm(r["cap"]),1e-12))))
    return {name:float(np.mean([r[name] for r in out])) for name in
            ("k","src","cap")} if out else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=Path("runs/geomfrac_init_pose42_kell_36g_r48_v1.npz"))
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_dyn_expand_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8expand_manifest.json"))
    p.add_argument("--old-plan",type=Path,default=Path("runs/geomfrac_pose42_plan.json"))
    p.add_argument("--old-manifest",type=Path,default=Path("runs/geomfrac_s5_v6_36g_hold6_manifest.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_init_v8_expand.pt.npz"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_init_v8_expand.json"))
    a=p.parse_args()
    old=GeomInit.load(a.base)
    if old.rank!=48:
        raise ValueError("rank48 atlas required")
    plan=read(a.plan)
    mf=read(a.manifest)
    extra=[]
    for spec in mf.get("train",[]):
        name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
        extra.append(pack(GeomHeat.load(spec["anchor"]),plan["cases"][name],name,48))
    if not extra:raise ValueError("no completed new training anchors")
    x=np.vstack((old.x*old.std+old.mu,np.stack([r["x"] for r in extra])))
    y=np.vstack((old.y,np.stack([r["y"] for r in extra])))
    k=np.concatenate((old.k,np.stack([r["k"] for r in extra])))
    src=np.concatenate((old.src,np.stack([r["src"] for r in extra])))
    ext=np.concatenate((old.ext,np.stack([r["ext"] for r in extra])))
    guard=np.concatenate((old.guard,np.stack([r["guard"] for r in extra])))
    fresh=GeomInit.fit(x,y,k,src,ext,ell=old.ell,lam=old.lam,kell=old.kell)
    fresh.set_guard(guard).set_kscale(old.kexp)
    for prop in ("sx","smu","sstd","sy","sbeta","sell","slam"):
        setattr(fresh,prop,getattr(old,prop))
    old_p=read(a.old_plan)
    old_m=read(a.old_manifest)
    dev=records(old_p,old_m,"val",48)
    new_dev=[]
    for spec in mf.get("val",[]):
        name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
        new_dev.append(pack(GeomHeat.load(spec["anchor"]),plan["cases"][name],name,48))
    result=dict(status="candidate",base=str(a.base),new_train=len(extra),
                new_val=len(new_dev),old_val=len(dev),
                old_dev_before=metrics(old,dev),
                old_dev_after=metrics(fresh,dev),
                new_dev_before=metrics(old,new_dev),
                new_dev_after=metrics(fresh,new_dev),
                checkpoint=str(a.out),
                note="Compare physical thermal K/src and closed-loop dynamics; no model promotion by this script.")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fresh.save(a.out)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V8_ATLAS_EXPAND",json.dumps(result),flush=True)


if __name__=="__main__":
    main()
