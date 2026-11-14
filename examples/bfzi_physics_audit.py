"""Audit frozen BFZI final reference histories without reusing surrogate scores.

This report preserves the original 24 NPZ reference files and does not imply
that an old high-temperature trajectory is a physically valid teacher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def first_cross(t, x, limit):
    ids=np.flatnonzero(np.max(x,axis=1)>=limit)
    return None if not len(ids) else float(t[ids[0]])


def one(path, case, *, sea_max=373.15, cu_max=1357.77, pkg_max=None):
    with np.load(path) as d:
        t=np.asarray(d["time_s"],float)
        volt=np.asarray(d["voltage_v"],float)
        vol=np.asarray(d["sea_vol"],float)
        temp=np.asarray(d["temp_truth"],float)
        psea=np.asarray(d["Psea_truth"],float)
        pcu=np.asarray(d["Pcu_truth"],float)
    ns=len(vol)
    if (temp.ndim!=2 or temp.shape[0]!=len(t) or
            temp.shape[1]<ns+66 or pcu.shape!=(len(t),2) or
            psea.shape!=(len(t),) or np.any(np.diff(t)<=0) or
            not np.isfinite(temp).all()):
        raise ValueError(f"invalid frozen reference: {path}")
    sea=np.concatenate((temp[:,:ns],temp[:,ns+66:]),axis=1)
    pkg=temp[:,ns:ns+64]
    cu=temp[:,ns+64:ns+66]
    dt=np.diff(t)
    # The thermal integrator is backward Euler with a source at t[k].
    e_cu=float(np.dot(dt,np.sum(pcu[1:],axis=1)))
    e_sea=float(np.dot(dt,psea[1:]))
    lim=dict(seawater_K=float(sea_max),copper_K=float(cu_max))
    hit=dict(seawater_s=first_cross(t,sea,sea_max),
             copper_s=first_cross(t,cu,cu_max))
    if pkg_max is not None:
        lim["package_K"]=float(pkg_max)
        hit["package_s"]=first_cross(t,pkg,pkg_max)
    valid=all(v is None for v in hit.values())
    return dict(id=case,reference_valid=valid,first_exceed_s=hit,
                limits=lim,voltage_max_V=float(np.max(abs(volt))),
                time_end_s=float(t[-1]),
                min_seawater_K=float(np.min(sea)),
                max_seawater_K=float(np.max(sea)),
                max_package_K=float(np.max(pkg)),
                max_copper_K=float(np.max(cu)),
                peak_copper_W=float(np.max(np.sum(pcu,axis=1))),
                peak_seawater_W=float(np.max(psea)),
                peak_total_joule_W=float(np.max(psea+pcu.sum(axis=1))),
                copper_joule_J=e_cu,seawater_joule_J=e_sea,
                total_joule_J=e_cu+e_sea)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--keys",nargs="*",default=None)
    p.add_argument("--sea-max-K",type=float,default=373.15,
                   help="Liquid seawater limit at selected water pressure; default 1 atm")
    p.add_argument("--cu-max-K",type=float,default=1357.77,
                   help="Solid copper melting threshold")
    p.add_argument("--pkg-max-K",type=float,default=None,
                   help="Optional enclosure material validity ceiling")
    p.add_argument("--out",type=Path,default=None)
    a=p.parse_args()
    if not (a.sea_max_K>0 and a.cu_max_K>0 and
            (a.pkg_max_K is None or a.pkg_max_K>0)):
        p.error("material bounds must be positive")
    report=json.loads((a.runs/"bfzi_final_accept.json").read_text())
    keys=a.keys or sorted(report["cases"])
    result={}
    for key in keys:
        if key not in report["cases"]:
            p.error(f"unknown frozen case: {key}")
        x=one(a.runs/f"bfzi_final_{key}.npz",key,
              sea_max=a.sea_max_K,cu_max=a.cu_max_K,
              pkg_max=a.pkg_max_K)
        result[key]=x
        print(key,"VALID" if x["reference_valid"] else "INVALID",
              "Tcu",round(x["max_copper_K"],2),
              "Twater",round(x["max_seawater_K"],2),
              "Pcu",round(x["peak_copper_W"],2),
              "Psea",round(x["peak_seawater_W"],2))
    out=dict(source="frozen BFZI V150 final 24-run reference NPZ",
             historical_copper_power="Pcu_truth includes in-package winding and exterior terminal leads, all previously deposited into package copper heat state",
             material_scope="single-phase liquid seawater, solid copper",
             cases=result,
             summary=dict(checked=len(result),
                          valid=sum(v["reference_valid"] for v in result.values()),
                          invalid=[k for k,v in result.items()
                                   if not v["reference_valid"]]))
    dest=a.out or a.runs/"bfzi_physics_audit.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2)+"\n")
    print("SAVED",dest)


if __name__=="__main__":
    main()
