"""Measure S5 full-temperature projection error independently of neural accuracy.

Use non-frozen physical teacher trajectories for basis development. Reports
bulk-water, eight-layer skin, separate packages and separate copper errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.geom_canon import basis, project, mode_scale
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import groups
from sdfmpneo_bfzi.s5_bridge import teacher_temp


def metrics(head, temp, rank=32):
    temp=np.asarray(temp,float)
    if temp.ndim!=2 or temp.shape[1]!=len(head.cap):
        raise ValueError("teacher must contain (times, thermal_nodes)")
    phi=basis(head,rank)
    state=project(head,temp-head.amb,phi)
    pred=head.amb+(state*mode_scale(head,rank))@phi.T
    err=pred-temp
    ns,npkg=head.ns,head.npkg

    def measure(sel):
        e=err[:,sel]
        if not e.size:
            return None
        t=temp[:,sel]
        p=pred[:,sel]
        return dict(max_abs_K=float(np.max(abs(e))),
                    rmse_K=float(np.sqrt(np.mean(e*e))),
                    hotspot_max_K=float(np.max(abs(
                        np.max(p,axis=1)-np.max(t,axis=1)))))

    parts=dict(seawater=measure(slice(0,ns)),
               package_tx=measure(slice(ns,ns+npkg)),
               package_rx=measure(slice(ns+npkg,ns+2*npkg)),
               copper_tx=measure(slice(ns+2*npkg,ns+2*npkg+1)),
               copper_rx=measure(slice(ns+2*npkg+1,ns+2*npkg+2)))
    if hasattr(head,"nwet"):
        nwet,nskin=int(head.nwet),int(head.nskin)
        parts["sea_bulk"]=measure(slice(0,nwet))
        parts["sea_skin"]=measure(slice(nwet,ns))
        pkg=(np.asarray(head.skin_pkg,int) if hasattr(head,"skin_pkg")
             else np.repeat(np.arange(2),nskin//2))
        for j,key in enumerate(("skin_tx","skin_rx")):
            parts[key]=measure(nwet+np.flatnonzero(pkg==j))
    cap=np.asarray(head.cap,float)
    means=[float(np.max(abs((err[:,sl]@cap[sl])/cap[sl].sum())))
           for sl in groups(head)]
    return dict(rank=rank,ntimes=len(temp),nheat=len(cap),
                global_max_K=float(np.max(abs(err))),
                global_rmse_K=float(np.sqrt(np.mean(err*err))),
                material_mean_max_K=max(means),regions=parts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--out",type=Path,default=None)
    p.add_argument("--rank",type=int,default=32,choices=(5,32))
    p.add_argument("--max-skin-K",type=float,default=None)
    p.add_argument("--max-pkg-K",type=float,default=None)
    a=p.parse_args()
    head=GeomHeat.load(a.anchor)
    with np.load(a.teacher) as f:
        temp=teacher_temp(f["temp_K"],head)
        if temp.ndim!=2:
            p.error("single S5 teacher history required")
        if "Tcu_K" in f and not np.allclose(
                temp[:,head.cu],f["Tcu_K"],atol=1e-9,rtol=0):
            p.error("S5 copper-temperature order mismatch")
        if "Tpkg_K" in f and not np.allclose(
                temp[:,head.ns:head.cu.start],f["Tpkg_K"],atol=1e-9,rtol=0):
            p.error("S5 package-temperature order mismatch")
    result=metrics(head,temp,rank=a.rank)
    result.update(anchor=str(a.anchor),teacher=str(a.teacher))
    checks={}
    if a.max_skin_K is not None:
        if "sea_skin" not in result["regions"]:
            p.error("a physical S5 skin anchor is required for --max-skin-K")
        checks["sea_skin"]=(
            result["regions"]["sea_skin"]["max_abs_K"]<=a.max_skin_K)
    if a.max_pkg_K is not None:
        for mat in ("package_tx","package_rx"):
            checks[mat]=result["regions"][mat]["max_abs_K"]<=a.max_pkg_K
    if checks:
        result["gates"]=checks
        result["passed"]=all(checks.values())
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    if checks and not result["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
