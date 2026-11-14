"""Native BFZI wet FV common-volume J/q comparison, no new EM solves.

Regenerate only two physical wet-FV grids to recover their cell boxes
and exact wet volume quadrature, then reuse archived 600 s fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_layer_study import GRIDS
from sdfmpneo_bfzi.field_transfer import shared, compare, project


def study(a):
    bf, v150 = originals(a.bfzi_root)
    geo_db=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    results={}
    for name in a.poses:
        pre=f"bfzi_pose_{name.lower()}"
        meta=json.loads((a.runs/f"{pre}_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(
            geo_db["geometries"][name]["new_shift_mm"],float)
        levels={}
        for label in ("medium","extra"):
            xz,y=GRIDS[label]
            data,_=create(bf,v150,"M4",**geo,
                          sea_void="oriented",physical_box_only=True,
                          grid_xz=xz,grid_y=y)
            levels[label]=data
        xyz,w,im,ie=shared(levels["medium"]["records"],
                           levels["extra"]["records"])
        data=dict(samples=len(w),volume_m3=float(w.sum()),modes={})
        for mask in a.source_masks:
            with np.load(a.runs/f"{pre}_subcell_{mask}_medium.npz") as m, \
                 np.load(a.runs/f"{pre}_subcell_{mask}_extra.npz") as e:
                if (len(m["sea_xyz"])!=len(levels["medium"]["records"]) or
                    len(e["sea_xyz"])!=len(levels["extra"]["records"])):
                    raise ValueError("archive and rebuilt BFZI wet FV grids differ")
                if (not np.allclose(m["sea_xyz"],levels["medium"]["points"]) or
                    not np.allclose(e["sea_xyz"],levels["extra"]["points"])):
                    raise ValueError("archived native BFZI FV ordering changed")
                times=m["field_times_s"]
                if not np.array_equal(times,e["field_times_s"]):
                    raise ValueError("BFZI comparison times differ")
                out={"times_s":times.tolist()}
                for attr in ("sea_J","sea_q"):
                    coarse=m[attr][:,im,...]
                    fine=e[attr][:,ie,...]
                    out[attr]=compare(coarse,fine,w)
                    avg=[]
                    row_vol=None
                    for snapshot in e[attr]:
                        projected,row_vol=project(
                            w,im,ie,len(m["sea_xyz"]),len(e["sea_xyz"]),snapshot)
                        avg.append(projected)
                    out[attr+"_coarse_resolved"]=compare(
                        m[attr],np.asarray(avg),row_vol)
                    if "row_vol_mismatch_rel" not in data:
                        data["row_vol_mismatch_rel"]=float(np.max(
                            abs(row_vol-m["sea_vol"])/m["sea_vol"]))
                data["modes"][mask]=out
                print(name,mask,"samples",len(w),"raw J",
                      out["sea_J"]["raw"],"q",out["sea_q"]["raw"],flush=True)
        results[name]=data
    return dict(method="original wet-FV shared physical quadrature",
                source="archived 600s BFZI P03/P07 native 3D J/q",
                sample_method="symmetric half-volume union of both wet FV quadratures",
                maps="exact containing original FV boxes; piecewise constant fields",
                no_new_em_solve=True,geometries=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),
                   default=("P03","P07"))
    p.add_argument("--source-masks",nargs="+",choices=("legacy","all","subcell"),
                   default=("legacy","all","subcell"))
    p.add_argument("--out",type=Path,default=Path("runs/bfzi_obb_common_field.json"))
    a=p.parse_args()
    out=study(a)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
