"""Recover physical BFZI near-source J(q) and weak-Joule q(q).

Uses archived P03/P07 600 s states; only source evaluation and fixed
geometry are needed, with no new warm Green/GMRES time integration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_layer_study import GRIDS
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.field_transfer import shared,rel
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.near_source import local_q


def unpack(z):
    return np.array([[z[0]+1j*z[1],z[2]+1j*z[3]],
                     [z[2]+1j*z[3],z[4]+1j*z[5]]],complex)


def field(em,rec,src,pts,ids,jbar,sigma,current,qcell):
    q,w,own,c=em.source_native()
    native=em.sample_j(jbar,sigma,current,q,own,source=c)
    shape,scale=local_q(native,sigma,qcell,em.vol,own,w)
    j=em.sample_j(jbar,sigma,current,pts,ids,source=src)
    qval,_=local_q(j,sigma,qcell,em.vol,ids,scale=scale)
    mean=np.zeros_like(jbar,dtype=complex)
    np.add.at(mean,own,w[:,None]*native)
    weight=np.bincount(own,weights=w,minlength=len(em.vol))
    mean/=weight[:,None]
    return dict(j=j,q=qval,
                max_j_mean_rel=float(np.linalg.norm(mean-jbar)/
                    max(np.linalg.norm(jbar),1e-30)),
                power_rel=float(abs(w@shape-qcell@em.vol)/
                    max(abs(qcell@em.vol),1e-30)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),default=("P03","P07"))
    p.add_argument("--out",type=Path,default=Path("runs/bfzi_obb_source_field.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    study=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    output=dict(
        method="original BFZI near-source C(q)-mean(C), conserved original mean J",
        q_mode="native weak Joule cell power, source-shaped nonnegative profile",
        no_warm_em_solve=True,geometries={})
    for name in a.poses:
        stamp=time.perf_counter()
        pre=f"bfzi_pose_{name.lower()}"
        meta=json.loads((a.runs/f"{pre}_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(study["geometries"][name]["new_shift_mm"],float)
        head=GeomHeat.load(a.runs/f"{pre}_disjoint_anchor.npz")
        models={}
        arcs={}
        for label in ("medium","extra"):
            xz,y=GRIDS[label]
            data,_=create(bf,v150,"M4",**geo,
                          sea_void="oriented",physical_box_only=True,
                          grid_xz=xz,grid_y=y)
            models[label]=BFZIV148Adapter(
                data,bf,source_mask="subcell",fv_faces="exact",
                heat_mode="port-weak",sea_only=True)
            arcs[label]=np.load(a.runs/f"{pre}_subcell_subcell_{label}.npz")
        em1,em2=models["medium"],models["extra"]
        a1,a2=arcs["medium"],arcs["extra"]
        pts,w,ia,ib=shared(em1.base["records"],em2.base["records"])
        source=em1.source_at(pts)
        if not np.allclose(em1.source_at(pts[:2]),
                           em2.source_at(pts[:2]),rtol=1e-10,atol=1e-10):
            raise ValueError("same physical coils have different BFZI source fields")
        rows=[]
        for ix,t in enumerate(a1["field_times_s"]):
            j=int(np.argmin(abs(a1["time_s"]-t)))
            if not np.isclose(a1["time_s"][j],t):
                raise ValueError("missing field time in original BFZI archive")
            if not np.array_equal(a1["field_times_s"],a2["field_times_s"]):
                raise ValueError("BFZI two grids saved different snapshots")
            pred=[]
            for em,arc,owner in ((em1,a1,ia),(em2,a2,ib)):
                z=unpack(arc["z6"][j])
                I=np.linalg.solve(z+head.ext,
                                  np.array([arc["voltage_v"][j],0],complex))
                sigma=em.sigma0*(1+em.alpha*(arc["sea_T_K"][j]-em.ambient))
                pred.append(field(em,em.base["records"],source,pts,owner,
                                  arc["sea_J"][ix],sigma,I,arc["sea_q"][ix]))
            x,y=pred
            row=dict(
                time_s=float(t),voltage_V=float(a1["voltage_v"][j]),
                J_center_rel=rel(a1["sea_J"][ix][ia],a2["sea_J"][ix][ib],w)
                    if np.linalg.norm(a2["sea_J"][ix])>1e-20 else None,
                J_source_rel=rel(x["j"],y["j"],w)
                    if np.linalg.norm(y["j"])>1e-20 else None,
                q_center_rel=rel(a1["sea_q"][ix][ia],a2["sea_q"][ix][ib],w)
                    if np.linalg.norm(a2["sea_q"][ix])>1e-20 else None,
                q_source_rel=rel(x["q"],y["q"],w)
                    if np.linalg.norm(y["q"])>1e-20 else None,
                max_j_mean_rel=max(x["max_j_mean_rel"],y["max_j_mean_rel"]),
                max_weak_power_rel=max(x["power_rel"],y["power_rel"]))
            rows.append(row)
            print(name,"t",int(t),"J",row["J_center_rel"],
                  row["J_source_rel"],"q",row["q_center_rel"],
                  row["q_source_rel"],flush=True)
        output["geometries"][name]=dict(
            common_points=len(w),source_calc_sec=time.perf_counter()-stamp,
            rows=rows)
        a1.close();a2.close()
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
