"""Independent BFZI V150 J/q and rotated package-seawater interface validation.

Re-evaluates the original BFZI full EM fields at archived physical 3-D
temperatures. Original and frozen-cold-field surrogate receive exactly the
same physical port currents; this isolates spatial field approximation error.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import originals
from examples.bfzi_tangent_study import cold_em
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_interface import report


def rel_field(a,b,vol):
    d=np.asarray(a)-np.asarray(b)
    weight=np.asarray(vol).reshape(-1,*([1]*(d.ndim-1)))
    return float(np.sqrt(np.sum(weight*abs(d)**2)/
                        max(np.sum(weight*abs(b)**2),1e-30)))


def evaluate(head,em,teacher):
    rows=[]
    with np.load(teacher) as d:
        t=np.asarray(d["time_s"],float)
        vv=np.asarray(d["voltage_v"],float)
        temps=np.asarray(d["temp_K"],float)
        zs=np.asarray(d["z6"],float)
        original_i=np.asarray(d["currents_A"],complex)
        cases=json.loads(str(d["cases_json"]))
    picks=np.unique(np.linspace(0,len(t)-1,5).astype(int))
    for i in range(len(vv)):
        for j in picks:
            T=temps[i,j]
            theta=T-head.amb
            state=em.solve(T[:head.ns],T[head.cu])
            I=np.linalg.solve(state.total_z+head.ext,
                              np.array([vv[i,j],0],complex))
            jq=np.einsum("npc,p->nc",state.unit_j,I,optimize=True)
            q_ref,_=em.heat(state,I)
            _,q,jfield,ps,_=head.source(theta,I,state.sea_z)
            pi=float(.5*np.vdot(I,state.sea_z@I).real)
            im=float(np.linalg.norm(I-original_i[i,j])/
                     max(np.linalg.norm(original_i[i,j]),1e-12))
            pk=np.argmax(q_ref) if len(q_ref) else 0
            pm=np.argmax(q) if len(q) else 0
            rows.append(dict(
                case=int(cases[i]["id"]),time_s=float(t[j]),
                voltage_V=float(vv[i,j]),physical_current_rel=im,
                original_dynamic_port_rel=float(
                    np.linalg.norm(pack(state.total_z)-zs[i,j])/
                    max(np.linalg.norm(zs[i,j]),1e-12)),
                J_volume_relative_L2=rel_field(jfield,jq,head.vol),
                q_volume_relative_L2=rel_field(q,q_ref,head.vol),
                q_hotspot_distance_m=float(np.linalg.norm(
                    head.xyz[pk]-head.xyz[pm])),
                q_original_peak_W_m3=float(np.max(q_ref)),
                q_proxy_peak_W_m3=float(np.max(q)),
                original_q_port_gap_W=float(abs(q_ref@head.vol-pi)),
                proxy_q_port_gap_W=float(abs(q@head.vol-ps)),
                interface=report(head,theta)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_spatial_boundary_study.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    out={"source":"fresh original BFZI V150 warm spatial J/q at archived physical 3-D T",
         "test_geom":["P03","P07"],"time_samples_per_case":5,
         "heldout":{}}
    for name in ("P03","P07"):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        head=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        ts=time.perf_counter()
        em=cold_em(bf,v150,head,meta)
        result={"geometry_setup_sec":time.perf_counter()-ts,
                "interface_cold":report(head),"cases":{}}
        teacher=[pre.with_name(pre.name+"_holdout.npz")]
        if name=="P07":
            teacher.append(a.runs/"bfzi_pose_p07_15s_holdout.npz")
        for file in teacher:
            ts=time.perf_counter()
            rows=evaluate(head,em,file)
            result["cases"][file.name]=rows
            result.setdefault("field_EM_sec",{})[file.name]=(
                time.perf_counter()-ts)
            print(name,file.name,
                  "max_J_rel",max(x["J_volume_relative_L2"] for x in rows),
                  "max_q_rel",max(x["q_volume_relative_L2"] for x in rows),
                  "warm_port",max(x["original_dynamic_port_rel"] for x in rows),
                  "inner_wet_links",[z["inside_box_links"]
                     for z in result["interface_cold"]],flush=True)
        out["heldout"][name]=result
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
