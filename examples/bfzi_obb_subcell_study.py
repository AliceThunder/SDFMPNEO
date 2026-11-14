"""P03/P07 600 s physical BFZI skin + subcell study.

Independently rebuild original BFZI V150 Green/GMRES for each wet FV
grid and source integration mode. Save full temperatures and impedance
history, selected native 3-D J/q fields and power-conservation metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_refine import load
from examples.bfzi_obb_layer_study import GRIDS, compare, view
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat, pack
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D


def measure(et, t, v):
    start=time.perf_counter()
    def drive(now):
        j=int(np.argmin(abs(t-now)))
        return np.array([v[j],0.],complex)
    hist=et.trajectory(t,drive)
    elapsed=time.perf_counter()-start
    state=np.array([h.temp for h in hist])
    temp=et.amb+state
    z=np.array([pack(h.impedance) for h in hist])
    sea=et.amb+np.array([et.sea_temp(h.temp) for h in hist])
    qpower=np.array([h.sea_power for h in hist])
    cupower=np.array([h.copper_power for h in hist])
    port=np.array([.5*np.vdot(h.currents,h.sea_z@h.currents).real
                   for h in hist])
    pick=np.unique([int(np.argmin(abs(t-s))) for s in (0.,60.,300.,600.)])
    js=np.array([hist[i].sea_j for i in pick])
    qs=np.array([hist[i].sea_q for i in pick])
    node=0.
    total=0.
    for k in range(1,len(hist)):
        dt=float(t[k]-t[k-1])
        heat=et.heat_vec(hist[k].sea_q,hist[k].copper_power)
        res=(et.cap*(state[k]-state[k-1])/dt+et.H@state[k]-heat)
        node=max(node,float(np.max(abs(res))))
        total=max(total,float(abs(np.sum(res))))
    mapped=0
    count=0
    if et._qmap:
        for ids in et._qmap.values():
            mapped+=int(np.count_nonzero(ids>=et.skin.start))
            count+=len(ids)
    data=dict(
        temp_K=temp,z6=z,sea_T_K=sea,
        sea_q=qs,sea_J=js,field_times_s=t[pick],
        sea_xyz=et.model.points,sea_vol=et.model.vol,
        Psea_W=qpower,Pcu_W=cupower,
        max_node_budget_W=node,max_total_budget_W=total,
        max_port_weak_diff_W=float(np.max(abs(port-qpower))),
        max_port_weak_rel=float(np.max(abs(port-qpower)/
                                    np.maximum(abs(port),1e-12))),
        q_skin_samples=mapped,q_samples=count,
        direct_sec=elapsed,et=et)
    return data


def field_diff(a,b):
    """Volume-weighted cross-grid differences at coarse native FV centers."""
    xyz=a["sea_xyz"]
    d,idx=cKDTree(b["sea_xyz"]).query(xyz,k=4)
    w=1/np.maximum(d,1e-9)**2
    w/=w.sum(axis=1)[:,None]
    vol=a["sea_vol"]
    out={}
    for attr in ("sea_J","sea_q"):
        aa=a[attr]
        bb=b[attr]
        row=[]
        for x,y in zip(aa,bb):
            interp=np.sum(w[(...,)+(None,)*(y.ndim-1)]*y[idx],axis=1)
            diff=x-interp
            if diff.ndim>1:
                num=float(np.sum(vol[:,None]*abs(diff)**2))
                den=float(np.sum(vol[:,None]*abs(interp)**2))
            else:
                num=float(vol@(abs(diff)**2))
                den=float(vol@(abs(interp)**2))
            row.append(float(np.sqrt(num/max(den,1e-30))))
        out[attr+"_cross_grid_rel_L2"]=row
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),
                   default=("P03","P07"))
    p.add_argument("--levels",nargs="+",choices=tuple(GRIDS),
                   default=("medium","extra"))
    p.add_argument("--source-masks",nargs="+",
                   choices=("legacy","all","subcell"),
                   default=("legacy","all","subcell"))
    p.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_obb_subcell_study.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    old=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    out=dict(method="original BFZI V150 600s skin thermal quadrature",
             source_masks=a.source_masks,
             skin_depth_m=.002,skin_layers=8,skin_map=2,
             wet_faces="exact",gap_bridge=False,geometries={})
    for name in a.poses:
        stem=f"bfzi_pose_{name.lower()}"
        meta=json.loads((a.runs/f"{stem}_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(
            old["geometries"][name]["new_shift_mm"],float)
        prev=GeomHeat.load(a.runs/f"{stem}_disjoint_anchor.npz")
        prog=load(a.runs/f"{stem}_disjoint_oriented_medium.npz")
        t,v=prog["time_s"],prog["voltage_v"]
        run={}
        group=dict(modes={})
        for mask in a.source_masks:
            g=dict(levels={})
            for label in a.levels:
                xz,y=GRIDS[label]
                start=time.perf_counter()
                data,_=create(bf,v150,"M4",**geo,
                              sea_void="oriented",physical_box_only=True,
                              grid_xz=xz,grid_y=y)
                em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                                   fv_faces="exact",source_mask=mask)
                et=BFZIV148Skin3D(
                    em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
                    skin_depth=.002,skin_layers=8,skin_ratio=1.4,
                    skin_quad=2,skin_near=4,skin_map=2,pkg_sea_h=20.)
                setup=time.perf_counter()-start
                item=measure(et,t,v)
                run[mask,label]=item
                f=a.runs/f"{stem}_subcell_{mask}_{label}.npz"
                np.savez_compressed(
                    f,time_s=t,voltage_v=v,
                    temp_K=item["temp_K"],z6=item["z6"],
                    sea_T_K=item["sea_T_K"],
                    sea_q=item["sea_q"],sea_J=item["sea_J"],
                    field_times_s=item["field_times_s"],
                    sea_xyz=item["sea_xyz"],sea_vol=item["sea_vol"],
                    Psea_W=item["Psea_W"],Pcu_W=item["Pcu_W"],
                    nsea=et.count,n_pkg=et.pkg_n)
                pkg,cu=view(item["temp_K"],et)
                g["levels"][label]=dict(
                    sea_cells=et.count,skin_states=et.skin.stop-et.skin.start,
                    physical_file=str(f),setup_sec=setup,
                    direct_sec=item["direct_sec"],
                    quadrature_skin_samples=item["q_skin_samples"],
                    quadrature_total_samples=item["q_samples"],
                    max_node_budget_W=item["max_node_budget_W"],
                    max_total_budget_W=item["max_total_budget_W"],
                    max_port_weak_diff_W=item["max_port_weak_diff_W"],
                    max_port_weak_rel=item["max_port_weak_rel"],
                    max_sea_T_K=float(np.max(item["sea_T_K"])),
                    max_pkg_T_K=float(np.max(pkg)),
                    max_copper_T_K=float(np.max(cu)),
                    sea_power_peak_W=float(np.max(item["Psea_W"])),
                    copper_power_peak_W=float(np.max(item["Pcu_W"])),
                    thermal_capacity_residual_J_K=et.skin_info[
                        "sea_cap_delta_J_K"])
                print(name,mask,label,"wet",et.count,
                      "qskin",item["q_skin_samples"],"energy",
                      round(item["max_node_budget_W"],8),
                      "direct_s",round(item["direct_sec"],2),flush=True)
            if all(k in run for k in ((mask,"medium"),(mask,"extra"))):
                m=run[mask,"medium"]
                e=run[mask,"extra"]
                g["cross_grid"]=dict(
                    **compare(m,e),**field_diff(m,e),
                    max_sea_mean_abs_K=float(np.max(abs(
                        (m["sea_T_K"]@m["sea_vol"])/m["sea_vol"].sum()-
                        (e["sea_T_K"]@e["sea_vol"])/e["sea_vol"].sum()))))
            group["modes"][mask]=g
            out["geometries"][name]=group
            a.out.parent.mkdir(parents=True,exist_ok=True)
            a.out.write_text(json.dumps(out,indent=2))
        for mask in a.source_masks:
            if mask=="legacy":
                continue
            group["modes"][mask]["vs_legacy"]={}
            for label in a.levels:
                x=run[mask,label]["z6"]
                b=run["legacy",label]["z6"] if ("legacy",label) in run else None
                if b is None:
                    continue
                group["modes"][mask]["vs_legacy"][label]=dict(
                    dynamic_delta_Z_relative_L2=float(
                        np.linalg.norm((x-x[0])-(b-b[0]))/
                        max(np.linalg.norm(b-b[0]),1e-30)))
        a.out.write_text(json.dumps(out,indent=2))
        print("CROSS",name,json.dumps({
            key:group["modes"][key].get("cross_grid",{})
            for key in group["modes"]}),flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
