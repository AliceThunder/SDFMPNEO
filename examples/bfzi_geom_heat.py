"""BFZI cold-geometry export and independent conservative neural 3-D rollout.

Full physical M3 is held out from neural training. Export the original
BFZI cold weak-form/J/thermal FV operator ONCE; every future time step is
solved from a frozen SOE fractional network and sparse thermal response.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def metrics(pred,true,ns,npkg,refz):
    d=abs(pred-true)
    z=refz
    return dict(
        sea_max_abs_K=float(d[:,:ns].max()),
        pkg_max_abs_K=float(d[:,ns:ns+2*npkg].max()),
        cu_max_abs_K=float(d[:,ns+2*npkg:].max()),
        sea_L2_relative=relative(pred[:,:ns]-293.15,true[:,:ns]-293.15),
        pkg_L2_relative=relative(pred[:,ns:ns+2*npkg]-293.15,
                               true[:,ns:ns+2*npkg]-293.15))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,default=None)
    p.add_argument("--sample",choices=("M1","M2","M3","M4"),default="M3")
    p.add_argument("--anchor",type=Path,
                   default=Path("runs/m3_bfzi_anchor.npz"))
    p.add_argument("--teacher",type=Path,
                   default=Path("runs/m3_frac_holdout.npz"))
    p.add_argument("--model",type=Path,
                   default=Path("runs/bfzi_geom_frac_v5.pt"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/m3_bfzi_heat_metrics.json"))
    p.add_argument("--bench-direct",action="store_true")
    p.add_argument("--field-check",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(2)
    start=time.perf_counter()
    em=None;et=None
    if args.bfzi_root:
        sys.path.insert(0,str(args.bfzi_root.resolve()))
        import run_v150 as original
        import bfzi_v148_results as calc
        from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
        from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
        data=original.build_data(args.sample)
        em=BFZIV148Adapter(data,calc,heat_mode="port-weak")
        ex=np.diag([.5-1j*em.omega*em.air[0,0],
                    .5-1j*em.omega*em.air[1,1]])
        et=BFZIV148Pkg3D(em,external=ex,pkg_shape=(4,2,4),pkg_k=.25)
        head=GeomHeat.from_original(em,et,ex)
        args.anchor.parent.mkdir(parents=True,exist_ok=True)
        head.save(args.anchor)
    else:
        head=GeomHeat.load(args.anchor)
    anchor_seconds=time.perf_counter()-start
    ck=torch.load(args.model,map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"])
    net.eval()
    with np.load(args.teacher) as d:
        t=np.asarray(d["time_s"],float)
        v=np.asarray(d["voltage_v"],float)
        true=np.asarray(d["temp_K"],float)
        ztrue=np.asarray(d["z6"],float)
        refI=np.asarray(d["currents_A"],complex)
        case_meta=json.loads(str(d["cases_json"]))
        g=np.asarray(d["geometry"],float)
        z0=np.asarray(d["z6"][0,0],float)
    if not np.allclose(g,head.geom,rtol=1e-9,atol=1e-10):
        raise ValueError("original BFZI holdout geometry differs from cached cold anchor")
    cold_err=relative(head.z0,z0)
    if cold_err>1e-8:
        raise ValueError(f"original BFZI cold complex port mismatch: {cold_err}")
    results=[];state=[];zpred=[];qpred=[];jpred=[]
    q_truth=[];j_truth=[]
    for i in range(len(v)):
        t0=time.perf_counter()
        pr=head.rollout(net,ck,v[i],t)
        sec=time.perf_counter()-t0
        one=metrics(pr["temp_K"],true[i],head.ns,head.npkg,ztrue[i])
        one.update(
            case=int(case_meta[i]["id"]),
            voltage_V=float(case_meta[i]["voltage_v"]),
            neural_trajectory_sec=sec,
            dynamic_Z_relative_L2=relative(pr["z6"]-z0,ztrue[i]-z0),
            max_FV_energy_residual_W=float(pr["fv_node_residual_W"].max()),
            max_sea_source_gap_W=float(np.max(abs(
                pr["qsea_W_m3"]@head.vol-pr["Psea_W"]))))
        if args.bench_direct:
            if et is None:
                raise ValueError("--bench-direct requires --bfzi-root")
            def drive(tm):
                idx=int(np.argmin(abs(t-float(tm))))
                return np.array([v[i,idx],0.],complex)
            t1=time.perf_counter()
            et.trajectory(t,drive)
            original_sec=time.perf_counter()-t1
            one["direct_trajectory_sec"]=original_sec
            one["speedup"]=original_sec/sec
        if args.field_check:
            if em is None:
                raise ValueError("--field-check requires --bfzi-root")
            fields=[]
            for k in (0,5,10,15,20):
                truth=em.solve(true[i,k,:head.ns],
                               true[i,k,head.cu])
                I=refI[i,k]
                tq,_=em.heat(truth,I)
                tj=np.einsum("npc,p->nc",truth.unit_j,I,optimize=True)
                _,aq,aj,_,_=head.source(true[i,k]-head.amb,I,truth.sea_z)
                fields.append(dict(time_s=float(t[k]),
                    cold_anchor_q_relative_L2=relative(aq,tq),
                    cold_anchor_J_relative_L2=relative(aj,tj)))
            one["original_BFZI_field_checks"]=fields
        results.append(one)
        state.append(pr["temp_K"])
        zpred.append(pr["z6"])
        qpred.append(pr["qsea_W_m3"])
        jpred.append(pr["Jsea_A_m2"])
        print(json.dumps(one,indent=2),flush=True)
    data=dict(sample=args.sample,teacher=str(args.teacher),
              model=str(args.model),anchor=str(args.anchor),
              anchor_build_seconds=anchor_seconds,
              cold_Z_relative_error=cold_err,
              sea_cells=head.ns,package_cells_per_box=head.npkg,
              physical_source="original BFZI cold unit J and port weak Joule",
              history="learned causal fractional SOE sea delta",
              heat="original 3-D BFZI FV conservative heat operator",
              no_dynamic_BFZI_EM_solve=True,
              results=results)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2),encoding="utf-8")
    np.savez_compressed(args.out.with_suffix(".npz"),time_s=t,
        temp_true_K=true,temp_pred_K=np.asarray(state),
        z_true=ztrue,z_pred=np.asarray(zpred),
        qsea_pred_W_m3=np.asarray(qpred),
        Jsea_pred_A_m2=np.asarray(jpred),
        sea_xyz_m=head.xyz,pkg_xyz_m=head.pkg_xyz,sea_vol_m3=head.vol)
    print("SAVED",args.out,flush=True)


if __name__=="__main__":
    main()
