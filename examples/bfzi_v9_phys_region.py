"""Calibrate seawater spatial susceptibility from complete BFZI physical histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_canon import labels
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_weakz import WeakSea


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def parts(row, init, alpha):
    a=np.asarray(row["a"],float)
    cap=np.asarray(init["cap"],float)
    src=np.asarray(init["src"],float)
    if a.shape[1]!=48 or cap.shape!=(5,) or src.shape!=(48,3):
        raise ValueError("expected rank48 physics inputs")
    x=a*np.sqrt(cap[labels(48)])[None]
    q=np.stack((x[:,:1]@src[:1],
                x[:,1:32]@src[1:32],
                x[:,32:48]@src[32:48]),axis=1)
    return float(alpha)*q


def gain_fit(rows, lam):
    A=np.zeros((3,3,3))
    b=np.zeros((3,3))
    for item in rows:
        p=parts(item["row"],item["init"],item["case"]["alpha_per_K"])[1:]
        y=np.asarray(item["row"]["y"],float)[1:,[0,2,4]]
        for k in range(3):
            x=p[:,:,k];target=y[:,k]
            den=max(float(np.linalg.norm(target)),1.e-6)
            scale=1./(den*den)
            A[k]+=scale*(x.T@x)
            b[k]+=scale*(x.T@target)
    gain=np.zeros((3,3))
    for k in range(3):
        d=np.maximum(np.diag(A[k]),1e-6*np.trace(A[k]))
        h=A[k]+lam*np.diag(d)
        coef=np.linalg.solve(h,b[k]+lam*d)
        gain[:,k]=np.clip(coef,.5,2.)
    return gain


def oracle(rows,gain):
    accum={"raw":np.zeros(3),"cal":np.zeros(3),"n":0}
    for item in rows:
        p=parts(item["row"],item["init"],item["case"]["alpha_per_K"])[1:]
        y=np.asarray(item["row"]["y"],float)[1:,[0,2,4]]
        raw=p.sum(axis=1)
        cal=(p*gain[None]).sum(axis=1)
        den=np.maximum(np.linalg.norm(y,axis=0),1e-9)
        accum["raw"]+=np.linalg.norm(raw-y,axis=0)/den
        accum["cal"]+=np.linalg.norm(cal-y,axis=0)/den
        accum["n"]+=1
    return {s:{k:float(v/accum["n"]) for k,v in
            zip(("R11","R12","R22"),accum[s])} for s in ("raw","cal")}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8_full_manifest.json"))
    cli.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_full_plan.json"))
    cli.add_argument("--heat-model",type=Path,default=Path("runs/geomfrac_et_v8_heat_tune.pt"))
    cli.add_argument("--ridge",type=float,default=.1)
    cli.add_argument("--out",type=Path,default=Path("runs/geomfrac_v9_phys_region.json"))
    a=cli.parse_args()
    torch.set_num_threads(2)
    plan=read(a.plan);mf=read(a.manifest)
    cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
    src=SrcFix.load(cfg["source_model"])
    tan=restore_rank48("runs/geomfrac_rank48_skin_tan_v3.pt",
                       "runs/geomfrac_et_v9_ztune.pt")
    par=ParZ48.load("runs/geomfrac_v8_par48.pt")
    ck=torch.load(a.heat_model,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=48,soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"]);net.eval()
    def collect(split):
        out=[];names=set()
        for spec in mf[split]:
            name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
            if name in names:raise ValueError("duplicate geometry")
            names.add(name)
            case=plan["cases"][name]
            init=thermal_mix(atlas,src.apply(case,cold.apply(
                case,atlas.predict(case))),.25)
            for row in prepare(spec,48):
                out.append(dict(name=name,case=case,init=init,row=row))
        return out,names
    tr,tn=collect("train")
    va,vn=collect("val")
    if tn & vn or len(tn)!=48 or len(vn)!=15 or len(tr)!=132 or len(va)!=41:
        raise ValueError("expected frozen 48/15, 132/41 geometry-disjoint histories")

    gain=gain_fit(tr,a.ridge)
    or_train=oracle(tr,gain);or_val=oracle(va,gain)
    print("REGION_GAINS",gain.tolist(),flush=True)
    print("REGION_ORACLE",json.dumps(or_val),flush=True)

    def validate(beta):
        vals=[]
        for item in va:
            ex=WeakSea(tan,par,item["init"],item["case"]["alpha_per_K"],
                       mix=beta,gain=gain)
            with torch.no_grad():
                r=item["row"]
                pred=rollout(item["case"],atlas,net,ck,r["time"],r["voltage"],
                    tan,init_override=item["init"],extra=ex,
                    shape_mix=float(cfg["shape_mix"]))
            vals.append(metric(pred,r))
        keys=("dynamic_z_rel_L2","R11_dynamic_rel_L2_self",
              "R22_dynamic_rel_L2_self","X11_dynamic_rel_L2_self",
              "X22_dynamic_rel_L2_self","material_mean_rmse_K",
              "current_rel_L2","sea_power_rel_L2","copper_power_rel_L2")
        return {k:float(np.mean([m[k] for m in vals])) for k in keys}

    results={}
    for beta in (0.,.25,.5,.75,1.):
        scores=validate(beta)
        results[str(beta)]=scores
        print("REGION_DEV",beta,json.dumps(scores),flush=True)
    baseline=results["0.0"]
    choice=None;best=baseline
    for key,r in results.items():
        if (r["dynamic_z_rel_L2"]<best["dynamic_z_rel_L2"]-.001
            and r["R11_dynamic_rel_L2_self"]<=1.02*baseline["R11_dynamic_rel_L2_self"]
            and r["R22_dynamic_rel_L2_self"]<=1.02*baseline["R22_dynamic_rel_L2_self"]
            and r["material_mean_rmse_K"]<=1.03*baseline["material_mean_rmse_K"]
            and r["current_rel_L2"]<=1.03*baseline["current_rel_L2"]
            and r["sea_power_rel_L2"]<=1.03*baseline["sea_power_rel_L2"]):
            best=r;choice=float(key)
    out=dict(status="development_candidate" if choice is not None else "not_promoted",
        method="shared three-region BFZI cold Joule susceptibility; nine fitted gains",
        train_geometries=48,train_histories=132,
        val_geometries=15,val_histories=41,
        region_modes={"mean":[0],"low_order":[1,31],"skin":[32,47]},
        ridge=a.ridge,gain=gain.tolist(),
        oracle_train=or_train,oracle_val=or_val,
        baseline=baseline,selected_mix=choice,best=best,trials=results,
        blind_not_used=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("REGION_RESULT",json.dumps({k:out[k] for k in
        ("status","selected_mix","baseline","best")}),flush=True)


if __name__=="__main__":
    main()
