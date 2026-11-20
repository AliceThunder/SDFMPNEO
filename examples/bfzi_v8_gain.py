"""Fit six global thermal-impedance tangent gains; retain the V8 physical circuit.

Train/validation geometries are disjoint; final core six-geometry histories
are only evaluated after checkpoint selection. No additional neural branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_cold_v7_parallel import build,collect,KEYS,mean
from examples.bfzi_cold_v7_train import read
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import rollout


def closed(rows,env,gain):
    cfg,_,_,ck,net,atlas,_,_,tan,_=env
    mul=torch.as_tensor(gain,dtype=torch.float32)
    def scaled(g,a):
        return tan(g,a)*mul
    out=[]
    with torch.no_grad():
        for r in rows:
            pred=rollout(r["case"],atlas,net,ck,r["row"]["time"],
                r["row"]["voltage"],scaled,init_override=r["init"],
                shape_mix=float(cfg["shape_mix"]))
            res=metric(pred,r["row"])
            out.append({k:float(res[k]) for k in KEYS})
    return mean(out)


def fit(rows,tan):
    s=np.zeros(6)
    t=np.zeros(6)
    for r in rows:
        h=torch.as_tensor(r["h"],dtype=torch.float32)
        g=torch.as_tensor(np.broadcast_to(r["g"],(len(h),45)).copy(),
                            dtype=torch.float32)
        with torch.no_grad():
            part=tan(g,h).numpy()
        x=part-part[:1]
        y=r["dr"]-r["dz"]+x
        den=max(float(np.linalg.norm(r["dr"])),1e-7)
        s+=np.sum(x*y,axis=0)/den**2
        t+=np.sum(x*x,axis=0)/den**2
    return s,t


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v8_gain.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    env=list(build())
    env[6]=ColdNet.load(a.cold)
    env=tuple(env)
    cfg,plan,hold,ck,net,atlas,cold,sf,tan,dom=env
    mf=read("runs/geomfrac_s5_v6_36g_hold6_manifest.json")
    tr,trg=collect(mf["train"],plan,env)
    va,vag=collect(mf["val"],plan,env)
    names_tr={n for n,_ in trg}
    names_va={n for n,_ in vag}
    if names_tr&names_va:
        raise ValueError("train/validation geometry overlap")
    s,t=fit(tr,tan)
    scores={}
    base=closed(va,env,np.ones(6))
    best=base["dynamic_z_rel_L2"]
    chosen=np.ones(6)
    for lam in (0.01,0.1,1.0,10.0,100.0):
        gain=np.clip((s+lam)/(t+lam),.6,1.4)
        val=closed(va,env,gain)
        scores[str(lam)]=dict(gain=gain.tolist(),metrics=val)
        print("V8_GAIN_DEV",lam,"dynamic",round(val["dynamic_z_rel_L2"],5),
              "sea",round(val["sea_power_rel_L2"],5),flush=True)
        if (val["dynamic_z_rel_L2"]<best-0.0001 and
            val["sea_power_rel_L2"]<=base["sea_power_rel_L2"]*1.05 and
            val["current_rel_L2"]<=base["current_rel_L2"]*1.05):
            chosen=gain
            best=val["dynamic_z_rel_L2"]
    tmf=read("runs/geomfrac_s5_cold_v7_core_manifest.json")
    te,teg=collect(tmf["test"],hold,env)
    if ({n for n,_ in teg}&(names_tr|names_va)) or len(te)!=12:
        raise RuntimeError("historical core geometries not disjoint")
    test=closed(te,env,chosen)
    result=dict(status="candidate" if np.max(abs(chosen-1))>1e-8 else "not_promoted",
        train_geometries=len(trg),val_geometries=len(vag),
        train_histories=len(tr),val_histories=len(va),
        baseline_dev=base,fit_numer=s.tolist(),fit_denom=t.tolist(),
        candidate_dev=scores,selected_gain=chosen.tolist(),historical_test=test)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V8_GAIN_FINAL",json.dumps({k:result[k] for k in
        ("status","selected_gain","baseline_dev","historical_test")}),flush=True)


if __name__=="__main__":
    main()
