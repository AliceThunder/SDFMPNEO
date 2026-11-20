"""Train a compact full-state parallel impedance residual for V8."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_cold_v7_parallel import collect, closed, mean
from examples.bfzi_cold_v7_train import read
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import geometry
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def pack(rows):
    gs=[]; aa=[]; tt=[]; vv=[]; yy=[]
    for r in rows:
        n=len(r["h"])
        gs.append(np.broadcast_to(r["g"],(n,45)).copy())
        aa.append(np.asarray(r["h"],float))
        tt.append(np.asarray(r["row"]["time"],float).reshape(-1,1))
        vv.append(np.asarray(r["row"]["voltage"],float).reshape(-1,1))
        yy.append(np.asarray(r["target"],float))
    return tuple(torch.as_tensor(np.concatenate(x),dtype=torch.float32)
                 for x in (gs,aa,tt,vv,yy))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("runs/geomfrac_free_rank48_v8_selection.json"))
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_full_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8_full_manifest.json"))
    p.add_argument("--model",type=Path,default=Path("runs/geomfrac_et_v8_heat_tune.pt"))
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--skin",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v3.pt"))
    p.add_argument("--skin-base",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v8_par48.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v8_par48.json"))
    p.add_argument("--epochs",type=int,default=200)
    p.add_argument("--patience",type=int,default=50)
    p.add_argument("--every",type=int,default=10)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--hidden",type=int,default=32)
    a=p.parse_args()

    torch.manual_seed(72311)
    np.random.seed(72311)
    torch.set_num_threads(2)
    cfg=read(a.config)
    plan=read(a.plan)
    mf=read(a.manifest)
    ck=torch.load(a.model,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=48,
        soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    net.eval()
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load(a.cold)
    sf=SrcFix.load(cfg["source_model"])
    tan=restore_rank48(a.skin,a.skin_base)
    env=(cfg,plan,{},ck,net,atlas,cold,sf,tan,None)

    tr,tg=collect(mf["train"],plan,env,gate_only=False)
    va,vg=collect(mf["val"],plan,env,gate_only=False)
    names_tr={n for n,_ in tg}; names_va={n for n,_ in vg}
    if len(tg)!=48 or len(vg)!=15 or names_tr&names_va:
        raise RuntimeError(f"expected disjoint 48/15 geometries, got {len(tg)}/{len(vg)}")

    gt,at,tt,vt,yt=pack(tr)
    gm=gt.mean(0); gs=gt.std(0).clamp_min(1e-3)
    am=at.mean(0); ass=at.std(0).clamp_min(.05)
    ys=yt.std(0).clamp_min(1e-6)
    model=ParZ48(gm,gs,am,ass,ys,hidden=a.hidden)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=5e-4)
    base=mean(closed(None,va,env))
    best=base["dynamic_z_rel_L2"]
    ep_best=0
    state=copy.deepcopy(model.state_dict())
    hist=[]
    print("PAR48_INIT",len(tg),len(vg),len(tr),len(va),
          json.dumps(base),flush=True)

    for ep in range(1,a.epochs+1):
        model.train()
        for ix in torch.randperm(len(gt)).split(a.batch):
            pred=model(gt[ix],at[ix],tt[ix],vt[ix])
            loss=((pred-yt[ix])/ys).square().mean()
            loss+=.01*(pred/ys).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        if ep%a.every:
            continue
        model.eval()
        val=mean(closed(model,va,env))
        hist.append(dict(epoch=ep,train_loss=float(loss.detach()),**val))
        good=(val["dynamic_z_rel_L2"]<best-.0002 and
              val["current_rel_L2"]<=base["current_rel_L2"]*1.05 and
              val["sea_power_rel_L2"]<=base["sea_power_rel_L2"]*1.1)
        if good:
            best=val["dynamic_z_rel_L2"]
            ep_best=ep
            state=copy.deepcopy(model.state_dict())
        print("PAR48_TRAIN",ep,"loss",round(float(loss.detach()),6),
              "dynamic",round(val["dynamic_z_rel_L2"],6),
              "best",round(best,6),"epoch",ep_best,flush=True)
        if ep-ep_best>=a.patience:
            break

    model.load_state_dict(state)
    model.eval()
    strengths={}
    for strength in (0.,.25,.5,.75,1.):
        model.strength=strength
        strengths[str(strength)]=mean(closed(model,va,env))
        print("PAR48_STRENGTH",strength,
              strengths[str(strength)]["dynamic_z_rel_L2"],flush=True)
    chosen=min(strengths,key=lambda k:strengths[k]["dynamic_z_rel_L2"])
    model.strength=float(chosen)
    selected=strengths[chosen]
    promoted=(model.strength>0. and
        selected["dynamic_z_rel_L2"]<base["dynamic_z_rel_L2"]-.0005 and
        selected["current_rel_L2"]<=base["current_rel_L2"]*1.05 and
        selected["sea_power_rel_L2"]<=base["sea_power_rel_L2"]*1.1)
    if promoted:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        model.save(a.out)
    report=dict(status="candidate" if promoted else "not_promoted",
        train_geometries=len(tg),val_geometries=len(vg),
        train_histories=len(tr),val_histories=len(va),
        best_epoch=ep_best,epochs_trained=ep,
        baseline_val=base,best_full_strength_dynamic_z=best,
        selected_strength=model.strength,closed_val_by_strength=strengths,
        checkpoint=str(a.out) if promoted else None,
        note="Frozen V8 heat/SOE, GeomInit, ColdNet, SrcFix and SkinTan V3; parallel head sees all 48 heat states, 45 geometry features, time and voltage.")
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("PAR48_RESULT",json.dumps({k:report[k] for k in
        ("status","best_epoch","selected_strength","baseline_val","closed_val_by_strength")}),flush=True)


if __name__=="__main__":
    main()
