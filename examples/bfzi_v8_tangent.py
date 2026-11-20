"""Fine-tune frozen V7 rank48 SkinTan on V8 autonomous heat trajectories.

Only impedance tangent weights are trained. ColdNet, geometry atlas and
fractional thermal dynamics stay frozen. Geometry-disjoint train/dev sets
select the checkpoint; six V7-core histories remain a final diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_cold_v7_parallel import build,collect,KEYS,mean
from examples.bfzi_cold_v7_train import read
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_skintan import SkinTanV3,restore_rank48


def load_skin(path):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if ck.get("model_version")!="v3" or int(ck["rank"])!=48:
        raise ValueError("V8 tangent expects SkinTanV3 rank48")
    model=SkinTanV3(ng=int(ck["ng"]),sa=ck["sa"],ss=ck["ss"])
    model.load_state_dict(ck["state"])
    return model,ck


def curves(model,ck,env):
    cfg,_,_,_,_,_,_,_,_,_=env
    from sdfmpneo_bfzi.geom_ztune import restore
    base=restore(cfg["z_fix_model"])
    sy=torch.as_tensor(ck["sy"][:6],dtype=torch.float32)
    def calc(g,a):
        return base(g,a[:,:32])+model(g,a[:,:32],a[:,32:])*sy
    return calc


@torch.no_grad()
def score(model,ck,rows,env):
    cfg,_,_,net_ck,net,atlas,_,_,_,_=env
    tan=curves(model,ck,env)
    out=[]
    model.eval()
    for r in rows:
        pred=rollout(r["case"],atlas,net,net_ck,
            r["row"]["time"],r["row"]["voltage"],tan,
            init_override=r["init"],shape_mix=float(cfg["shape_mix"]))
        item=metric(pred,r["row"])
        out.append({k:float(item[k]) for k in KEYS})
    return mean(out)


def fit(args):
    torch.manual_seed(72117)
    torch.set_num_threads(2)
    env=list(build())
    env[6]=ColdNet.load(args.cold)
    env=tuple(env)
    cfg,plan,hold,_,_,_,_,_,_,_=env
    mf=read("runs/geomfrac_s5_v6_36g_hold6_manifest.json")
    if a.expand_plan and a.expand_manifest:
        ex=read(a.expand_plan)
        em=read(a.expand_manifest)
        plan["cases"].update(ex["cases"])
        for part in ("train","val"):
            mf[part].extend(em.get(part,[]))
        print("V8_EXPAND_USED",len(em.get("train",[])),
              len(em.get("val",[])),flush=True)
    tr,gt=collect(mf["train"],plan,env)
    va,gv=collect(mf["val"],plan,env)
    names_tr={n for n,_ in gt}
    names_va={n for n,_ in gv}
    if len(names_tr)<10 or len(names_va)<3 or names_tr&names_va:
        raise RuntimeError("insufficient geometry-disjoint development histories")
    model,ck=load_skin(args.skin)
    sy=torch.as_tensor(ck["sy"][:6],dtype=torch.float32)
    g=torch.cat([torch.as_tensor(np.broadcast_to(r["g"],(len(r["h"]),45)).copy(),
                                  dtype=torch.float32) for r in tr])
    h=torch.cat([torch.as_tensor(r["h"].copy(),dtype=torch.float32) for r in tr])
    target=torch.cat([torch.as_tensor(r["target"],dtype=torch.float32) for r in tr])
    with torch.no_grad():
        old=model(g,h[:,:32],h[:,32:]).detach()*sy
    den=torch.cat([torch.full((len(r["h"]),1),
                 max(float(np.linalg.norm(r["dr"])),1e-6)) for r in tr])
    baseline=score(model,ck,va,env)
    best=baseline["dynamic_z_rel_L2"]
    ep_best=0
    state=copy.deepcopy(model.state_dict())
    hist=[dict(epoch=0,**baseline)]
    print("V8_TAN_BASE",json.dumps(baseline),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    for ep in range(1,args.epochs+1):
        model.train()
        corr=(model(g,h[:,:32],h[:,32:])*sy-old)
        loss=((corr-target)/den).square().sum(-1).mean()
        loss+=.03*(corr/den).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if ep%args.every:continue
        val=score(model,ck,va,env)
        hist.append(dict(epoch=ep,**val))
        good=(val["dynamic_z_rel_L2"]<best-0.0002 and
              val["sea_power_rel_L2"]<=baseline["sea_power_rel_L2"]*1.1 and
              val["current_rel_L2"]<=baseline["current_rel_L2"]*1.05)
        if good:
            best=val["dynamic_z_rel_L2"]
            ep_best=ep
            state=copy.deepcopy(model.state_dict())
        print("V8_TAN_DEV",ep,"loss",round(float(loss.detach()),6),
              "dynamic",round(val["dynamic_z_rel_L2"],5),
              "best",round(best,5),"epoch",ep_best,flush=True)
        if ep-ep_best>=args.patience:break
    model.load_state_dict(state)
    model.eval()
    selected=score(model,ck,va,env)
    status="candidate" if ep_best else "not_promoted"
    if ep_best:
        ck["state"]=model.state_dict()
        ck["best_epoch"]=ep_best
        ck["source_skin"]=str(args.skin)
        ck["cold_model"]=str(args.cold)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(ck,args.out)
    # Historical holdout is run only after development checkpoint selection.
    hold_mf=read("runs/geomfrac_s5_cold_v7_core_manifest.json")
    te,ge=collect(hold_mf["test"],hold,env)
    if (len(ge)!=6 or len(te)!=12 or
            {n for n,_ in ge}&(names_tr|names_va)):
        raise RuntimeError("historical core holdout overlaps with development")
    test=score(model,ck,te,env)
    report=dict(status=status,cold=str(args.cold),skin=str(args.skin),
        checkpoint=str(args.out) if ep_best else None,
        train_geometries=len(gt),val_geometries=len(gv),
        train_histories=len(tr),val_histories=len(va),
        historical_test_geometries=len(ge),historical_test_histories=len(te),
        best_epoch=ep_best,epochs_trained=ep,
        baseline_val=baseline,selected_val=selected,
        historical_test=test,history=hist,
        note="Six V7 core geometries only for post-selection historical comparison.")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V8_TAN_RESULT",json.dumps({k:report[k] for k in
        ("status","best_epoch","baseline_val","selected_val","historical_test")}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v8_split.pt"))
    p.add_argument("--skin",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v3.pt"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v8.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v8_tangent.json"))
    p.add_argument("--epochs",type=int,default=160)
    p.add_argument("--every",type=int,default=10)
    p.add_argument("--patience",type=int,default=50)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--expand-plan",type=Path,default=None)
    p.add_argument("--expand-manifest",type=Path,default=None)
    a=p.parse_args()
    fit(a)


if __name__=="__main__":
    main()
