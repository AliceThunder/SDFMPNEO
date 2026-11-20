"""Fine-tune the rank48 impedance tangent on the complete V8 expanded geometry split."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_cold_v7_parallel import collect,KEYS,mean
from examples.bfzi_cold_v7_train import read
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_skintan import SkinTanV3,restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def load_skin(path):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if ck.get("model_version")!="v3" or int(ck["rank"])!=48:
        raise ValueError("V8 tangent expects SkinTanV3 rank48")
    model=SkinTanV3(ng=int(ck["ng"]),sa=ck["sa"],ss=ck["ss"])
    model.load_state_dict(ck["state"])
    return model,ck


def curves(model,ck,env):
    cfg=env[0]
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


def fit(a):
    torch.manual_seed(72117)
    np.random.seed(72117)
    torch.set_num_threads(2)
    cfg=read(a.config)
    plan=read(a.plan)
    plan["cases"].update(read(a.expand_plan)["cases"])
    mf=read(a.manifest)
    ex=read(a.expand_manifest)
    for key in ("train","val"):
        mf[key].extend(ex.get(key,[]))

    net_ck=torch.load(a.model,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(net_ck["ng"]),rank=48,
        soe_quad=net_ck.get("soe_quad","trap"),
        heat_dt=float(net_ck.get("heat_dt",1.875)),
        heat_z=bool(net_ck.get("heat_z",False)))
    net.load_state_dict(net_ck["state"])
    net.eval()
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load(a.cold)
    sf=SrcFix.load(cfg["source_model"])
    base_tan=restore_rank48(a.skin,a.skin_base)
    env=(cfg,plan,{},net_ck,net,atlas,cold,sf,base_tan,None)

    tr,gt=collect(mf["train"],plan,env,gate_only=False)
    va,gv=collect(mf["val"],plan,env,gate_only=False)
    names_tr={n for n,_ in gt}
    names_va={n for n,_ in gv}
    if len(gt)!=48 or len(gv)!=15 or names_tr&names_va:
        raise RuntimeError(f"expected disjoint 48/15 geometries, got {len(gt)}/{len(gv)}")

    model,ck=load_skin(a.skin)
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
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-3)
    for ep in range(1,a.epochs+1):
        model.train()
        corr=(model(g,h[:,:32],h[:,32:])*sy-old)
        loss=((corr-target)/den).square().sum(-1).mean()
        loss+=.03*(corr/den).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if ep%a.every:
            continue
        val=score(model,ck,va,env)
        hist.append(dict(epoch=ep,**val))
        good=(val["dynamic_z_rel_L2"]<best-0.0002 and
              val["sea_power_rel_L2"]<=baseline["sea_power_rel_L2"]*1.1 and
              val["current_rel_L2"]<=baseline["current_rel_L2"]*1.05)
        if good:
            best=val["dynamic_z_rel_L2"]
            ep_best=ep
            state=copy.deepcopy(model.state_dict())
        print("V8_TAN_TRAIN",ep,"loss",round(float(loss.detach()),6),
              "dynamic",round(val["dynamic_z_rel_L2"],6),
              "best",round(best,6),"epoch",ep_best,flush=True)
        if ep-ep_best>=a.patience:
            break

    model.load_state_dict(state)
    model.eval()
    selected=score(model,ck,va,env)
    status="candidate" if ep_best else "not_promoted"
    if ep_best:
        ck["state"]=model.state_dict()
        ck["best_epoch"]=ep_best
        ck["source_skin"]=str(a.skin)
        ck["cold_model"]=str(a.cold)
        ck["atlas"]=cfg["atlas"]
        ck["source_model"]=cfg["source_model"]
        a.out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(ck,a.out)
    report=dict(status=status,cold=str(a.cold),skin=str(a.skin),
        checkpoint=str(a.out) if ep_best else None,
        train_geometries=len(gt),val_geometries=len(gv),
        train_histories=len(tr),val_histories=len(va),
        best_epoch=ep_best,epochs_trained=ep,
        baseline_val=baseline,selected_val=selected,history=hist,
        atlas=cfg["atlas"],source_model=cfg["source_model"],
        note="Only rank48 SkinTan impedance residual is updated; heat/SOE networks and geometry operators are frozen.")
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V8_TAN_RESULT",json.dumps({k:report[k] for k in
        ("status","best_epoch","baseline_val","selected_val")}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("runs/geomfrac_free_rank48_v8_selection.json"))
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_pose42_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v6_36g_hold6_manifest.json"))
    p.add_argument("--expand-plan",type=Path,default=Path("runs/geomfrac_v8_dyn_expand_plan.json"))
    p.add_argument("--expand-manifest",type=Path,default=Path("runs/geomfrac_s5_v8expand_manifest.json"))
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--model",type=Path,default=Path("runs/geomfrac_et_v8_heat_tune.pt"))
    p.add_argument("--skin",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v3.pt"))
    p.add_argument("--skin-base",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v8_expand.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v8_tangent_expand.json"))
    p.add_argument("--epochs",type=int,default=160)
    p.add_argument("--every",type=int,default=10)
    p.add_argument("--patience",type=int,default=50)
    p.add_argument("--lr",type=float,default=2e-4)
    fit(p.parse_args())


if __name__=="__main__":
    main()
