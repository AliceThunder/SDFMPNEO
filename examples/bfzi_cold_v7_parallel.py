"""Train a small V7 parallel self-impedance head on frozen-main-model errors.

Training and validation geometries are disjoint. The six V7 core acceptance
geometries are evaluated only after the validation-selected model is frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_cold_v7_train import read
from examples.bfzi_geom_free_eval import metric
from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_canon import labels
from sdfmpneo_bfzi.geom_cold_domain import check,load
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import geometry,rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_parallel import ParZ
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix

IDX=[0,1,4,5]
KEYS=("dynamic_z_rel_L2","full_z_rel_L2","current_rel_L2",
      "material_mean_rmse_K","material_mean_max_K",
      "sea_power_rel_L2","copper_power_rel_L2","energy_budget_W")


def build():
    cfg=read("runs/geomfrac_free_rank48_selection.json")
    plan=read("runs/geomfrac_pose42_plan.json")
    hold=read("runs/geomfrac_cold_v7_dyn_plan.json")
    ck=torch.load("runs/geomfrac_et_pose42_kell_r48_thermal_v1.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=int(ck["rank"]),
                  soe_quad=ck.get("soe_quad","trap"),
                  heat_dt=float(ck.get("heat_dt",1.875)),
                  heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"]);net.eval()
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load("runs/geomfrac_cold_v7.pt")
    sf=SrcFix.load(cfg["source_model"])
    tan=restore_rank48("runs/geomfrac_rank48_skin_tan_v3.pt",cfg["z_fix_model"])
    dom=load("runs/geomfrac_cold_v7_domain.json")
    return cfg,plan,hold,ck,net,atlas,cold,sf,tan,dom


def collect(specs,plan,env,gate_only=True):
    cfg,_,_,ck,net,atlas,cold,sf,tan,dom=env
    rows=[];geoms=[]
    for spec in specs:
        name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
        case=plan["cases"][name]
        if gate_only and not check(case,dom)["accepted"]:
            continue
        init=sf.apply(case,cold.apply(case,atlas.predict(case)))
        _,g=geometry(case,init)
        geoms.append((name,g))
        for row in prepare(spec,48):
            pred=rollout(case,atlas,net,ck,row["time"],row["voltage"],tan,
                         init_override=init,shape_mix=float(cfg["shape_mix"]))
            dr=row["z6"]-row["z6"][0]
            dz=pred["z6"]-pred["z6"][0]
            rows.append(dict(name=name,case=case,row=row,init=init,g=g,
                             h=pred["heat_state"],dr=dr,dz=dz,
                             target=dr-dz))
    return rows,geoms


def arrays(rows):
    g=np.concatenate([np.broadcast_to(r["g"],(len(r["h"]),45)) for r in rows])
    h=np.concatenate([r["h"] for r in rows])
    y=np.concatenate([r["target"][:,IDX] for r in rows])
    return g.astype(np.float32),h.astype(np.float32),y.astype(np.float32)


@torch.no_grad()
def predict(model,rows):
    scores=[]
    model.eval()
    for r in rows:
        n=len(r["h"])
        g=torch.as_tensor(np.broadcast_to(r["g"],(n,45)).copy(),dtype=torch.float32)
        h=torch.as_tensor(r["h"].copy(),dtype=torch.float32)
        corr=model(g,h).cpu().numpy()
        e=np.linalg.norm(r["dz"]+corr-r["dr"])/max(np.linalg.norm(r["dr"]),1e-12)
        scores.append(float(e))
    return float(np.mean(scores))


def closed(model,rows,env):
    cfg,_,_,ck,net,atlas,_,_,tan,_=env
    out=[]
    for r in rows:
        pred=rollout(r["case"],atlas,net,ck,r["row"]["time"],r["row"]["voltage"],tan,
                     init_override=r["init"],extra=model,shape_mix=float(cfg["shape_mix"]))
        m=metric(pred,r["row"])
        out.append(dict(name=r["name"],**{k:float(m[k]) for k in KEYS}))
    return out


def mean(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in KEYS}


def train(args):
    torch.manual_seed(72031);np.random.seed(72031);torch.set_num_threads(2)
    env=build();cfg,plan,hold,ck,net,atlas,cold,sf,tan,dom=env
    mf=read("runs/geomfrac_s5_v6_36g_hold6_manifest.json")
    tr,tg=collect(mf["train"],plan,env)
    va,vg=collect(mf["val"],plan,env)
    if len(tg)<10 or len(vg)<3 or {n for n,g in tg}&{n for n,g in vg}:
        raise RuntimeError("insufficient disjoint V7 core residual geometries")
    gt,at,yt=arrays(tr)
    gm=np.stack([g for n,g in tg])
    xm=gm.mean(0);xs=np.maximum(gm.std(0),1e-3)
    asc=np.maximum(np.percentile(np.abs(at[:,:5]),75,axis=0),0.5)
    ys=np.maximum(yt.std(0),1e-5)
    model=ParZ(xm,xs,asc,ys,hidden=args.hidden)
    g=torch.as_tensor(gt);a=torch.as_tensor(at);target=torch.as_tensor(yt)
    yscale=model.y_scale
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    base=predict(model,va);best=base;epbest=0
    state=copy.deepcopy(model.state_dict());record=[]
    print("PAR_INIT",len(tg),len(vg),len(tr),len(va),"baseline",base,flush=True)
    for ep in range(1,args.epochs+1):
        model.train()
        for ix in torch.randperm(len(g)).split(args.batch):
            corr=model(g[ix],a[ix])[:,IDX]
            loss=(((corr-target[ix])/yscale)**2).mean()
            loss+=1e-3*(corr/yscale).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if ep%5:continue
        val=predict(model,va)
        record.append(dict(epoch=ep,val_dynamic_z=val,train_loss=float(loss.detach())))
        if val<best:
            best=val;epbest=ep;state=copy.deepcopy(model.state_dict())
        if ep%25==0:
            print("PAR_TRAIN",ep,"val",round(val,5),"best",round(best,5),"epoch",epbest,flush=True)
        if ep-epbest>=args.patience:break
    model.load_state_dict(state);model.eval()

    # Select the correction strength only from development geometries.
    scales={}
    for alpha in (0.,0.5,1.):
        model.strength=alpha
        rows=closed(model,va,env)
        scales[str(alpha)]=mean(rows)
        print("PAR_VAL_CLOSED",alpha,scales[str(alpha)]["dynamic_z_rel_L2"],flush=True)
    chosen=min(scales,key=lambda k:scales[k]["dynamic_z_rel_L2"])
    model.strength=float(chosen)
    model.save(args.out)

    mf_test=read("runs/geomfrac_s5_cold_v7_core_manifest.json")
    te,ge=collect(mf_test["test"],hold,env)
    if len(ge)!=6 or len(te)!=12 or {n for n,g in ge}&({n for n,g in tg}|{n for n,g in vg}):
        raise RuntimeError("independent V7 six-core holdout partition failed")
    final=closed(model,te,env)
    by={}
    for name,g in ge:
        by[name]=mean([x for x in final if x["name"]==name])
    summary=mean(final)
    print("PAR_TEST",json.dumps(summary),flush=True)
    result=dict(status="candidate" if model.strength else "not_promoted",
        train_geometries=len(tg),val_geometries=len(vg),test_geometries=len(ge),
        train_histories=len(tr),val_histories=len(va),test_histories=len(te),
        best_epoch=epbest,epochs_trained=ep,
        baseline_val_dynamic_z=base,best_val_dynamic_z=best,
        selected_strength=model.strength,closed_val=scales,
        test=summary,test_by_geometry=by,
        checkpoint=str(args.out),history=record,
        note="All main networks frozen; six unseen core geometries used only once after validation selection.")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("PAR_RESULT",json.dumps({k:result[k] for k in
        ("best_epoch","selected_strength","test")}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v7_parallel.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v7_parallel.json"))
    p.add_argument("--epochs",type=int,default=240)
    p.add_argument("--patience",type=int,default=50)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--hidden",type=int,default=32)
    args=p.parse_args()
    train(args)


if __name__=="__main__":main()
