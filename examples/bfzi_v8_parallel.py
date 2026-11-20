"""Train a compact parallel dynamic-impedance correction on the expanded V8 set."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_cold_v7_parallel import arrays, collect, closed, mean, predict
from examples.bfzi_cold_v7_train import read
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_parallel import ParZ
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_pose42_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v6_36g_hold6_manifest.json"))
    p.add_argument("--expand-plan",type=Path,default=Path("runs/geomfrac_v8_dyn_expand_plan.json"))
    p.add_argument("--expand-manifest",type=Path,default=Path("runs/geomfrac_s5_v8expand_manifest.json"))
    p.add_argument("--config",type=Path,default=Path("runs/geomfrac_free_rank48_v8_selection.json"))
    p.add_argument("--cold",type=Path,default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--model",type=Path,default=Path("runs/geomfrac_et_v8_heat_tune.pt"))
    p.add_argument("--skin-tangent",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v3.pt"))
    p.add_argument("--skin-base",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v8_parallel.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v8_parallel.json"))
    p.add_argument("--epochs",type=int,default=240)
    p.add_argument("--patience",type=int,default=50)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--hidden",type=int,default=32)
    a=p.parse_args()

    torch.manual_seed(72031)
    np.random.seed(72031)
    torch.set_num_threads(2)
    cfg=read(a.config)
    plan=read(a.plan)
    plan["cases"].update(read(a.expand_plan)["cases"])
    mf=read(a.manifest)
    ex=read(a.expand_manifest)
    for key in ("train","val"):
        mf[key].extend(ex.get(key,[]))
    ck=torch.load(a.model,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=int(ck["rank"]),
        soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    net.eval()
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load(a.cold)
    sf=SrcFix.load(cfg["source_model"])
    tan=restore_rank48(a.skin_tangent,a.skin_base)
    env=(cfg,plan,{},ck,net,atlas,cold,sf,tan,None)

    tr,tg=collect(mf["train"],plan,env,gate_only=False)
    va,vg=collect(mf["val"],plan,env,gate_only=False)
    names_tr={n for n,_ in tg}
    names_va={n for n,_ in vg}
    if len(tg)!=48 or len(vg)!=15 or names_tr & names_va:
        raise RuntimeError(f"expected disjoint 48/15 geometries, got {len(tg)}/{len(vg)}")
    gt,at,yt=arrays(tr)
    gm=np.stack([g for _,g in tg])
    xm=gm.mean(0)
    xs=np.maximum(gm.std(0),1e-3)
    asc=np.maximum(np.percentile(np.abs(at[:,:5]),75,axis=0),.5)
    ys=np.maximum(yt.std(0),1e-5)
    model=ParZ(xm,xs,asc,ys,hidden=a.hidden)
    g=torch.as_tensor(gt)
    h=torch.as_tensor(at)
    target=torch.as_tensor(yt)
    scale=model.y_scale
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-3)
    base=predict(model,va)
    best=base
    ep_best=0
    state=copy.deepcopy(model.state_dict())
    hist=[]
    print("V8_PAR_INIT",len(tg),len(vg),len(tr),len(va),base,flush=True)
    for ep in range(1,a.epochs+1):
        model.train()
        for ix in torch.randperm(len(g)).split(a.batch):
            corr=model(g[ix],h[ix])[:,[0,1,4,5]]
            loss=(((corr-target[ix])/scale)**2).mean()
            loss+=1e-3*(corr/scale).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep%5:
            continue
        val=predict(model,va)
        hist.append(dict(epoch=ep,val_dynamic_z=val,
                         train_loss=float(loss.detach())))
        if val<best:
            best=val
            ep_best=ep
            state=copy.deepcopy(model.state_dict())
        if ep%25==0:
            print("V8_PAR_TRAIN",ep,"val",round(val,6),
                  "best",round(best,6),"epoch",ep_best,flush=True)
        if ep-ep_best>=a.patience:
            break

    model.load_state_dict(state)
    model.eval()
    closed_val={}
    for strength in (0.,.25,.5,.75,1.):
        model.strength=strength
        closed_val[str(strength)]=mean(closed(model,va,env))
        print("V8_PAR_CLOSED",strength,
              closed_val[str(strength)]["dynamic_z_rel_L2"],flush=True)
    chosen=min(closed_val,key=lambda k:closed_val[k]["dynamic_z_rel_L2"])
    model.strength=float(chosen)
    model.save(a.out)
    selected=closed_val[chosen]
    report=dict(status="candidate" if model.strength else "not_promoted",
        train_geometries=len(tg),val_geometries=len(vg),
        train_histories=len(tr),val_histories=len(va),
        best_epoch=ep_best,epochs_trained=ep,
        baseline_teacher_forced_dynamic_z=base,
        best_teacher_forced_dynamic_z=best,
        selected_strength=model.strength,closed_val=closed_val,
        selected_val=selected,checkpoint=str(a.out),history=hist,
        atlas=str(a.config),cold=str(a.cold),source_model=cfg["source_model"],
        note="Parallel TX/RX self-impedance residual; main heat and SOE networks frozen.")
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("V8_PAR_RESULT",json.dumps({k:report[k] for k in
        ("status","best_epoch","selected_strength","selected_val")}),flush=True)


if __name__=="__main__":
    main()
